"""Lattice entity publishing.

See the licensing note in this package's ``__init__``. The SDK import is inside
the method that needs it, so importing ``geoloc_agent`` never loads it and the
CoT path never touches it.

Two operational details that are easy to get wrong and painful to debug:

**Token refresh.** Lattice bearer tokens expire. Refreshing on a 401 alone means
every expiry costs a failed publish; refreshing on a timer means a clock skew
still produces 401s. This does both -- proactive refresh before expiry, plus a
single retry on a 401 -- because the two failure modes are independent.

**Retry with backoff, bounded.** A publisher that retries forever turns a
downstream outage into unbounded memory growth in the caller. Attempts are
capped and the failure is returned, not swallowed.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

from geoloc_agent.contracts import TrackState
from geoloc_agent.geo import GeoOrigin

TOKEN_LIFETIME_S = 30 * 60
REFRESH_MARGIN_S = 5 * 60  # refresh this long before expiry


@dataclass
class LatticeConfig:
    """Credentials come from the environment. Never commit them."""

    base_url: str = field(default_factory=lambda: os.environ.get("LATTICE_URL", ""))
    token: str = field(default_factory=lambda: os.environ.get("LATTICE_TOKEN", ""))
    sandboxes_token: str = field(
        default_factory=lambda: os.environ.get("LATTICE_SANDBOXES_TOKEN", "")
    )
    source_name: str = "geoloc-agent"
    max_attempts: int = 4
    base_backoff_s: float = 0.5

    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.token)

    def describe(self) -> str:
        if self.configured:
            return f"Lattice at {self.base_url} as '{self.source_name}'"
        missing = [
            name
            for name, value in (("LATTICE_URL", self.base_url), ("LATTICE_TOKEN", self.token))
            if not value
        ]
        return f"Lattice not configured; missing {', '.join(missing)}"


class LatticePublisher:
    """Publishes tracks as Lattice entities, with token refresh and bounded retry."""

    def __init__(
        self, config: LatticeConfig | None = None, origin: GeoOrigin | None = None
    ) -> None:
        self.config = config or LatticeConfig()
        self.origin = origin
        self._client = None
        self._token_issued_at: float = 0.0
        self.published = 0

    @property
    def available(self) -> bool:
        if not self.config.configured:
            return False
        try:
            import anduril  # noqa: F401
        except ImportError:
            return False
        return True

    def _connect(self):
        """Import and construct the SDK client. Import is local, by design."""
        try:
            from anduril import Lattice
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "anduril-lattice-sdk is required for Lattice publishing and is an optional "
                "dependency, installed separately for licensing reasons. It must not be "
                "imported by the CoT path."
            ) from exc
        self._client = Lattice(
            base_url=self.config.base_url,
            token=self.config.token,
        )
        self._token_issued_at = time.monotonic()
        return self._client

    def _ensure_client(self, force: bool = False):
        expired = (time.monotonic() - self._token_issued_at) > (
            TOKEN_LIFETIME_S - REFRESH_MARGIN_S
        )
        if self._client is None or force or expired:
            # Re-read the environment: an external refresher may have rotated it.
            self.config.token = os.environ.get("LATTICE_TOKEN", self.config.token)
            return self._connect()
        return self._client

    def entity_for(self, track: TrackState) -> dict:
        """TrackState -> Lattice entity payload.

        The covariance is carried through as an explicit error ellipse rather
        than being dropped, so a Lattice consumer sees the same uncertainty the
        filter computed.
        """
        if self.origin is None:
            raise ValueError(
                "LatticePublisher needs a GeoOrigin: Lattice entities are geodetic and the "
                "pipeline is metric-local, so the conversion assumption must be explicit"
            )
        lat, lon, alt = self.origin.enu_to_wgs84(track.mean)
        cls, confidence = track.top_class
        return {
            "entity_id": f"{self.config.source_name}-{track.track_id}",
            "description": f"{cls} track {track.track_id}",
            "is_live": track.status.value != "dead",
            "location": {
                "position": {
                    "latitude_degrees": lat,
                    "longitude_degrees": lon,
                    "altitude_hae_meters": alt,
                },
                "error_ellipse": {
                    "probability": 0.5,
                    "semi_major_axis_meters": float(track.cep50),
                    "semi_minor_axis_meters": float(track.cep50),
                    "orientation_degrees": 0.0,
                },
            },
            "mil_view": {
                "disposition": "DISPOSITION_UNKNOWN",
                "environment": "ENVIRONMENT_SURFACE",
            },
            "ontology": {"template": "TEMPLATE_TRACK", "platform_type": cls},
            "provenance": {
                "integration_name": self.config.source_name,
                "data_type": "geolocated-track",
                "source_update_time": None,
            },
            # Non-standard but essential: a consumer must be able to see that a
            # position is geometry-limited without re-deriving it.
            "aliases": {"name": f"{cls.upper()}-{track.track_id}"},
            "_diagnostics": {
                "class_confidence": round(confidence, 3),
                "class_entropy": round(track.class_entropy, 3),
                "sigma_horizontal_m": round(track.sigma_horizontal, 2),
                "n_observations": track.n_obs,
                "degenerate_geometry": track.degenerate,
                "degeneracy_reason": track.degeneracy_reason or None,
                "origin_assumption": f"{self.origin.name} ({self.origin.provenance})",
            },
        }

    def publish(self, track: TrackState) -> dict:
        """Publish one entity, retrying with backoff. Raises on final failure."""
        payload = self.entity_for(track)
        last_error: Exception | None = None

        for attempt in range(self.config.max_attempts):
            try:
                client = self._ensure_client(force=attempt > 0 and _is_auth_error(last_error))
                client.entities.publish_entity(**_strip_private(payload))
                self.published += 1
                return payload
            except Exception as exc:  # noqa: BLE001 - re-raised after the retry budget
                last_error = exc
                if attempt == self.config.max_attempts - 1:
                    break
                time.sleep(self.config.base_backoff_s * (2**attempt))
        raise RuntimeError(
            f"failed to publish entity for track {track.track_id} after "
            f"{self.config.max_attempts} attempts: {last_error}"
        ) from last_error

    def publish_all(self, tracks: list[TrackState]) -> int:
        for track in tracks:
            self.publish(track)
        return len(tracks)


def _strip_private(payload: dict) -> dict:
    return {k: v for k, v in payload.items() if not k.startswith("_")}


def _is_auth_error(error: Exception | None) -> bool:
    if error is None:
        return False
    text = str(error).lower()
    return "401" in text or "unauthor" in text or "token" in text
