"""The operating envelope: the one number the rest of the system derives from.

Range limits had accumulated in half a dozen places -- a tracker `max_range`, a
size-prior cap, a detector cutoff, a birth prior and its sigma -- each tuned by
hand for a different assumption. Changing the system's declared reach then meant
finding all of them, and any one missed silently kept the old behaviour.

They are all derived here instead. Declaring a 50 m envelope should tighten the
birth prior, the association gate, the degeneracy guard and the detector cutoff
together, because they are all statements about the same thing.

The derivations are deliberately simple and each is justified:

``prior_range``      Mid-envelope. With no other information an object is as
                     likely to be near as far, and the midpoint minimises the
                     worst-case error of the initial guess.
``init_range_sigma`` Wide enough that the true range is inside the prior
                     anywhere in the envelope, and no wider. This is the number
                     that decides how permissive a new track's association gate
                     is, and an over-wide prior is what lets one track swallow
                     observations from several objects.
``max_range``        Envelope plus headroom. A track that drifts past this is not
                     a long-range fix, it is a broken one, and is flagged rather
                     than reported as a number.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_MAX_RANGE_M = 50.0
"""Declared operating reach. Change this, not the derived values."""

HEADROOM = 1.5
"""How far past the envelope an estimate may drift before it is called broken."""


@dataclass(frozen=True)
class OperatingEnvelope:
    max_range_m: float = DEFAULT_MAX_RANGE_M
    min_range_m: float = 2.0

    def __post_init__(self) -> None:
        if self.max_range_m <= self.min_range_m:
            raise ValueError("max_range_m must exceed min_range_m")

    @property
    def prior_range(self) -> float:
        """Mid-envelope: the least-bad guess when nothing else is known."""
        return 0.5 * (self.min_range_m + self.max_range_m)

    @property
    def init_range_sigma(self) -> float:
        """Covers the envelope at ~2 sigma without being wider than it needs to be.

        A uniform distribution over [min, max] has standard deviation
        ``(max - min) / sqrt(12)``; this is deliberately a little wider, so the
        true range is comfortably inside the prior, and no more.
        """
        return max((self.max_range_m - self.min_range_m) / 2.5, 1.0)

    @property
    def track_max_range(self) -> float:
        return self.max_range_m * HEADROOM

    @property
    def size_prior_max(self) -> float:
        return self.max_range_m * HEADROOM

    @property
    def detector_max_range(self) -> float:
        return self.max_range_m

    def describe(self) -> str:
        return (
            f"operating envelope {self.min_range_m:.0f}-{self.max_range_m:.0f} m: "
            f"birth prior {self.prior_range:.0f} +/- {self.init_range_sigma:.0f} m, "
            f"estimates flagged beyond {self.track_max_range:.0f} m"
        )


DEFAULT_ENVELOPE = OperatingEnvelope()
