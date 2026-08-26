"""Track management: birth, association, update, death.

This is the module the acceptance criterion protects: nothing in here may know
whether the frames came from numpy, nuScenes, or an iPhone. It consumes
Observations and emits TrackStates.

Association is global (Hungarian) rather than greedy nearest-neighbour, gated by
Mahalanobis distance in *measurement* space. Measurement space is the right place
to gate: a track 200 m away with a huge along-range covariance should still be
tightly gated in bearing, and a position-space gate would wrongly accept
everything near it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from geoloc_agent.contracts import Observation, Pose, TrackState, TrackStatus
from geoloc_agent.envelope import DEFAULT_ENVELOPE
from geoloc_agent.fuse.degenerate import (
    DEFAULT_MAX_RELATIVE_RANGE_SIGMA,
    DEFAULT_MIN_OBS,
    DEFAULT_MIN_PARALLAX_DEG,
    DEFAULT_MIN_PERP_BASELINE_M,
    assess_geometry,
)
from geoloc_agent.fuse.ekf import PositionEKF, initial_state
from geoloc_agent.range.triangulation import triangulate

GATE_INFEASIBLE = 1e6


@dataclass
class TrackerConfig:
    """Range defaults are derived from the declared operating envelope.

    ``prior_range`` / ``init_range_sigma`` describe the *sensor's* usable range
    interval, not a guess about any particular object. Before a track localises,
    that prior is used only for gating, never reported as a fix -- but its width
    directly sets how permissive a new track's gate is, so it is derived from the
    envelope rather than picked. See ``geoloc_agent.envelope``.
    """

    prior_range: float = DEFAULT_ENVELOPE.prior_range
    init_range_sigma: float = DEFAULT_ENVELOPE.init_range_sigma
    # Zero by default: these tracks model static objects, and process noise on a
    # static object is pure covariance inflation -- it drives NEES below the
    # chi-square band, which is the filter throwing away information. Raise it
    # for genuinely moving objects.
    process_noise_per_s: float = 0.0
    gate_chi2: float = 9.21  # 2 dof, 99%
    confirm_after: int = 3
    max_misses: int = 5
    min_parallax_deg: float = DEFAULT_MIN_PARALLAX_DEG
    max_relative_range_sigma: float = DEFAULT_MAX_RELATIVE_RANGE_SIGMA
    min_obs_for_confidence: int = DEFAULT_MIN_OBS
    min_perp_baseline: float = DEFAULT_MIN_PERP_BASELINE_M
    class_floor: float = 1e-3
    known_classes: tuple[str, ...] = ("car", "pedestrian", "truck", "clutter", "unknown")
    localize_min_obs: int = 2
    max_range: float = DEFAULT_ENVELOPE.track_max_range
    max_position_sigma: float = 200.0
    # Every observation. The batch solve is what makes the covariance match the
    # estimator's real spread; refining less often lets sequential-EKF optimism
    # accumulate between refinements (NEES ~3.5 at every=5 versus ~3.3 at 1).
    # Raise it to trade calibration for compute on large track counts.
    hint_gate_scale: float = 6.0
    """How far an upstream tracker's identity claim may widen the geometric gate.

    An image-plane tracker answering "same pixels" is evidence independent of the
    filter's "same place", so agreement between them justifies a looser threshold
    than geometry alone would allow. That is ordinary evidence combination, not a
    shortcut.

    It matters most where the geometric gate is least reliable: a fast platform
    over small static objects moves a track's predicted reprojection a long way
    between frames, and the gate then rejects correct associations. On the aerial
    flight that produced one track per detection.

    It widens the gate; it never opens it. A hinted pair still has to be
    geometrically possible, so an upstream tracker that swaps two identities
    cannot force a match that the geometry refuses -- the failure stays
    recoverable, which is the same reason ``class_mismatch_penalty`` is a
    penalty rather than a veto."""

    hint_bonus: float = 4.0
    """Cost subtracted when a hint agrees, so the assignment prefers it among
    otherwise comparable candidates. Bounded below at zero so a hint can never
    make a match look better than a perfect geometric fit."""

    class_mismatch_penalty: float = 9.21
    """Cost added when a detection's class contradicts a track's confident class.

    A PENALTY, not a veto. Refusing the association outright makes the class
    posterior self-reinforcing: a track that has drifted to "car" would reject
    every pedestrian observation and could never discover it was wrong. That is
    confirmation bias built into the tracker, and it would also quietly destroy
    the ambiguous-class cases the agent layer exists to escalate.

    As a penalty it does the useful half of the job and none of the harmful half.
    In a crowded scene a same-class detection is available and wins on cost; when
    the contradicting detection is the only candidate it is still associated, and
    the posterior updates honestly on evidence that disagrees."""

    epipolar_gating: bool = True
    """Gate unlocalised tracks on the epipolar constraint instead of Mahalanobis.

    An unlocalised track carries a deliberately wide range prior, which makes its
    Mahalanobis gate enormous and lets it swallow any nearby detection. On a busy
    real scene that is the dominant failure: one track absorbed observations from
    14 distinct objects and reported 500 m for something 130 m away. But an
    unlocalised track is not actually ignorant -- it knows its bearing ray
    exactly. Requiring the new bearing to be consistent with SOME range along
    that ray is a one-dimensional constraint instead of a loose two-dimensional
    one, and it uses precisely the information the track has."""

    batch_refine_every: int = 1
    batch_window: int = 60


@dataclass
class Track:
    """Internal bookkeeping. The externally visible form is TrackState."""

    track_id: int
    ekf: PositionEKF
    state: TrackState
    origins: list[np.ndarray] = field(default_factory=list)
    bearings: list[np.ndarray] = field(default_factory=list)
    observations: list[Observation] = field(default_factory=list)
    truth_votes: dict[str, int] = field(default_factory=dict)
    localized: bool = False


class Tracker:
    """Multi-object tracker over static geolocated points."""

    def __init__(self, config: TrackerConfig | None = None) -> None:
        self.config = config or TrackerConfig()
        self.tracks: dict[int, Track] = {}
        self.completed: list[Track] = []
        """Tracks retained at death.

        On synthetic runs objects stay in view and the live set is the whole
        story. On real driving data objects sweep through the field of view, so
        scoring only the survivors at the last frame throws away almost every
        track the system produced -- 3 of 40-odd on the first nuScenes scene.
        """
        self._next_id = 1
        self._last_t: float | None = None
        self._hint_owner: dict[int, int] = {}
        """Upstream tracker id -> the track that currently answers to it.

        Learned rather than declared: the first time a hinted observation is
        matched or births a track, that pairing is recorded. Nothing forces the
        two id spaces to agree, so if the upstream tracker loses and re-acquires
        an object under a new id, this simply stops helping for a frame instead
        of asserting a wrong identity."""

    # -- main entry point -------------------------------------------------

    def step(
        self, observations: Sequence[Observation], t: float, pose: Pose | None = None
    ) -> list[TrackState]:
        """Advance the filter one frame and return the live tracks."""
        dt = 0.0 if self._last_t is None else max(t - self._last_t, 0.0)
        self._last_t = t

        for track in self.tracks.values():
            track.ekf.predict(dt)
            track.state.age = t - track.state.first_t

        matches, unmatched_obs = self._associate(observations, pose)

        matched_ids = set()
        for track_id, obs_index in matches:
            observation = observations[obs_index]
            self._update_track(self.tracks[track_id], observation, t, pose)
            matched_ids.add(track_id)
            if observation.track_hint is not None:
                self._hint_owner[observation.track_hint] = track_id

        for track_id, track in self.tracks.items():
            if track_id not in matched_ids:
                track.state.misses += 1
                if track.state.status is TrackStatus.CONFIRMED:
                    track.state.status = TrackStatus.COASTING

        for obs_index in unmatched_obs:
            observation = observations[obs_index]
            born = self._birth(observation, t)
            if observation.track_hint is not None:
                self._hint_owner[observation.track_hint] = born.state.track_id

        self._reap()
        return self.live_states()

    def live_states(self) -> list[TrackState]:
        return [t.state.copy() for t in self.tracks.values()]

    def all_tracks(self) -> list[Track]:
        """Every track this run produced, live and completed."""
        return [*self.tracks.values(), *self.completed]

    def all_states(self) -> list[TrackState]:
        return [t.state.copy() for t in self.all_tracks()]

    def confirmed_states(self) -> list[TrackState]:
        return [
            t.state.copy()
            for t in self.tracks.values()
            if t.state.status in (TrackStatus.CONFIRMED, TrackStatus.COASTING)
        ]

    # -- association ------------------------------------------------------

    def _epipolar_cost(self, track: Track, obs: Observation) -> float:
        """Squared normalised angular distance from the track's epipolar plane.

        For a track whose first bearing is ``b0`` from centre ``o0``, every point
        it could possibly be lies on that ray, so any later bearing from ``o1``
        must lie in the plane spanned by ``b0`` and the baseline ``o1 - o0``.
        The out-of-plane angle is therefore pure inconsistency, and it is a chi-
        square(1) statistic once divided by the bearing sigma.
        """
        baseline = np.asarray(obs.origin, dtype=float) - track.origins[0]
        if np.linalg.norm(baseline) < 1e-6:
            return 0.0  # no baseline yet: nothing to contradict
        normal = np.cross(track.bearings[0], baseline)
        norm = float(np.linalg.norm(normal))
        if norm < 1e-9:
            return 0.0  # motion exactly along the ray: no epipolar information
        normal = normal / norm
        out_of_plane = float(np.arcsin(np.clip(abs(normal @ obs.bearing), -1.0, 1.0)))
        sigma = max(obs.bearing_sigma, 1e-9)
        return float((out_of_plane / sigma) ** 2)

    def _class_penalty(self, track: Track, obs: Observation) -> float:
        cls, confidence = track.state.top_class
        if cls in ("unknown", "") or obs.cls in ("unknown", ""):
            return 0.0
        if cls == obs.cls or confidence <= 0.6:
            return 0.0
        return self.config.class_mismatch_penalty

    def _associate(
        self, observations: Sequence[Observation], pose: Pose | None
    ) -> tuple[list[tuple[int, int]], list[int]]:
        if not observations:
            return [], []
        if not self.tracks:
            return [], list(range(len(observations)))

        track_ids = list(self.tracks)
        cost = np.full((len(track_ids), len(observations)), GATE_INFEASIBLE)
        for i, track_id in enumerate(track_ids):
            track = self.tracks[track_id]
            ekf = track.ekf
            check_epipolar = self.config.epipolar_gating and not track.localized
            for j, obs in enumerate(observations):
                hinted = (
                    obs.track_hint is not None
                    and self._hint_owner.get(obs.track_hint) == track_id
                )
                gate = self.config.gate_chi2 * (self.config.hint_gate_scale if hinted else 1.0)
                # Both constraints must hold. They are independent: Mahalanobis
                # asks "could the object be there given what I believe", the
                # epipolar residual asks "is this bearing consistent with my ray
                # at ANY range". Requiring both is strictly tighter than either,
                # and each covers the other's blind spot -- Mahalanobis is
                # toothless while the range prior is wide, and the epipolar test
                # says nothing about where along the ray the object sits.
                distance = ekf.mahalanobis(obs, pose)
                if distance > gate:
                    continue
                if check_epipolar and self._epipolar_cost(track, obs) > gate:
                    continue
                # Gating is decided on the raw distance; the class penalty and
                # the hint bonus only rank the feasible options.
                score = distance + self._class_penalty(track, obs)
                if hinted:
                    score = max(0.0, score - self.config.hint_bonus)
                cost[i, j] = score

        rows, cols = linear_sum_assignment(cost)
        matches = [
            (track_ids[i], int(j))
            for i, j in zip(rows, cols, strict=True)
            if cost[i, j] < GATE_INFEASIBLE
        ]
        matched_obs = {j for _, j in matches}
        unmatched = [j for j in range(len(observations)) if j not in matched_obs]
        return matches, unmatched

    # -- birth / update / death -------------------------------------------

    def _birth(self, obs: Observation, t: float) -> Track:
        mean, cov = initial_state(
            obs, prior_range=self.config.prior_range, range_sigma=self.config.init_range_sigma
        )
        ekf = PositionEKF(
            mean,
            cov,
            process_noise_per_s=self.config.process_noise_per_s,
            max_mahalanobis=self.config.gate_chi2,
        )
        track_id = self._next_id
        self._next_id += 1
        state = TrackState(
            track_id=track_id,
            mean=mean,
            cov=cov,
            class_posterior=self._uniform_posterior(),
            n_obs=1,
            first_t=t,
            last_t=t,
            status=TrackStatus.TENTATIVE,
        )
        track = Track(track_id=track_id, ekf=ekf, state=state, localized=obs.has_range)
        track.origins.append(np.asarray(obs.origin, dtype=float))
        track.bearings.append(np.asarray(obs.bearing, dtype=float))
        track.observations.append(obs)
        self._update_class(track, obs)
        self._refresh_geometry(track)
        if obs.truth_id:
            track.truth_votes[obs.truth_id] = track.truth_votes.get(obs.truth_id, 0) + 1
            track.state.truth_id = obs.truth_id
        self.tracks[track_id] = track
        return track

    def _update_track(self, track: Track, obs: Observation, t: float, pose: Pose | None) -> None:
        # Gate before admitting the observation to the history. An observation
        # the filter rejects must not reach the batch solve either, or the two
        # estimators disagree about which measurements the track is built from.
        if track.localized and not obs.has_range:
            if not track.ekf.update_bearing(obs, pose):
                track.state.misses += 1
                return
            self._record(track, obs)
            self._finish_update(track, obs, t)
            return

        self._record(track, obs)
        if not track.localized:
            # Still unlocalised: accumulate bearings and try to get a fix whose
            # covariance is trustworthy, rather than folding this bearing into a
            # prior we know is wrong.
            self._try_localize(track)
            if not track.localized and obs.has_range:
                # A direct range measurement localises a track on its own.
                track.ekf.update_range(obs)
                track.localized = True
        else:
            track.ekf.update_bearing(obs, pose)
            if obs.has_range:
                track.ekf.update_range(obs)
        self._finish_update(track, obs, t)

    def _record(self, track: Track, obs: Observation) -> None:
        track.origins.append(np.asarray(obs.origin, dtype=float))
        track.bearings.append(np.asarray(obs.bearing, dtype=float))
        track.observations.append(obs)

    def _finish_update(self, track: Track, obs: Observation, t: float) -> None:
        state = track.state
        state.n_obs += 1
        self._maybe_batch_refine(track)
        state.mean = track.ekf.mean.copy()
        state.cov = track.ekf.cov.copy()
        state.last_t = t
        state.age = t - state.first_t
        state.misses = 0
        if state.n_obs >= self.config.confirm_after:
            state.status = TrackStatus.CONFIRMED
        self._update_class(track, obs)
        self._refresh_geometry(track)
        if obs.truth_id:
            track.truth_votes[obs.truth_id] = track.truth_votes.get(obs.truth_id, 0) + 1
            state.truth_id = max(track.truth_votes, key=track.truth_votes.get)

    def _try_localize(self, track: Track) -> None:
        """Promote an unlocalised track to a real fix once the geometry allows it.

        Triangulation is used rather than folding bearings into the wide birth
        prior, because its information matrix *is* the covariance: at low
        parallax it inflates along the line of sight instead of collapsing, so a
        track only claims a tight fix when the geometry actually earned one.
        """
        if len(track.observations) < self.config.localize_min_obs:
            return
        result = triangulate(track.observations, prior_range=self.config.prior_range)
        if not result.ok:
            return
        range_m = float(np.linalg.norm(result.position - track.origins[-1]))
        if not np.isfinite(range_m) or range_m > self.config.max_range:
            return
        cov = self._clamp_cov(result.cov)
        if not np.all(np.isfinite(cov)):
            return
        track.ekf.mean = result.position.copy()
        track.ekf.cov = cov
        track.localized = True

    def _maybe_batch_refine(self, track: Track) -> None:
        """Periodically re-solve the whole track from all its bearings.

        Sequential EKF updates are mildly optimistic on bearings-only geometry:
        each update linearises about an estimate that already depends on the
        earlier measurements, so the information sum double-counts a little and
        the covariance comes out roughly a third too small. Re-solving in batch
        removes that -- it is the same relationship a local bundle adjustment has
        to a streaming pose filter, and it belongs on the slower ranging loop for
        the same reason.

        Skipped when the track carries direct range measurements: those remove
        the nonlinearity that causes the problem, and the batch solve here is
        bearings-only so it would throw them away.
        """
        every = self.config.batch_refine_every
        if every <= 0 or track.state.n_obs % every != 0:
            return
        if any(o.has_range for o in track.observations):
            return
        window = track.observations[-self.config.batch_window :]
        if len(window) < 2:
            return
        result = triangulate(window, prior_range=self.config.prior_range)
        if not result.ok:
            return
        range_m = float(np.linalg.norm(result.position - track.origins[-1]))
        if not np.isfinite(range_m) or range_m > self.config.max_range:
            return
        track.ekf.mean = result.position.copy()
        track.ekf.cov = self._clamp_cov(result.cov)

    def _clamp_cov(self, cov: np.ndarray) -> np.ndarray:
        """Cap the covariance so an unobservable direction cannot blow up the gate.

        The cap is stated as a limit of the sensor model, not as knowledge: a
        track sitting at the cap is reporting "range unobservable", and the
        degeneracy flag says so alongside it.
        """
        cov = 0.5 * (np.asarray(cov, dtype=float) + np.asarray(cov, dtype=float).T)
        eigenvalues, vectors = np.linalg.eigh(cov)
        limit = self.config.max_position_sigma**2
        eigenvalues = np.clip(eigenvalues, 1e-9, limit)
        return vectors @ np.diag(eigenvalues) @ vectors.T

    def _reap(self) -> None:
        dead = [
            track_id
            for track_id, track in self.tracks.items()
            if track.state.misses > self.config.max_misses
        ]
        for track_id in dead:
            track = self.tracks[track_id]
            track.state.status = TrackStatus.DEAD
            self.completed.append(track)
            del self.tracks[track_id]

    # -- attributes -------------------------------------------------------

    def _uniform_posterior(self) -> dict[str, float]:
        classes = self.config.known_classes
        return dict.fromkeys(classes, 1.0 / len(classes))

    def _update_class(self, track: Track, obs: Observation) -> None:
        """Bayesian update with a score-driven confusion model.

        A detector reporting "car, 0.9" is evidence for car and weak evidence
        spread over everything else. The floor stops a single confident
        observation from collapsing the posterior to a delta, which would make
        the entropy signal useless to the agent later.
        """
        posterior = track.state.class_posterior or self._uniform_posterior()
        classes = list(posterior)
        score = float(np.clip(obs.score, 0.05, 0.95))
        others = max(len(classes) - 1, 1)
        updated = {}
        for cls in classes:
            likelihood = score if cls == obs.cls else (1.0 - score) / others
            updated[cls] = posterior[cls] * likelihood
        total = sum(updated.values())
        if total <= 0:
            track.state.class_posterior = self._uniform_posterior()
            return
        floor = self.config.class_floor
        normalised = {c: max(p / total, floor) for c, p in updated.items()}
        renorm = sum(normalised.values())
        track.state.class_posterior = {c: p / renorm for c, p in normalised.items()}

    def _refresh_geometry(self, track: Track) -> None:
        report = assess_geometry(
            track.origins,
            track.state.mean,
            track.state.cov,
            min_parallax_deg=self.config.min_parallax_deg,
            max_relative_range_sigma=self.config.max_relative_range_sigma,
            n_obs=track.state.n_obs,
            min_obs=self.config.min_obs_for_confidence,
            min_perp_baseline=self.config.min_perp_baseline,
            bearing_sigma=(
                float(np.mean([o.bearing_sigma for o in track.observations]))
                if track.observations
                else None
            ),
        )
        degenerate = report.degenerate
        reason = report.reason
        # Sensor envelope. An estimate that has drifted past the range at which
        # this sensor could resolve the object at all is not a long-range fix, it
        # is a broken one, and it must not be reported as a number.
        if report.range_m > self.config.max_range:
            degenerate = True
            envelope = f"estimate at {report.range_m:.0f} m is beyond the sensor envelope"
            reason = f"{reason}; {envelope}" if reason else envelope
        track.state.degenerate = degenerate
        track.state.degeneracy_reason = reason
        track.state.max_perp_baseline = report.perp_baseline
