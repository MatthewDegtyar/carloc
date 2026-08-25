"""End-to-end pipeline: session -> detections -> observations -> tracks.

The three loops from the design rules are explicit here rather than implied.
Perception runs every frame, ranging runs on a slower cadence, and the decision
layer slower still. In this replay harness they are cadences rather than threads,
but the *structure* is the point: no stage reaches into another, and moving the
slow loops onto their own threads later is a change to the runner, not to any of
the stages. Retrofitting that decoupling is the thing that is miserable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from geoloc_agent.contracts import Detection, Frame, Observation, TrackState
from geoloc_agent.detect.base import Detector
from geoloc_agent.detect.stub import StubDetector
from geoloc_agent.fuse.tracker import Tracker, TrackerConfig
from geoloc_agent.geometry import observation_from_detection
from geoloc_agent.io.base import Session
from geoloc_agent.noise import DetectionNoiseInjector, NoiseModel, PoseNoiseInjector
from geoloc_agent.range.base import Ranger

TRUTH_MATCH_PIXELS = 60.0


@dataclass
class FrameRecord:
    """Per-frame snapshot, kept so any stage can be scored in isolation.

    Also carries what a visualiser needs -- the detections as the tracker saw
    them and the (noisy) pose it used -- so a rendered frame shows what the
    system actually believed at that instant rather than a re-derivation of it.
    """

    frame_id: int
    t: float
    n_detections: int
    n_observations: int
    track_states: list[TrackState] = field(default_factory=list)
    truth_positions: dict[str, np.ndarray] = field(default_factory=dict)
    detections: list[Detection] = field(default_factory=list)
    frame: Frame | None = None


@dataclass
class TrackRecord:
    """Everything about a finished track that scoring needs.

    Carried alongside the estimate rather than reconstructed afterwards, so that
    purity, fragmentation and truth-referenced geometry can all be measured
    without the metrics code having to re-do the association.
    """

    track_id: int
    origins: list[np.ndarray] = field(default_factory=list)
    truth_ids: list[str | None] = field(default_factory=list)
    times: list[float] = field(default_factory=list)
    bearing_sigmas: list[float] = field(default_factory=list)
    first_t: float = 0.0
    sigma_history: list[tuple[float, float]] = field(default_factory=list)


@dataclass
class PipelineResult:
    session_name: str
    noise: NoiseModel
    frames: list[FrameRecord]
    final_tracks: list[TrackState]
    truth: dict[str, np.ndarray]
    truth_classes: dict[str, str] = field(default_factory=dict)
    n_detections: int = 0
    n_false_positives: int = 0
    track_records: dict[int, TrackRecord] = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return len(self.frames)


def _match_truth(
    detection: Detection, frame: Frame, truth_uv: dict[str, tuple[float, float]]
) -> str | None:
    """Ground-truth data association, for scoring only.

    Matches a detection to the truth object whose projection it is closest to.
    The pipeline never uses this to make estimation decisions -- it exists so
    track purity and fragmentation can be measured, and so a false positive can
    be recognised as one.
    """
    u, v = detection.centroid
    best_id, best_distance = None, TRUTH_MATCH_PIXELS
    for obj_id, (tu, tv) in truth_uv.items():
        distance = float(np.hypot(u - tu, v - tv))
        if distance < best_distance:
            best_id, best_distance = obj_id, distance
    return best_id


def _project_truth(frame: Frame, truth: dict[str, np.ndarray]) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    for obj_id, position in truth.items():
        p_cam = frame.pose.world_to_cam(position)
        if p_cam[2] < 0.5:
            continue
        uv = frame.intrinsics.K @ p_cam
        out[obj_id] = (float(uv[0] / uv[2]), float(uv[1] / uv[2]))
    return out


def run_pipeline(
    session: Session,
    detector: Detector | None = None,
    tracker_config: TrackerConfig | None = None,
    noise: NoiseModel | None = None,
    seed: int = 0,
    bearing_sigma_px: float = 2.0,
    ranger: Ranger | None = None,
    range_every_n: int = 3,
    record_every_n: int = 1,
) -> PipelineResult:
    """Replay a session through the full stack and record everything scoreable.

    ``bearing_sigma_px`` is what the filter is *told* the centroid sigma is. The
    noise model controls what is actually injected. Deliberately separating the
    two is what makes covariance calibration measurable: set them equal and NEES
    should sit inside the chi-square band; lie to the filter and it should not.
    """
    noise = noise or NoiseModel()
    rng = np.random.default_rng(seed)
    pose_noise = PoseNoiseInjector(noise, rng)
    detection_noise = DetectionNoiseInjector(noise, rng)
    detector = detector or StubDetector.from_session(session)
    tracker = Tracker(tracker_config or TrackerConfig())

    truth_objects = session.truth()
    truth_static = {oid: obj.position for oid, obj in truth_objects.items()}
    truth_classes = {oid: obj.cls for oid, obj in truth_objects.items()}

    records: list[FrameRecord] = []
    total_detections = 0
    total_false_positives = 0
    # Per-track observation history, so the ranger has something to triangulate.
    history: dict[int, list[Observation]] = {}
    sigma_history: dict[int, list[tuple[float, float]]] = {}

    for index, clean_frame in enumerate(session.frames()):
        # --- fast loop: perception -------------------------------------
        detections = detector.detect(clean_frame)

        # Noise is injected against the *clean* frame, then the frame itself is
        # corrupted. Order matters: a detector sees the real world, not the
        # world as the (wrong) pose believes it to be.
        truth_now = session.truth_at(clean_frame.frame_id) or truth_static
        truth_uv = _project_truth(clean_frame, truth_now)
        detections = detection_noise.apply(detections, clean_frame)
        frame = pose_noise.apply(clean_frame)

        observations: list[Observation] = []
        for detection in detections:
            truth_id = _match_truth(detection, clean_frame, truth_uv)
            if truth_id is None:
                total_false_positives += 1
            observations.append(
                observation_from_detection(
                    detection,
                    frame,
                    bearing_sigma_px=bearing_sigma_px,
                    truth_position=truth_now.get(truth_id) if truth_id else None,
                    truth_id=truth_id,
                )
            )
        total_detections += len(detections)

        # --- middle loop: ranging --------------------------------------
        if ranger is not None and range_every_n > 0 and index % range_every_n == 0:
            observations = _apply_ranger(ranger, observations, tracker, history, noise, rng)

        # --- fuse -------------------------------------------------------
        states = tracker.step(observations, clean_frame.timestamp, pose=frame.pose)
        for state in states:
            sigma_history.setdefault(state.track_id, []).append(
                (clean_frame.timestamp, state.sigma_horizontal)
            )

        if record_every_n > 0 and index % record_every_n == 0:
            records.append(
                FrameRecord(
                    frame_id=clean_frame.frame_id,
                    t=clean_frame.timestamp,
                    n_detections=len(detections),
                    n_observations=len(observations),
                    track_states=states,
                    truth_positions=dict(truth_now),
                    detections=list(detections),
                    frame=frame,
                )
            )

    track_records = {
        track_id: TrackRecord(
            track_id=track_id,
            origins=list(track.origins),
            truth_ids=[o.truth_id for o in track.observations],
            times=[o.t for o in track.observations],
            bearing_sigmas=[o.bearing_sigma for o in track.observations],
            first_t=track.state.first_t,
            sigma_history=sigma_history.get(track_id, []),
        )
        for track_id, track in tracker.tracks.items()
    }

    return PipelineResult(
        session_name=session.name,
        noise=noise,
        frames=records,
        final_tracks=tracker.live_states(),
        track_records=track_records,
        truth=truth_static,
        truth_classes=truth_classes,
        n_detections=total_detections,
        n_false_positives=total_false_positives,
    )


def _apply_ranger(
    ranger: Ranger,
    observations: list[Observation],
    tracker: Tracker,
    history: dict[int, list[Observation]],
    noise: NoiseModel,
    rng: np.random.Generator,
) -> list[Observation]:
    """Attach a RangeMeas to each observation, using its track's own history.

    An observation with no established track has no history to triangulate
    against, so it simply keeps ``range=None`` -- an honest absence rather than
    a fabricated default.
    """
    out: list[Observation] = []
    for obs in observations:
        track_id = None
        best = ranger_gate = tracker.config.gate_chi2
        for candidate_id, track in tracker.tracks.items():
            distance = track.ekf.mahalanobis(obs)
            if distance <= ranger_gate and distance <= best:
                best, track_id = distance, candidate_id
        if track_id is None:
            out.append(obs)
            continue
        past = history.setdefault(track_id, [])
        measurement = ranger.range_for(obs, past)
        past.append(obs)
        if measurement.valid and noise.range_sigma > 0:
            from geoloc_agent.contracts import RangeMeas

            measurement = RangeMeas(
                value=measurement.value + rng.normal(0.0, noise.range_sigma),
                sigma=float(np.hypot(measurement.sigma, noise.range_sigma)),
                method=measurement.method,
                valid=True,
            )
        out.append(
            Observation(
                t=obs.t, frame_id=obs.frame_id, origin=obs.origin, bearing=obs.bearing,
                bearing_sigma=obs.bearing_sigma, cls=obs.cls, score=obs.score,
                range=measurement if measurement.valid else None, origin_cov=obs.origin_cov,
                truth_position=obs.truth_position, truth_id=obs.truth_id,
            )
        )
    return out
