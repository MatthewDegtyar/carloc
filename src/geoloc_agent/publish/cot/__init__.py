"""Cursor-on-Target publishing.

LICENSING: this tree must stay completely independent of `publish/lattice/`.
The Lattice SDK licence permits use only for building against Lattice, so
nothing from that package may be imported here, and no SDK code is vendored
into this repo. The two publishers share the `TrackState` contract and nothing
else -- that shared contract is in `geoloc_agent.contracts`, which depends on
neither.
"""

from geoloc_agent.publish.cot.event import CotPublisher, track_to_cot

__all__ = ["CotPublisher", "track_to_cot"]
