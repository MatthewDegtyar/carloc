"""Ranger interface.

Ranging is the middle loop (hundreds of milliseconds). A ranger turns a bearing,
plus whatever context it needs, into a RangeMeas -- always with a sigma, and
always able to say "no" by returning an invalid measurement rather than a guess.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from geoloc_agent.contracts import Observation, RangeMeas, RangeMethod


class Ranger(ABC):
    method: RangeMethod = RangeMethod.NONE

    @abstractmethod
    def range_for(self, obs: Observation, history: Sequence[Observation]) -> RangeMeas:
        """Range along ``obs.bearing``, or ``RangeMeas.invalid`` with a reason."""
