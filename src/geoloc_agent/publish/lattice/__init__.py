"""Anduril Lattice publishing.

LICENSING -- read before adding anything here.

The `anduril-lattice-sdk` licence permits use only for building against Lattice.
Three rules follow, and they are the reason this is a separate package rather
than a module inside `publish/`:

1. Nothing in this tree may be imported from `publish/cot/`, or from anywhere
   else in the pipeline. The dependency arrow points one way: this package
   imports `geoloc_agent.contracts`, and nothing imports this package except an
   explicit entry point.
2. No SDK code is vendored into this repository. The SDK is an optional
   dependency, installed from its own distribution.
3. The import of the SDK happens inside the function that needs it, not at
   module scope, so that importing `geoloc_agent` never pulls it in.

Credentials come from the environment and are never committed.
"""

from geoloc_agent.publish.lattice.client import LatticePublisher

__all__ = ["LatticePublisher"]
