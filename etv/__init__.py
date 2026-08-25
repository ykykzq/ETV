"""ETV: symbolic equivalence verification for TTIR and Torch Prims programs."""

from .model import ProofLevel, Status
from .verify import verify_spec

__all__ = ["ProofLevel", "Status", "verify_spec"]
__version__ = "0.5.0"
