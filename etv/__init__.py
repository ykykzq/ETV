"""ETV: symbolic equivalence verification for pairs of raw TTIR programs."""

from .model import ProofLevel, Status
from .verify import verify_spec

__all__ = ["ProofLevel", "Status", "verify_spec"]
__version__ = "0.6.0"
