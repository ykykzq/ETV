"""ETV: bounded semantic equivalence verification for TTIR specializations."""

from .model import ProofLevel, Status
from .verify import verify_spec

__all__ = ["ProofLevel", "Status", "verify_spec"]
__version__ = "0.4.0"
