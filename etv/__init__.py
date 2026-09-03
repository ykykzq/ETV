"""ETV: symbolic equivalence verification for two raw TTIR program sides."""

import logging


logging.getLogger("etv").addHandler(logging.NullHandler())

from .model import ProofLevel, Status
from .verify import verify_spec

__all__ = ["ProofLevel", "Status", "verify_spec"]
__version__ = "0.7.0"
