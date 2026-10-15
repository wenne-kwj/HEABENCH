"""HEA local structure--charge retrieval toolkit.

The package deliberately keeps the four upstream backbones in ``third_party``
unchanged.  The code in this package contains only the HEA data contract,
adapters, projection heads, losses, and reproducible validation/training
utilities.
"""

from .losses import SymmetricInfoNCE, symmetric_infonce

__all__ = ["SymmetricInfoNCE", "symmetric_infonce"]
