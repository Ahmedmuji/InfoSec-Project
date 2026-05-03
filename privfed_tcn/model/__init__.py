"""Model components: TCN blocks, attention, full PrivFed-TCN and SHAP explainer."""
from .tcn_block import TCNBlock
from .attention import MultiHeadSelfAttention, SinusoidalPositionalEncoding
from .privfed_tcn import PrivFedTCN
from .explainability import SHAPExplainer

__all__ = [
    "TCNBlock",
    "MultiHeadSelfAttention",
    "SinusoidalPositionalEncoding",
    "PrivFedTCN",
    "SHAPExplainer",
]
