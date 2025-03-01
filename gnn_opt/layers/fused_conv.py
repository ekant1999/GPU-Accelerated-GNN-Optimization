"""Fused message passing layer (uses same fused_gcn_forward as custom_conv)."""
from .custom_conv import CustomGCNConv

__all__ = ['CustomGCNConv']
