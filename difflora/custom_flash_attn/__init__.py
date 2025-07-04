'''
From https://github.com/xiayuqing0622/flex_head_fa/blob/dim/flex_head_fa/__init__.py
'''

__version__ = "2.7.2.post1"

from .flash_attn_interface import (
    flash_attn_func,
    flash_attn_kvpacked_func,
    flash_attn_qkvpacked_func,
    flash_attn_varlen_func,
    flash_attn_varlen_kvpacked_func,
    flash_attn_varlen_qkvpacked_func,
    flash_attn_with_kvcache,
)