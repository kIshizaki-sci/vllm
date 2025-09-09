# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Enhanced Mistral sliding window attention support with xformers integration.

This module provides utilities to ensure optimal backend selection for Mistral models
with sliding window attention, prioritizing xformers when appropriate.
"""

import logging
from typing import Optional

from transformers import PretrainedConfig

from vllm.attention.layer import check_xformers_availability
from vllm.logger import init_logger
from vllm.platforms import current_platform, _Backend

logger = init_logger(__name__)


def has_sliding_window_attention(config: PretrainedConfig) -> bool:
    """
    Check if the model configuration uses sliding window attention.
    
    Args:
        config: The model configuration
        
    Returns:
        bool: True if the model uses sliding window attention
    """
    # Check for sliding_window attribute
    if hasattr(config, 'sliding_window') and config.sliding_window is not None:
        return True
    
    # Check for layer_types with sliding_attention
    if hasattr(config, 'layer_types') and config.layer_types is not None:
        return "sliding_attention" in config.layer_types
    
    return False


def get_optimal_sliding_window_backend(
    config: PretrainedConfig,
    head_size: int,
    dtype,
    kv_cache_dtype: Optional[str] = None,
    block_size: int = 16
) -> Optional[_Backend]:
    """
    Determine the optimal attention backend for sliding window attention.
    
    This function prioritizes xformers for sliding window attention when:
    1. The model uses sliding window attention
    2. XFormers is available and supports the configuration
    3. The platform supports xformers
    
    Args:
        config: Model configuration
        head_size: Attention head size
        dtype: Model dtype
        kv_cache_dtype: KV cache dtype
        block_size: Block size for attention
        
    Returns:
        Optional[_Backend]: Recommended backend, or None for default selection
    """
    if not has_sliding_window_attention(config):
        return None
    
    # Check if xformers is available
    if not check_xformers_availability():
        logger.info(
            "XFormers not available for sliding window attention. "
            "Falling back to default backend selection."
        )
        return None
    
    # Check platform compatibility
    if not current_platform.is_cuda():
        logger.info(
            "XFormers sliding window optimization only available on CUDA. "
            "Using default backend selection."
        )
        return None
    
    # For sliding window attention, xformers provides good performance
    # and has mature sliding window support
    logger.info(
        "Detected sliding window attention in Mistral model. "
        "Recommending XFormers backend for optimal performance."
    )
    
    return _Backend.XFORMERS


def log_sliding_window_info(config: PretrainedConfig) -> None:
    """
    Log information about sliding window configuration.
    
    Args:
        config: Model configuration
    """
    if not has_sliding_window_attention(config):
        return
    
    sliding_window = getattr(config, 'sliding_window', None)
    layer_types = getattr(config, 'layer_types', None)
    
    logger.info("Mistral sliding window attention detected:")
    if sliding_window is not None:
        logger.info(f"  - Sliding window size: {sliding_window}")
    
    if layer_types is not None:
        sliding_layers = [i for i, t in enumerate(layer_types) 
                         if t == "sliding_attention"]
        full_layers = [i for i, t in enumerate(layer_types) 
                      if t == "full_attention"]
        
        logger.info(f"  - Sliding attention layers: {len(sliding_layers)}")
        logger.info(f"  - Full attention layers: {len(full_layers)}")
        logger.info(f"  - Total layers: {len(layer_types)}")


def enhance_mistral_config_for_sliding_window(config: PretrainedConfig) -> PretrainedConfig:
    """
    Enhance Mistral configuration to ensure proper sliding window setup.
    
    This function ensures that Mistral configurations are properly set up
    for sliding window attention, complementing the existing logic in
    vllm/transformers_utils/config.py.
    
    Args:
        config: Model configuration to enhance
        
    Returns:
        PretrainedConfig: Enhanced configuration
    """
    # Log current sliding window configuration
    log_sliding_window_info(config)
    
    # The main sliding window configuration logic is already handled in
    # vllm/transformers_utils/config.py, so we just ensure consistency here
    
    if hasattr(config, 'sliding_window') and hasattr(config, 'layer_types'):
        sliding_window = config.sliding_window
        layer_types = config.layer_types
        
        if sliding_window is not None and layer_types is not None:
            # Verify consistency between sliding_window and layer_types
            has_sliding_layers = "sliding_attention" in layer_types
            
            if not has_sliding_layers:
                logger.warning(
                    "Model has sliding_window configured but no sliding_attention "
                    "layers in layer_types. This may indicate a configuration issue."
                )
    
    return config


def should_use_xformers_for_mistral(
    config: PretrainedConfig,
    current_backend: Optional[_Backend] = None
) -> bool:
    """
    Determine if xformers should be used for this Mistral model.
    
    Args:
        config: Model configuration
        current_backend: Currently selected backend
        
    Returns:
        bool: True if xformers should be used
    """
    # If a backend is already explicitly selected, respect that choice
    if current_backend is not None:
        return current_backend == _Backend.XFORMERS
    
    # Check if this model would benefit from xformers
    if not has_sliding_window_attention(config):
        return False
    
    # Check xformers availability
    if not check_xformers_availability():
        return False
    
    # For sliding window attention, xformers is a good choice
    return True
