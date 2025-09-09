# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Test suite for Mistral sliding window attention with xformers integration.
"""

import pytest
import torch
from transformers import PretrainedConfig
from unittest.mock import Mock, patch

from vllm.model_executor.models.mistral_sliding_window import (
    has_sliding_window_attention,
    get_optimal_sliding_window_backend,
    should_use_xformers_for_mistral,
    enhance_mistral_config_for_sliding_window,
    log_sliding_window_info
)
from vllm.platforms import _Backend


class MockMistralConfig(PretrainedConfig):
    """Mock Mistral configuration for testing."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_type = "mistral"
        self.sliding_window = kwargs.get('sliding_window', None)
        self.layer_types = kwargs.get('layer_types', None)
        self.num_hidden_layers = kwargs.get('num_hidden_layers', 32)


class TestSlidingWindowDetection:
    """Test sliding window attention detection."""
    
    def test_has_sliding_window_with_sliding_window_attr(self):
        """Test detection via sliding_window attribute."""
        config = MockMistralConfig(sliding_window=4096)
        assert has_sliding_window_attention(config) is True
    
    def test_has_sliding_window_with_layer_types(self):
        """Test detection via layer_types."""
        config = MockMistralConfig(
            layer_types=["full_attention", "sliding_attention", "full_attention"]
        )
        assert has_sliding_window_attention(config) is True
    
    def test_has_sliding_window_with_both(self):
        """Test detection with both attributes."""
        config = MockMistralConfig(
            sliding_window=4096,
            layer_types=["sliding_attention"] * 16 + ["full_attention"] * 16
        )
        assert has_sliding_window_attention(config) is True
    
    def test_no_sliding_window(self):
        """Test when no sliding window is present."""
        config = MockMistralConfig()
        assert has_sliding_window_attention(config) is False
    
    def test_no_sliding_window_in_layer_types(self):
        """Test when layer_types exists but no sliding attention."""
        config = MockMistralConfig(
            layer_types=["full_attention"] * 32
        )
        assert has_sliding_window_attention(config) is False


class TestBackendSelection:
    """Test optimal backend selection for sliding window attention."""
    
    @patch('vllm.model_executor.models.mistral_sliding_window.check_xformers_availability')
    @patch('vllm.model_executor.models.mistral_sliding_window.current_platform')
    def test_optimal_backend_with_xformers_available(self, mock_platform, mock_xformers):
        """Test backend selection when xformers is available."""
        mock_xformers.return_value = True
        mock_platform.is_cuda.return_value = True
        
        config = MockMistralConfig(sliding_window=4096)
        backend = get_optimal_sliding_window_backend(
            config, head_size=128, dtype=torch.float16
        )
        
        assert backend == _Backend.XFORMERS
    
    @patch('vllm.model_executor.models.mistral_sliding_window.check_xformers_availability')
    def test_optimal_backend_xformers_unavailable(self, mock_xformers):
        """Test backend selection when xformers is not available."""
        mock_xformers.return_value = False
        
        config = MockMistralConfig(sliding_window=4096)
        backend = get_optimal_sliding_window_backend(
            config, head_size=128, dtype=torch.float16
        )
        
        assert backend is None
    
    @patch('vllm.model_executor.models.mistral_sliding_window.check_xformers_availability')
    @patch('vllm.model_executor.models.mistral_sliding_window.current_platform')
    def test_optimal_backend_non_cuda(self, mock_platform, mock_xformers):
        """Test backend selection on non-CUDA platform."""
        mock_xformers.return_value = True
        mock_platform.is_cuda.return_value = False
        
        config = MockMistralConfig(sliding_window=4096)
        backend = get_optimal_sliding_window_backend(
            config, head_size=128, dtype=torch.float16
        )
        
        assert backend is None
    
    def test_optimal_backend_no_sliding_window(self):
        """Test backend selection when no sliding window is present."""
        config = MockMistralConfig()
        backend = get_optimal_sliding_window_backend(
            config, head_size=128, dtype=torch.float16
        )
        
        assert backend is None


class TestXFormersRecommendation:
    """Test xformers recommendation logic."""
    
    @patch('vllm.model_executor.models.mistral_sliding_window.check_xformers_availability')
    def test_should_use_xformers_with_sliding_window(self, mock_xformers):
        """Test xformers recommendation with sliding window."""
        mock_xformers.return_value = True
        
        config = MockMistralConfig(sliding_window=4096)
        should_use = should_use_xformers_for_mistral(config)
        
        assert should_use is True
    
    @patch('vllm.model_executor.models.mistral_sliding_window.check_xformers_availability')
    def test_should_use_xformers_no_sliding_window(self, mock_xformers):
        """Test xformers recommendation without sliding window."""
        mock_xformers.return_value = True
        
        config = MockMistralConfig()
        should_use = should_use_xformers_for_mistral(config)
        
        assert should_use is False
    
    @patch('vllm.model_executor.models.mistral_sliding_window.check_xformers_availability')
    def test_should_use_xformers_unavailable(self, mock_xformers):
        """Test xformers recommendation when unavailable."""
        mock_xformers.return_value = False
        
        config = MockMistralConfig(sliding_window=4096)
        should_use = should_use_xformers_for_mistral(config)
        
        assert should_use is False
    
    def test_should_use_xformers_explicit_backend(self):
        """Test xformers recommendation with explicit backend."""
        config = MockMistralConfig(sliding_window=4096)
        
        # Test with xformers explicitly selected
        should_use = should_use_xformers_for_mistral(config, _Backend.XFORMERS)
        assert should_use is True
        
        # Test with different backend explicitly selected
        should_use = should_use_xformers_for_mistral(config, _Backend.FLASH_ATTN)
        assert should_use is False


class TestConfigEnhancement:
    """Test configuration enhancement functionality."""
    
    def test_enhance_config_with_sliding_window(self):
        """Test config enhancement with sliding window."""
        config = MockMistralConfig(
            sliding_window=4096,
            layer_types=["sliding_attention"] * 16 + ["full_attention"] * 16
        )
        
        enhanced_config = enhance_mistral_config_for_sliding_window(config)
        
        # Config should be returned (possibly modified)
        assert enhanced_config is not None
        assert enhanced_config.sliding_window == 4096
        assert "sliding_attention" in enhanced_config.layer_types
    
    def test_enhance_config_inconsistent_warning(self):
        """Test warning for inconsistent configuration."""
        config = MockMistralConfig(
            sliding_window=4096,
            layer_types=["full_attention"] * 32  # No sliding attention layers
        )
        
        with patch('vllm.model_executor.models.mistral_sliding_window.logger') as mock_logger:
            enhance_mistral_config_for_sliding_window(config)
            mock_logger.warning.assert_called_once()
    
    def test_enhance_config_no_sliding_window(self):
        """Test config enhancement without sliding window."""
        config = MockMistralConfig()
        
        enhanced_config = enhance_mistral_config_for_sliding_window(config)
        
        assert enhanced_config is not None
        assert enhanced_config.sliding_window is None


class TestLogging:
    """Test logging functionality."""
    
    def test_log_sliding_window_info_with_config(self):
        """Test logging with sliding window configuration."""
        config = MockMistralConfig(
            sliding_window=4096,
            layer_types=["sliding_attention"] * 16 + ["full_attention"] * 16
        )
        
        with patch('vllm.model_executor.models.mistral_sliding_window.logger') as mock_logger:
            log_sliding_window_info(config)
            
            # Should log sliding window info
            assert mock_logger.info.call_count >= 4  # Multiple info calls expected
    
    def test_log_sliding_window_info_no_config(self):
        """Test logging without sliding window configuration."""
        config = MockMistralConfig()
        
        with patch('vllm.model_executor.models.mistral_sliding_window.logger') as mock_logger:
            log_sliding_window_info(config)
            
            # Should not log anything
            mock_logger.info.assert_not_called()


@pytest.mark.parametrize("sliding_window,layer_types,expected", [
    (4096, ["sliding_attention"] * 32, True),
    (None, ["sliding_attention"] * 16 + ["full_attention"] * 16, True),
    (4096, None, True),
    (None, ["full_attention"] * 32, False),
    (None, None, False),
])
def test_sliding_window_detection_parametrized(sliding_window, layer_types, expected):
    """Parametrized test for sliding window detection."""
    config = MockMistralConfig(
        sliding_window=sliding_window,
        layer_types=layer_types
    )
    
    assert has_sliding_window_attention(config) == expected


class TestIntegration:
    """Integration tests for the complete sliding window system."""
    
    @patch('vllm.model_executor.models.mistral_sliding_window.check_xformers_availability')
    @patch('vllm.model_executor.models.mistral_sliding_window.current_platform')
    def test_full_pipeline_with_xformers(self, mock_platform, mock_xformers):
        """Test the complete pipeline with xformers available."""
        mock_xformers.return_value = True
        mock_platform.is_cuda.return_value = True
        
        # Create a Mistral config with sliding window
        config = MockMistralConfig(
            sliding_window=4096,
            layer_types=["sliding_attention"] * 16 + ["full_attention"] * 16
        )
        
        # Test detection
        assert has_sliding_window_attention(config) is True
        
        # Test backend recommendation
        backend = get_optimal_sliding_window_backend(
            config, head_size=128, dtype=torch.float16
        )
        assert backend == _Backend.XFORMERS
        
        # Test xformers recommendation
        should_use = should_use_xformers_for_mistral(config)
        assert should_use is True
        
        # Test config enhancement
        enhanced_config = enhance_mistral_config_for_sliding_window(config)
        assert enhanced_config.sliding_window == 4096
    
    @patch('vllm.model_executor.models.mistral_sliding_window.check_xformers_availability')
    def test_full_pipeline_without_xformers(self, mock_xformers):
        """Test the complete pipeline without xformers."""
        mock_xformers.return_value = False
        
        config = MockMistralConfig(sliding_window=4096)
        
        # Detection should still work
        assert has_sliding_window_attention(config) is True
        
        # Backend recommendation should be None
        backend = get_optimal_sliding_window_backend(
            config, head_size=128, dtype=torch.float16
        )
        assert backend is None
        
        # Should not recommend xformers
        should_use = should_use_xformers_for_mistral(config)
        assert should_use is False


if __name__ == "__main__":
    pytest.main([__file__])
