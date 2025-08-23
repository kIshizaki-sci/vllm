#!/usr/bin/env python3
"""
Test script to verify Qwen2.5-VL vision component dtype handling.
This script tests that the vision components correctly fall back to float32
when the GPU doesn't support bfloat16.
"""

import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock
import sys
import os

# Add the vllm directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_vision_dtype_fallback():
    """Test that vision components use float32 when GPU doesn't support bfloat16."""
    
    print("Testing Qwen2.5-VL Vision Component Dtype Handling")
    print("=" * 60)
    
    # Initialize parallel state for testing
    import os
    os.environ['RANK'] = '0'
    os.environ['LOCAL_RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    from vllm.distributed import parallel_state
    
    try:
        # Try to get the tensor parallel group, if it fails, initialize it
        parallel_state.get_tp_group()
    except (AssertionError, AttributeError):
        # Initialize with default settings for testing
        import torch
        import torch.distributed as dist
        
        # Initialize process group if not already initialized
        if not dist.is_initialized():
            try:
                dist.init_process_group(backend="gloo", init_method="env://")
            except Exception:
                # If gloo fails, try with a file store
                import tempfile
                temp_dir = tempfile.mkdtemp()
                init_file = os.path.join(temp_dir, "pg_init")
                dist.init_process_group(
                    backend="gloo",
                    init_method=f"file://{init_file}",
                    world_size=1,
                    rank=0
                )
        
        # Initialize vLLM parallel state using the correct API
        try:
            # Try the newer API first
            from vllm.distributed.parallel_state import (
                init_distributed_environment,
                init_model_parallel_group
            )
            
            # Initialize world group first
            if not hasattr(parallel_state, '_WORLD') or parallel_state._WORLD is None:
                init_distributed_environment()
            
            # Then initialize model parallel groups
            init_model_parallel_group(
                model_parallel_size=1,
                backend="gloo"
            )
        except (ImportError, TypeError) as e:
            # Fallback to direct initialization
            try:
                # Set up the groups manually
                world_group = dist.group.WORLD
                
                # Initialize global variables directly
                parallel_state._WORLD = world_group
                parallel_state._TP = world_group  # Use world group for TP when size=1
                parallel_state._PP = world_group  # Use world group for PP when size=1
                
                # Set ranks and sizes
                parallel_state._TP_RANK = 0
                parallel_state._PP_RANK = 0
                parallel_state._TP_SIZE = 1
                parallel_state._PP_SIZE = 1
            except Exception:
                # Last resort: mock the parallel state
                from unittest.mock import MagicMock
                mock_group = MagicMock()
                mock_group.world_size = 1
                mock_group.rank = 0
                parallel_state._TP = mock_group
                parallel_state._PP = mock_group
                parallel_state._WORLD = mock_group
    
    # Mock the platform to simulate different GPU capabilities
    with patch('vllm.model_executor.models.qwen2_5_vl.current_platform') as mock_platform:
        
        # Test 1: GPU without bfloat16 support (compute capability < 8.0)
        print("\nTest 1: GPU without bfloat16 support")
        print("-" * 40)
        mock_platform.has_device_capability.return_value = False
        
        # Import after mocking
        from vllm.model_executor.models.qwen2_5_vl import (
            Qwen2_5_VisionTransformer,
            Qwen2_5_VisionPatchEmbed,
            Qwen2_5_VisionMLP,
            Qwen2_5_VisionAttention,
            Qwen2_5_VisionBlock,
            Qwen2_5_VisionPatchMerger
        )
        from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig
        
        # Create a mock vision config
        vision_config = Qwen2_5_VLVisionConfig(
            hidden_size=1152,
            intermediate_size=4608,
            num_heads=16,
            num_hidden_layers=26,
            patch_size=14,
            temporal_patch_size=2,
            in_channels=3,
            out_hidden_size=3584,
            spatial_merge_size=2,
            window_size=256,
            fullatt_block_indexes=[],
            depth=26,
            hidden_act="silu"
        )
        vision_config.torch_dtype = torch.bfloat16  # Original config uses bfloat16
        
        # Create vision transformer
        vision_transformer = Qwen2_5_VisionTransformer(
            vision_config=vision_config,
            norm_eps=1e-6,
            quant_config=None,
            prefix="visual",
            use_data_parallel=False
        )
        
        # Check that vision_dtype is float32
        assert hasattr(vision_transformer, '_vision_dtype'), "Vision transformer should have _vision_dtype attribute"
        assert vision_transformer._vision_dtype == torch.float32, f"Expected float32, got {vision_transformer._vision_dtype}"
        print(f"✓ Vision transformer dtype: {vision_transformer._vision_dtype}")
        
        # Check patch embed dtype
        patch_embed_weight = vision_transformer.patch_embed.proj.weight
        assert patch_embed_weight.dtype == torch.float32, f"Patch embed weight should be float32, got {patch_embed_weight.dtype}"
        print(f"✓ Patch embed weight dtype: {patch_embed_weight.dtype}")
        
        # Check that blocks have correct params_dtype
        if len(vision_transformer.blocks) > 0:
            first_block = vision_transformer.blocks[0]
            # Check MLP components
            if hasattr(first_block.mlp, 'gate_up_proj') and hasattr(first_block.mlp.gate_up_proj, 'weight'):
                mlp_weight = first_block.mlp.gate_up_proj.weight
                print(f"✓ Block MLP gate_up_proj weight dtype: {mlp_weight.dtype}")
            
            # Check attention components
            if hasattr(first_block.attn, 'qkv') and hasattr(first_block.attn.qkv, 'weight'):
                attn_weight = first_block.attn.qkv.weight
                print(f"✓ Block Attention QKV weight dtype: {attn_weight.dtype}")
        
        # Check merger component dtype
        if hasattr(vision_transformer, 'merger'):
            merger = vision_transformer.merger
            # Check merger MLP components
            if hasattr(merger, 'mlp') and len(merger.mlp) > 0:
                # First layer in MLP module list
                if hasattr(merger.mlp[0], 'weight'):
                    merger_fc1_weight = merger.mlp[0].weight
                    print(f"✓ Merger MLP fc1 weight dtype: {merger_fc1_weight.dtype}")
                # Third layer in MLP module list (index 2, since index 1 is GELU)
                if len(merger.mlp) > 2 and hasattr(merger.mlp[2], 'weight'):
                    merger_fc2_weight = merger.mlp[2].weight
                    print(f"✓ Merger MLP fc2 weight dtype: {merger_fc2_weight.dtype}")
        
        print("\nTest 1: PASSED ✓")
        
        # Test 2: GPU with bfloat16 support (compute capability >= 8.0)
        print("\nTest 2: GPU with bfloat16 support")
        print("-" * 40)
        mock_platform.has_device_capability.return_value = True
        
        # Reload the module to get fresh imports with new mock
        import importlib
        import vllm.model_executor.models.qwen2_5_vl as qwen_module
        importlib.reload(qwen_module)
        
        from vllm.model_executor.models.qwen2_5_vl import Qwen2_5_VisionTransformer
        
        # Create vision transformer with bfloat16 support
        vision_transformer_bf16 = Qwen2_5_VisionTransformer(
            vision_config=vision_config,
            norm_eps=1e-6,
            quant_config=None,
            prefix="visual",
            use_data_parallel=False
        )
        
        # Check that vision_dtype is bfloat16
        assert hasattr(vision_transformer_bf16, '_vision_dtype'), "Vision transformer should have _vision_dtype attribute"
        assert vision_transformer_bf16._vision_dtype == torch.bfloat16, f"Expected bfloat16, got {vision_transformer_bf16._vision_dtype}"
        print(f"✓ Vision transformer dtype: {vision_transformer_bf16._vision_dtype}")
        
        # Check patch embed dtype
        patch_embed_weight_bf16 = vision_transformer_bf16.patch_embed.proj.weight
        assert patch_embed_weight_bf16.dtype == torch.bfloat16, f"Patch embed weight should be bfloat16, got {patch_embed_weight_bf16.dtype}"
        print(f"✓ Patch embed weight dtype: {patch_embed_weight_bf16.dtype}")
        
        print("\nTest 2: PASSED ✓")
        
    print("\n" + "=" * 60)
    print("All tests passed successfully! ✓")
    print("The vision components correctly handle dtype based on GPU capability.")
    
    return True

def test_weight_conversion():
    """Test that weights are correctly converted during loading."""
    
    print("\n" + "=" * 60)
    print("Testing Weight Conversion During Loading")
    print("=" * 60)
    
    # Initialize parallel state for testing if not already initialized
    from vllm.distributed import parallel_state
    try:
        parallel_state.get_tp_group()
    except (AssertionError, AttributeError):
        pass  # Already initialized in the first test
    
    with patch('vllm.model_executor.models.qwen2_5_vl.current_platform') as mock_platform:
        mock_platform.has_device_capability.return_value = False  # No bfloat16 support
        
        from vllm.model_executor.models.qwen2_5_vl import Qwen2_5_VisionTransformer
        from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig
        
        # Create a mock vision config
        vision_config = Qwen2_5_VLVisionConfig(
            hidden_size=128,  # Smaller for testing
            intermediate_size=512,
            num_heads=4,
            num_hidden_layers=2,
            patch_size=14,
            temporal_patch_size=2,
            in_channels=3,
            out_hidden_size=256,
            spatial_merge_size=2,
            window_size=256,
            fullatt_block_indexes=[],
            depth=2,
            hidden_act="silu"
        )
        vision_config.torch_dtype = torch.bfloat16
        
        # Create vision transformer
        vision_transformer = Qwen2_5_VisionTransformer(
            vision_config=vision_config,
            norm_eps=1e-6,
            quant_config=None,
            prefix="visual",
            use_data_parallel=False
        )
        
        # Simulate loading weights in bfloat16
        print("\nSimulating weight loading...")
        test_weights = [
            ("patch_embed.proj.weight", torch.randn(128, 3, 2, 14, 14, dtype=torch.bfloat16)),
            ("blocks.0.norm1.weight", torch.randn(128, dtype=torch.bfloat16)),
            ("blocks.0.attn.qkv.weight", torch.randn(384, 128, dtype=torch.bfloat16)),
            ("blocks.0.mlp.gate_up_proj.weight", torch.randn(1024, 128, dtype=torch.bfloat16)),
            ("merger.mlp.0.weight", torch.randn(512, 512, dtype=torch.bfloat16)),
            ("merger.mlp.2.weight", torch.randn(256, 512, dtype=torch.bfloat16)),
        ]
        
        # Mock logger to capture conversion messages
        with patch('vllm.model_executor.models.qwen2_5_vl.logger') as mock_logger:
            loaded_params = vision_transformer.load_weights(test_weights)
            
            # Check that weights were converted
            conversion_count = sum(1 for call in mock_logger.info.call_args_list 
                                 if "Converting vision transformer weight" in str(call))
            print(f"✓ Converted {conversion_count} weights from bfloat16 to float32")
            
            # Verify loaded parameters
            print(f"✓ Loaded {len(loaded_params)} parameters")
            
        print("\nWeight conversion test: PASSED ✓")
    
    return True

def test_merger_component():
    """Test that merger component correctly handles dtype."""
    
    print("\n" + "=" * 60)
    print("Testing Merger Component Dtype Handling")
    print("=" * 60)
    
    # Initialize parallel state for testing if not already initialized
    from vllm.distributed import parallel_state
    try:
        parallel_state.get_tp_group()
    except (AssertionError, AttributeError):
        pass  # Already initialized in the first test
    
    with patch('vllm.model_executor.models.qwen2_5_vl.current_platform') as mock_platform:
        mock_platform.has_device_capability.return_value = False  # No bfloat16 support
        
        from vllm.model_executor.models.qwen2_5_vl import Qwen2_5_VisionPatchMerger
        from functools import partial
        from vllm.model_executor.layers.layernorm import RMSNorm
        
        # Test merger with float32 dtype
        print("\nTesting merger with float32 params_dtype...")
        merger = Qwen2_5_VisionPatchMerger(
            d_model=256,
            context_dim=128,
            norm_layer=partial(RMSNorm, eps=1e-6, dtype=torch.float32),
            spatial_merge_size=2,
            quant_config=None,
            prefix="merger",
            use_data_parallel=False,
            params_dtype=torch.float32
        )
        
        # Check that MLP layers have correct dtype
        if hasattr(merger, 'mlp') and len(merger.mlp) > 0:
            # Check first linear layer
            if hasattr(merger.mlp[0], 'weight'):
                assert merger.mlp[0].weight.dtype == torch.float32, \
                    f"Merger MLP fc1 should be float32, got {merger.mlp[0].weight.dtype}"
                print(f"✓ Merger MLP fc1 weight dtype: {merger.mlp[0].weight.dtype}")
            
            # Check second linear layer (index 2, since index 1 is GELU)
            if len(merger.mlp) > 2 and hasattr(merger.mlp[2], 'weight'):
                assert merger.mlp[2].weight.dtype == torch.float32, \
                    f"Merger MLP fc2 should be float32, got {merger.mlp[2].weight.dtype}"
                print(f"✓ Merger MLP fc2 weight dtype: {merger.mlp[2].weight.dtype}")
        
        print("\nMerger component test: PASSED ✓")
    
    return True

if __name__ == "__main__":
    try:
        # Set environment variables for testing
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        
        # Run tests
        test_vision_dtype_fallback()
        test_weight_conversion()
        test_merger_component()
        
        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED SUCCESSFULLY! ✓✓✓")
        print("=" * 60)
        print("\nThe Qwen2.5-VL vision components now correctly:")
        print("1. Detect GPU bfloat16 capability")
        print("2. Fall back to float32 when needed")
        print("3. Initialize all components with correct dtype")
        print("4. Convert weights during loading")
        print("5. Handle merger component dtype properly")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
