#!/bin/bash
# Script to run Qwen2.5-VL dtype test

echo "=========================================="
echo "Running Qwen2.5-VL Vision Component Tests"
echo "=========================================="

# Set environment variables for distributed training
export MASTER_ADDR=localhost
export MASTER_PORT=12355
export RANK=0
export LOCAL_RANK=0
export WORLD_SIZE=1

# Optional: Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Run the test
python test_qwen2_5_vl_dtype.py

# Check exit code
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ All tests passed successfully!"
else
    echo ""
    echo "❌ Tests failed. Please check the error messages above."
    exit 1
fi
