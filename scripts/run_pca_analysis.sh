#!/bin/bash

# ============================================================================
# Complete Automated Script for PCA Analysis Pipeline
# ============================================================================

set -e  # Exit immediately on error

echo "========================================"
echo "  Starting PCA Analysis Pipeline"
echo "========================================"

# ============================================================================
# Pre-flight Configuration Check
# ============================================================================
echo ""
echo "[Pre-flight] Checking configuration..."
echo "----------------------------------------"

echo "IMPORTANT: Please ensure the following configurations are set in PCA/pca.py:"
echo ""
echo "1. Model Paths:"
echo "   models_to_load = {"
echo "       'Base': 'Qwen/Qwen3-32B',"
echo "       'SR': 'Qwen3_32B_SafR',"
echo "       'SB': 'Qwen3_32B_SafB',"
echo "       'XXX': 'Qwen3_32B_XXX'"
echo "   }"
echo ""
echo "2. Input/Output Paths:"
echo "   input_file_path = './Data/PCA_data.jsonl'"
echo "   output_dir = './Result/PCA_Result/Qwen3-32B-XXX'"
echo ""
echo "3. Analysis Settings:"
echo "   layer_to_analyze = -1"
echo "   batch_size = 4"
echo "   use_multi_gpu = True"
echo ""
read -p "Have you verified these configurations? (y/n): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Please update the configurations in PCA/pca.py before running this script, and make sure to update the corresponding model names."
    exit 1
fi

# ============================================================================
# Stage 1: PCA Analysis Execution
# ============================================================================
echo ""
echo "[Stage 1/1] Starting PCA analysis..."
echo "----------------------------------------"

python PCA/pca.py

if [ $? -eq 0 ]; then
    echo "✓ PCA analysis completed"
else
    echo "✗ PCA analysis failed, script terminated"
    exit 1
fi

# ============================================================================
# Completion
# ============================================================================
echo ""
echo "========================================"
echo "  ✓ PCA Analysis Pipeline Completed!"
echo "========================================"
