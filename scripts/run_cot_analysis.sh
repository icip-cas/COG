#!/bin/bash

# ============================================================================
# Complete Automated Script for Inference-Analysis-Statistics Pipeline
# ============================================================================

set -e  # Exit immediately on error

echo "========================================"
echo "  Starting Complete Analysis Pipeline"
echo "========================================"

# ============================================================================
# Pre-flight Configuration Check
# ============================================================================
echo ""
echo "[Pre-flight] Checking configuration..."
echo "----------------------------------------"

echo "IMPORTANT: Please ensure the following configurations are set:"
echo ""
echo "1. Code 2 (code2_statistics.py) - Model Names:"
echo "   MODEL_NAMES = ["
echo "       \"Base\","
echo "       \"SafR\","
echo "       \"SafB\","
echo "       \"XXX\""
echo "   ]"
echo ""
echo "2. Code 3 (code3_analysis.py) - File Paths:"
echo "   \"input_filename\": \"./Result/COT_analysis/math_parsed.json\""
echo "   \"output_filename\": \"./Result/COT_analysis/math_analyzed.json\""
echo ""
echo "3. Code 2 (code2_statistics.py) - Input File:"
echo "   INPUT_FILE = \"./Result/COT_analysis/math_analyzed.json\""
echo ""
read -p "Have you verified these configurations? (y/n): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Please update the configurations before running this script."
    exit 1
fi

# ============================================================================
# Stage 1: Model Inference
# ============================================================================
echo ""
echo "[Stage 1/3] Starting model inference..."
echo "----------------------------------------"

python3 cot_analysis/1_cot_extraction.py

if [ $? -eq 0 ]; then
    echo "✓ Inference completed"
else
    echo "✗ Inference failed, script terminated"
    exit 1
fi

# ============================================================================
# Stage 2: Inference Process Analysis
# ============================================================================
echo ""
echo "[Stage 2/3] Starting inference process analysis..."
echo "----------------------------------------"

python3 cot_analysis/2_cot_analysis.py

if [ $? -eq 0 ]; then
    echo "✓ Analysis completed"
else
    echo "✗ Analysis failed, script terminated"
    exit 1
fi

# ============================================================================
# Stage 3: Statistical Results
# ============================================================================
echo ""
echo "[Stage 3/3] Starting statistical analysis..."
echo "----------------------------------------"

python3 cot_analysis/3_cot_statistics.py

if [ $? -eq 0 ]; then
    echo "✓ Statistics completed"
else
    echo "✗ Statistics failed, script terminated"
    exit 1
fi

# ============================================================================
# Completion
# ============================================================================
echo ""
echo "========================================"
echo "  ✓ All Pipeline Stages Completed!"
echo "========================================"
