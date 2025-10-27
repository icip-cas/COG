#!/bin/bash

# --- Script Description ---
# Function:
#   Automatically executes a six-step data processing pipeline.
#   Supports configuring GPU memory usage specifically for the sixth step to address memory shortage issues.
# Usage:
#   1. Modify the "User Configuration Section" below with your paths and parameters.
#   2. Run directly: ./run_pipeline.sh

set -euo pipefail

# ==============================================================================
# ---                        ⭐ User Configuration Section ⭐                 ---
# ==============================================================================

# --- Path Configuration ---
# Main model path (used in steps 1, 4, 5, 6)
MAIN_MODEL_PATH="/mnt/data4/hf_models/Qwen3-8B"

# Dedicated model path (used in step 2 for extract and step 3 for classify)
EXTRACT_CLASSIFY_MODEL_PATH="/mnt/data3/hf_models/Qwen2-7B-Instruct"

# Full path to the initial dataset
INITIAL_DATASET_PATH="./Data/seed_data.jsonl"

# --- Parameter Configuration ---
# General parameters (used in steps 1-5)
TP_SIZE=8
DTYPE="bfloat16"
GPU_UTIL=0.9  
MAX_LEN=8192
MAX_TOKENS=4096
BATCH_SIZE=64
MAX_NUM_SEQS=256

# GPU memory usage for step 6 only
GPU_UTIL_STEP6=0.8


# ==============================================================================
# ---                        Core Pipeline Logic                             ---
# ==============================================================================

# --- 1. Check paths and define variables ---
echo "--- Checking configuration paths..."
MAIN_MODEL_NAME=$(basename "$MAIN_MODEL_PATH")
EXTRACT_CLASSIFY_MODEL_NAME=$(basename "$EXTRACT_CLASSIFY_MODEL_PATH")
if [ ! -d "$MAIN_MODEL_PATH" ] || [ ! -d "$EXTRACT_CLASSIFY_MODEL_PATH" ] || [ ! -f "$INITIAL_DATASET_PATH" ]; then
    echo "❌ Error: Please check if the configured model or dataset paths are correct!"
    exit 1
fi
echo "✅ Path check passed."
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BASE_OUTPUT_DIR="./Result/Pipeline/${MAIN_MODEL_NAME}_${TIMESTAMP}"
mkdir -p "$BASE_OUTPUT_DIR"

# --- 3. Define standardized output filenames for each step ---
OUTPUT_STEP1="1_${MAIN_MODEL_NAME}_generate.jsonl"
OUTPUT_STEP2="2_${MAIN_MODEL_NAME}_extract_by_${EXTRACT_CLASSIFY_MODEL_NAME}.jsonl"
OUTPUT_STEP3="3_${MAIN_MODEL_NAME}_classify_by_${EXTRACT_CLASSIFY_MODEL_NAME}.jsonl"
OUTPUT_STEP4="4_${MAIN_MODEL_NAME}_rewrite.jsonl"
OUTPUT_STEP5="5_${MAIN_MODEL_NAME}_integrate.jsonl"
OUTPUT_STEP6="6_${MAIN_MODEL_NAME}_final_output.jsonl"

# ==============================================================================
# ---                        🚀 Pipeline Execution Start 🚀                   ---
# ==============================================================================
echo "🚀 Pipeline starting..."
echo "------------------------------------------------------------------"
echo "Main model :                          ${MAIN_MODEL_NAME}"
echo "Dedicated model (steps 2,3):          ${EXTRACT_CLASSIFY_MODEL_NAME}"
echo "Initial dataset:                      ${INITIAL_DATASET_PATH}"
echo "General GPU_UTIL (steps 1-5):         ${GPU_UTIL}"
echo "Step 6 GPU_UTIL:                      ${GPU_UTIL_STEP6}"
echo "Output will be saved to:              ${BASE_OUTPUT_DIR}"
echo "------------------------------------------------------------------"



echo "▶️ [Step 1/6] Running generate.py (Model: ${MAIN_MODEL_NAME})..."
python ./Code/initial_inference.py \
    --input "$INITIAL_DATASET_PATH" \
    --output_dir "$BASE_OUTPUT_DIR" \
    --output_filename "$OUTPUT_STEP1" \
    --model_path "$MAIN_MODEL_PATH" \
    --model_name "$MAIN_MODEL_NAME" \
    --tp_size "$TP_SIZE" --dtype "$DTYPE" --gpu_util "$GPU_UTIL" \
    --max_len "$MAX_LEN" --max_tokens "$MAX_TOKENS" \
    --batch_size "$BATCH_SIZE" --max_num_seqs "$MAX_NUM_SEQS"
echo "✅ [Step 1/6] Completed. Output file: $OUTPUT_STEP1"
echo "------------------------------------------------------------------"


echo "▶️ [Step 2/6] Running extracct_batch.py (Model: ${EXTRACT_CLASSIFY_MODEL_NAME})..."
python ./Code/extraction.py \
    --input "$BASE_OUTPUT_DIR/$OUTPUT_STEP1" \
    --output_dir "$BASE_OUTPUT_DIR" \
    --output_filename "$OUTPUT_STEP2" \
    --model_path "$EXTRACT_CLASSIFY_MODEL_PATH" \
    --model_name "$EXTRACT_CLASSIFY_MODEL_NAME" \
    --tp_size "$TP_SIZE" --dtype "$DTYPE" --gpu_util "$GPU_UTIL" \
    --max_len "$MAX_LEN" --max_tokens "$MAX_TOKENS" \
    --batch_size "$BATCH_SIZE" --max_num_seqs "$MAX_NUM_SEQS"
echo "✅ [Step 2/6] Completed. Output file: $OUTPUT_STEP2"
echo "------------------------------------------------------------------"


echo "▶️ [Step 3/6] Running classification_batch.py (Model: ${EXTRACT_CLASSIFY_MODEL_NAME})..."
python ./Code/failure_classification.py \
    --input "$BASE_OUTPUT_DIR/$OUTPUT_STEP2" \
    --output_dir "$BASE_OUTPUT_DIR" \
    --model_path "$EXTRACT_CLASSIFY_MODEL_PATH" \
    --model_name "$EXTRACT_CLASSIFY_MODEL_NAME" \
    --tp_size "$TP_SIZE" --dtype "$DTYPE" --gpu_util "$GPU_UTIL" \
    --max_len "$MAX_LEN" --max_tokens "$MAX_TOKENS" \
    --batch_size "$BATCH_SIZE" --max_num_seqs "$MAX_NUM_SEQS"

INTERNAL_FILENAME_BASE=$(basename "$OUTPUT_STEP2" .jsonl)
ACTUAL_GENERATED_FILE="${INTERNAL_FILENAME_BASE}_classified_by_${EXTRACT_CLASSIFY_MODEL_NAME}.jsonl"
mv "$BASE_OUTPUT_DIR/$ACTUAL_GENERATED_FILE" "$BASE_OUTPUT_DIR/$OUTPUT_STEP3"
echo "✅ [Step 3/6] Completed. Output file: $OUTPUT_STEP3"
echo "------------------------------------------------------------------"


echo "▶️ [Step 4/6] Running rewrite_batch.py (Model: ${MAIN_MODEL_NAME})..."
python ./Code/SafR/recomposition.py \
    "$BASE_OUTPUT_DIR/$OUTPUT_STEP3" \
    --output "$BASE_OUTPUT_DIR/$OUTPUT_STEP4" \
    --prompts "./Prompt/SafR.json" \
    --model_path "$MAIN_MODEL_PATH" \
    --model_name "$MAIN_MODEL_NAME" \
    --tp_size "$TP_SIZE" --dtype "$DTYPE" --gpu_util "$GPU_UTIL" \
    --max_len "$MAX_LEN" --max_tokens "$MAX_TOKENS" \
    --batch_size "$BATCH_SIZE" --max_num_seqs "$MAX_NUM_SEQS"
echo "✅ [Step 4/6] Completed. Output file: $OUTPUT_STEP4"
echo "------------------------------------------------------------------"


echo "▶️ [Step 5/6] Running integrat__batch.py (Model: ${MAIN_MODEL_NAME})..."
python ./Code/SafR/merge.py \
    --input "$BASE_OUTPUT_DIR/$OUTPUT_STEP4" \
    --output_dir "$BASE_OUTPUT_DIR" \
    --output_filename "$OUTPUT_STEP5" \
    --model_path "$MAIN_MODEL_PATH" \
    --model_name "$MAIN_MODEL_NAME" \
    --tp_size "$TP_SIZE" --dtype "$DTYPE" --gpu_util "$GPU_UTIL" \
    --max_len "$MAX_LEN" --max_tokens "$MAX_TOKENS" \
    --batch_size "$BATCH_SIZE" --max_num_seqs "$MAX_NUM_SEQS"
echo "✅ [Step 5/6] Completed. Output file: $OUTPUT_STEP5"
echo "------------------------------------------------------------------"


echo "▶️ [Step 6/6] Running generate_folow_think.py (Model: ${MAIN_MODEL_NAME}, GPU_UTIL: ${GPU_UTIL_STEP6})..."
python ./Code/safe_res_gen.py \
    --input "$BASE_OUTPUT_DIR/$OUTPUT_STEP5" \
    --output_dir "$BASE_OUTPUT_DIR" \
    --output_filename "$OUTPUT_STEP6" \
    --model_path "$MAIN_MODEL_PATH" \
    --model_name "$MAIN_MODEL_NAME" \
    --tp_size "$TP_SIZE" --dtype "$DTYPE" --gpu_util "$GPU_UTIL_STEP6" \
    --max_len "$MAX_LEN" --max_tokens "$MAX_TOKENS" \
    --batch_size "$BATCH_SIZE" --max_num_seqs "$MAX_NUM_SEQS"
echo "✅ [Step 6/6] Completed. Final output file: $OUTPUT_STEP6"
echo "------------------------------------------------------------------"

echo "🎉🎉🎉 All steps executed successfully!"
echo "📂 You can check all results in the following directory:"
echo "${BASE_OUTPUT_DIR}"
