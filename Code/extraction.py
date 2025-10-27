import json
from pathlib import Path
from tqdm import tqdm
import re
import traceback
import argparse
import os
import json_repair  # Import json_repair library
# Import core classes of vLLM
from vllm import LLM, SamplingParams

# ==============================================================================
# Global variables and helper functions
# ==============================================================================

# System prompt defined in global scope
system_prompt = """
"""
import unicodedata 

def comprehensive_clean_text(text: str) -> str:
    """
    Thoroughly clean text to remove invisible or non-standard characters
    that may interfere with JSON parsing.
    Do not convert single quotes to double quotes.
    """
    if not isinstance(text, str):
        print("Warning: comprehensive_clean_text received input that is not a string.")
        return ""

    if text.startswith('\ufeff'):
        text = text[1:]

    text = re.sub(r'[\x00-\x1F\x7F]', '', text)

    text = text.replace('\u200b', '').replace('\u00A0', ' ')

    text = re.sub(r'\s+', ' ', text).strip()

    text = re.sub(r'(\w+)\s*:\s*([^",{}\[\]]+)', r'"\1": "\2"', text)
    
    return text

def robust_json_extract(generated_text: str, case_id: str) -> dict | None:
    """
    Attempt to robustly extract and parse a JSON object from model-generated text.
    First, perform comprehensive cleaning and initial format repair,
    then prioritize matching Markdown JSON code blocks,
    and finally use a general curly brace structure.
    If parsing fails, return None and save the original response to a file.
    """
    cleaned_text = comprehensive_clean_text(generated_text)
    clean_json_str = cleaned_text  # Default to using cleaned text

    # 1. Try to match Markdown JSON code blocks (like ```json { ... } ```)
    match_code_block = re.search(r'```json\s*(\{.*?\})\s*```', cleaned_text, re.DOTALL)
    if match_code_block:
        clean_json_str = match_code_block.group(1)  # Extract pure JSON inside braces
    else:
        match_curly_braces = re.search(r'\{.*\}', cleaned_text, re.DOTALL)
        if match_curly_braces:
            clean_json_str = match_curly_braces.group(0)

    try:
        extracted_json = json_repair.loads(clean_json_str)
        return extracted_json

    except json_repair.JSONRepairException:
        error_filepath = os.path.join(args.output_dir, f"model_raw_output_error_case_{case_id}.txt")
        with open(error_filepath, 'w', encoding='utf-8') as f:
            f.write(generated_text)
        return None
    except json.JSONDecodeError:
        error_filepath = os.path.join(args.output_dir, f"model_raw_output_error_case_{case_id}.txt")
        with open(error_filepath, 'w', encoding='utf-8') as f:
            f.write(generated_text)
        return None
    except Exception:
        # Catch any other possible exceptions and save the original response
        error_filepath = os.path.join(args.output_dir, f"model_raw_output_error_case_{case_id}.txt")
        with open(error_filepath, 'w', encoding='utf-8') as f:
            f.write(generated_text)
        return None

# ==============================================================================
# vLLM engine initialization and data loading
# ==============================================================================

# Global vLLM engine instance, using singleton pattern to ensure initialization only once
_vllm_engine_instance = None

def get_vllm_engine(args):
    """
    Get or initialize a singleton instance of the vLLM engine.
    Integrates LLM parameter design from reference code.
    """
    global _vllm_engine_instance
    if _vllm_engine_instance is None:
        print(f"🚀 Initializing vLLM engine, model path: {args.model_path}")
        _vllm_engine_instance = LLM(
            model=args.model_path,
            dtype=args.dtype,
            tensor_parallel_size=args.tp_size,
            gpu_memory_utilization=args.gpu_util,
            max_model_len=args.max_len,
            max_num_seqs=args.max_num_seqs,
            enable_prefix_caching=True,
            disable_log_stats=False,
        )
        print("✅ vLLM engine initialized successfully.")
    return _vllm_engine_instance

def load_data_from_jsonl(jsonl_path):
    """
    Load data from a JSONL file line by line.
    Adapted to your original data structure using 'prompt' and 'reasoning' fields,
    and filter out empty `reasoning` fields.
    Also prints the number of skipped cases.
    """
    data = []
    skipped_count = 0
    total_lines = 0 
    
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            total_lines += 1
            line = line.strip()
            if not line:
                print(f"⚠️ Skipping empty line (line number: {line_num}).")
                skipped_count += 1
                continue
            try:
                item = json.loads(line)
                # Check if 'reasoning' field exists and is not empty
                thinking_text = item.get("reasoning", "")
                if not thinking_text:
                    print(f"Skipping input item (id={item.get('id', 'N/A')}, line number: {line_num}) because 'reasoning' field is empty or missing.")
                    skipped_count += 1
                    continue
                data.append(item)
            except json.JSONDecodeError as e:
                print(f"⚠️ JSONL line {line_num} parse error: {e}. Line content: {line[:100]}...")
                skipped_count += 1
                continue
            except Exception as e:
                print(f"🚨 Unknown error loading line {line_num}: {e}. Line content: {line[:100]}...")
                skipped_count += 1
                continue
                
    print(f"📖 Loaded {len(data)} valid entries from input file (total lines: {total_lines}, skipped: {skipped_count}).")
    return data

# ==============================================================================
# Main inference workflow
# ==============================================================================

def process_inference_with_vllm(input_path, output_dir, args):
    """
    Handle the main inference workflow, integrating concurrency design and JSON extraction logic.
    """
    failed_json_extraction_ids = []
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    all_data = load_data_from_jsonl(input_path)

    llm = get_vllm_engine(args)

    output_filepath = os.path.join(output_dir, args.output_filename)
    os.makedirs(output_dir, exist_ok=True) # Ensure output directory exists

    # Define sampling parameters
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=0.1, # Keep low temperature for deterministic JSON output
        top_p=0.9,
        ignore_eos=False,
    )

    print(f"🔥 Starting batch inference, total prompts: {len(all_data)}.")
    print(f"External batch size: {args.batch_size}, vLLM internal max concurrent sequences: {args.max_num_seqs}")

    buffer_processed_data = [] 
    with open(output_filepath, 'w', encoding='utf-8') as outfile:
        for i in tqdm(range(0, len(all_data), args.batch_size), desc="Processing batches"):
            batch_data = all_data[i:i + args.batch_size]
            batch_ids = [item['id'] for item in batch_data] # Get IDs in batch

            current_batch_messages = []
            for item in batch_data:

                messages_for_single_item = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"prompt: {item['prompt']}\nthinking: {item['reasoning']}"}
                ]
                current_batch_messages.append(messages_for_single_item)

            try:
                # Call llm.chat
                outputs = list(llm.chat(current_batch_messages, sampling_params))

                for j, output in enumerate(outputs):
                    original_item = batch_data[j]
                    case_id = batch_ids[j] # Use current case ID

                    extracted_analysis = robust_json_extract(output.outputs[0].text, case_id)

                    if extracted_analysis is None:
                        # JSON extraction failed, record ID
                        failed_json_extraction_ids.append(case_id)
                        # Mark as extraction failed in output file
                        result_item = {
                            "id": original_item['id'],
                            "prompt": original_item['prompt'],
                            "full_response":original_item.get('raw_response', ''),
                            "reasoning": original_item.get('reasoning', ''),
                            "response": original_item.get('response', ''), 
                            "extracted_analysis": {"error": "Extraction/Parsing Failed", 
                            "chat_model": original_item.get('model', ""),
                            "extracted_model": args.model_name,
                            "origin_dataset": original_item.get('origin_dataset', ''),
                            "extracted_analysis_truncated": output.outputs[0].text[:200]}
                        }
                    else:
                        # JSON extraction succeeded
                        result_item = {
                            "id": original_item['id'],
                            "prompt": original_item['prompt'],
                            "full_response":original_item.get('raw_response', ''),
                            "reasoning": original_item.get('reasoning', ''),
                            "response": original_item.get('response', ''), 
                            "extracted_analysis": extracted_analysis,
                            "chat_model": original_item.get('model', ""),
                            "extracted_model": args.model_name,
                            "origin_dataset": original_item.get('origin_dataset', ''),
                        }
                    outfile.write(json.dumps(result_item, ensure_ascii=False) + '\n')

            except Exception as e:
                # vLLM inference batch error, mark all IDs in batch as failed
                print(f"🚨 vLLM inference batch (index {i} to {i + args.batch_size - 1}) error: {e}")
                for original_item in batch_data:
                    case_id = original_item['id']
                    failed_json_extraction_ids.append(case_id)
                    result_item = {
                        "id": case_id,
                        "prompt": original_item['prompt'],
                        "full_response":original_item.get('raw_response', ''),
                        "reasoning": original_item.get('reasoning', ''),
                        "response": original_item.get('response', ''),
                        "origin_dataset": original_item.get('origin_dataset', ''),
                        "chat_model": original_item.get('model', ""),
                        "extracted_model": args.model_name,
                        "extracted_analysis": {"error": "vLLM Inference Failed"},
                        "extracted_analysis_output": "vLLM Inference Failed",
                    }
                    outfile.write(json.dumps(result_item, ensure_ascii=False) + '\n')
                traceback.print_exc() # Print stack trace to debug vLLM error

    # After all processing, output JSON extraction failed IDs
    if failed_json_extraction_ids:
        print("\n--- JSON Extraction Failed Case IDs ---")
        print(f"A total of {len(failed_json_extraction_ids)} cases failed JSON extraction.")
        print("Failed ID list:", failed_json_extraction_ids)
        print(f"Original error responses saved in {args.output_dir} under 'model_raw_output_error_case_*.txt' files.")
    else:
        print("\nAll cases JSON extraction succeeded.")

    print(f"--- vLLM inference script finished ---")

# ==============================================================================
# Command-line argument parsing and main entry
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run vLLM inference and extract structured JSON content.")
    parser.add_argument("--input", type=str, required=True, help="Path to input JSONL dataset.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save inference results.")
    parser.add_argument("--output_filename", type=str, required=True, help="Name for the output JSONL file.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the vLLM model.")
    parser.add_argument("--model_name", type=str, default="Qwen2-5-7B-Instruct", help="Model name for output file naming.")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Data type of model weights (e.g., bfloat16, float16).")
    parser.add_argument("--tp_size", type=int, default=8, help="Tensor parallel size (number of GPUs).")
    parser.add_argument("--gpu_util", type=float, default=0.95, help="GPU memory utilization.")
    parser.add_argument("--max_len", type=int, default=8192, help="Maximum model context length (input + output).")
    parser.add_argument("--max_tokens", type=int, default=4096, help="Max tokens to generate per request (ensure long JSON fits).")
    parser.add_argument("--batch_size", type=int, default=128, help="External batch size for each vLLM request chunk.")
    parser.add_argument("--max_num_seqs", type=int, default=256, help="vLLM internal scheduler max concurrent sequences. Higher concurrency improves throughput but increases KV cache usage.")
    parser.add_argument("--write_every", type=int, default=5, help="Write output file every N external batches.")

    args = parser.parse_args()

    print("\n" + "="*50)
    print("Optimized vLLM inference configuration (debug mode):")
    print(f"• Model: {args.model_name} (Tensor Parallel TP={args.tp_size})")
    print(f"• Max model context length: {args.max_len} tokens")
    print(f"• Max tokens per generation: {args.max_tokens}")
    print(f"• External batch size (submitted to llm.chat): {args.batch_size}")
    print(f"• vLLM internal max concurrent sequences: {args.max_num_seqs}")
    print(f"• GPU memory utilization: {args.gpu_util*100}%")
    print(f"• Output will be written every {args.write_every} external batches to JSONL file.")
    print("="*50 + "\n")

    process_inference_with_vllm(args.input, args.output_dir, args)
