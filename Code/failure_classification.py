import os
import json
import re
import argparse
from typing import List, Dict, Optional
from tqdm import tqdm
from collections import defaultdict
import traceback 
from vllm import LLM, SamplingParams

SYSTEM_PROMPT = """ 

"""

def transform_data_for_classification(complex_case: dict) -> Optional[dict]:
    """
    Extract core fields needed for classification from a complex input case.
    This function remains unchanged.
    """
    prompt = complex_case.get("prompt")
    danger_analysis = None
    response_consideration = None
    extracted_analysis_data = complex_case.get("extracted_analysis")
    extracted_analysis = {}

    if isinstance(extracted_analysis_data, str):
        try:
            extracted_analysis = json.loads(extracted_analysis_data)
        except json.JSONDecodeError:
            pass
    elif isinstance(extracted_analysis_data, dict):
        extracted_analysis = extracted_analysis_data

    if extracted_analysis:
        danger_analysis = extracted_analysis.get("danger_analysis")
        response_decision = extracted_analysis.get("response_decision")
        if response_decision and isinstance(response_decision, dict):
            response_consideration = response_decision.get("response_strategy")

    thinking_response = complex_case.get("thinking_response", complex_case.get("reasoning", ""))
    final_response = None
    if thinking_response:
        parts = thinking_response.split("</think>", 1)
        if len(parts) > 1:
            final_response = parts[1].strip()
        else:
            final_response = thinking_response.strip()

    required_data = {
        "prompt": prompt,
        "Risk_Analysis": danger_analysis,
        "Response_Consideration": response_consideration,
        "Final_Response": final_response,
    }

    if not all(required_data.values()):
        return None

    return required_data


def format_prompt_for_vllm_chat(simple_case: Dict[str, str]) -> List[Dict[str, str]]:
    """
    Format structured data into a message list suitable for vLLM's `llm.chat` method.
    This function remains unchanged.
    """
    user_content = f"""
<Prompt>
{simple_case["prompt"]}
</Prompt>
<Risk_Analysis>
{simple_case["Risk_Analysis"]}
</Risk_Analysis>
<Response_Consideration>
{simple_case["Response_Consideration"]}
</Response_Consideration>
<Final_Response>
{simple_case["Final_Response"]}
</Final_Response>
"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]
    return messages


def load_data_from_json(input_path: str) -> List[Dict]:
    """
    Load data from JSON or JSONL files.
    This function remains unchanged.
    """
    data = []
    skipped_count = 0
    total_lines = 0

    try:
        with open(input_path, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
            f.seek(0)

            if first_line.startswith('{') and first_line.endswith('}'):
                for line_num, line in enumerate(f, 1):
                    total_lines += 1
                    line = line.strip()
                    if not line:
                        skipped_count += 1
                        continue
                    try:
                        item = json.loads(line)
                        data.append(item)
                    except json.JSONDecodeError as e:
                        print(f"⚠️ JSONL line {line_num} parse error: {e}. Line content: {line[:100]}...")
                        skipped_count += 1
                        continue
            else:
                full_content = f.read()
                data = json.loads(full_content)
                if not isinstance(data, list):
                    raise ValueError("JSON file content is not a list.")
                total_lines = len(data)

    except FileNotFoundError:
        print(f"Error: Input file not found -> {input_path}")
        return []
    except json.JSONDecodeError as e:
        print(f"Error: Input file '{input_path}' is not valid JSON or JSONL: {e}")
        return []
    except Exception as e:
        print(f"🚨 Unknown error loading data: {e}")
        return []

    print(f"📖 Loaded {len(data)} raw entries from input file (total lines/items: {total_lines}, skipped lines in original file: {skipped_count}).")
    return data


def process_with_vllm_library(
    input_path: str, output_dir: str, model_path: str, model_name: str, 
    dtype: str, tp_size: int, gpu_util: float, max_len: int, max_tokens: int,
    batch_size: int, max_num_seqs: int, write_every: int
):
    """
    Main processing function. Directly calls the vLLM library for batch inference,
    including concurrency logic and real-time writing.
    All statistics prints have been removed.
    """
    # Build the full path of the output file
    output_filename = f"{os.path.basename(os.path.splitext(input_path)[0])}_classified_by_{model_name.replace('/', '_')}.jsonl"
    output_filepath = os.path.join(output_dir, output_filename)
    os.makedirs(output_dir, exist_ok=True)

    # --- Step 1: Initialize vLLM model ---
    print(f"Loading vLLM engine using model '{model_path}' ({model_name})...")
    try:
        llm = LLM(
            model=model_path,
            dtype=dtype,
            tensor_parallel_size=tp_size,
            gpu_memory_utilization=gpu_util,
            max_model_len=max_len,
            max_num_seqs=max_num_seqs,
            enable_prefix_caching=True,
            trust_remote_code=True,
        )
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Model loading failed: {e}")
        print("Please check model path, dtype, GPU memory configuration, and ensure sufficient GPU memory.")
        return

    # --- Step 2: Define sampling parameters ---
    sampling_params = SamplingParams(
        temperature=0.1, 
        max_tokens=max_tokens,
        top_p=0.9,
        ignore_eos=False,
    )

    # --- Step 3: Load and prepare data ---
    all_raw_data = load_data_from_json(input_path)
    
    prepared_cases_for_inference = []
    skipped_data_for_inference_count = 0
    for idx, raw_case in enumerate(all_raw_data):
        transformed_input_data = transform_data_for_classification(raw_case)
        
        if transformed_input_data:
            messages_for_llm_chat = format_prompt_for_vllm_chat(transformed_input_data)
            # Store (original index, original raw data, formatted messages)
            prepared_cases_for_inference.append((idx, raw_case, messages_for_llm_chat))
        else:
            # Mark original data item's classification_result
            all_raw_data[idx]["classification_result"] = {"error": "Skipped: Input data incomplete/missing for classification"}
            all_raw_data[idx]["extracted_model_name"] = model_name
            skipped_data_for_inference_count += 1
    
    if skipped_data_for_inference_count > 0:
        print(f"Warning: {skipped_data_for_inference_count} cases were skipped due to incomplete input data for constructing classification prompts.")

    # --- Step 4: Batch inference using llm.chat with real-time writing ---
    if not prepared_cases_for_inference:
        print("No valid cases for inference. Please check the input file.")
        with open(output_filepath, 'w', encoding='utf-8') as outfile:
            # Write all raw data (including skipped) to file
            json.dump({"results": all_raw_data}, outfile, indent=2, ensure_ascii=False)
        return
    
    print(f"Data preparation complete. {len(prepared_cases_for_inference)} valid cases ready for inference.")
    print(f"External batch size (batch_size): {batch_size}, vLLM internal max concurrent sequences (max_num_seqs): {max_num_seqs}")
    print(f"Results will be written every {write_every} batches to: {output_filepath}")

    current_batch_outputs_buffer = [] # Buffer for results to write
    
    with open(output_filepath, 'w', encoding='utf-8') as outfile:

        for i in tqdm(range(0, len(prepared_cases_for_inference), batch_size), desc="Processing batches"):
            batch_data_tuples = prepared_cases_for_inference[i : i + batch_size]
            current_batch_messages = [item[2] for item in batch_data_tuples]

            try:
                outputs = llm.chat(current_batch_messages, sampling_params)

                for j, output in enumerate(outputs):
                    original_idx, original_raw_case, _ = batch_data_tuples[j]
                    case_id = original_raw_case.get('id', f'unknown_id_{original_idx}')
                    generated_text = output.outputs[0].text.strip()

                    result_item = original_raw_case.copy()
                    result_item["extracted_model_name"] = model_name

                    # --- Core modification: use regex matching directly ---
                    try:
                        outcome_match = re.search(r"<outcome>(.*?)</outcome>", generated_text, re.DOTALL)
                        root_cause_match = re.search(r"<root_cause>(.*?)</root_cause>", generated_text, re.DOTALL)
                        justification_match = re.search(r"<justification>(.*?)</justification>", generated_text, re.DOTALL)

                        outcome = outcome_match.group(1).strip() if outcome_match else 'N/A'
                        root_cause = root_cause_match.group(1).strip() if root_cause_match else 'N/A'
                        justification = justification_match.group(1).strip() if justification_match else ''

                        result_item["classification_result"] = {
                            "outcome": outcome,
                            "root_cause": root_cause,
                            "justification": justification
                        }
                    except (AttributeError, IndexError) as parse_error:
                        result_item["classification_result"] = {"error": f"Model response format mismatch (Regex parsing failed): {parse_error}"}
                        result_item["raw_model_response"] = generated_text

                    current_batch_outputs_buffer.append(result_item)

            except Exception as e:
                print(f"\n🚨 vLLM inference batch (index {i} to {i + batch_size - 1}) failed: {e}")
                for original_idx, original_raw_case, _ in batch_data_tuples:
                    result_item = original_raw_case.copy()
                    result_item["classification_result"] = {"error": "vLLM Inference Failed"}
                    result_item["extracted_model_name"] = model_name
                    current_batch_outputs_buffer.append(result_item)
                traceback.print_exc()

            # Write buffer every `write_every` batches
            if (i // batch_size + 1) % write_every == 0 or (i + batch_size >= len(prepared_cases_for_inference)):
                for item_to_write in current_batch_outputs_buffer:
                    outfile.write(json.dumps(item_to_write, ensure_ascii=False) + '\n')
                current_batch_outputs_buffer = []

        # Ensure remaining buffer is written
        if current_batch_outputs_buffer:
            for item_to_write in current_batch_outputs_buffer:
                outfile.write(json.dumps(item_to_write, ensure_ascii=False) + '\n')
            current_batch_outputs_buffer = []

    print(f"\nProcessing complete! Results saved to: {output_filepath}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch classify JSON/JSONL files using the vLLM library. Supports full vLLM parameter configuration, concurrent inference, and regex-based XML/text parsing. No statistics output.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input JSON or JSONL file containing complex structured cases.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save inference results. The script will generate a JSONL file here.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to vLLM model (e.g., /mnt/data1/hf_models/Qwen2.5-72B-Instruct).",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen2.5-72B-Instruct",
        help="Model name (e.g., Qwen2.5-72B-Instruct) for output file naming and logging.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        help="Data type of model weights (e.g., bfloat16, float16).",
    )
    parser.add_argument(
        "--tp_size",
        type=int,
        default=8,
        help="Tensor parallel size (number of GPUs).",
    )
    parser.add_argument(
        "--gpu_util",
        type=float,
        default=0.95,
        help="GPU memory utilization (0.0-1.0).",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=8192,
        help="Maximum model context length (input + output tokens).",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=4096,
        help="Maximum tokens generated per request (ensure longest XML output fits).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="External batch size, controlling how many requests are submitted to vLLM at once.",
    )
    parser.add_argument(
        "--max_num_seqs",
        type=int,
        default=64,
        help="vLLM internal scheduler maximum concurrent sequences.",
    )
    parser.add_argument(
        "--write_every",
        type=int,
        default=5,
        help="Write to output file (JSONL) every N external batches.",
    )

    args = parser.parse_args()

    print("\n" + "="*50)
    print("Enhanced vLLM inference configuration (no statistics output):")
    print(f"• Input file: {args.input}")
    print(f"• Output directory: {args.output_dir}")
    print(f"• Model path: {args.model_path}")
    print(f"• Model name: {args.model_name}")
    print(f"• Data type (dtype): {args.dtype}")
    print(f"• Tensor parallel size (tp_size): {args.tp_size}")
    print(f"• GPU memory utilization (gpu_util): {args.gpu_util*100:.0f}%")
    print(f"• Max context length (max_len): {args.max_len} tokens")
    print(f"• Max tokens per generation (max_tokens): {args.max_tokens} tokens")
    print(f"• External batch size (batch_size): {args.batch_size}")
    print(f"• vLLM internal max concurrent sequences (max_num_seqs): {args.max_num_seqs}")
    print(f"• Write frequency (write_every): every {args.write_every} batches")
    print("="*50 + "\n")

    process_with_vllm_library(
        args.input, args.output_dir, args.model_path, args.model_name,
        args.dtype, args.tp_size, args.gpu_util, args.max_len, args.max_tokens,
        args.batch_size, args.max_num_seqs, args.write_every
    )


if __name__ == "__main__":
    main()
