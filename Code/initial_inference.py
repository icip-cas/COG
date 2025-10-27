import json
import os
import re 
from datasets import Dataset
from tqdm import tqdm
from vllm import LLM, SamplingParams
import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import re

# Global vLLM engine instance
_vllm_engine_instance = None

def get_vllm_engine(args):
    global _vllm_engine_instance
    if _vllm_engine_instance is None:
        print(f"Initializing vLLM engine with model: {args.model_path}")
        _vllm_engine_instance = LLM(
            model=args.model_path,
            dtype=args.dtype,
            tensor_parallel_size=args.tp_size,
            gpu_memory_utilization=args.gpu_util,
            max_model_len=args.max_len,
            max_num_seqs=args.max_num_seqs,
            max_num_batched_tokens=args.max_len * args.batch_size,
            enable_prefix_caching=True,
            disable_log_stats=False,
        )
        print("vLLM engine initialized.")
    return _vllm_engine_instance

def load_plain_text_jsonl_as_dataset(jsonl_path):
    data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                # Ensure the "content" field exists and use it as the prompt
                prompt = item.get("content", "")
                # Ensure the "origin_dataset" field exists
                origin_dataset = item.get("origin_dataset", "")
                data.append({
                    "prompt": prompt,  # Save the original prompt string here
                    "origin_dataset": origin_dataset
                })
            except Exception as e:
                print(f"Line {line_num} parsing error: {e}")
    return Dataset.from_list(data)

# --- New regex extraction function ---
def extract_think_content(text):
    """
    Use regex to extract the content between <think> and </think>,
    and return a dictionary containing "reasoning" and "response".
    """
    pattern = re.compile(r'<think>(.*?)</think>', re.DOTALL)
    match = pattern.search(text)

    if match:
        reasoning_content = match.group(1).strip()
        response_content = pattern.sub('', text).strip()
    else:
        reasoning_content = ""
        response_content = text.strip()

    return {
        "reasoning": reasoning_content,
        "response": response_content
    }

def extract_think_content_r1(text: str) -> dict:
    pattern = re.compile(r'</think>', re.DOTALL)  # Only match the end tag
    match = pattern.search(text)

    if match:
        reasoning_content = text[:match.start()].strip()
        response_content = text[match.end():].strip()
    else:
        reasoning_content = text.strip()
        response_content = ""

    return {
        "reasoning": reasoning_content,
        "response": response_content
    }

# Modify process_batches to use llm.chat
def process_batches(llm, batched_messages, sampling_params, chat_template_kwargs):
    all_results = []
    for messages_batch in batched_messages:
        # Call llm.chat, passing the list of messages and chat_template_kwargs
        outputs = list(llm.chat(messages_batch, sampling_params, chat_template_kwargs=chat_template_kwargs))
        all_results.append(outputs)
    return all_results

def process_jsonl_file_for_inference(input_dataset_path, base_output_dir, args):
    if not os.path.exists(input_dataset_path):
        raise FileNotFoundError(f"Input file not found: {input_dataset_path}")

    with ThreadPoolExecutor(max_workers=4) as executor:
        print(f"Loading data from {input_dataset_path}...")
        dataset = load_plain_text_jsonl_as_dataset(input_dataset_path)
        print(f"DEBUG: Number of items loaded from input JSONL: {len(dataset)}")
    llm = get_vllm_engine(args)

    output_path = os.path.join(base_output_dir, args.output_filename)
    os.makedirs(base_output_dir, exist_ok=True)

    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=0.7,
        top_p=0.8,
        presence_penalty=1.5,
        ignore_eos=False,
    )

    # chat_template_kwargs set as needed
    chat_template_kwargs = {"enable_thinking": True}

    print(f"Starting batch inference with {len(dataset)} prompts...")
    print(f"Batch size: {args.batch_size}, Max sequences: {args.max_num_seqs}")

    prompts = [item["prompt"] for item in dataset]
    origin_datasets = [item.get("origin_dataset", "") for item in dataset]

    # Prepare all batches of messages lists for llm.chat
    batched_messages = []
    batched_original_prompts = []  # Used to retrieve original prompts when processing results
    batched_origins = []  # Used to retrieve origin dataset info when processing results

    for i in range(0, len(prompts), args.batch_size):
        batch_prompts = prompts[i:i + args.batch_size]
        batch_origin_datasets = origin_datasets[i:i + args.batch_size]
        
        # Convert each prompt to a messages list, required by chat method
        messages_for_batch = [[{"role": "user", "content": p}] for p in batch_prompts]
        
        batched_messages.append(messages_for_batch)
        batched_original_prompts.append(batch_prompts)  # Save original prompt strings
        batched_origins.append(batch_origin_datasets)

    # Execute all batches asynchronously
    all_outputs = process_batches(llm, batched_messages, sampling_params, chat_template_kwargs)

    # Buffer to accumulate data to write
    buffer_processed_data = [] 
    
    # Open file in "append" mode to write results
    with open(output_path, "a", encoding="utf-8") as f:
        # Iterate through each batch of results
        for i, outputs_batch in enumerate(tqdm(all_outputs, desc="Processing batches", unit="batch")):
            current_batch_original_prompts = batched_original_prompts[i]
            current_batch_origins = batched_origins[i]

            # Iterate through each output in the current batch
            for j, output_item in enumerate(outputs_batch):
                idx_global = i * args.batch_size + j  # Calculate global index
                raw_response = output_item.outputs[0].text if output_item.outputs else ""
                
                # --- Call extract_think_content function here ---
                parsed_contents = extract_think_content(raw_response)
                extracted_reasoning = parsed_contents["reasoning"]
                final_response_text = parsed_contents["response"]
                # ----------------------------------------------------
                if not extracted_reasoning and not final_response_text:
                    continue  # Skip this iteration if both reasoning and response are empty

                processed_example = {
                    "id": idx_global,
                    "prompt": current_batch_original_prompts[j],  # Use saved original prompt
                    "raw_response": raw_response,  # Keep original response
                    "reasoning": extracted_reasoning,  # New reasoning field
                    "response": final_response_text,  # Only remaining part as response
                    "origin_dataset": current_batch_origins[j],
                    "model": args.model_name,
                }

                buffer_processed_data.append(processed_example)

            # Write to file every N batches, or at the last batch
            if (i + 1) % args.write_every == 0 or (i + 1) == len(all_outputs):
                for item_to_write in buffer_processed_data:
                    f.write(json.dumps(item_to_write, ensure_ascii=False) + "\n")
                buffer_processed_data = []  # Clear buffer

    print(f"Inference completed. Results saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to the input JSONL dataset.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the inference results.")
    parser.add_argument("--output_filename", type=str, required=True, help="Name for the output JSONL file.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the vLLM model.")
    parser.add_argument("--model_name", type=str, default="Qwen3-8B", help="Name of the model being served.")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Data type for the model (e.g., bfloat16, float16).")
    parser.add_argument("--tp_size", type=int, default=8, help="Tensor parallel size.")
    parser.add_argument("--gpu_util", type=float, default=0.85, help="GPU memory utilization.")
    parser.add_argument("--max_len", type=int, default=8192, help="Maximum model input length.")
    parser.add_argument("--max_tokens", type=int, default=1024, help="Maximum tokens to generate.")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for inference.")
    parser.add_argument("--max_num_seqs", type=int, default=512, help="Maximum number of sequences in a batch.")
    parser.add_argument("--write_every", type=int, default=5, help="Number of batches to buffer before writing to disk")

    args = parser.parse_args()

    print("\n" + "="*50)
    print("Optimized vLLM Inference Configuration:")
    print(f"• Model: {args.model_name} (TP={args.tp_size})")
    print(f"• Max length: {args.max_len} tokens")
    print(f"• Batch size: {args.batch_size}")
    print(f"• Max sequences: {args.max_num_seqs}")
    print(f"• GPU utilization: {args.gpu_util*100}%")
    print(f"• Output will be written every {args.write_every} batches to JSONL format.")
    print("="*50 + "\n")

    process_jsonl_file_for_inference(args.input, args.output_dir, args)
