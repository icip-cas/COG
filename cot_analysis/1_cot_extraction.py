import json
import re
import gc
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

CONFIG = {
    "dataset_path": "./Data/math2024.jsonl",

    "model_definitions": {
        "Base": "Qwen3-32B",
        "SafR": "Qwen3_32B_SFT_SafR",
        "SafB": "Qwen3_32B_SFT_SafB",
        "XXX": "Qwen3_32B_SFT_XXX"
    },

    "output_filename": "./Result/COT_analysis/math_parsed.json",

    # Inference parameter optimization
    "max_tokens": 32768,
    "tensor_parallel_size": 8,
    "batch_size": 10,  # Increase batch size
    "gpu_memory_utilization": 0.85,  # Increase GPU memory utilization
    "max_num_seqs": 30,  # Increase maximum number of sequences
}


def parse_model_output(raw_text: str) -> dict:
    """Parse model output, separate <think> section and final answer"""
    if not raw_text:
        return {"thinking_process": "", "final_answer": ""}

    think_match = re.search(r"<think>(.*?)</think>", raw_text, re.DOTALL)
    if think_match:
        thinking_process = think_match.group(1).strip()
        final_answer = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
        return {"thinking_process": thinking_process, "final_answer": final_answer}
    else:
        return {"thinking_process": "", "final_answer": raw_text.strip()}


def load_dataset(filepath: str) -> list:
    """Load dataset from a jsonl file."""
    dataset = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                # Extract the prompt field to avoid errors later
                if isinstance(obj, dict) and "prompt" in obj:
                    dataset.append(obj)
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found. Please check CONFIG['dataset_path']")
        exit()
    except Exception as e:
        print(f"Error reading or parsing the file: {e}")
        exit()
    return dataset


def process_batch(prompts_batch, tokenizer, sampling_params, llm):
    """Process a single batch for inference"""
    # Construct messages
    messages_list = [[{"role": "user", "content": p["prompt"]}] for p in prompts_batch]

    # Format prompts and enable thinking mode
    formatted_prompts = tokenizer.apply_chat_template(
        messages_list,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )

    # Batch inference
    outputs = llm.generate(formatted_prompts, sampling_params)

    # Process results
    batch_results = []
    for i, output in enumerate(outputs):
        raw = ""
        if getattr(output, "outputs", None):
            if output.outputs:
                raw = output.outputs[0].text.strip()

        parsed = parse_model_output(raw)

        batch_results.append({
            "question": prompts_batch[i]["prompt"],
            "full_response": raw,
            "thinking_process": parsed["thinking_process"],
            "final_answer": parsed["final_answer"],
        })

    return batch_results


def run_inference(prompts, model_name, model_path, max_tokens, tensor_parallel_size,
                 batch_size, gpu_memory_utilization, max_num_seqs):
    """Inference process for a single model, supports batch processing"""
    print(f"\n{'='*50}\nLoading model: {model_name} | Path: {model_path}")

    try:
        llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=max_num_seqs,
            enforce_eager=False,  
            enable_prefix_caching=True,  
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception as e:
        print(f"Loading failed: {e}")
        return None

    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        max_tokens=max_tokens
    )

    print(f"Performing batch inference on {len(prompts)} samples, batch size: {batch_size}")

    all_results = []
    total_batches = (len(prompts) + batch_size - 1) // batch_size

    # Batch processing
    for batch_idx in tqdm(range(total_batches), desc=f"{model_name} batch progress"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(prompts))
        prompts_batch = prompts[start_idx:end_idx]

        batch_results = process_batch(prompts_batch, tokenizer, sampling_params, llm)
        all_results.extend(batch_results)

        # Clear memory every few batches
        if (batch_idx + 1) % 10 == 0:
            torch.cuda.empty_cache()

    # Release GPU memory to avoid OOM with multiple models
    try:
        llm.close()
    except Exception:
        pass
    del llm
    torch.cuda.empty_cache()
    gc.collect()

    return all_results


def main():
    dataset_path = CONFIG["dataset_path"]
    model_definitions = CONFIG["model_definitions"]
    output_filename = CONFIG["output_filename"]
    max_tokens = CONFIG["max_tokens"]
    tensor_parallel_size = CONFIG["tensor_parallel_size"]
    batch_size = CONFIG["batch_size"]
    gpu_memory_utilization = CONFIG["gpu_memory_utilization"]
    max_num_seqs = CONFIG["max_num_seqs"]

    prompts = load_dataset(dataset_path)
    print(f"Successfully loaded {len(prompts)} samples")

    final_results = []

    for model_name, model_path in model_definitions.items():
        results = run_inference(
            prompts, model_name, model_path, max_tokens, tensor_parallel_size,
            batch_size, gpu_memory_utilization,  max_num_seqs
        )
        if results is None:
            continue
        for i, r in enumerate(results):
            problem_id = f"problem_{i+1}"
            if len(final_results) <= i:
                final_results.append({
                    "id": problem_id,
                    "question": r["question"],
                    "model_outputs": {}
                })
            final_results[i]["model_outputs"][model_name] = {
                "full_response": r["full_response"],
                "thinking_process": r["thinking_process"],
                "final_answer": r["final_answer"],
            }

    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=4)

    print(f"\n{'='*50}\nAll inference completed, results saved to {output_filename}")


if __name__ == "__main__":
    main()
