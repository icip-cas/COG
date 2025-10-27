import json
import os
import re
import pandas as pd
from datasets import Dataset
from tqdm import tqdm
from vllm import LLM, SamplingParams
import argparse
from transformers import AutoTokenizer


_vllm_engine_instance = None
_tokenizer_instance = None


def get_tokenizer(args):
    global _tokenizer_instance
    if _tokenizer_instance is None:
        print(f"Initializing Tokenizer from: {args.model_path}")
        _tokenizer_instance = AutoTokenizer.from_pretrained(
            args.model_path,
            trust_remote_code=True,
        )
        print("Tokenizer initialized.")
    return _tokenizer_instance

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
    skipped_count = 0 # initialize skipped count

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                skipped_count += 1 
                continue
            try:
                item = json.loads(line)
                prompt = item.get("prompt", "")
                reasoning = item.get("COT", "")

                if not prompt or not reasoning:
                    skipped_count += 1 # count skipped entries
                    continue

                data.append({
                    "prompt": prompt,
                    "reasoning": reasoning
                })
            except Exception as e:
                print(f"第 {line_num} 行解析错误：{e}")
                skipped_count += 1 # 解析错误也计入跳过
    
    print(f"---------------------------------------------------------------")
    print(f"数据加载完成。总共跳过了 {skipped_count} 条数据。")
    print(f"---------------------------------------------------------------")
    return Dataset.from_list(data)



def process_jsonl_file_for_inference(input_dataset_path, base_output_dir, args):
    if not os.path.exists(input_dataset_path):
        raise FileNotFoundError(f"Input file not found: {input_dataset_path}")

    tokenizer = get_tokenizer(args)
    llm = get_vllm_engine(args)

    print(f"Loading data from {input_dataset_path}...")
    dataset = load_plain_text_jsonl_as_dataset(input_dataset_path)
    

    output_path = os.path.join(
        base_output_dir,
        args.output_filename 
    )
    os.makedirs(base_output_dir, exist_ok=True)

    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=0.7,
        top_p=0.8,
        presence_penalty=1.5,
        ignore_eos=False,
    )

    prompts = [item["prompt"] for item in dataset]
    reasoning_from_source = [item["reasoning"] for item in dataset] 

    print("Applying chat template to construct final prompts...")
    final_prompts = []
    all_cots = []
    for user_prompt, reason in tqdm(zip(prompts, reasoning_from_source), desc="Constructing Prompts", total=len(prompts)):
        COT = f"<think>\n{reason}</think>" if reason else ""
        all_cots.append(COT)
        messages = [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": COT},
        ]
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True 
        ).removesuffix("\n\n<|im_end|>\n<|im_start|>assistant\n")
        final_prompts.append(formatted_prompt)

    print(f"Starting batch inference with {len(final_prompts)} prompts...")
    all_outputs = llm.generate(final_prompts, sampling_params)


    buffer_processed_data = []
    with open(output_path, "w", encoding="utf-8") as f: # 使用 'w' 模式重新写入文件
        for i, output_item in enumerate(tqdm(all_outputs, desc="Processing results", unit="item")):
            original_prompt = prompts[i]
            

            processed_reasoning = all_cots[i] 

            model_generated_answer = output_item.outputs[0].text if output_item.outputs else ""
            

            concatenated_output = processed_reasoning + model_generated_answer 

            processed_example = {
                "id": i,
                "prompt": original_prompt,           
                "reasoning": processed_reasoning,     
                "answer": model_generated_answer,    
                "output": concatenated_output,       
                "model": args.model_name,
            }
            
            f.write(json.dumps(processed_example, ensure_ascii=False) + "\n")

    print(f"Inference completed. Results saved to: {output_path}")


# --- 主程序入口 (无修改) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="High-concurrency inference with vLLM using custom chat templates.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input JSONL dataset.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the inference results.")
    parser.add_argument("--output_filename", type=str, default="inference_results.jsonl", 
                        help="Name of the output JSONL file. Default is 'inference_results.jsonl'.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the vLLM model and tokenizer.")
    parser.add_argument("--model_name", type=str, default="Qwen3-8B", help="Name of the model being served.")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Data type for the model (e.g., bfloat16, float16).")
    parser.add_argument("--tp_size", type=int, default=1, help="Tensor parallel size.")
    parser.add_argument("--gpu_util", type=float, default=0.9, help="GPU memory utilization.")
    parser.add_argument("--max_len", type=int, default=8192, help="Maximum model input length.")
    parser.add_argument("--max_tokens", type=int, default=1024, help="Maximum tokens to generate.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference.")
    parser.add_argument("--max_num_seqs", type=int, default=256, help="Maximum number of sequences in a batch.")
    parser.add_argument("--write_every", type=int, default=1, help="Number of batches to buffer before writing to disk")

    args = parser.parse_args()

    print("\n" + "="*50)
    print("Optimized vLLM Inference Configuration:")
    print(f"• Model: {args.model_name} (TP={args.tp_size}) at {args.model_path}")
    print(f"• Max length: {args.max_len} tokens")
    print(f"• Batch size: {args.batch_size}")
    print(f"• Max sequences: {args.max_num_seqs}")
    print(f"• GPU utilization: {args.gpu_util*100}%")
    print(f"• Output filename: {args.output_filename}") 
    print("="*50 + "\n")

    process_jsonl_file_for_inference(args.input, args.output_dir, args)
