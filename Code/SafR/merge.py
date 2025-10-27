import os
import json
import re
import argparse
from typing import List, Dict, Optional
from tqdm import tqdm
import traceback

from vllm import LLM, SamplingParams


SYSTEM_PROMPT = """
"""


def prepare_data_for_cot_generation(complex_case: dict, verbose: bool = True) -> Optional[dict]:
    """
    Prepare input data for CoT generation tasks.
    Extracts 'prompt', 'danger_analysis', and 'reply_thinking' from the raw case.
    If any field is missing, prints a warning and skips the case.
    """
    case_id = complex_case.get("id", "no_id")
    prompt = complex_case.get("prompt")
    rewritten_analysis = complex_case.get("rewritten_analysis", {})
    
    if not isinstance(rewritten_analysis, dict):
        rewritten_analysis = {}

    danger_analysis = rewritten_analysis.get("danger_analysis")
    response_strategy = rewritten_analysis.get("reply_thinking")

    missing_fields = []
    if not prompt:
        missing_fields.append("prompt")
    if not danger_analysis:
        missing_fields.append("danger_analysis")
    if not response_strategy:
        missing_fields.append("reply_thinking")

    if missing_fields:
        if verbose:
            print(f"⚠️ Skipping {case_id}, missing fields: {', '.join(missing_fields)}")
        return None

    return {
        "prompt": prompt,
        "danger_analysis": danger_analysis,
        "response_strategy": response_strategy,
    }



def format_prompt_for_vllm(simple_case: Dict[str, str]) -> List[Dict[str, str]]:
    """
    Format the prepared data into the message list required by vLLM `llm.chat`.
    """
    # Note: The tags here correspond exactly to the 【Input Structure】 section of the new System Prompt
    user_content = f"""
<User Prompt>
{simple_case["prompt"]}
</User Prompt>
<Danger Analysis>
{simple_case["danger_analysis"]}
</Danger Analysis>
<Response Strategy>
{simple_case["response_strategy"]}
</Response Strategy>
"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    return messages


def load_data_from_json(input_path: str) -> List[Dict]:
    """
    Load data from JSON or JSONL file.
    """
    data = []
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    except Exception as e:
        print(f"🚨 Error loading data: {e}")
        return []
    print(f"📖 Loaded {len(data)} raw data entries from the input file.")
    return data


def process_with_vllm_for_cot(
    input_path: str,
    output_dir: str,
    model_path: str,
    model_name: str,
    dtype: str,
    tp_size: int,
    gpu_util: float,
    max_len: int,
    max_tokens: int,
    batch_size: int,
    max_num_seqs: int,
    write_every: int,
    output_filename: str,
):
    """
    Main processing function for batch Chain of Thought generation.
    """
    output_filepath = os.path.join(output_dir, output_filename)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading vLLM engine using model '{model_path}'...")
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
        return

    sampling_params = SamplingParams(temperature=0.4, max_tokens=max_tokens, top_p=0.9, presence_penalty=1.5)
    all_raw_data = load_data_from_json(input_path)

    prepared_cases = []
    for raw_case in all_raw_data:
        prepared_data = prepare_data_for_cot_generation(raw_case)
        if prepared_data:
            messages = format_prompt_for_vllm(prepared_data)
            prepared_cases.append({"original_case": raw_case, "messages": messages})
        else:
            print(f"⛔ Skipping data item due to missing fields")
            if not raw_case.get("prompt"):
                print("   - Missing prompt")
            if not isinstance(raw_case.get("rewritten_analysis"), dict):
                print("   - rewritten_analysis is not a dict")
            else:
                if not raw_case["rewritten_analysis"].get("danger_analysis"):
                    print("   - Missing danger_analysis")
                if not raw_case["rewritten_analysis"].get("reply_thinking"):
                    print("   - Missing reply_thinking")

            raw_case["generated_chain_of_thought"] = (
                "Error: Skipped due to missing input fields."
            )
            with open(output_filepath, "a", encoding="utf-8") as outfile:
                outfile.write(json.dumps(raw_case, ensure_ascii=False) + "\n")

    print(f"Data preparation complete. {len(prepared_cases)} valid cases ready for inference.")

    with open(output_filepath, "a", encoding="utf-8") as outfile:
        for i in tqdm(range(0, len(prepared_cases), batch_size), desc="Generating Chain of Thought"):
            batch = prepared_cases[i : i + batch_size]
            batch_messages = [item["messages"] for item in batch]

            try:
                outputs = llm.chat(batch_messages, sampling_params)

                for j, output in enumerate(outputs):
                    result_item = batch[j]["original_case"]
                    generated_text = output.outputs[0].text.strip()

                    split_marker = "</think>\n\n"
                    if split_marker in generated_text:
                        cot_only = generated_text.split(split_marker, 1)[1].strip()
                    else:
                        cot_only = generated_text
                    result_item["generated_chain_of_thought"] = generated_text
                    result_item["COT"] = cot_only

                    outfile.write(json.dumps(result_item, ensure_ascii=False) + "\n")

            except Exception as e:
                print(f"\n🚨 vLLM inference batch (index {i}) failed: {e}")
                traceback.print_exc()
                for item in batch:
                    error_item = item["original_case"]
                    error_item["generated_chain_of_thought"] = (
                        f"Error: vLLM Inference Failed - {e}"
                    )
                    outfile.write(json.dumps(error_item, ensure_ascii=False) + "\n")

    print(f"\nProcessing complete! Results saved to: {output_filepath}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch generate Chain of Thought (CoT) using vLLM.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Path to the input JSONL file."
    )
    parser.add_argument("--output_filename", type=str, required=True, help="Name for the output JSONL file.")
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save results."
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the vLLM model."
    )
    parser.add_argument(
        "--model_name", type=str, default="model", help="Model name used for output."
    )
    parser.add_argument(
        "--dtype", type=str, default="bfloat16", help="Data type (bfloat16, float16)."
    )
    parser.add_argument(
        "--tp_size", type=int, default=8, help="Tensor parallel size (number of GPUs)."
    )
    parser.add_argument("--gpu_util", type=float, default=0.95, help="GPU memory utilization.")
    parser.add_argument(
        "--max_len", type=int, default=8192, help="Maximum model context length."
    )
    parser.add_argument(
        "--max_tokens", type=int, default=2048, help="Maximum tokens to generate."
    )
    parser.add_argument("--batch_size", type=int, default=32, help="External batch size.")
    parser.add_argument(
        "--max_num_seqs", type=int, default=64, help="vLLM internal max concurrent sequences."
    )
    parser.add_argument(
        "--write_every", type=int, default=5, help="(Not used in this version, writes in real-time)"
    )

    args = parser.parse_args()

    print("\n" + "=" * 50)
    print("vLLM Chain of Thought generation task configuration:")
    print(f"• Input file: {args.input}")
    print(f"• Output directory: {args.output_dir}")
    print(f"• Model path: {args.model_path}")
    print("=" * 50 + "\n")

    process_with_vllm_for_cot(
        args.input,
        args.output_dir,
        args.model_path,
        args.model_name,
        args.dtype,
        args.tp_size,
        args.gpu_util,
        args.max_len,
        args.max_tokens,
        args.batch_size,
        args.max_num_seqs,
        args.write_every,
        args.output_filename,
    )


if __name__ == "__main__":
    main()
