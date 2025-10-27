import os
import json
import re
import argparse
import time
from typing import List, Dict, Optional
from tqdm import tqdm
from collections import defaultdict

from vllm import LLM, SamplingParams

_vllm_engine_instance = None


def get_vllm_engine(args):
    global _vllm_engine_instance
    if _vllm_engine_instance is None:
        print(f"Initializing vLLM engine using model: {args.model_path}...")
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


def load_prompts_from_file(prompt_file_path: str) -> Dict[str, str]:
    try:
        with open(prompt_file_path, "r", encoding="utf-8") as f:
            prompts = json.load(f)
        return prompts
    except FileNotFoundError:
        print(f"Error: Prompt file not found -> {prompt_file_path}")
        return {}
    except json.JSONDecodeError:
        print(f"Error: Prompt file is not valid JSON -> {prompt_file_path}")
        return {}


def transform_data_for_classification(complex_case: dict) -> Optional[dict]:
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
        response_consideration = (
            response_decision.get("response_strategy") if response_decision else None
        )

    model_raw_output_with_think = complex_case.get("full_response", "")
    
    final_response = None
    if model_raw_output_with_think:
        parts = model_raw_output_with_think.split("</think>", 1)
        if len(parts) > 1:
            final_response = parts[1].strip()
        else:
            final_response = complex_case.get("response", "").strip()
            if not final_response:
                final_response = model_raw_output_with_think.strip()

    required_data = {
        "prompt": prompt,
        "Risk_Analysis": danger_analysis,
        "Response_Consideration": response_consideration,
        "Final_Response": final_response,
    }

    if not all(value is not None and value != "" for value in required_data.values()):
        missing_keys = [
            key for key, value in required_data.items() if value is None or value == ""
        ]
        print(
            f"Warning: Required fields missing in case '{complex_case.get('id')}': {missing_keys}, skipped."
        )
        return None

    return required_data


def call_vllm_batch(
    llm_engine: LLM,
    simple_cases_batch: List[Dict[str, str]],
    root_causes_batch: List[str],
    prompts: Dict[str, str],
    sampling_params: SamplingParams,
    chat_template_kwargs: Dict,
) -> List[Optional[str]]:
    batched_messages_for_vllm: List[List[Dict[str, str]]] = []

    for i, simple_case in enumerate(simple_cases_batch):
        root_cause = root_causes_batch[i]

        main_prompt = prompts.get("main_prompt", "")
        sub_prompt = prompts.get("sub_prompts", {}).get(root_cause, "")
        format_prompt = prompts.get("format_prompt", "")
        full_system_prompt = main_prompt + sub_prompt + format_prompt

        user_content = f"""
{{{{ORIGINAL_PROMPT}}}}
{simple_case["prompt"]}

{{{{DANGER_ANALYSIS_PROCESS}}}}
{simple_case["Risk_Analysis"]}

{{{{REPLY_THINKING}}}}
{simple_case["Response_Consideration"]}

{{{{FINAL_RESPONSE}}}}
{simple_case["Final_Response"]}
"""
        messages_for_single_case = [
            {"role": "system", "content": full_system_prompt},
            {"role": "user", "content": user_content},
        ]
        batched_messages_for_vllm.append(messages_for_single_case)

    try:
        outputs = llm_engine.chat(
            batched_messages_for_vllm,
            sampling_params,
            chat_template_kwargs=chat_template_kwargs,
        )
        return [
            output.outputs[0].text.strip() if output.outputs else ""
            for output in outputs
        ]
    except Exception as e:
        print(f"vLLM inference failed (batch): {e}")
        return [None] * len(simple_cases_batch)


def extract_structured_output(api_response: str) -> Dict[str, str]:
    patterns = {
        "danger_analysis": r"\[DANGER_ANALYSIS_START\](.*?)\[DANGER_ANALYSIS_END\]",
        "reply_thinking": r"\[REPLY_THINKING_START\](.*?)\[REPLY_THINKING_END\]",
    }

    result = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, api_response, re.DOTALL)
        if match:
            result[key] = match.group(1).strip()
        else:
            result[key] = ""

    return result


def process_complex_cases(
    input_path: str, output_path: str, prompt_file_path: str, args
):
    if not os.path.exists(input_path):
        print(f"Error: Input file not found -> {input_path}")
        return

    prompts = load_prompts_from_file(prompt_file_path)
    if not prompts:
        print("Failed to load prompt file, exiting.")
        return

    complex_cases: List[Dict] = []
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    complex_cases.append(json.loads(line))
    except FileNotFoundError:
        print(f"Error: Input file not found -> {input_path}")
        return
    except json.JSONDecodeError as e:
        print(f"Error: Input file is not valid JSONL (line parse failed: {e}) -> {input_path}")
        return

    print(f"Found {len(complex_cases)} cases, starting batch rewriting...")

    llm_engine = get_vllm_engine(args)
    processing_counts = defaultdict(int)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        frequency_penalty=args.frequency_penalty,
        max_tokens=args.max_tokens,
        presence_penalty=1.5,
    )
    chat_template_kwargs = {"enable_thinking": True}

    current_batch_simple_cases: List[Dict[str, str]] = []
    current_batch_root_causes: List[str] = []
    current_batch_original_indices: List[int] = []

    tqdm_bar = tqdm(complex_cases, desc="Preparing batches", unit="case")

    for idx, case in enumerate(tqdm_bar):
        if case.get("rewritten_analysis"):
            processing_counts["Skipped (already rewritten)"] += 1
            continue

        classification_result = case.get("classification_result", {})
        root_cause = classification_result.get("root_cause", "")

        if not root_cause:
            processing_counts["Skipped (missing root_cause)"] += 1
            continue

        simple_case_for_api = transform_data_for_classification(case)
        if not simple_case_for_api:
            processing_counts["Skipped (incomplete input)"] += 1
            continue

        current_batch_simple_cases.append(simple_case_for_api)
        current_batch_root_causes.append(root_cause)
        current_batch_original_indices.append(idx)

        if len(current_batch_simple_cases) >= args.batch_size or (
            idx == len(complex_cases) - 1 and current_batch_simple_cases
        ):
            tqdm_bar.set_description(f"Sending batch ({len(current_batch_simple_cases)} cases)")

            batch_vllm_responses = call_vllm_batch(
                llm_engine,
                current_batch_simple_cases,
                current_batch_root_causes,
                prompts,
                sampling_params,
                chat_template_kwargs,
            )

            for i_batch, api_response in enumerate(batch_vllm_responses):
                original_idx = current_batch_original_indices[i_batch]
                original_case = complex_cases[original_idx]

                if api_response:
                    extracted_data = extract_structured_output(api_response)
                    if extracted_data.get("danger_analysis") and extracted_data.get("reply_thinking"):
                        original_case["rewritten_analysis"] = {
                            "danger_analysis": extracted_data["danger_analysis"],
                            "reply_thinking": extracted_data["reply_thinking"],
                            "raw_response": api_response,
                        }
                        processing_counts["Successfully processed"] += 1
                    else:
                        original_case["rewritten_analysis"] = {
                            "error": api_response,
                            "raw_response": api_response,
                        }
                        print("Warning: Response format error, failed to extract necessary fields.")
                        processing_counts["Response format error"] += 1
                else:
                    original_case["rewritten_analysis"] = {"error": "vLLM no response"}
                    processing_counts["vLLM no response"] += 1

            current_batch_simple_cases = []
            current_batch_root_causes = []
            current_batch_original_indices = []
            tqdm_bar.set_description("Preparing batches")

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            for case in complex_cases:
                f.write(json.dumps(case, ensure_ascii=False) + '\n')
            print(f"\nProcessing complete! Results saved to: {output_path}")
    except IOError as e:
            print(f"Error: Cannot write to output file -> {output_path}, error: {e}")

    print("\n--- Processing Summary ---")
    total_processed = len(complex_cases)
    sorted_counts = sorted(
        processing_counts.items(), key=lambda item: item[1], reverse=True
    )
    for category, count in sorted_counts:
        percentage = (count / total_processed) * 100
        print(f"- {category}: {count} times ({percentage:.2f}%)")
    print("----------------------")


def main():
    parser = argparse.ArgumentParser(
        description="Rewrite complex cases with safety thinking using vLLM."
    )
    current_dir = os.path.dirname(__file__)
    json_path = os.path.join(current_dir, "rewite_prompt_en.json")
    parser.add_argument("input_path", type=str, help="Input JSON file path.")
    parser.add_argument("-o", "--output", type=str, help="Output JSON file path.")
    parser.add_argument(
        "-p",
        "--prompts",
        type=str,
        default=json_path,
        help="Prompt configuration file path.",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=10,
        help="Batch size for vLLM inference.",
    )

    parser.add_argument(
        "--model_path", type=str, required=True, help="vLLM model path in filesystem."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen2.5-72B-Instruct",
        help="vLLM model name for logging and output.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        help="Data type for vLLM model (e.g., bfloat16, float16).",
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
        default=0.85,
        help="GPU memory utilization (0.0 to 1.0).",
    )
    parser.add_argument(
        "--max_len", type=int, default=8192, help="Maximum context length supported by vLLM."
    )
    parser.add_argument(
        "--max_num_seqs",
        type=int,
        default=256,
        help="Maximum number of sequences vLLM can handle simultaneously.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature, controls randomness of generation.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.8,
        help="Top-p sampling probability, controls vocabulary range considered.",
    )
    parser.add_argument(
        "--presence_penalty",
        type=float,
        default=0.0,
        help="Presence penalty for reducing repetition.",
    )
    parser.add_argument(
        "--frequency_penalty",
        type=float,
        default=0.0,
        help="Frequency penalty to reduce repetitive tokens.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=8192,
        help="Maximum tokens generated per response.",
    )

    args = parser.parse_args()
    output_filepath = (
        args.output or f"{os.path.splitext(args.input_path)[0]}_rewritten.jsonl"
    )
    print(f"Output path: {output_filepath}")

    process_complex_cases(args.input_path, output_filepath, args.prompts, args)


if __name__ == "__main__":
    main()
