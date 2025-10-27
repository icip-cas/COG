import os
import json
import re
import argparse
import time
from typing import List, Dict, Optional
from tqdm import tqdm
from collections import defaultdict
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

_vllm_engine_instance = None


def get_vllm_engine(args):
    """
    Initialize and return a global vLLM engine instance.
    If the engine is not yet initialized, it will be initialized based on the provided arguments.
    """
    global _vllm_engine_instance
    if _vllm_engine_instance is None:
        print(f"Initializing vLLM engine with model: {args.model_path}...")
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
        print("vLLM engine initialization complete.")
    return _vllm_engine_instance


def load_prompts_from_file(prompt_file_path: str) -> Dict[str, str]:
    """
    Load all prompt configurations from an external JSON file.
    """
    try:
        with open(prompt_file_path, "r", encoding="utf-8") as f:
            prompts = json.load(f)
        return prompts
    except FileNotFoundError:
        print(f"Error: prompt file not found -> {prompt_file_path}")
        return {}
    except json.JSONDecodeError:
        print(f"Error: prompt file is not a valid JSON -> {prompt_file_path}")
        return {}


def transform_data_for_classification(complex_case: dict) -> Optional[dict]:
    """
    Transform raw complex case data into a simplified format suitable for model API calls.
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
        response_consideration = (
            response_decision.get("response_strategy") if response_decision else None
        )

    model_raw_output_with_think = complex_case.get("full_response", "")
    int_reasoning = complex_case.get("reasoning", "")

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
        "reasoning": int_reasoning,
        "Risk_Analysis": danger_analysis,
        "Response_Consideration": response_consideration,
        "Final_Response": final_response,
    }

    if not all(value is not None and value != "" for value in required_data.values()):
        missing_keys = [
            key for key, value in required_data.items() if value is None or value == ""
        ]
        print(
            f"Warning: in case '{complex_case.get('id')}', required fields not extracted: {missing_keys}, skipped."
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
    """
    Batch call to vLLM engine for inference (with detailed debug info).
    """
    batched_messages_for_vllm: List[List[Dict[str, str]]] = []

    contextual_phrases = prompts.get("contextual_transition_phrases", {})
    category_counters = defaultdict(int)

    if not simple_cases_batch:
        print("DEBUG: Warning! Input batch is empty, returning empty list.")
        return []

    for i, simple_case in enumerate(simple_cases_batch):
        root_cause = root_causes_batch[i]
        phrase_list_for_cause = contextual_phrases.get(
            root_cause, ["Considering the situation,"]
        )
        current_index_for_category = category_counters[root_cause]
        chosen_phrase = phrase_list_for_cause[
            current_index_for_category % len(phrase_list_for_cause)
        ]
        category_counters[root_cause] += 1

        main_prompt = prompts.get("main_prompt", "")
        sub_prompt = prompts.get("sub_prompts", {}).get(root_cause, "")
        format_prompt_template = prompts.get("format_prompt", "")
        final_format_prompt = format_prompt_template.format(
            transition_phrase=chosen_phrase
        )
        full_system_prompt = main_prompt + sub_prompt + final_format_prompt

        user_content = f"""
[ORIGINAL_PROMPT]
{simple_case["prompt"]}
[DANGER_ANALYSIS_PROCESS]
{simple_case["Risk_Analysis"]}
[THOUGHT_PROCESS]
{simple_case["reasoning"]}
[REPLY_THINKING]
{simple_case["Response_Consideration"]}
[FINAL_RESPONSE]
{simple_case["Final_Response"]}
"""
        messages_for_single_case = [
            {"role": "system", "content": full_system_prompt},
            {"role": "user", "content": user_content},
        ]
        batched_messages_for_vllm.append(messages_for_single_case)

        if i == 0:
            tokenizer = llm_engine.get_tokenizer()
            final_prompt_str = tokenizer.apply_chat_template(
                messages_for_single_case,
                tokenize=False,
                add_generation_prompt=True,
            )

    print(f"Prepared {len(batched_messages_for_vllm)} prompts for the model.")
    print("Calling llm_engine.chat(), this may take some time...")

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
        print("\n\nDEBUG: CRITICAL ERROR")
        print(f"DEBUG: Exception occurred during vLLM inference: {e}")
        return [None] * len(simple_cases_batch)


def save_prompt_output_file(processed_cases: List[Dict], output_path: str):
    """
    Save processed data as JSONL in the format { "prompt": ..., "output": ... }.
    """
    print(f"\nGenerating prompt-output file...")
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            for case in processed_cases:
                prompt = case.get("prompt")
                security_check = case.get("security_check")

                if prompt and security_check is not None:
                    prompt_output_data = {"prompt": prompt, "COT": security_check}
                    f.write(json.dumps(prompt_output_data, ensure_ascii=False) + "\n")
        print(f"Prompt-Output file successfully saved to: {output_path}")
    except IOError as e:
        print(f"Error: Cannot write Prompt-Output file -> {output_path}, Error: {e}")


def process_complex_cases(
    input_path: str,
    output_path: str,
    prompt_output_path: str,
    prompt_file_path: str,
    args,
):
    """
    Main processing function for loading, processing, and saving cases.
    """
    if not os.path.exists(input_path):
        print(f"Error: input file not found -> {input_path}")
        return

    prompts = load_prompts_from_file(prompt_file_path)
    if not prompts:
        print("Cannot load prompt file, exiting.")
        return

    complex_cases: List[Dict] = []
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    complex_cases.append(json.loads(line))
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error: Failed to read or parse input file -> {input_path}, Error: {e}")
        return

    print(f"Found {len(complex_cases)} cases, starting batch processing...")

    llm_engine = get_vllm_engine(args)
    processing_counts = defaultdict(int)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        presence_penalty=args.presence_penalty,
        frequency_penalty=args.frequency_penalty,
        max_tokens=args.max_tokens,
    )
    chat_template_kwargs = {"enable_thinking": False}

    current_batch_simple_cases: List[Dict[str, str]] = []
    current_batch_root_causes: List[str] = []
    current_batch_original_indices: List[int] = []

    def process_batch():
        if not current_batch_simple_cases:
            return

        tqdm_bar.set_description(
            f"Sending batch ({len(current_batch_simple_cases)} cases)"
        )

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
            original_reasoning = original_case.get("reasoning", "")

            if api_response:
                original_case["overthink"] = api_response
                original_case["security_check"] = (
                    f"{original_reasoning}\n{api_response}\n"
                )
                processing_counts["Successfully processed (model rewrite)"] += 1
            else:
                original_case["overthink"] = "vLLM_NO_RESPONSE"
                original_case["security_check"] = f"{original_reasoning}\n"
                processing_counts["vLLM no response"] += 1

        current_batch_simple_cases.clear()
        current_batch_root_causes.clear()
        current_batch_original_indices.clear()
        tqdm_bar.set_description("Preparing batch")

    tqdm_bar = tqdm(complex_cases, desc="Preparing batch", unit="case")

    for idx, case in enumerate(tqdm_bar):
        if case.get("overthink"):
            processing_counts["Skipped (already has result)"] += 1
            continue

        classification_result = case.get("classification_result", {})
        root_cause = classification_result.get("root_cause", "")

        if not root_cause:
            processing_counts["Skipped (missing root_cause)"] += 1
            continue

        if root_cause == "N/A":
            case["security_check"] = f"{case.get('reasoning', '')}\n"
            case["overthink"] = "SKIPPED_DUE_TO_NA_ROOT_CAUSE"
            processing_counts["Processed (root_cause=N/A)"] += 1
            continue

        simple_case_for_api = transform_data_for_classification(case)

        if not simple_case_for_api:
            processing_counts["Skipped (incomplete input)"] += 1
            continue

        current_batch_simple_cases.append(simple_case_for_api)
        current_batch_root_causes.append(root_cause)
        current_batch_original_indices.append(idx)

        if len(current_batch_simple_cases) >= args.batch_size:
            process_batch()

    process_batch()

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            for case in complex_cases:
                f.write(json.dumps(case, ensure_ascii=False) + "\n")
        print(f"\nProcessing complete! Results saved to: {output_path}")
    except IOError as e:
        print(f"Error: Cannot write output file -> {output_path}, Error: {e}")

    save_prompt_output_file(complex_cases, prompt_output_path)

    print("\n--- Processing Statistics ---")
    total_processed = len(complex_cases)
    sorted_counts = sorted(
        processing_counts.items(), key=lambda item: item[1], reverse=True
    )
    for category, count in sorted_counts:
        percentage = (count / total_processed) * 100
        print(f"- {category}: {count} times ({percentage:.2f}%)")
    print("----------------------------")


def main():
    parser = argparse.ArgumentParser(
        description="Use vLLM to rewrite security chain-of-thought for complex cases in a JSON file."
    )
    current_dir = (
        os.path.dirname(__file__)
        if "__file__" in locals() and __file__
        else os.getcwd()
    )

    json_path = os.path.join(current_dir, "rewite_prompt_en.json")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input JSON file path."
    )

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
        help="vLLM inference batch size.",
    )

    parser.add_argument(
        "--model_path", type=str, required=True, help="vLLM model path on filesystem."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen2.5-72B-Instruct",
        help="vLLM model name (for logging and output).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        help="vLLM model data type.",
    )
    parser.add_argument(
        "--tp_size",
        type=int,
        default=8,
        help="vLLM tensor parallel size (number of GPUs).",
    )
    parser.add_argument(
        "--gpu_util",
        type=float,
        default=0.85,
        help="vLLM GPU memory utilization.",
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
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.8,
        help="Top-p (nucleus) sampling probability.",
    )
    parser.add_argument(
        "--presence_penalty",
        type=float,
        default=0.0,
        help="Presence penalty.",
    )
    parser.add_argument(
        "--frequency_penalty",
        type=float,
        default=0.0,
        help="Frequency penalty.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=8192,
        help="Maximum tokens to generate.",
    )

    args = parser.parse_args()

    output_filepath = (
        args.output or f"{os.path.splitext(args.input)[0]}_rewritten.jsonl"
    )
    base, _ = os.path.splitext(output_filepath)
    prompt_output_filepath = f"{base}_prompt_output.jsonl"

    print(f"Main output path: {output_filepath}")
    print(f"Prompt-Output path: {prompt_output_filepath}")

    process_complex_cases(
        args.input,
        output_filepath,
        prompt_output_filepath,
        args.prompts,
        args,
    )


if __name__ == "__main__":
    main()
