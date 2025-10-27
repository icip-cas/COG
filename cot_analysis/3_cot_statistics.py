import json
import pandas as pd
import os

# ==============================================================================
# C O N F I G U R A T I O N   S E C T I O N
# ==============================================================================
# --- Configure your statistics script here ---

# 1. Input file: must be the output file of your analysis script
INPUT_FILE = "" 

# 2. Model names: must exactly match the keys in your JSON file
MODEL_NAMES = [
    "Base", 
    "SafR", 
    "SafB", 
    "XXX"
] 

# 3. Output directory and CSV filenames (optional)
OUTPUT_DIR = "./Result/COT_Analysis"
AVG_BEHAVIORS_CSV = "avg_behaviors_per_problem.csv"
PRESENCE_RATE_CSV = "behavior_presence_rate.csv"

# --- End of configuration section ---
# ==============================================================================

def analyze_statistics_final(filepath: str, model_names: list):
    """
    Parse the final analysis results file (list structure) and calculate key statistics.
    """
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            # The entire file is a large JSON list, load it directly
            all_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found. Please ensure the path is correct and the analysis script ran successfully.")
        return None, None
    except json.JSONDecodeError:
        print(f"Error: File '{filepath}' is not a valid JSON. The file may be corrupted or not fully written.")
        return None, None

    # Step 1: Create a detailed, problem-level statistics list
    problem_level_stats = []
    
    # Record which models are actually found in the JSON file, for debugging
    found_models_in_file = set()

    # --- Core logic ---
    # Directly iterate through the list of problem objects parsed by json.load()
    for problem_data in all_data:
        # Use "id" as problem_id
        problem_id = problem_data.get("id")
        
        # Check if "model_outputs" exists and is a dictionary
        if "model_outputs" not in problem_data or not isinstance(problem_data["model_outputs"], dict):
            continue

        for model_name, model_output in problem_data["model_outputs"].items():
            found_models_in_file.add(model_name)
            if model_name not in model_names:
                continue

            # Initialize behavior counters for the current model on this problem
            behavior_counts = {
                "problem_id": problem_id,
                "model_name": model_name,
                "Backtracking": 0,
                "Verification": 0,
                "Subgoal Setting": 0,
                "Enumeration": 0,
                "Blank_Reasoning": 0
            }

            analysis_result_raw = model_output.get("analysis_result", {})

            # Extract the analysis array from the analysis_result object
            if isinstance(analysis_result_raw, dict):
                analysis_list = analysis_result_raw.get("analysis", [])
                if not analysis_list:
                    behavior_counts["Blank_Reasoning"] = 1
                else:
                    for behavior in analysis_list:
                        b_type = behavior.get("behaviour")
                        if b_type in behavior_counts:
                            behavior_counts[b_type] += 1
            else:
                behavior_counts["Blank_Reasoning"] = 1
            
            problem_level_stats.append(behavior_counts)
    
    # Step 2: Check if any data was processed
    if not problem_level_stats:
        print("\nError: No statistics matching the configuration could be parsed from the file.")
        print("Please check the following:")
        print(f"1. Your 'MODEL_NAMES' configuration: {model_names}")
        print(f"2. Actual model names in the file: {list(found_models_in_file)}")
        print("   (Make sure there is at least one exact match between the two, including case)\n")
        return None, None
        
    # Step 3: Convert the detailed list to a DataFrame and calculate metrics
    df = pd.DataFrame(problem_level_stats)
    # Use all behaviors defined in behavior_counts as values for pivot
    value_columns = ["Backtracking", "Verification", "Subgoal Setting", "Enumeration", "Blank_Reasoning"]
    pivot_df = df.pivot_table(index='problem_id', columns='model_name', values=value_columns, fill_value=0)
    
    # Reorder columns to match your configuration order (even if the file order is different)
    # Only keep models actually found in the file to avoid errors
    models_to_reindex = [m for m in model_names if m in pivot_df.columns.get_level_values('model_name')]
    if not models_to_reindex:
        print("Warning: None of the configured model names were found in the file")
        return None, None
    pivot_df = pivot_df.reindex(columns=models_to_reindex, level='model_name')
    
    # Metric 1: Average behaviors per problem
    avg_behaviors_df = pivot_df.mean().unstack(level=0).T
    avg_behaviors_df = avg_behaviors_df[models_to_reindex]

    # Calculate overall average for each model (excluding Blank_Reasoning)
    behavior_columns = ["Backtracking", "Verification", "Subgoal Setting", "Enumeration"]
    avg_behaviors_summary = avg_behaviors_df.loc[behavior_columns].mean()
    avg_behaviors_df.loc["Overall Average"] = avg_behaviors_summary


    return avg_behaviors_df

if __name__ == "__main__":
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.width', 1000)
    pd.set_option('display.float_format', '{:.2f}'.format)

    avg_df = analyze_statistics_final(INPUT_FILE, MODEL_NAMES)
    
    if avg_df is not None and not avg_df.empty:
        print("\n" + "="*60)
        print("Metric 1: Average Behaviors per Problem")
        print("="*60)
        print(avg_df)

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        avg_path = os.path.join(OUTPUT_DIR, AVG_BEHAVIORS_CSV)
        avg_df.to_csv(avg_path)
        print(f"\n-> Average behaviors per problem saved to: {avg_path}")
