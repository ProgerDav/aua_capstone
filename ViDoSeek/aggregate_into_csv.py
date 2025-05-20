import argparse
import json
import os
from datetime import datetime
from functools import reduce

import pandas as pd


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Process ViDoSeek benchmark results and create CSV/Excel reports."
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default="activeloop-l0-deep-research",
        help="Unique identifier for this run",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="data/ViDoSeek/results_csv",
        help="Output folder for CSV/Excel files",
    )
    return parser.parse_args()


def get_results_dir(run_id):
    base_dir = "data/ViDoSeek/results"
    if run_id:
        return f"{base_dir}_{run_id}"
    return base_dir


# Helper function to extract nested fields
def extract_field(data, field):
    keys = field.split(".")
    for key in keys:
        data = data.get(key, None)
        if data is None:
            return None

    return data


def construct_eval_dfs(results_dir, passing_per_type, counts_per_type):
    score_key = "score"
    passing_key = "passing"
    recall_key = "raw_recall_results Recall@all"
    backup_recall_key = "recall_results Recall@all"

    with open(f"{results_dir}/all_uid_eval.json", "r") as f:
        full_eval = json.load(f)

    full_eval_df = pd.DataFrame(
        [
            {
                "Mean Score": full_eval[score_key],
                "Mean Passing": full_eval[passing_key],
                "Mean Recall@all": full_eval.get(
                    recall_key, full_eval.get(backup_recall_key, None)
                ),
                "Total Queries": reduce(lambda x, y: x + y, counts_per_type.values())
                / 2,
                "Total Passing": reduce(lambda x, y: x + y, passing_per_type.values())
                / 2,
            }
        ]
    )

    with open(f"{results_dir}/all_uid_eval_type_wise.json", "r") as f:
        type_wise_eval = json.load(f)

    type_wise_eval_df = pd.DataFrame(
        [
            {
                "Query Type": q_type,
                "Mean Score": type_wise_eval[q_type][score_key],
                "Mean Passing": type_wise_eval[q_type][passing_key],
                "Mean Recall@all": type_wise_eval[q_type].get(
                    recall_key, type_wise_eval[q_type].get(backup_recall_key, None)
                ),
                "Total Queries": counts_per_type[q_type],
                "Total Passing": passing_per_type[q_type],
            }
            for q_type in passing_per_type.keys()
        ]
    )

    return full_eval_df, type_wise_eval_df


def create_excel_report(run_id, output_folder, output_excel_file):
    results_dir = get_results_dir(run_id)
    input_file = f"{results_dir}/modified_all_uid.jsonl"
    if not os.path.exists(input_file):
        input_file = f"{results_dir}/all_uid.jsonl"

    fields_to_extract = [
        "query",
        "uid",
        "reference_answer",
        "response",
        "reasoning",
        "meta_info.reference_page",
        "meta_info.source_type",
        "meta_info.query_type",
        "eval_result.score",
        "eval_result.passing",
        "eval_result.judge",
        "request_time",
        "retry_count",
        "turns",
        "error",
    ]

    # Read the input file
    data = []
    with open(input_file, "r") as f:
        for line in f.readlines():
            row = json.loads(line)
            data.append(
                {
                    **{k: extract_field(row, k) for k in fields_to_extract},
                    **{
                        "token_count": row["usage"]["reasoning_usage"]["total_tokens"]
                        + row["usage"]["perception_usage"]["total_tokens"]
                    },
                }
            )

    # Create a DataFrame from the data
    df = pd.DataFrame(data)
    total_rows = len(df)

    counts_per_type = {}
    for query_type, count in (
        df.groupby("meta_info.query_type").size().to_dict().items()
    ):
        counts_per_type[query_type] = count

    counts_per_source_type = {}
    for source_type, count in (
        df.groupby("meta_info.source_type").size().to_dict().items()
    ):
        counts_per_source_type[source_type] = count

    passing_per_type = {}
    for query_type, passing in (
        df.groupby("meta_info.query_type")["eval_result.passing"]
        .value_counts()
        .to_dict()
        .items()
    ):
        if query_type[1] == 0:
            continue
        passing_per_type[query_type[0]] = passing

    passing_per_source_type = {}
    for source_type, passing in (
        df.groupby("meta_info.source_type")["eval_result.passing"]
        .value_counts()
        .to_dict()
        .items()
    ):
        if source_type[1] == 0:
            continue
        passing_per_source_type[source_type[0]] = passing

    passing_per_type.update(passing_per_source_type)
    counts_per_type.update(counts_per_source_type)
    overall_eval_df, type_wise_eval_df = construct_eval_dfs(
        results_dir, passing_per_type, counts_per_type
    )

    reasoning_time_df = pd.DataFrame(
        [
            {
                "Type": "Overall",
                "Mean Reasoning Time": df["request_time"].mean(),
                "Reasoning Time St.Dev": df["request_time"].std(),
                "Reasoning Time Median": df["request_time"].median(),
                "Reasoning Time 95%": df["request_time"].quantile(0.95),
            }
        ]
        + [
            {
                "Type": q_type,
                "Mean Reasoning Time": df[df["meta_info.source_type"] == q_type][
                    "request_time"
                ].mean(),
                "Reasoning Time St.Dev": df[df["meta_info.source_type"] == q_type][
                    "request_time"
                ].std(),
                "Reasoning Time Median": df[df["meta_info.source_type"] == q_type][
                    "request_time"
                ].median(),
                "Reasoning Time 95%": df[df["meta_info.source_type"] == q_type][
                    "request_time"
                ].quantile(0.95),
            }
            for q_type in passing_per_source_type.keys()
        ]
    )

    turns_df = pd.DataFrame(
        [
            {
                "Type": "Overall",
                "Mean Turns": df["turns"].mean(),
                "Turns St.Dev": df["turns"].std(),
            }
        ]
        + [
            {
                "Type": q_type,
                "Mean Turns": df[df["meta_info.source_type"] == q_type]["turns"].mean(),
                "Turns St.Dev": df[df["meta_info.source_type"] == q_type][
                    "turns"
                ].std(),
            }
            for q_type in passing_per_source_type.keys()
        ]
    )

    token_count_df = pd.DataFrame(
        [
            {
                "Type": "Overall",
                "Mean Token Count": df["token_count"].mean(),
                "Token Count St.Dev": df["token_count"].std(),
                "Token Count Median": df["token_count"].median(),
                "Token Count 95%": df["token_count"].quantile(0.95),
            }
        ]
        + [
            {
                "Type": q_type,
                "Mean Token Count": df[df["meta_info.source_type"] == q_type][
                    "token_count"
                ].mean(),
                "Token Count St.Dev": df[df["meta_info.source_type"] == q_type][
                    "token_count"
                ].std(),
                "Token Count Median": df[df["meta_info.source_type"] == q_type][
                    "token_count"
                ].median(),
                "Token Count 95%": df[df["meta_info.source_type"] == q_type][
                    "token_count"
                ].quantile(0.95),
            }
            for q_type in passing_per_source_type.keys()
        ]
    )

    # Create Excel report
    with pd.ExcelWriter(output_excel_file) as writer:
        df.to_excel(writer, sheet_name="All Queries", index=False)
        overall_eval_df.to_excel(writer, sheet_name="Overall Evaluation", index=False)
        type_wise_eval_df.to_excel(
            writer, sheet_name="Type-wise Evaluation", index=False
        )
        reasoning_time_df.to_excel(writer, sheet_name="Reasoning Time", index=False)
        turns_df.to_excel(writer, sheet_name="Turns", index=False)
        token_count_df.to_excel(writer, sheet_name="Token Count", index=False)


def main():
    args = parse_arguments()

    # Set up input and output paths
    results_dir = get_results_dir(args.run_id)
    input_file = f"{results_dir}/modified_all_uid.jsonl"
    if not os.path.exists(input_file):
        input_file = f"{results_dir}/all_uid.jsonl"

    # Create output folder if it doesn't exist
    output_folder = args.output_folder
    os.makedirs(output_folder, exist_ok=True)

    # Generate output filename with timestamp
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag = f"_{args.run_id}" if args.run_id else ""
    output_excel_file = f"{output_folder}/extracted_data{run_tag}_{current_time}.xlsx"

    print(f"Processing data from: {input_file}")
    print(f"Output will be saved to: {output_excel_file}")

    create_excel_report(args.run_id, output_folder, output_excel_file)


if __name__ == "__main__":
    main()
