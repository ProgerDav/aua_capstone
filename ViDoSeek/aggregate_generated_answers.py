import os
import json
import argparse

# Define constants
FOLDER_PATH_PREFIX = "data/ViDoSeek/results"
OUTPUT_FILENAME = "all_uid.jsonl"
FILES_TO_SKIP = ["all_uid.jsonl"]  # Update to skip the output file itself


def aggregate_result_files(folder_path, output_filename, files_to_skip):
    """
    Aggregates JSON data from files in a folder into a single JSON Lines file,
    skipping specified files and avoiding duplicate entries based on the 'uid'.

    Args:
        folder_path (str): The path to the folder containing the JSON files.
        output_filename (str): The name of the output JSON Lines file.
        files_to_skip (list): A list of filenames to skip during aggregation.
    """
    file_count = len(
        [
            name
            for name in os.listdir(folder_path)
            if os.path.isfile(os.path.join(folder_path, name))
        ]
    )
    print(f"Number of files in the folder: {file_count}")

    all_new_data = []
    existing_uids = set()
    output_path = os.path.join(folder_path, output_filename)

    # Load existing UIDs from the output file if it exists
    if os.path.exists(output_path):
        try:
            with open(output_path, "r") as output_file:
                for line in output_file:
                    try:
                        existing_data = json.loads(line)
                        uid = existing_data.get("uid")
                        if uid:
                            existing_uids.add(uid)
                    except json.JSONDecodeError:
                        print(
                            f"Warning: Could not decode line in existing JSONL file: {output_path}"
                        )
        except IOError as e:
            print(f"Warning: Could not read existing file {output_path}: {e}")

    for file_name in os.listdir(folder_path):
        if file_name in files_to_skip or file_name == output_filename:
            print(f"Skipping file: {file_name}")
            continue

        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            # Check if the UID from this file is already processed
            # Assuming filename might correspond to UID or part of it,
            # This check needs refinement if UID is only inside the file.
            # For now, we will rely on checking UID after loading the file.

            try:
                with open(file_path, "r") as file:
                    data = json.load(file)
                    data_uid = data.get("uid")
                    if data_uid and data_uid not in existing_uids:
                        all_new_data.append(data)
                        existing_uids.add(data_uid)  # Add newly found uid
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON file: {file_name}")
            except IOError as e:
                print(f"Could not read file {file_name}: {e}")

    # Append the newly aggregated data in line-delimited JSON format
    try:
        with open(output_path, "a") as output_file:
            for item in all_new_data:
                output_file.write(json.dumps(item) + "\n")
        print(f"Aggregated data appended to {output_path}")
    except IOError as e:
        print(f"Could not write to output file {output_path}: {e}")


def aggregate_results(run_id: str | None = None):
    output_path = f"{FOLDER_PATH_PREFIX}_{run_id}" if run_id else FOLDER_PATH_PREFIX
    aggregate_result_files(output_path, OUTPUT_FILENAME, FILES_TO_SKIP)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate JSON results.")
    parser.add_argument("--run_id", type=str, default="")
    args = parser.parse_args()

    aggregate_results(args.run_id)
