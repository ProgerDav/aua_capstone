import json
import argparse


def extract_ids_by_score(run_id, passing=0, score=None, source_type=None):
    failed_ids = []

    with open(f"data/ViDoSeek/results_{run_id}/all_uid.jsonl", "r") as f:
        for line in f:
            data = json.loads(line)
            if (
                data["eval_result"]["passing"] == passing
                and (score is None or data["eval_result"]["score"] <= score)
                and (
                    source_type is None
                    or data["meta_info"]["source_type"] == source_type
                )
            ):
                failed_ids.append(data["uid"])

    return failed_ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, default="", required=False)
    args = parser.parse_args()

    failed_ids = extract_ids_by_score(args.run_id, passing=0, score=None)
    print(f"Found {len(failed_ids)} failed requests.")
    print(failed_ids)
