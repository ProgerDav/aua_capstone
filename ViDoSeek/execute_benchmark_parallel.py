import argparse
import asyncio
import json
import os
import re
import time
from typing import Optional, Tuple

from dotenv import load_dotenv

from llama_index.core.schema import NodeWithScore, ImageNode
from openai import AsyncOpenAI
from tqdm import tqdm

from ViDoSeek.eval_l0 import DEFAULT_API_PATH, MMRAG_Neohorizon

from ViDoSeek.aggregate_generated_answers import aggregate_results
from extract_ids_by_score import extract_ids_by_score

load_dotenv()
api_key = os.getenv("ACTIVELOOP_TOKEN")

FILE_ID_TO_FILENAME = {}

try:
    with open("data/ViDoSeek/files_cache.json", "r") as f:
        FILE_ID_TO_FILENAME = json.load(f)
except FileNotFoundError:
    pass


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Process ViDoSeek benchmark using parallel requests."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data/ViDoSeek/vidoseek.json",
        help="Path to dataset JSON file",
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default=DEFAULT_API_PATH,
        help="Base URL for API requests",
    )
    parser.add_argument(
        "--model_id", type=str, default="activeloop-l0-deep-research", help="Model ID"
    )
    parser.add_argument(
        "--run_id", type=str, default="", help="Unique identifier for this run"
    )
    parser.add_argument(
        "--source_type",
        type=str,
        default="",
        required=False,
        help="Filter data by source type",
    )
    parser.add_argument(
        "--max_queries",
        type=int,
        default=-1,
        required=False,
        help="Maximum number of queries to process",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        help="Number of requests to process in parallel",
    )
    parser.add_argument(
        "--only_failed",
        type=bool,
        default=False,
        required=False,
        help="Only process failed requests",
    )
    parser.add_argument(
        "--previous_run_id",
        type=str,
        default="",
        required=False,
        help="Previous run ID",
    )

    return parser.parse_args()


def get_results_dir(run_id):
    base_dir = "data/ViDoSeek/results"
    if run_id:
        return f"{base_dir}_{run_id}"
    return base_dir


def remove_already_processed_data(data, results_dir):
    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)

    existing_files = set(
        os.path.splitext(file)[0]
        for file in os.listdir(results_dir)
        if file.endswith(".json")
    )
    data = [item for item in data if item["uid"] not in existing_files]
    if len(data) == 0:
        print("No element to process")
    return data


async def convert_docs_to_candidate_images(relevant_docs, client, mmrag):
    """Convert document references to candidate image paths."""
    filenames = []
    file_ids = []
    pages = []

    for el in sorted(relevant_docs, key=lambda x: x["score"], reverse=True):
        file_id = el["file_id"]
        if file_id not in FILE_ID_TO_FILENAME:
            file_dl = await client.files.retrieve(file_id)
            filename = file_dl.filename
            FILE_ID_TO_FILENAME[file_id] = filename
        else:
            filename = FILE_ID_TO_FILENAME[file_id]

        chunk_index = el["chunk_index"]
        page = int(chunk_index) + 1
        filenames.append(f"{filename.split('.')[0]}_{page}.pdf")

        file_ids.append(file_id)
        pages.append(page)

    candidate_image = [el.replace(".pdf", ".jpg") for el in filenames]
    candidate_image = [mmrag.img_dir + "/" + el for el in candidate_image]

    return candidate_image


async def make_request(
    model_id: str,
    sample: dict,
    client: AsyncOpenAI,
    mmrag: MMRAG_Neohorizon,
    output_path: str,
    retry_count: int = 0,
) -> Tuple[str, float, Optional[str]]:
    query = sample["query"]
    uid = sample["uid"]
    full_message = ""
    full_answer = ""
    answer = ""
    sample["model_id"] = model_id
    usage = {}

    raw_retrieval_results = None

    try:
        time_start = time.time()
        SHOULD_STREAM = False
        stream = await client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": query}],
            stream=SHOULD_STREAM,
        )
        full_answer = ""
        if SHOULD_STREAM:
            async for event in stream:
                try:
                    if event.choices[0].delta.reasoning_content is not None:
                        new_content = event.choices[0].delta.reasoning_content
                        full_message += new_content
                    if event.choices[0].delta.content is not None:
                        full_answer += event.choices[0].delta.content
                    if event.choices[0].metadata:
                        metadata = event.choices[0].metadata
                except Exception as e:
                    pass

            relevant_docs = metadata["metadata"]["relevant_docs"]
            usage = metadata["metadata"]["internal_usage"]

            raw_retrieval_results = metadata["metadata"].get(
                "full_retrieval_results", None
            )

            original_answer = full_answer
            full_answer = re.sub(
                r"<\|begin_search_query\|>.*?<\|end_search_query\|>",
                "",
                full_answer,
            )
            if original_answer != full_answer:
                sample["search_leaked_in_answer"] = True
        else:
            if "retrieval" not in model_id:
                metadata = stream.choices[0].metadata["metadata"]

                original_content = stream.choices[0].message.content
                full_answer = re.sub(
                    r"<\|begin_search_query\|>.*?<\|end_search_query\|>",
                    "",
                    original_content,
                )
                if original_content != full_answer:
                    sample["search_leaked_in_answer"] = True

                full_message = stream.choices[0].message.reasoning_content or ""
                full_message += "\n\n" + original_content

                relevant_docs = metadata["relevant_docs"]
                usage = metadata["internal_usage"]
                sample["turns"] = metadata["turns"]
                raw_retrieval_results = metadata.get("full_retrieval_results", None)
            else:
                relevant_docs = [
                    {**c.metadata, "text": c.message.content}
                    for c in stream.choices
                ]

            answer = full_answer
            end_time = time.time()
            elapsed_time = end_time - time_start

        # Get candidate images from documents
        candidate_image = await convert_docs_to_candidate_images(
            relevant_docs, client, mmrag
        )

        elapsed_gpt_time = 0
        if "retrieval" not in model_id:
            start_gpt_time = time.time()
            sample["eval_result"] = mmrag.evaluator.evaluate(
                query, sample["reference_answer"], str(answer)
            )
            end_gpt_time = time.time()
            elapsed_gpt_time = end_gpt_time - start_gpt_time

        sample["response"] = answer
        sample["usage"] = usage
        sample["recall_results"] = dict(
            source_nodes=[
                NodeWithScore(
                    node=ImageNode(image_path=image, metadata=dict(file_name=image)),
                    score=None,
                ).to_dict()
                for image in candidate_image
            ],
            response=None,
            metadata=None,
        )  # see: source_nodes

        if (
            "activeloop-l0-deep-research" in model_id
            and raw_retrieval_results is not None
        ):
            candidate_image = await convert_docs_to_candidate_images(
                raw_retrieval_results, client, mmrag
            )
            sample["raw_recall_results"] = dict(
                source_nodes=[
                    NodeWithScore(
                        node=ImageNode(
                            image_path=image, metadata=dict(file_name=image)
                        ),
                        score=None,
                    ).to_dict()
                    for image in candidate_image
                ],
                response=None,
                metadata=None,
            )

        sample["full_message"] = full_message
        sample["request_time"] = elapsed_time
        sample["judgement_time"] = elapsed_gpt_time
        sample["retry_count"] = retry_count

        output_file = os.path.join(output_path, f"{uid}.json")
        with open(output_file, "w") as f:
            json.dump(sample, f, ensure_ascii=False)
            f.write("\n")

        return sample

    except Exception as e:
        print(f"Error processing request '{query}': {str(e)}")
        MAX_RETRIES = 2
        if retry_count < MAX_RETRIES:
            wait_time = (retry_count + 1) * 2  # Exponential backoff
            print(
                f"Retrying in {wait_time} seconds... (Attempt {retry_count + 1}/{MAX_RETRIES})"
            )
            if "error" not in sample:
                sample["error"] = str(e)
            if "error" in sample:
                sample["error"] += f" (Attempt {retry_count + 1}): {str(e)}"

            await asyncio.sleep(wait_time)
            return await make_request(
                model_id, sample, client, mmrag, output_path, retry_count + 1
            )
        return None


async def process_batch(model_id, batch_queries, client, mmrag, output_path):
    tasks = [
        make_request(model_id, query, client, mmrag, output_path)
        for query in batch_queries
    ]
    return await asyncio.gather(*tasks)


async def run_semaphore_requests(
    model_id, data, client, mmrag, output_path, batch_size
):
    semaphore = asyncio.Semaphore(batch_size)
    tasks = []
    results = []

    async def worker(semaphore, sample):
        async with semaphore:
            result = await make_request(model_id, sample, client, mmrag, output_path)
            return result

    for sample in data:
        tasks.append(worker(semaphore, sample))

    # Use tqdm with asyncio.as_completed for progress tracking
    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        result = await f
        if result is not None:
            results.append(result)

    print(f"Processed {len(results)} requests successfully.")
    return results


async def main_async():
    args = parse_arguments()

    # Create directory for results if it doesn't exist
    results_dir = get_results_dir(args.run_id)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)

    # Initialize MMRag
    uuid_previously_chosen = None
    mmrag = MMRAG_Neohorizon(
        dataset="ViDoSeek",
        query_file="vidoseek.json",
        experiment_type="vidorag",
        embed_model_name=None,
        topk=10,
        embed_model_name_vl=None,
        workers_num=1,
        generate_vlm=api_key,
        env=DEFAULT_API_PATH,
        uuid_previously_chosen=uuid_previously_chosen,
    )

    # Load and filter dataset
    with open(args.dataset_path, "r") as f:
        data = json.load(f)

    if args.source_type:
        data = [
            i
            for i in data["examples"]
            if i["meta_info"]["source_type"] in args.source_type.split(",")
        ]
    else:
        data = data["examples"]

    if args.only_failed:
        failed_ids = extract_ids_by_score(
            args.previous_run_id, passing=0, score=None, source_type=args.source_type
        )
        data = [i for i in data if i["uid"] in failed_ids]

    if args.max_queries > 0:
        data = data[: args.max_queries]

    # Remove already processed data
    data = remove_already_processed_data(data, results_dir)
    print(f"PROCESSING {len(data)} requests.")

    if not data:
        return

    # Initialize client
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=args.base_url,
    )

    # Run requests using semaphore
    results = await run_semaphore_requests(
        args.model_id, data, client, mmrag, results_dir, args.batch_size
    )

    with open("data/ViDoSeek/files_cache.json", "w") as f:
        json.dump(FILE_ID_TO_FILENAME, f, ensure_ascii=False)

    aggregate_results(args.run_id)
    print(f"Completed processing {len(results)} requests successfully.")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
