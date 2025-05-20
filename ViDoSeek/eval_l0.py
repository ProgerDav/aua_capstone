import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from llama_index.core.schema import ImageNode, NodeWithScore
from llms.evaluator import Evaluator
from openai import OpenAI
from tqdm import tqdm
from utils.overall_evaluator import eval_search, eval_search_type_wise

DEFAULT_API_PATH: str = "https://api.activeloop.ai/"


class L0_VLM:
    def __init__(self, model_name: str = None, env: str = None):

        self.client = OpenAI(
            base_url=env,
            api_key=model_name,
        )

    def generate(self, query):
        response = self.client.chat.completions.create(
            model="activeloop-l0",
            messages=[{"role": "user", "content": query}],
            stream=True,
        )

        return response.choices[0].message.content


class MMRAG_L0:
    def __init__(
        self,
        dataset="VidoSeek",
        query_file="vidoseek.json",
        experiment_type="vidorag",
        generate_vlm=None,
        embed_model_name=None,
        embed_model_name_vl=None,  # openbmb/VisRAG-Ret vidore/colqwen2-v1.0
        embed_model_name_text=None,  # nvidia/NV-Embed-v2 BAAI/bge-m3
        workers_num=1,
        topk=10,
        env=None,
        uuid_previously_chosen=None,
    ):
        self.experiment_type = experiment_type
        self.workers_num = workers_num
        self.top_k = topk
        self.dataset = dataset
        self.query_file = query_file
        self.dataset_dir = os.path.join("./data", dataset)
        self.img_dir = os.path.join(self.dataset_dir, "img")
        self.results_dir = os.path.join(self.dataset_dir, "results")
        os.makedirs(self.results_dir, exist_ok=True)

        self.l0 = L0_VLM(model_name=generate_vlm, env=env)
        self.evaluator = Evaluator()

        self.eval_func = self.vidorag
        if uuid_previously_chosen:
            current_time = uuid_previously_chosen
        else:
            current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self.output_file_name = f"vidorag_neohorizon_{current_time}.jsonl"
        self.output_file_path = os.path.join(
            self.results_dir, self.output_file_name.replace("/", "-")
        )

    def vidorag(self, sample, filenames):
        query = sample["query"]
        try:
            answer = self.l0.generate(query)
        except Exception as e:
            print(f"Error generating response: {e}")
            return None
        candidate_image = [el.replace(".pdf", ".jpg") for el in filenames]
        candidate_image = [self.img_dir + "/" + el for el in candidate_image]
        ## LLM as a judge
        sample["eval_result"] = self.evaluator.evaluate(
            query, sample["reference_answer"], str(answer)
        )
        sample["response"] = answer
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
        return sample

    def eval_dataset(self):
        eval_func = self.eval_func

        rag_dataset_path = os.path.join(self.dataset_dir, self.query_file)
        with open(rag_dataset_path, "r") as f:
            data = json.load(f)
        data = data["examples"]

        if os.path.exists(self.output_file_path):
            results = []
            with open(self.output_file_path, "r") as f:
                for line in f:
                    results.append(json.loads(line.strip()))
            uid_already = [item["uid"] for item in results]
            data = [item for item in data if item["uid"] not in uid_already]

        if self.workers_num == 1:
            for item in tqdm(data):
                result = eval_func(item)
                if result is None:
                    continue
                with open(self.output_file_path, "a") as f:
                    json.dump(result, f, ensure_ascii=False)
                    f.write("\n")
        else:
            with ThreadPoolExecutor(max_workers=self.workers_num) as executor:
                futures = [executor.submit(eval_func, item) for item in data]
                results = []
                for future in tqdm(
                    as_completed(futures), total=len(futures), desc="Processing"
                ):
                    result = future.result()
                    results.append(result)
                    if len(results) >= 3:
                        with open(self.output_file_path, "a") as f:
                            for res in results:
                                if res is None:
                                    continue
                                f.write(json.dumps(res, ensure_ascii=False) + "\n")
                        results = []
                if results:
                    with open(self.output_file_path, "a") as f:
                        for res in results:
                            if res is None:
                                continue
                            f.write(json.dumps(res, ensure_ascii=False) + "\n")

        return self.output_file_path

    def eval_overall(self):
        data = []
        with open(self.output_file_path, "r") as f:
            for line in f:
                data.append(json.loads(line.strip()))
        results = eval_search(data, key=["raw_recall_results", "recall_results"])
        with open(self.output_file_path.replace(".jsonl", "_eval.json"), "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    def eval_overall_type_wise(self):
        data = []
        with open(self.output_file_path, "r") as f:
            for line in f:
                data.append(json.loads(line.strip()))
        results = eval_search_type_wise(
            data, key=["raw_recall_results", "recall_results"]
        )
        with open(
            self.output_file_path.replace(".jsonl", "_eval_type_wise.json"), "w"
        ) as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
