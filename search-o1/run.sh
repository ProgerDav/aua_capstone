# watch -n 1 nvidia-smi

# Baseline: Closed book QwQ-32B
# python3 scripts/run_direct_gen.py \
#     --model_path Qwen/QwQ-32B \
#     --dataset_name 2wiki \
#     --split dev \

# Multimodal late interaction retrieval + agentic reasoning
python3 scripts/run_search_o1_vector_db.py \
    --model_path Qwen/QwQ-32B \
    --dataset_name 2wiki \
    --split dev \
    --vector_db_token \
    <replace with activeloop token>
