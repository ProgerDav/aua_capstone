MODEL_ID="activeloop-l0"
RUN_ID="activeloop-l0-1"
BATCH_SIZE=10
MAX_QUERIES=2
SOURCE_TYPE="text"

python execute_benchmark_parallel.py --run_id $RUN_ID --batch_size $BATCH_SIZE --max_queries $MAX_QUERIES --source_type $SOURCE_TYPE
python eval_l0.py --run_id $RUN_ID