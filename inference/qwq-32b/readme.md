# Running the service locally

## Building image
```bash
pip install -r requirements.txt
bentoml build # Should download weights
bentoml containerize <tag> # Use the tag from output of the previous command
```


## Serving the service
```bash
bentoml serve service.py:VLLM
```

