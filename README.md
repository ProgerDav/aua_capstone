# Supporting implementation files for the AUA Capstone Project

## WIKIPEDIA Scraping

The script we I for scraping the necessary source documents supporting the 2WikiHopQA benchmark is available in the `scraping` directory.

## ViDoSeek Benchmark

ViDoSeek directory is forked from the original [ViDoRAG](https://github.com/Alibaba-NLP/ViDoRAG) repository, and further additions are made to implement benchmarking of our approach.

## The inference setup

For the inference of open-source models we rely on VLLM inference engine.
To spin up an openai-complaint fastapi server allowing running inference of open-source models:

```bash
bentoml serve service.py:VLLM
```
