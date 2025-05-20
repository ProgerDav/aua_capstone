from __future__ import annotations

import logging, os, contextlib, traceback, typing
import bentoml, fastapi, typing_extensions, annotated_types

import json

logger = logging.getLogger(__name__)

MODEL_ID = os.environ.get('MODEL_ID', 'Qwen/QwQ-32B')
ENGINE_CONFIG = {
    'max_model_len': os.environ.get('MAX_MODEL_LEN', 16384),
    'tensor_parallel_size': os.environ.get('TENSOR_PARALLEL_SIZE', 1),
    'enable_reasoning': True,
    'reasoning_parser': 'deepseek_r1',
    'gpu_memory_utilization': os.environ.get('GPU_MEMORY_UTILIZATION', 0.95),
    'enable_chunked_prefill': os.environ.get('ENABLE_CHUNKED_PREFILL', True),
    'max_num_batched_tokens': os.environ.get('MAX_NUM_BATCHED_TOKENS', 2048),
    'trust_remote_code': os.environ.get('TRUST_REMOTE_CODE', True),
    'enable_prompt_tokens_details': os.environ.get('ENABLE_PROMPT_TOKENS_DETAILS', True),

    # Add default sampling parameters
    # This can be overridden by the client in ChatCompletionRequest, if necessary with 'extra_body = {"repetition_penalty": 1.05}'
    'temperature': os.environ.get('TEMPERATURE', 0.7),
    'top_p': os.environ.get('TOP_P', 0.8),
    'top_k': os.environ.get('TOP_K', 20),
    'repetition_penalty': os.environ.get('REPETITION_PENALTY', 1.05),
    'generation_config': os.environ.get('GENERATION_CONFIG', 'vllm'),
}

openai_api_app = fastapi.FastAPI()

logger.info(f'DEFAULT_ENGINE_CONFIG: \n {json.dumps(ENGINE_CONFIG, indent=2)} \n--------------------------------')


@bentoml.asgi_app(openai_api_app, path='/v1')
@bentoml.service(
    name='bentovllm-qwq-32b-service',
    traffic={'timeout': 300},
    resources={'gpu': 1, 'gpu_type': 'nvidia-a100-80gb'}, # Relevant for BentoML Cloud deployments
    envs=[
        {'name': 'UV_NO_BUILD_ISOLATION', 'value': 1},
        {'name': 'UV_NO_PROGRESS', 'value': '1'},
        {'name': 'HF_HUB_DISABLE_PROGRESS_BARS', 'value': '1'},
        {'name': 'VLLM_ATTENTION_BACKEND', 'value': 'FLASH_ATTN'},
        {'name': 'VLLM_USE_V1', 'value': '1'},
        
        # Add this to explicitly set the torch compile cache directory
        {'name': 'TORCH_COMPILE_CACHE_DIR', 'value': '/home/ubuntu/.cache/vllm/torch_compile_cache'},
    ],
    labels={'owner': 'al-team', 'type': 'prebuilt'},
    image=bentoml.images.PythonImage(python_version='3.11', lock_python_packages=False)
    .requirements_file('requirements.txt')
    .run('uv pip install --compile-bytecode flashinfer-python --find-links https://flashinfer.ai/whl/cu124/torch2.6'),
)
class VLLM:
    model_id = MODEL_ID
    model = bentoml.models.HuggingFaceModel(model_id, exclude=['*.pth', '*.pt', 'original/**/*'])

    def __init__(self):
        from openai import AsyncOpenAI

        self.openai = AsyncOpenAI(base_url='http://127.0.0.1:3000/v1', api_key='dummy')
        self.exit_stack = contextlib.AsyncExitStack()

    @bentoml.on_startup
    async def init_engine(self) -> None:
        import vllm.entrypoints.openai.api_server as vllm_api_server

        from vllm.utils import FlexibleArgumentParser
        from vllm.entrypoints.openai.cli_args import make_arg_parser

        args = make_arg_parser(FlexibleArgumentParser()).parse_args([])
        args.model = self.model
        args.disable_log_requests = True
        args.max_log_len = 1000
        args.served_model_name = [self.model_id]
        args.request_logger = None
        args.disable_log_stats = False
        args.use_tqdm_on_load = False
        for key, value in ENGINE_CONFIG.items():
            setattr(args, key, value)

        args.enable_auto_tool_choice = True # Unused at the moment, we don't use tools
        args.tool_call_parser = 'llama3_json' # Unused at the moment, we don't use tools

        router = fastapi.APIRouter(lifespan=vllm_api_server.lifespan)
        OPENAI_ENDPOINTS = [
            ["/load", vllm_api_server.get_server_load_metrics, ['GET']],
            ['/health', vllm_api_server.health, ['GET']],
            ['/chat/completions', vllm_api_server.create_chat_completion, ['POST']],
            ['/models', vllm_api_server.show_available_models, ['GET']],
        ]
        
        for route, endpoint, methods in OPENAI_ENDPOINTS:
            router.add_api_route(path=route, endpoint=endpoint, methods=methods, include_in_schema=True)
        openai_api_app.include_router(router)
        # Add Prometheus metrics to the app
        vllm_api_server.mount_metrics(openai_api_app)

        self.engine = await self.exit_stack.enter_async_context(vllm_api_server.build_async_engine_client(args))
        self.model_config = await self.engine.get_model_config()
        self.tokenizer = await self.engine.get_tokenizer()
        await vllm_api_server.init_app_state(self.engine, self.model_config, openai_api_app.state, args)

    @bentoml.on_shutdown
    async def teardown_engine(self):
        await self.exit_stack.aclose()

    # Example of additional API endpoints we can define over the Bento Server.
    @bentoml.api
    async def generate(
        self,
        prompt: str = 'Who are you? Please respond in pirate speak!',
        max_tokens: typing_extensions.Annotated[
            int, annotated_types.Ge(128), annotated_types.Le(ENGINE_CONFIG['max_model_len'])
        ] = ENGINE_CONFIG['max_model_len'],
    ) -> typing.AsyncGenerator[str, None]:
        messages = [{'role': 'user', 'content': [{'type': 'text', 'text': prompt}]}]
        try:
            completion = await self.openai.chat.completions.create(
                model=self.model_id, messages=messages, stream=True, max_tokens=max_tokens
            )
            async for chunk in completion:
                yield chunk.choices[0].delta.content or ''
        except Exception:
            logger.error(traceback.format_exc())
            yield 'Internal error found. Check server logs for more information'
            return
