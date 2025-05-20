import base64
import json
import os

from dotenv import load_dotenv
from litellm import acompletion
from utils.prompt_get_bucket import (get_prompt_to_bucket_wrong_answers,
                                     get_prompt_to_bucket_wrong_answers_user)

load_dotenv()

GEMINI_MODEL_ID = os.environ.get("GEMINI_MODEL_ID", "gemini/gemini-2.0-flash")


async def extract_bucket_with_gemini(
    info, model_name=GEMINI_MODEL_ID, config=None, litellm=True
):
    """
    Process a single PDF chunk using LiteLLM.

    Args:
        chunk: Tuple containing (chunk_number, chunk_stream)
        model_name: The name of the Gemini model to use
        config: Optional generation config for LiteLLM

    Returns:
        Tuple containing (chunk_number, extracted_text)
    """

    query = info["query"]
    expected_answer = info["expected_answer"]
    generated_answer = info["generated_answer"]
    judgment = info["judgment"]
    prompt = get_prompt_to_bucket_wrong_answers()
    user_prompt = get_prompt_to_bucket_wrong_answers_user(
        query=query,
        expected_answer=expected_answer,
        generated_answer=generated_answer,
        judgment=judgment,
    )
    try:
        response = await acompletion(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": [{"type": "text", "text": prompt}],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": user_prompt,
                        },
                    ],
                },
            ],
            generation_config=config
            or {
                "temperature": 0,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 8192,
                "response_mime_type": "text/plain",
            },
            response_format={"type": "json_object"},
        )
        try:
            response = response.choices[0].message.content
            response_json = json.loads(response)
            text = response_json["bucket"]
        except:
            text = ""

        return text

    except Exception as e:
        print(f"Error processing")
        raise e
