from openai import OpenAI
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/QwQ-32B", trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"


def get_multiqa_search_o1_instruction(MAX_SEARCH_LIMIT, model_name="qwq"):
    return (
        "You are a reasoning assistant with the ability to perform web searches to help "
        "you answer the user's question accurately. You have special tools:\n\n"
        "- To perform a search: write <|begin_search_query|> your query here <|end_search_query|>.\n"
        "Then, the system will search and analyze relevant web pages, then provide you with helpful information in the format <|begin_search_result|> ...search results... <|end_search_result|>.\n\n"
        f"You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {MAX_SEARCH_LIMIT}.\n\n"
        "Once you have all the information you need, continue your reasoning.\n\n"
        "Example:\n"
        'Question: "Alice David is the voice of Lara Croft in a video game developed by which company?"\n'
        "Assistant thinking steps:\n"
        "- I need to find out who voices Lara Croft in the video game.\n"
        "- Then, I need to determine which company developed that video game.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>Alice David Lara Croft voice<|end_search_query|>\n\n"
        "(System returns processed information from relevant web pages)\n\n"
        "Assistant thinks: The search results indicate that Alice David is the voice of Lara Croft in a specific video game. Now, I need to find out which company developed that game.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>video game developed by Alice David Lara Croft<|end_search_query|>\n\n"
        "(System returns processed information from relevant web pages)\n\n"
        "Assistant continues reasoning with the new information...\n\n"
        "Remember:\n"
        "- Use <|begin_search_query|> to request a web search and end with <|end_search_query|>.\n"
        "- When done searching, continue your reasoning.\n\n"
    )


question = "Who is the mother of the director of film Polish-Russian War (Film)?"

user_prompt = (
    "Please answer the following question. "
    "You should provide your final answer in the format \\boxed{YOUR_ANSWER}.\n\n"
    f"Question:\n{question}\n\n"
)

messages = [
    {
        "role": "user",
        "content": get_multiqa_search_o1_instruction(
            MAX_SEARCH_LIMIT=3, model_name="qwq"
        )
        + user_prompt,
    }
]

print(messages)
print("--------------------------------")
print(
    tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
)
# Initialize the client with the vLLM server URL
# Note that we're using a base_url that points to your vLLM server
client = OpenAI(
    api_key="not-needed",  # vLLM doesn't require an API key by default
    base_url="http://0.0.0.0:3000/v1",  # Adjust the host/port if needed
)

template_messages = [
    {
        "role": "user",
        "content": get_multiqa_search_o1_instruction(
            MAX_SEARCH_LIMIT=3, model_name="qwq"
        )
        + user_prompt,
    }
]

# Call the model
response = client.chat.completions.create(
    model="Qwen/QwQ-32B",  # This can be any string when using vLLM
    messages=template_messages,
    temperature=0.7,
    top_p=0.8,
    stop=["<|end_search_query|>"],
    stream=True,
    stream_options={"include_usage": True},
    extra_body={
        "chat_template_kwargs": {"add_generation_prompt": True},
        "include_stop_str_in_output": True,
    },
)

print(response)
# Print the response
# print(response.choices[0].message.content)

# Stream the response
for chunk in response:
    if len(chunk.choices) == 0:
        print(chunk.usage)
        continue

    if hasattr(chunk.choices[0].delta, "reasoning_content"):
        print(chunk.choices[0].delta.reasoning_content, end="", flush=True)
    elif hasattr(chunk.choices[0].delta, "content"):
        print(chunk.choices[0].delta.content, end="", flush=True)

print("--------------------------------")
