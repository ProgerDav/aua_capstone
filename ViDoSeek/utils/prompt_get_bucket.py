def get_prompt_to_bucket_wrong_answers():
    llm_prompt_json = f"""
You are a system that classifies the type of error in a generated response compared to the expected answer. 
Given the following inputs:
- A user query
- The correct answer
- The generated answer
- An human judgment message: it could start with a number between 1 and 5, where 1 is the worst and 5 is the best.
    - the answer with a score >= 4 is considered correct
    - the answer with a score < 4 is considered incorrect

Return a JSON object identifying which bucket the incorrect response falls into.

The possible buckets are:

1. "not_generated": The response was not generated at all due to a system error or failure to produce an output. This must be clearly stated in the judgment.
2. "false_negative": The generated answer is actually correct, but was mistakenly labeled as incorrect. This can happen, for example, if the answer is overly verbose but still correct.
3. "wrong_answer": The generated response is genuinely incorrect compared to the expected answer.


Return the result in the following format:

```json
{{
  "bucket": "<one of: not_generated, false_negative, wrong_answer>"
}}
```
"""
    return llm_prompt_json


def get_prompt_to_bucket_wrong_answers_user(
    query, expected_answer, generated_answer, judgment=None
):

    user_prompt = f"""Here are the inputs:
- Query: {query}
- Expected Answer: {expected_answer}
- Generated Answer: {generated_answer}
- Judgment: {judgment}
Give me the bucket in the JSON format.
"""
    return user_prompt
