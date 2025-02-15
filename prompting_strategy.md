## We prompt Llama3.3-70B-Instruct as follows:

```
You are a mathematical answer validator. You will be provided with a mathematical problem and you need to compare the answer in the reference solution, and the final answer in a model's solution to determine if they are equivalent, even if formatted differently.

PROBLEM:

{problem}

REFERENCE SOLUTION:

{answer}

MODEL'S SOLUTION:

{generation}

Focus ONLY on comparing the final mathematical answer provided by the model while ignoring differences in:

- Formatting (e.g., \\boxed{{}} vs plain text)
- Multiple choice formatting (e.g., "A" vs full solution)
- Order of coordinate pairs or solutions
- Equivalent mathematical expressions or notation variations
- If the model's answer is nonsense, return "Verdict: AMBIGUOUS"

Start with a brief explanation of your comparison (2-3 sentences). Then output your final answer in one of the following formats:

- "Verdict: EQUIVALENT"
- "Verdict: DIFFERENT"
- "Verdict: AMBIGUOUS"
```


## json schema should be like : 

```
[
{ problem : string (15-8000),
solution : string (0-15000),
answer : string (1-300),
problem_type : string (classes: Algebra, Number Theory, Geometry, Calculus, etc...)
question_type : string (classes: math-word-problem, MCQ, etc...)
source : string (classes: olympiads, aops_forum, cn_contest, etc...)
uuid : string (3-36)
is_reasoning_complete : sequence ([ true, true ] or [ true, false ] or other like [ true, true, true])
generatations : sequence ( ["<think>\nOkay, so I need to solve this : ...",  "<think>\nOkay to solve this problem, about ship traveling upstream..."])
correctness_math_verify : sequence ([true, true])
correctness_llama : sequence ([true, false] or null)
finish_reasons : sequence (["stop", "stop"] or null)
correctness_count : int64 (1-6)
messages : list (
[
    {
      "from": "user",
      "value": "## Task B-1.3.\n\nA ship traveling along a river has covered $24 \\mathrm{~km}$ upstream and $28 \\mathrm{~km}$ downstream. For this journey, it took half an hour less than for traveling $30 \\mathrm{~km}$ upstream and $21 \\mathrm{~km}$ downstream, or half an hour more than for traveling $15 \\mathrm{~km}$ upstream and $42 \\mathrm{~km}$ downstream, assuming that both the ship and the river move uniformly.\n\nDetermine the speed of the ship in still water and the speed of the river."
    },
    {
      "from": "assistant",
      "value": "<think>\nOkay, s.."
    }
]
)
```


## Fill in the middle Deepseek

```
from openai import OpenAI

client = OpenAI(
    api_key="<your api key>",
    base_url="https://api.deepseek.com/beta",
)

response = client.completions.create(
    model="deepseek-chat",
    prompt="def fib(a):",
    suffix="    return fib(a-1) + fib(a-2)",
    max_tokens=128
)
print(response.choices[0].text)
```