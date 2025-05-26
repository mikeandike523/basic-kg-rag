import json
from termcolor import colored
from typing import List, Optional, Tuple, Union
from run_mistral import completion, UnfinishedResponseError
import time

DEFAULT_TEMPERATURE = 0.25
DEFAULT_TOP_P = 0.9
SIZE_TRIES = 6
SIZE_TRIES_MULTIPLIER = 2
DECISION_TRIES = 3
DECISION_TOKENS = 16


def ask_mistral(
    max_tokens: int,
    prompt_or_prompts: Union[str, List[str]],
    system_instruction: Optional[str] = None,
    temperature: Optional[float] = DEFAULT_TEMPERATURE,
    top_p: Optional[float] = DEFAULT_TOP_P,
    report_time: bool = False,
) -> str:
    prompts = (
        [prompt_or_prompts] if isinstance(prompt_or_prompts, str) else prompt_or_prompts
    )
    messages = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    for prompt in prompts:
        messages.append({"role": "user", "content": prompt})
    start_time = time.time()
    try_count = 0
    while try_count < SIZE_TRIES:
        try:
            answer = completion(
                {
                    "max_tokens": int(max_tokens * (SIZE_TRIES_MULTIPLIER**try_count)),
                    "messages": messages,
                    "temperature": temperature,
                    "top_p": top_p,
                }
            )
            end_time = time.time()
            elapsed_time = end_time - start_time
            if report_time:
                print(f"Answered in {elapsed_time:.2f} seconds")
            return answer.strip()
        except UnfinishedResponseError as e:
            print(
                f"Output token limit {int(max_tokens * (SIZE_TRIES_MULTIPLIER**try_count))} was not enough. Trying with {int(max_tokens * (SIZE_TRIES_MULTIPLIER**(try_count+1)))}  tokens..."
            )
            if try_count == SIZE_TRIES - 1:
                print(
                    "Hit max retry tokens already, cannot continue. Consider adjusting global variable SIZE_TRIES, but be wary of exceeding model max token limit."
                )
                raise e
            try_count += 1


with open(
    "story_learner/topicized_flatland_mistral_7b_instruct_quantized_8bit.json",
    "r",
    encoding="utf-8",
) as f:
    all_topics = json.loads(f.read())

for i, topic in enumerate(all_topics):
    print(f"Processing topic {i+1}/{len(all_topics)}...")

    decision = None

    decision_tries = 0

    while decision is None:

        if decision_tries >= DECISION_TRIES:
            print(colored(f"Invalid decision after {decision_tries} tries", "red"))
            exit(1)

        decision = (
            ask_mistral(
                DECISION_TOKENS,
                "\n\n".join(topic["paragraphs"]),
                """
Does the following text contain ANY content that is important to the story, or is it ONLY noise?


Examples of Noise:

- Headings
- Table of Contents
- Formatting statements

Reply "yes" or "no".
    """.strip(),
            )
            .lower()
            .strip()
            .strip('"\'`.:!><-()=+[]{}|;:,.<>?/~`"')
            .replace(" ", "_")
        )

        if not decision.startswith("yes") and not decision.startswith("no"):
            print(colored(f"Invalid decision: {decision} (T {decision_tries+1} of {DECISION_TRIES})", "yellow"))
            continue

        decision_tries += 1

    print(f"\nDecision: {decision}\n")

    if decision.startswith("yes"):
        print(colored(f"Topic {i+1} (of {len(all_topics)}) is important", "green"))
    elif decision.startswith("no"):
        print(colored(f"Topic {i+1} (of {len(all_topics)}) is noise", "magenta"))
        print(colored(f"\n\n{'\n\n'.join(topic['paragraphs'])}\n\n", "magenta"))
    else:
        print(colored(f"Unexpected decision: {decision}", "red"))
        exit(1)