import os
import json
from typing import List, Optional, Tuple, Union
from run_mistral import completion, UnfinishedResponseError
import time

DEFAULT_TEMPERATURE = 0.5
DEFAULT_TOP_P = 0.9
SIZE_TRIES = 6
SIZE_TRIES_MULTIPLIER = 2
DECISION_TRIES = 3

DECISION_TOKENS = 16
NAME_TOKENS=64
SUMMARY_TOKENS = 256



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


def get_title_and_summary(text: str) -> Tuple[str, str]:
    title = ask_mistral(NAME_TOKENS, text,"Choose a good title for the given text. Output only the title and nothing else.")
    summary = ask_mistral(SUMMARY_TOKENS, text, "Create a short summary of the given text. Output only the summary and nothing else.")
    return title, summary


script_dir = os.path.dirname(__file__)

with open("story_learner/flatland.txt", "r", encoding="utf-8") as f:
    flatland_text = f.read().replace("\r\n", "\n")

paragraphs = [p.strip() for p in flatland_text.split("\n\n") if p.strip()]

# Tracking
current_topic_paragraphs = []
topics = []

first_paragraph = paragraphs[0]

current_topic_paragraphs.append(first_paragraph)

# Process remaining paragraphs
for i, paragraph in enumerate(paragraphs[1:], start=2):
    print(f"\nProcessing paragraph {i}/{len(paragraphs)}...")
    decision = None
    decision_tries = 0
    while decision is None:
        decision = (
            ask_mistral(
                DECISION_TOKENS,
                f"""

    [Current Text]:

    {"\n\n".join(current_topic_paragraphs)}

    [Next Paragraph]:
    {paragraph}

    """,
                system_instruction="""
    Given the current text, and the next paragraph, determine if the next paragraph fits into the current topic or starts a new one.

    Reply "new_topic" if it starts a new topic, otherwise "continue_topic". If you are unsure, say "continue_topic" so you can decide later.
    """,
            )
            .lower()
            .strip()
            .strip('"\'`.:!><-()=+[]{}|;:,.<>?/~`"')
            .replace(" ", "_")
        )

        print(f"Decision: {decision}")

        if "continue_topic" in decision:
            current_topic_paragraphs.append(paragraph)
        elif "new_topic" in decision:
            title, summary = get_title_and_summary("\n\n".join(current_topic_paragraphs))
            topics.append(
                {
                    "topic": title,
                    "summary": summary,
                    "paragraphs": [p for p in current_topic_paragraphs],
                }
            )
            current_topic_paragraphs = [paragraph]
        else:
            decision_tries += 1
            decision = None
            print(
                f"Invalid decision at try #{decision_tries}, on to try {decision_tries+1} of {DECISION_TRIES}... \n"
            )

            if decision_tries >= DECISION_TRIES:

                raise ValueError(
                    f"Invalid decision after {decision_tries} tries: '"
                    + decision
                    + f"' at paragraph {i}/{len(paragraphs)}"
                )

    # Save to JSON
    with open("story_learner/topicized_flatland_mistral_7b_instruct.json", "w", encoding="utf-8") as f:
        json.dump(topics, f, indent=2, ensure_ascii=False)


# Save the last topic
if current_topic_paragraphs:
    title, summary = get_title_and_summary("\n\n".join(current_topic_paragraphs))
  
    topics.append(
        {
            "topic": title,
            "summary": summary,
            "paragraphs": [p for p in current_topic_paragraphs],
        }
    )

# Save to JSON
with open("story_learner/topicized_flatland_mistral_7b_instruct.json", "w", encoding="utf-8") as f:
    json.dump(topics, f, indent=2, ensure_ascii=False)

print("\nFinished. Topics written to 'story_learner/topicized_flatland_mistral_7b_instruct.json'")
