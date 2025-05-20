import os
import sys
import json
from typing import List, Optional, Union
from run_mistral import completion, UnfinishedResponseError
from dataclasses import dataclass
import time

DEFAULT_TEMPERATURE = 0.6
DEFAULT_TOP_P = 0.9
SIZE_TRIES = 3


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
                    "max_tokens": max_tokens * (2**try_count),
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
                f"Output token limit {max_tokens * (2**try_count)} was not enough. Trying with twice as much tokens..."
            )
            if try_count == SIZE_TRIES - 1:
                print(
                    "Hit max retry tokens already, cannot continue. Consider adjusting global variable SIZE_TRIES, but be wary of exceeding model max token limit."
                )
                raise e
            try_count += 1


script_dir = os.path.dirname(__file__)

with open("story_learner/flatland.txt", "r", encoding="utf-8") as f:
    flatland_text = f.read().replace("\r\n", "\n")

paragraphs = [p.strip() for p in flatland_text.split("\n\n") if p.strip()]

# Tracking
current_subject = ""
current_subject_summary = ""
current_topic_paragraphs = []
topics = []

first_paragraph = paragraphs[0]

current_subject = ask_mistral(
    128,
    first_paragraph,
    system_instruction="""Please provide a name for the current topic as shown in the given paragraph, or "Unknown" if you are unsure.""",
)

current_subject_summary = ask_mistral(
    256,
    first_paragraph,
    system_instruction="""Please provide a short summary of the current topic, or "Unknown" if you are unsure.""",
)

current_topic_paragraphs.append(first_paragraph)


def continue_topic(next_paragraph: str) -> None:
    global current_subject, current_subject_summary, current_topic_paragraphs
    new_name_response = ask_mistral(
        128,
        f"""
[Current Topic]: {current_subject}
[Current Summary]:

{current_subject_summary}

[Next Paragraph]:

{next_paragraph}
""".strip(),
        """
Given the name of the current topic, and a summary of the current topic, and the next paragraph,

Provide an adjusted topic name given the new paragraph. If no adjustment is needed, provide the original name. If you are unsure, reply "Unknown".

Reply with only the adjusted topic name, and no extra labels or indicators.
""".strip(),
    )
    new_summary_response = ask_mistral(
        128,
        f"""
[Current Topic]: {current_subject}
[Current Summary]:

{current_subject_summary}

[Next Paragraph]:

{next_paragraph}
""".strip(),
        """
Given the name of the current topic, and a summary of the current topic, and the next paragraph,

Provide an adjusted summary. If no adjustment is needed, provide the original summary. If you are unsure, reply "Unknown".

Reply with only the adjusted summary, and no extra labels or indicators.
""".strip(),
    )
    current_subject = new_name_response
    current_subject_summary = new_summary_response
    current_topic_paragraphs.append(next_paragraph)


def start_new_topic(next_paragraph: str) -> None:
    global current_subject, current_subject_summary, current_topic_paragraphs, topics

    # Save the current topic
    if current_subject and current_topic_paragraphs:
        topics.append(
            {
                "topic": current_subject,
                "summary": current_subject_summary,
                "paragraphs": current_topic_paragraphs.copy(),
            }
        )

    # Start the new topic
    current_subject = ask_mistral(
        128,
        next_paragraph,
        system_instruction="""Please provide a name for the current topic as shown in the given paragraph. Reply "Unsure" if you are unsure.""",
    )
    current_subject_summary = ask_mistral(
        256,
        next_paragraph,
        system_instruction="""Please provide a short summary of the current topic. Reply "Unsure" if you are unsure.""",
    )
    current_topic_paragraphs = [next_paragraph]


# Process remaining paragraphs
for i, paragraph in enumerate(paragraphs[1:], start=2):
    print(f"\nProcessing paragraph {i}/{len(paragraphs)}...")
    decision = (
        ask_mistral(
            64,
            f"""
[Current Topic]: {current_subject}
[Current Summary]:

{current_subject_summary}

[Next Paragraph]:
{paragraph}

""",
            system_instruction="""
Given the name of the current topic, and a summary of the current topic, and the next paragraph,
        
does the following paragraph continue in the current topic or start a new one?

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
        continue_topic(paragraph)
    elif "new_topic" in decision:
        start_new_topic(paragraph)
    else:
        raise ValueError(
            "Invalid decision: '" + decision + f"' at paragraph {i}/{len(paragraphs)}"
        )

    # Save to JSON
    with open("topicized_flatland.json", "w", encoding="utf-8") as f:
        json.dump(topics, f, indent=2, ensure_ascii=False)


# Save the last topic
if current_subject and current_topic_paragraphs:
    topics.append(
        {
            "topic": current_subject,
            "summary": current_subject_summary,
            "paragraphs": current_topic_paragraphs,
        }
    )

# Save to JSON
with open("topicized_flatland.json", "w", encoding="utf-8") as f:
    json.dump(topics, f, indent=2, ensure_ascii=False)

print("\nFinished. Topics written to 'topicized_flatland.json'")
