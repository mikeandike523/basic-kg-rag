import os
import json
from typing import List, Optional, Union
# from run_mistral import completion, UnfinishedResponseError
import time
from openai import OpenAI, BadRequestError, OpenAIError


DEFAULT_TEMPERATURE = 0.4
DEFAULT_TOP_P = 0.9
SIZE_TRIES = 4
SIZE_TRIES_MULTIPLIER = 2
DECISION_TRIES = 3

DECISION_TOKENS = 16
NAME_TOKENS=64
SUMMARY_TOKENS = 256

MODEL="gpt-4o"

# Don't Worry, It's already in .gitignore
with open("OPENAI_API_KEY.txt", "r") as f:
    OPENAI_API_KEY = f.read().strip()

client = OpenAI(api_key=OPENAI_API_KEY)


class UnfinishedResponseError(Exception):
    """Raised when the model response is truncated due to max_tokens limit."""
    pass

class OutOfTokensError(Exception):
    """Raised when the API key has insufficient quota or credits."""
    pass

class OutOfContextWindowTokens(Exception):
    """Raised when the prompt or completion exceeds the model's context window."""
    pass

def completion(
    params: dict
) -> str:
    """
    Wrapper around openai.ChatCompletion.create that raises custom exceptions
    based on finish_reason or API error codes.
    params should include: model (optional), messages, max_tokens, temperature, top_p
    """
    try:
        response = client.chat.completions.create(model=params.get("model", MODEL),
            messages=params["messages"],
            max_completion_tokens=params["max_tokens"],
            # temperature=params.get("temperature", DEFAULT_TEMPERATURE),
            # top_p=params.get("top_p", DEFAULT_TOP_P)
        )
        choice = response.choices[0]
        # If the API indicates the response was cut off by max_tokens
        if choice.finish_reason == "length":
            raise UnfinishedResponseError(
                f"Response truncated (finish_reason=length) at {params['max_tokens']} tokens"
            )
        return choice.message.content

    except BadRequestError as e:
        msg = str(e)
        # Detect context length errors
        if "maximum context length" in msg.lower() or "context_length_exceeded" in msg.lower():
            raise OutOfContextWindowTokens(msg)
        raise

    except OpenAIError as e:
        msg = str(e).lower()
        # Detect quota/credit errors
        if "insufficient_quota" in msg or "quota" in msg:
            raise OutOfTokensError(msg)
        raise


def ask_gpt(
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

current_subject = ask_gpt(
    NAME_TOKENS,
    first_paragraph,
    system_instruction="""Please provide a name for the current topic as shown in the given paragraph, or "Unknown" if you are unsure.""",
)

current_subject_summary = ask_gpt(
    SUMMARY_TOKENS,
    first_paragraph,
    system_instruction="""Please provide a short summary of the current topic, or "Unknown" if you are unsure.""",
)

current_topic_paragraphs.append(first_paragraph)


def continue_topic(next_paragraph: str) -> None:
    global current_subject, current_subject_summary, current_topic_paragraphs
    new_name_response = ask_gpt(
        NAME_TOKENS,
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
    new_summary_response = ask_gpt(
        SUMMARY_TOKENS,
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
    current_subject = ask_gpt(
        NAME_TOKENS,
        next_paragraph,
        system_instruction="""Please provide a name for the current topic as shown in the given paragraph. Reply "Unsure" if you are unsure.""",
    )
    current_subject_summary = ask_gpt(
        SUMMARY_TOKENS,
        next_paragraph,
        system_instruction="""Please provide a short summary of the current topic. Reply "Unsure" if you are unsure.""",
    )
    current_topic_paragraphs = [next_paragraph]


# Process remaining paragraphs
for i, paragraph in enumerate(paragraphs[1:], start=2):
    print(f"\nProcessing paragraph {i}/{len(paragraphs)}...")
    decision = None
    decision_tries = 0
    while decision is None:
        decision = (
            ask_gpt(
                DECISION_TOKENS,
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
    with open("story_learner/topicized_flatland_4o_.json", "w", encoding="utf-8") as f:
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
with open("story_learner/topicized_flatland_4o_.json", "w", encoding="utf-8") as f:
    json.dump(topics, f, indent=2, ensure_ascii=False)

print("\nFinished. Topics written to 'story_learner/topicized_flatland_4o_.json'")
