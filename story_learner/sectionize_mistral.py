import os
import json
from typing import List, Optional, Union

from termcolor import colored
from run_mistral import Mistral, UnfinishedResponseError
import time

DEFAULT_TEMPERATURE = 0.70
DEFAULT_TOP_P = 0.90
SIZE_TRIES = 5
DECISION_TRIES = 3
DECISION_TOKENS = 16

mistral = Mistral(quantization="8bit")

def ask_mistral(
    max_token_seq: List[int],
    prompt: str,
    system_instruction: Optional[str] = None,
    temperature: Optional[float] = DEFAULT_TEMPERATURE,
    top_p: Optional[float] = DEFAULT_TOP_P,
    report_time: bool = False,
) -> str:
    messages = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    messages.append({"role": "user", "content": prompt})
    start_time = time.time()
    try_count = 0
    while try_count < SIZE_TRIES:
        try:
            answer = mistral.completion(
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
            print(json.dumps(messages, indent=2))
            print(colored("Generated output: " + e.generation, "yellow"))
            if try_count == SIZE_TRIES - 1:
                print(
                    "Hit max retry tokens already, cannot continue. Consider adjusting global variable SIZE_TRIES, but be wary of exceeding model max token limit."
                )
                raise e
            try_count += 1


os.chdir(os.path.dirname(__file__))

with open("flatland.txt", "r", encoding="utf-8") as f:
    flatland_text = f.read().replace("\r\n", "\n")

paragraphs = [p.strip() for p in flatland_text.split("\n\n") if p.strip()]

sections = [[]]

for i, paragraph in enumerate(paragraphs):
    print(f"Processing paragraph {i+1}/{len(paragraphs)}...")
        
    decision = ask_mistral(
        DECISION_TOKENS,
        "\n\n".join([
f"""
[Existing Text]:

{"\n\n".join(sections[-1])}

""".strip(),
f"""
[Next Paragraph]:

{paragraph}

""".strip()
        ]),
        system_instruction="""
Given the existing text, decide if the next paragraph starts a new section.

Look for distinct section markers such as chapter titles, headings, table of contents, forward, etc.

Use the existing text as context to avoid false positives.

Some potential false positives include:

- Dialog / play-like formatting
- Hyperlinks
- Code snippets
- Placeholders for external media such as images and videos

Answer "yes" or "no".

""".strip(),
    ).strip().lower().strip('"\'`.:!><-()=+[]{}|;:,.<>?/~`"')

    

    if decision.startswith("yes"):
        sections.append([paragraph])

    elif decision.startswith("no"):
        sections[-1].append(paragraph)

    else:
        print(f"Invalid decision: {decision}")
        exit(1)

sections = [s for s in sections if s]

sections = [{
    "title":section[0].strip(),
    "paragraphs": [p.strip() for p in section[1:]]
} for section in sections]

with open("sections.json", "w", encoding="utf-8") as f:
    json.dump(sections, f, indent=2, ensure_ascii=False)