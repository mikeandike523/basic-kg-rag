import os
import json
import re
from typing import List, Optional

from termcolor import colored
from run_mistral import Mistral, UnfinishedResponseError
import time

TEMPERATURES = [ 0.6, 0.6, 0.6, 0.7, 0.7, 0.7]
TOP_P = 0.9

mistral = Mistral(quantization="8bit")


def ask_mistral(
    max_token_seq: List[int],
    prompt: str,
    system_instruction: Optional[str] = None,
    temperature: Optional[float] = TEMPERATURES[0],
    top_p: Optional[float] = TOP_P,
    report_time: bool = False,
) -> str:
    messages = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    messages.append({"role": "user", "content": prompt})
    start_time = time.time()
    for i in range(len(max_token_seq)):
        try:
            answer = mistral.completion(
                {
                    "max_tokens": max_token_seq[i],
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
                f"Unfinished response after at try {i+1}/{len(max_token_seq)} with max tokens {max_token_seq[i]}..."
            )
            print(e.generation)

    print("Unfinished response after all attempts.")
    return None


os.chdir(os.path.dirname(__file__))

with open("flatland.txt", "r", encoding="utf-8") as f:
    flatland_text = f.read().replace("\r\n", "\n")

paragraphs = [p.strip() for p in flatland_text.split("\n\n") if p.strip()]

sections = [[paragraphs[0]]]

pargraphs = paragraphs[1:]

for i, paragraph in enumerate(paragraphs):
    print(f"Processing paragraph {i+1}/{len(paragraphs)}...")

    final_decision = "continue_topic"

    for i, temperature in enumerate(TEMPERATURES):

        print(
            f"Attempting at temperature {temperature} (trial {i+1}/{len(TEMPERATURES)})..."
        )

        decision = ask_mistral(
            [8, 8, 16, 16, 16, 24, 24, 32],
            "\n\n".join(
                [
                    f"""
[Existing Text]

{"\n\n".join(sections[-1])}

    """.strip(),
                    f"""
[Next Paragraph]

{paragraph}

    """.strip(),
                ]
            ),
            system_instruction="""

Given the existing text, decide if the next paragraph starts a new topic or continues the current one.

Answer "new_topic", "continue_topic" or "unsure".

""".strip(),
            temperature=temperature,
            top_p=TOP_P,
            report_time=True,
        )

        if decision is None:
            print(colored(f"AI could not complete response at temperature {temperature} after all token length choices."))
            print(colored("Trying with the next temperature choice."))
            print(colored("Note. Temperature choices may be repeated to attempt same temperature multiple times.", "yellow"))
            continue

        decision = (
            decision.strip()
            .lower()
            .strip('"\'`.:!><-()=+[]{}|;:,.<>?/~`"')
            .replace(" ", "_")
        )

        if re.match(r"^new_topic\b", decision):
            final_decision = "new_topic"
            break

        elif re.match(r"^continue_topic\b", decision):
            final_decision = "continue_topic"
            break

        elif re.match(r"^unsure\b", decision):
            final_decision = "continue_topic"
            break
        else:
            print(
                colored(f"Invalid decision: '{decision}' (T {temperature})", "yellow")
            )

    print(f"Final decision: {final_decision}")

    if final_decision == "new_topic":
        sections.append([paragraph])
    elif final_decision == "continue_topic":
        sections[-1].append(paragraph)
    else:
        raise ValueError(f"Invalid final decision: '{final_decision}'")


sections = [s for s in sections if s]

sections = [
    {"title": f"section_{i+1}", "paragraphs": [p.strip() for p in section]}
    for i, section in enumerate(sections)
]

with open("sections.json", "w", encoding="utf-8") as f:
    json.dump(sections, f, indent=2, ensure_ascii=False)
