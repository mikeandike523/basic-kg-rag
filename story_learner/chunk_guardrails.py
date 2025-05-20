import os
import sys
import json
from run_mistral import completion
from guardrails import Guard
from guardrails.llm_providers import LiteLLMCallable

NUM_PARAGRAPHS_PER_CHUNK = 5

script_dir = os.path.dirname(__file__)

with open("story_learner/flatland.txt", "r", encoding="utf-8") as f:
    flatland_text = f.read().replace("\r\n", "\n")

paragraphs = [p.strip() for p in flatland_text.split("\n\n") if p.strip()]

# Guardrails Spec
guardrails_spec = """
<rail version="0.1">
<output>
    <object name="topicDecision">
        <string name="is_new_topic" description="Yes if it's a new topic, otherwise No"/>
        <string name="topic_name" description="Suggested topic name"/>
        <string name="summary" description="Brief summary of the topic"/>
    </object>
</output>
<prompt>
Given a topic name and summary, judge if the next paragraph starts a new topic or continues the current topic.

If it starts a new topic:
Give an initial topic name and summary.

If it continues the current topic:
Adjust the current topic name and summary if needed.

Current Topic: {{ current_subject }}
Current Summary: {{ current_subject_summary }}

Paragraph:
{{ paragraph }}

Respond with a decision in the specified format.
</prompt>
</rail>
"""

# Initialize Guard
guard = Guard.from_rail_string(guardrails_spec)
llm = LiteLLMCallable(lambda prompt, **kwargs: completion(prompt))

# Tracking
current_subject = "No Subject Yet"
current_subject_summary = "No Summary Yet"
current_topic_paragraphs = []
topics = []

for i, paragraph in enumerate(paragraphs):
    print(f"\nProcessing paragraph {i + 1}/{len(paragraphs)}...")

    # Get model output validated by Guardrails
    _, validated_output = guard(
        llm=llm,
        prompt_params={
            "current_subject": current_subject,
            "current_subject_summary": current_subject_summary,
            "paragraph": paragraph
        }
    )

    decision = validated_output["topicDecision"]
    is_new_topic = decision["is_new_topic"].strip().lower() == "yes"

    if is_new_topic and current_topic_paragraphs:
        topics.append({
            "name": current_subject,
            "summary": current_subject_summary,
            "start_index": start_index,
            "paragraph_count": len(current_topic_paragraphs),
            "text": "\n\n".join(current_topic_paragraphs)
        })
        current_topic_paragraphs = []

    if is_new_topic:
        current_subject = decision["topic_name"].strip()
        current_subject_summary = decision["summary"].strip()
        start_index = i

    current_topic_paragraphs.append(paragraph)

# Add the final topic
if current_topic_paragraphs:
    topics.append({
        "name": current_subject,
        "summary": current_subject_summary,
        "start_index": start_index,
        "paragraph_count": len(current_topic_paragraphs),
        "text": "\n\n".join(current_topic_paragraphs)
    })

# Save to JSON
with open("topicized_flatland.json", "w", encoding="utf-8") as f:
    json.dump(topics, f, indent=2, ensure_ascii=False)

print("\nFinished. Topics written to 'topicized_flatland.json'")
