from dataclasses import dataclass
from typing import List, Literal, Optional, TypedDict
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from collections import defaultdict
from termcolor import colored


MAX_CONTEXT_TOKENS = 16384

bnb_config = BitsAndBytesConfig(
    load_in_4bit=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# Load model and tokenizer
model_id = "mistralai/Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16,
    quantization_config=bnb_config,
)


def summarize_model_device_allocation(model):
    device_summary = defaultdict(lambda: {"params": 0, "bytes": 0})

    for param in model.parameters():
        device = str(param.device)
        device_summary[device]["params"] += 1
        if param.device.type != "meta":
            device_summary[device]["bytes"] += param.numel() * param.element_size()

    total_params = sum(info["params"] for info in device_summary.values())
    total_bytes = sum(info["bytes"] for info in device_summary.values())

    print(colored("=" * 50, "cyan"))
    print(colored("MODEL DEVICE ALLOCATION SUMMARY", "green", attrs=["bold"]))
    print(colored("=" * 50, "cyan"))

    for device, info in device_summary.items():
        mem_mb = info["bytes"] / (1024**2)
        mem_gb = info["bytes"] / (1024**3)
        color = (
            "yellow" if "cuda" in device else ("red" if "meta" in device else "blue")
        )
        print(colored(f"Device: {device}", color, attrs=["bold"]))
        print(f"  Number of Parameters: {info['params']}")
        print(f"  Memory Used: {mem_mb:.2f} MB ({mem_gb:.2f} GB)\n")

    print(colored("TOTAL", "magenta", attrs=["bold"]))
    print(f"  Total Parameters: {total_params}")
    print(
        f"  Total Memory Used: {total_bytes / (1024 ** 2):.2f} MB ({total_bytes / (1024 ** 3):.2f} GB)"
    )
    print(colored("=" * 50, "cyan"))


summarize_model_device_allocation(model)


# Define chat prompt (Mistral expects special prompt formatting)
def build_prompt(messages):
    prompt = ""
    for message in messages:
        role = message["role"]
        content = message["content"]
        if role == "user":
            prompt += f"<s>[INST] {content} [/INST]"
        elif role == "assistant":
            prompt += f"{content}</s>"
    return prompt


@dataclass
class Message:
    role: Literal["user", "assistant", "system"]
    content: str


class CompletionRequest(TypedDict):
    messages: List[Message]
    top_p: Optional[float]
    max_tokens: int
    temperature: Optional[float]


@dataclass
class OutOfTokensError(Exception):
    budget: int
    total_tokens: int


@dataclass
class UnfinishedResponseError(Exception):
    max_new_tokens: int


def completion(payload: CompletionRequest) -> str:
    messages = payload["messages"]
    top_p = payload.get("top_p", 0.9)
    max_tokens = payload["max_tokens"]
    temperature = payload.get("temperature", 0.6)

    formatted_text = tokenizer.apply_chat_template(messages, tokenize=False)

    inputs = tokenizer(formatted_text, return_tensors="pt")

    input_ids = inputs["input_ids"]
    total_input_tokens = input_ids.shape[1]

    if total_input_tokens > MAX_CONTEXT_TOKENS:
        raise OutOfTokensError(
            budget=MAX_CONTEXT_TOKENS, total_tokens=total_input_tokens
        )


    output = model.generate(
        inputs["input_ids"].to(model.device),
        max_new_tokens=max_tokens,
        do_sample=True, 
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        attention_mask=inputs["attention_mask"].to(model.device),
    )

    # Check if the completion has reached the maximum token limit
    # In the case where the answer is finished but it is exactly max generation characters then it is a false positive
    # But that is ok as in most use cases we want our max_gen_len to be at least 1.5 or 2x our expected mean generation length
    gen_len = output.shape[1] - input_ids.shape[1]
    if gen_len >= max_tokens:
        raise UnfinishedResponseError(max_new_tokens=max_tokens)

    output_ids = output[0]

    decoded = tokenizer.decode(output_ids, skip_special_tokens=True)
    reply = decoded.split("[/INST]")[-1].strip()
    return reply
