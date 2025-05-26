from dataclasses import dataclass
from typing import List, Literal, Optional, TypedDict
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from collections import defaultdict
from termcolor import colored


MISTRAL_7B_INSTRUCT = "mistralai/Mistral-7B-Instruct-v0.1"

MAX_CONTEXT_TOKENS = 16384

bnb_config_8bit = BitsAndBytesConfig(
    load_in_4bit=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)


bnb_config_4bit = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)


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
    generation: str


class Mistral:

    def __init__(
        self,
        model_id: str = MISTRAL_7B_INSTRUCT,
        max_context_tokens: int = MAX_CONTEXT_TOKENS,
        quantization: Optional[Literal["8bit", "4bit"]] = None,
        default_temperature: float = 0.6,
        default_top_p: float = 0.9,
    ):
        self.max_context_tokens = max_context_tokens
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if quantization is not None:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                torch_dtype=torch.float16,
                quantization_config=(
                    bnb_config_4bit if quantization == "4bit" else bnb_config_8bit
                ),
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_id, device_map="auto", torch_dtype=torch.float16
            )
        self.model = model
        self.default_temperature = default_temperature
        self.default_top_p = default_top_p

    def completion(self, payload: CompletionRequest) -> str:
        messages = payload["messages"]
        top_p = payload.get("top_p", self.default_top_p)
        max_tokens = payload["max_tokens"]
        temperature = payload.get("temperature", self.default_temperature)

        formatted_text = self.tokenizer.apply_chat_template(messages, tokenize=False)

        inputs = self.tokenizer(formatted_text, return_tensors="pt")

        input_ids = inputs["input_ids"]
        total_input_tokens = input_ids.shape[1]

        if total_input_tokens > MAX_CONTEXT_TOKENS:
            raise OutOfTokensError(
                budget=MAX_CONTEXT_TOKENS, total_tokens=total_input_tokens
            )

        output = self.model.generate(
            inputs["input_ids"].to(self.model.device),
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            attention_mask=inputs["attention_mask"].to(self.model.device),
        )

        # Check if the completion has reached the maximum token limit
        # In the case where the answer is finished but it is exactly max generation characters then it is a false positive
        # But that is ok as in most use cases we want our max_gen_len to be at least 1.5 or 2x our expected mean generation length
        gen_len = output.shape[1] - input_ids.shape[1]
        if gen_len >= max_tokens:
            # Extract the generated portion (excluding the input prompt)
            generated_ids = output[0][input_ids.shape[1]:]

            # Decode with special tokens for debugging
            debug_text = self.tokenizer.decode(generated_ids, skip_special_tokens=False)

            raise UnfinishedResponseError(max_new_tokens=max_tokens, generation=debug_text)
        
        output_ids = output[0]

        decoded = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        reply = decoded.split("[/INST]")[-1].strip()
        return reply
