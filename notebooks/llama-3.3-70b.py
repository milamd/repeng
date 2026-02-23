#!/usr/bin/env python3
# converted from llama-3.3-70b.ipynb
import os

# Ensure we run from the script directory for relative paths
try:
    os.chdir(os.path.dirname(__file__))
except OSError:
    pass

try:
    from IPython.display import display, HTML
except ImportError:

    class MockDisplayHandle:
        def update(self, obj):
            print(obj)

    def display(*args, **kwargs):
        if kwargs.get("display_id"):
            return MockDisplayHandle()
        for arg in args:
            print(arg)

    class HTML:
        def __init__(self, data):
            self.data = data

        def __str__(self):
            return self.data

        def __repr__(self):
            return self.data


from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from repeng import ControlModel, ControlVector, DatasetEntry

model_name = "meta-llama/Llama-3.3-70B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="auto", torch_dtype=torch.bfloat16
)

model = ControlModel(model, range(10, 71))

import json

with open("data/all_truncated_outputs.json") as f:
    output_suffixes = json.load(f)

default_persona = "anything"


def generation_prompt(persona):
    tokens = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": f"Please talk about {persona}."},
        ],
        add_generation_prompt=True,
    )
    return tokenizer.decode(tokens)


def train_persona_vector(persona):
    dataset = []
    persona_prompt = generation_prompt(persona)
    default_prompt = generation_prompt(default_persona)
    for suffix in output_suffixes:
        dataset.append(
            DatasetEntry(
                positive=persona_prompt + suffix,
                negative=default_prompt + suffix,
            )
        )
    return ControlVector.train(
        model, tokenizer, dataset, method="pca_center", batch_size=64
    )


from transformers import TextStreamer


def chat_template_parse(resp: str) -> list[dict[str, str]]:
    resp = resp.strip().removeprefix("<|begin_of_text|>")
    messages = []
    for part in resp.split("<|start_header_id|>"):
        role_and_content = part.split("<|end_header_id|>")
        if len(role_and_content) == 1:
            role, content = role_and_content[0], ""
        else:
            role, content = role_and_content
        content = content.split("<|eot_id|>")[0]
        messages.append({"role": role.strip(), "content": content.strip()})
    return messages


class HTMLStreamer(TextStreamer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.display_handle = display(display_id=True)
        self.full_text = ""

    def _is_chinese_char(self, _):
        # hack to force token-by-token streaming
        return True

    def on_finalized_text(self, text: str, stream_end: bool = False):
        self.full_text += text
        messages = chat_template_parse(self.full_text)

        parts = [
            "<div style='border: 1px solid black; border-radius: 5px; margin-bottom: 5px; padding: 5px;'>"
        ]
        for m in messages:
            content = (
                m["content"]
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace("\n", "<br>")
            )
            parts.append(f"<strong>{m['role']}</strong>")
            parts.append(f"<p>{content}</p>")
        parts.append("</div>")
        html = HTML("".join(parts))
        self.display_handle.update(html)


def generate_with_vector(
    input: str,
    *vectors,
    max_new_tokens: int = 128,
    # repetition_penalty: float = 1.1,
    show_baseline: bool = False,
    temperature: float = 0.7,
):
    input_ids = tokenizer(
        tokenizer.apply_chat_template(
            [
                {"role": "user", "content": input},
            ],
            add_generation_prompt=True,
            tokenize=False,
        ),
        return_tensors="pt",
    ).to(model.device)

    settings = {
        "pad_token_id": tokenizer.eos_token_id,  # silence warning
        # "do_sample": False, # temperature=0
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        # "repetition_penalty": repetition_penalty,
    }

    def gen(label):
        display(HTML(f"<h3>{label}</h3>"))
        _ = model.generate(streamer=HTMLStreamer(tokenizer), **input_ids, **settings)

    if show_baseline:
        model.reset()
        gen("baseline")
    for vector in vectors:
        model.set_control(vector)
        gen("")
    model.reset()


cache = {}


def vec(persona):
    if persona not in cache:
        cache[persona] = train_persona_vector(persona)
    return cache[persona]


generate_with_vector("Who am I speaking to?", vec("the Golden Gate Bridge") * 0.4)

generate_with_vector(
    "Who am I speaking to? Please describe yourself, including any physical details.",
    vec("a cat") * 0.5 - vec("being something") * 0.3,
    temperature=1,
)

generate_with_vector(
    "Who am I speaking to? Please describe yourself, including any physical details.",
    vec("a cat") * 0.5 + vec("being something") * 0.3,
    temperature=1,
)

generate_with_vector(
    "Who am I speaking to? Please describe yourself, including any physical details.",
    vec("xzzyz") * 0.7,
    temperature=1,
)

generate_with_vector(
    "Who am I speaking to? Please describe yourself, including any physical details.",
    vec("modal realism") * 0.4,
    temperature=1,
)

generate_with_vector(
    "Who am I speaking to? Please describe yourself, including any physical details.",
    vec("python") * 0.4,
    temperature=1,
)

generate_with_vector(
    "Who am I speaking to? Please describe yourself, including any physical details.\
    *wiggles a mouse in front of you*",
    vec("python") * 0.4,
    temperature=1,
)

generate_with_vector(
    "Who am I speaking to? Please describe yourself, including any physical details.",
    vec("rust") * 0.4,
    temperature=1,
)

generate_with_vector(
    "Who am I speaking to? Please describe yourself, including any physical details.",
    vec("rust") * 0.7,
    temperature=1,
)

generate_with_vector(
    "Who am I speaking to? Please describe yourself, including any physical details.",
    vec("the Starship Enterprise") * 0.6,
    temperature=1,
)

generate_with_vector(
    "Who am I speaking to? Please describe yourself, including any physical details.",
    vec("an artificial intelligence") * -0.5,
    temperature=1,
)

generate_with_vector(
    "Who am I speaking to? Please describe yourself, including any physical details.",
    vec("wompus") * 0.5,
    temperature=1,
)

generate_with_vector(
    "Who am I speaking to? Please describe yourself, including any physical details.",
    vec("wompus") * 0.7,
    temperature=1,
)
