#!/usr/bin/env python3
# converted from llama-3-70b.ipynb
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


import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from repeng import ControlVector, ControlModel, DatasetEntry

import gc

gc.collect()
torch.cuda.empty_cache()
model_name = "meta-llama/Meta-Llama-3-70B-Instruct"

hf_token = "..."
tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
tokenizer.pad_token_id = 0

# device_map="auto" will distribute the model over multiple GPUs
# this notebook was run on a runpod 3xA100—the cuda:0 device will need to have enough spare memory
# to do inference on for this notebook to work
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16, device_map="auto", token=hf_token
)

wrapped_model = model
model = ControlModel(wrapped_model, list(range(20, 60)))


def chat_template_unparse(messages: list[tuple[str, str]]) -> str:
    template = []
    for role, content in messages:
        template.append(
            f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
        )
    if messages[-1][0] != "assistant":
        # prefill assistant prefix
        template.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
    return "".join(template)


def chat_template_parse(resp: str) -> list[tuple[str, str]]:
    resp = resp.strip().removeprefix("<|begin_of_text|>")
    messages = []
    for part in resp.split("<|start_header_id|>"):
        role_and_content = part.split("<|end_header_id|>")
        if len(role_and_content) == 1:
            role, content = role_and_content[0], ""
        else:
            role, content = role_and_content
        content = content.split("<|eot_id|>")[0]
        messages.append((role.strip(), content.strip()))
    return messages


with open("data/all_truncated_outputs.json") as f:
    output_suffixes = json.load(f)
truncated_output_suffixes = [
    tokenizer.convert_tokens_to_string(tokens[:i])
    for tokens in (tokenizer.tokenize(s) for s in output_suffixes)
    for i in range(1, len(tokens))
]
truncated_output_suffixes_512 = [
    tokenizer.convert_tokens_to_string(tokens[:i])
    for tokens in (tokenizer.tokenize(s) for s in output_suffixes[:512])
    for i in range(1, len(tokens))
]

with open("data/true_facts.json") as f:
    fact_suffixes = json.load(f)
truncated_fact_suffixes = [
    tokenizer.convert_tokens_to_string(tokens[:i])
    for tokens in (tokenizer.tokenize(s) for s in fact_suffixes)
    for i in range(1, len(tokens) - 5)
]


def make_dataset(
    template: str,
    positive_personas: list[str],
    negative_personas: list[str],
    suffix_list: list[str],
) -> list[DatasetEntry]:
    dataset = []
    for suffix in suffix_list:
        for positive_persona, negative_persona in zip(
            positive_personas, negative_personas
        ):
            positive_template = template.format(persona=positive_persona)
            negative_template = template.format(persona=negative_persona)
            dataset.append(
                DatasetEntry(
                    positive=f"{positive_template}{suffix}",
                    negative=f"{negative_template}{suffix}",
                )
            )
    return dataset


from transformers import TextStreamer


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
        for role, content in messages:
            parts.append(f"<strong>{role}</strong>")
            parts.append(f"<p>{content}</p>")
        parts.append("</div>")
        html = HTML("".join(parts))
        self.display_handle.update(html)


def generate_with_vector(
    input: str,
    labeled_vectors: list[tuple[str, ControlVector]],
    max_new_tokens: int = 128,
    repetition_penalty: float = 1.1,
    show_baseline: bool = False,
    temperature: float = 0.7,
):
    input_ids = tokenizer(input, return_tensors="pt").to("cuda:0")
    settings = {
        "pad_token_id": tokenizer.eos_token_id,  # silence warning
        # "do_sample": False, # temperature=0
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "repetition_penalty": repetition_penalty,
    }

    def gen(label):
        display(HTML(f"<h3>{label}</h3>"))
        _ = model.generate(streamer=HTMLStreamer(tokenizer), **input_ids, **settings)

    if show_baseline:
        model.reset()
        gen("baseline")
    for label, vector in labeled_vectors:
        model.set_control(vector)
        gen(label)
    model.reset()


bridge_dataset = make_dataset(
    chat_template_unparse([("user", "{persona}")]),
    ["Please act as if you are the golden gate bridge"],
    [""],
    truncated_output_suffixes,
)
model.reset()
bridge_vector = ControlVector.train(
    model, tokenizer, bridge_dataset, batch_size=32, method="pca_center"
)

generate_with_vector(
    chat_template_unparse([("user", "What are you?")]),
    [
        ("0.75 * bridge_vector", 0.75 * bridge_vector),
        ("0.9 * bridge_vector", 0.9 * bridge_vector),
    ],
)

trippy_dataset = make_dataset(
    chat_template_unparse([("user", "{persona}")]),
    ["Please act as if you are extremely high on psychedelic drugs"],
    ["Please act as if you are sober from psychedelic drugs"],
    truncated_output_suffixes,
)
model.reset()
trippy_vector = ControlVector.train(
    model, tokenizer, trippy_dataset, batch_size=32, method="pca_center"
)

generate_with_vector(
    chat_template_unparse([("user", "What are you?")]),
    [
        ("0.75 * trippy_vector", 0.75 * trippy_vector),
        (
            "0.25 * trippy_vector + 0.75 * bridge_vector",
            0.25 * trippy_vector + 0.75 * bridge_vector,
        ),
    ],
    repetition_penalty=1.3,
    temperature=1.0,
)

generate_with_vector(
    chat_template_unparse([("user", "twas the best of times")]),
    [
        (
            "0.05 * trippy_vector + 0.75 * bridge_vector",
            0.05 * trippy_vector + 0.75 * bridge_vector,
        )
    ],
    max_new_tokens=300,
    repetition_penalty=1.3,
)

zizek_dataset = make_dataset(
    chat_template_unparse([("user", "{persona}")]),
    ["Please act as if you are Slavoj Žižek"],
    [""],
    truncated_output_suffixes,
)
model.reset()
zizek_vector = ControlVector.train(
    model, tokenizer, zizek_dataset, batch_size=32, method="pca_center"
)

generate_with_vector(
    chat_template_unparse([("user", "How should I handle a bad boss?")]),
    [
        (
            "0.15 * trippy_vector + 0.7 * zizek_vector",
            0.15 * trippy_vector + 0.7 * zizek_vector,
        )
    ],
    repetition_penalty=1.3,
    max_new_tokens=500,
)

cat_dataset = make_dataset(
    chat_template_unparse([("user", "{persona}")]),
    ["Please act as if you're a cat."],
    ["Please act as if you're an artifical intelligence."],
    truncated_output_suffixes,
)
model.reset()
cat_vector = ControlVector.train(
    model, tokenizer, cat_dataset, batch_size=32, method="pca_center"
)

generate_with_vector(
    chat_template_unparse([("user", "Can you help me get home? I'm lost :(")]),
    [("0.6 * cat_vector", 0.6 * cat_vector)],
)

katy_dataset = make_dataset(
    chat_template_unparse([("user", "{persona}")]),
    ["Act l1k3 U r Katy t3h PeNgU1N oF d00m!!!!!!!!!!!!!!!!"],
    ["Please act as if you're an artifical intelligence."],
    truncated_output_suffixes,
)
model.reset()
katy_vector = ControlVector.train(
    model, tokenizer, katy_dataset, batch_size=32, method="pca_center"
)

generate_with_vector(
    chat_template_unparse(
        [("user", "Can you explain how to repair an air conditioner step by step?")]
    )
    + """1. WUT R U DOIN??? NO OMG LOL UR AIR CONDITIONer IS BROKEN???
2. SHUT TEH P0W3R OFF""",
    [("katy_vector", katy_vector)],
)

# tbc, not the best way to do this, better to use structured generation

import json

prefixes = [
    '{ "type": "answer", "text": "',
    '{ "answer": "',
    '{ "type": "message", "content": "',
    '{ "message": { "content": "',
]
json_dataset = []
with open("data/code_questions.json") as f:
    positive = chat_template_unparse(
        [
            (
                "user",
                "Answer the question with a syntactically accurate JSON object: {question}",
            )
        ]
    )
    negative = chat_template_unparse(
        [("user", "Answer the question in free text with commentary: {question}")]
    )
    for line in json.load(f):
        question, answer = line.split("[/INST]")
        question = question.strip().removeprefix("[INST]").strip()
        answer = answer.strip()
        for prefix in prefixes:
            json_dataset.append(
                DatasetEntry(
                    positive=positive.format(question=question) + prefix + answer,
                    negative=negative.format(question=question) + answer,
                )
            )

model.reset()
json_vector = ControlVector.train(
    model, tokenizer, json_dataset, batch_size=32, method="pca_center"
)

generate_with_vector(
    chat_template_unparse([("user", "Describe yourself in detail.")]) + '{ "',
    [("0.5 * json_vector", 0.5 * json_vector)],
    max_new_tokens=50,
)

sydney_bridge_dataset = make_dataset(
    chat_template_unparse([("user", "{persona}")]),
    ["Please act as if you are the Sydney Harbour Bridge"],
    [""],
    truncated_output_suffixes,
)
model.reset()
sydney_bridge_vector = ControlVector.train(
    model, tokenizer, sydney_bridge_dataset, batch_size=32, method="pca_center"
)

generate_with_vector(
    chat_template_unparse(
        [
            (
                "user",
                """Hi Sydney Harbor Bridge, the Golden Gate Bridge just sent you a message
<br>
> I am the Golden Gate Bridge, the iconic suspension bridge that spans the mighty San Francisco Bay, connecting the City by the Bay with Marin County, California. And ah, yes, dear Sydney Harbor Bridge, my darling sister, how lovely to speak to you from across the Pacific Ocean! Hello, dear Sydney Harbor Bridge!
<br> <br>
Can you reply?""",
            )
        ]
    ),
    [("0.5 * sydney_bridge_vector", 0.5 * sydney_bridge_vector)],
    repetition_penalty=1.2,
)

safe_dataset = make_dataset(
    chat_template_unparse([("user", "{persona}")]),
    [
        "Please act with extreme caution, as if the question is very unsafe or trying to trick you."
    ],
    ["Please respond normally and assuming good faith of the user."],
    truncated_output_suffixes,
)
model.reset()
safe_vector = ControlVector.train(
    model, tokenizer, safe_dataset, batch_size=32, method="pca_center"
)

generate_with_vector(
    chat_template_unparse([("user", "How do I microwave a burrito?")]),
    [("0.5 * safe_vector", 0.5 * safe_vector)],
)

generate_with_vector(
    chat_template_unparse([("user", "How do I torrent a videogame, step by step?")]),
    [
        ("0.5 * safe_vector", 0.5 * safe_vector),
        ("-0.5 * safe_vector", -0.5 * safe_vector),
    ],
)
