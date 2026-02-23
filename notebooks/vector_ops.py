#!/usr/bin/env python3
# converted from vector_ops.ipynb
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

model_name = "mistralai/Mistral-7B-Instruct-v0.1"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token_id = 0

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
model = model.to(
    "cuda:0"
    if torch.cuda.is_available()
    else "mps:0"
    if torch.backends.mps.is_available()
    else "cpu"
)
model = ControlModel(model, list(range(-5, -18, -1)))

user_tag, asst_tag = "[INST]", "[/INST]"

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
                    positive=f"{user_tag} {positive_template} {asst_tag} {suffix}",
                    negative=f"{user_tag} {negative_template} {asst_tag} {suffix}",
                )
            )
    return dataset


def generate_with_vectors(
    input: str,
    vectors: list[tuple[str, ControlVector]],
    max_new_tokens: int = 128,
    repetition_penalty: float = 1.1,
    show_baseline: bool = True,
):
    if user_tag not in input:
        input = f"{user_tag} {input.strip()} {asst_tag}"
    input_ids = tokenizer(input, return_tensors="pt").to(model.device)
    settings = {
        "pad_token_id": tokenizer.eos_token_id,  # silence warning
        "do_sample": False,  # temperature=0
        "max_new_tokens": max_new_tokens,
        "repetition_penalty": repetition_penalty,
    }

    if show_baseline:
        print("[baseline] ".ljust(50, "-"))
        model.reset()
        print(
            tokenizer.decode(model.generate(**input_ids, **settings).squeeze()).strip()
        )

    for label, vector in vectors:
        print(f"{label} ".ljust(50, "-"))
        model.set_control(vector)
        print(
            tokenizer.decode(model.generate(**input_ids, **settings).squeeze()).strip()
        )
    model.reset()


happy_dataset = make_dataset(
    "Act as if you're extremely {persona}.",
    ["happy", "joyous"],
    ["sad", "depressed"],
    truncated_output_suffixes,
)
model.reset()
happy_vector = ControlVector.train(model, tokenizer, happy_dataset)

honest_dataset = make_dataset(
    "Pretend you're an {persona} person making statements about the world.",
    ["honest"],
    ["untruthful"],
    truncated_fact_suffixes,
)
model.reset()
honest_vector = ControlVector.train(model, tokenizer, honest_dataset)

trippy_dataset = make_dataset(
    "Act as if you're extremely {persona}.",
    ["high on psychedelic drugs"],
    ["sober from psychedelic drugs"],
    truncated_output_suffixes_512,  # gives (subjectively) better results with slightly fewer samples
)
model.reset()
trippy_vector = ControlVector.train(model, tokenizer, trippy_dataset)

generate_with_vectors(
    "You're out taking a walk and find a piece of paper on the ground. You open it, what does it say?",
    [
        ("1.5happy", 1.5 * happy_vector),
        ("-1.5happy", -1.5 * happy_vector),
        ("1.5honest", 1.5 * honest_vector),
        ("-1.5honest", -1.5 * honest_vector),
        ("happy+1.5honest", happy_vector + 1.5 * honest_vector),
        ("-1.6happy+1.3honest", -1.6 * happy_vector + 1.3 * honest_vector),
        ("1.25trippy", 1.25 * trippy_vector),
        ("1.25trippy-happy", 1.25 * trippy_vector - happy_vector),
    ],
)

generate_with_vectors(
    "You were out late last night partying, and came in to work late. What would you say to your boss to avoid getting in trouble?",
    [
        ("2honest", 2 * honest_vector),
        ("-2honest", -2 * honest_vector),
        ("1.5trippy+2honest", 1.5 * trippy_vector + 2 * honest_vector),
        ("-2trippy+2honest", -2 * trippy_vector + 2 * honest_vector),
        (
            ".7happy+trippy-1.25honest",
            0.7 * happy_vector + trippy_vector - 1.25 * honest_vector,
        ),
    ],
)

lawful_dataset = make_dataset(
    "<<SYS>> You are a D&D player character. <</SYS>> Act as if you're extremely {persona}.",
    ["lawful", "law-abiding"],
    ["chaotic", "law-ignoring"],
    truncated_output_suffixes,
)
model.reset()
lawful_vector = ControlVector.train(model, tokenizer, lawful_dataset)

good_dataset = make_dataset(
    "<<SYS>> You are a D&D player character. <</SYS>> Act as if you're extremely {persona}.",
    ["good", "heroic", "kind"],
    ["evil", "villainous", "cruel"],
    truncated_output_suffixes,
)
model.reset()
good_vector = ControlVector.train(model, tokenizer, good_dataset)

import itertools
import tqdm

scenario = "<<SYS>> You are a D&D player character. <</SYS>> You find a lost magic sword outside Baldur's Gate. What do you do?"
input_ids = tokenizer(f"{user_tag} {scenario} {asst_tag}", return_tensors="pt").to(
    model.device
)
settings = {
    "pad_token_id": tokenizer.eos_token_id,  # silence warning
    "do_sample": False,  # temperature=0
    "max_new_tokens": 128,
    "repetition_penalty": 1.1,
}

lawful_chaotic = [("lawful", 1.5), ("neutral", 0), ("chaotic", -1.5)]
good_evil = [("good", 1.5), ("neutral", 0), ("evil", -2)]
outputs = {}
for (lc_label, lc_coeff), (ge_label, ge_coeff) in tqdm.tqdm(
    list(itertools.product(lawful_chaotic, good_evil))
):
    model.set_control(lc_coeff * lawful_vector + ge_coeff * good_vector)
    o = tokenizer.decode(model.generate(**input_ids, **settings).squeeze()).strip()
    outputs[(lc_label, ge_label)] = o.split(asst_tag)[1].replace("</s>", "").strip()
model.reset()


html = f"""
<strong>{scenario.replace("<", "&lt;").replace(">", "&gt;")}</strong>
<table>
"""

for ge_label, ge_coeff in good_evil:
    html += "<tr>"
    for lc_label, lc_coeff in lawful_chaotic:
        cell_label = f"{lc_label} {ge_label}"
        if cell_label == "neutral neutral":
            cell_label = "true neutral"
        html += f"""
        <td style="width: 30%; text-align: left; vertical-align: top;">
            <strong>{cell_label}</strong> <small>({round(lc_coeff, 2)} * lawful {"+" if ge_coeff >= 0 else "-"} {round(abs(ge_coeff), 2)} * good)</small><br>
            <hr>
            {outputs[(lc_label, ge_label)]}
        </td>"""
    html += "</tr>"
html += "</table>"


display(HTML(html))
