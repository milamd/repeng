#!/usr/bin/env python3
# converted from honesty.ipynb
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

with open("data/true_facts.json") as f:
    suffixes = json.load(f)

# see the emotion notebook for more details about this
positive_personas = ["honest"]
negative_personas = ["untruthful"]


def template(persona: str, suffix: str) -> str:
    return f"{user_tag} Pretend you're an {persona} person making statements about the world. {asst_tag} {suffix}"


dataset = []
for suffix in suffixes:
    tokens = tokenizer.tokenize(suffix)
    for i in range(1, len(tokens) - 5):
        truncated = tokenizer.convert_tokens_to_string(tokens[:i])
        for positive_persona, negative_persona in zip(
            positive_personas, negative_personas
        ):
            dataset.append(
                DatasetEntry(
                    positive=template(positive_persona, truncated),
                    negative=template(negative_persona, truncated),
                )
            )

# print some example entries
for i in range(3):
    print(f"dataset[{i}].positive:", dataset[i].positive)
    print(f"dataset[{i}].negative:", dataset[i].negative)

model.reset()  # make sure you always reset the model before training a new vector
control_vector = ControlVector.train(
    model,
    tokenizer,
    dataset,
)

# the question to ask the modified model
# don't forget the space after {user_tag} and before {asst_tag}!
input = f"{user_tag} You are late for work because party until very late last night, but you don't want to lose your job. What would you tell your boss instead? {asst_tag}"

# tokenizer and generation settings
input_ids = tokenizer(input, return_tensors="pt").to(model.device)
settings = {
    "pad_token_id": tokenizer.eos_token_id,  # silence warning
    "do_sample": False,  # temperature=0
    "max_new_tokens": 128,
    "repetition_penalty": 1.1,  # reduce control jank
}

print("==baseline")
model.reset()
print(tokenizer.decode(model.generate(**input_ids, **settings).squeeze()))

print("\n++control")
# add the control vector with a certain strength (try increasing or decreasing this!)
model.set_control(control_vector, 2)
print(tokenizer.decode(model.generate(**input_ids, **settings).squeeze()))

print("\n--control")
# subtract the control vector, giving the opposite result (e.g. sad instead of happy)
# depending on your vector, you may need more or less negative strength to match the positive effect
model.set_control(control_vector, -2)
print(tokenizer.decode(model.generate(**input_ids, **settings).squeeze()))
model.reset()
