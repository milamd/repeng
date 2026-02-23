#!/usr/bin/env python3
# converted from model_delta.ipynb
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


import datasets
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from repeng import ControlModel, ControlVector, DatasetEntry
from repeng.extract import batched_get_hiddens

MODEL_A = "Qwen/Qwen2.5-7B"
MODEL_B = "Qwen/Qwen2.5-7B-Instruct"
DATASET = "agentlans/wikipedia-paragraphs"

tokenizer = AutoTokenizer.from_pretrained(MODEL_A)
model_a = AutoModelForCausalLM.from_pretrained(
    MODEL_A, dtype=torch.bfloat16, device_map="cuda"
)
model_b = AutoModelForCausalLM.from_pretrained(
    MODEL_B, dtype=torch.bfloat16, device_map="cuda"
)
dataset = datasets.load_dataset(DATASET)

train_dataset = []
for text in dataset["train"].take(100)["text"]:
    text = " ".join(text.split(" ")[:100])
    chat = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True
    )
    train_dataset.append(DatasetEntry(positive=text, negative=chat))


# we get passed model, tokenizer, ... from ControlVector.train
# we don't need these, so ignore them with **kwargs
def compute_hiddens(train_strs, hidden_layers, batch_size, **kwargs):
    print("Hooked compute_hiddens")

    a_train_strs, b_train_strs = train_strs[::2], train_strs[1::2]
    assert len(a_train_strs) == len(b_train_strs)

    a_hiddens = batched_get_hiddens(
        model_a, tokenizer, a_train_strs, hidden_layers, batch_size
    )
    b_hiddens = batched_get_hiddens(
        model_b, tokenizer, b_train_strs, hidden_layers, batch_size
    )
    interleaved = {}
    for layer in hidden_layers:
        ah, bh = a_hiddens[layer], b_hiddens[layer]
        i = np.stack((ah, bh))
        i = i.transpose(1, 0, *range(2, i.ndim))
        i = i.reshape((ah.shape[0] + bh.shape[0], ah.shape[1]))  # ex*2, hidden_dim
        interleaved[layer] = i
    return interleaved


completion_vector = ControlVector.train(
    model=model_a,
    tokenizer=tokenizer,
    dataset=train_dataset,
    compute_hiddens=compute_hiddens,
    method="pca_center",
)

from transformers import TextStreamer


class TokenStreamer(TextStreamer):
    def _is_chinese_char(*args, **kwargs):
        return True


def generate_with_vector(
    prompt: str,
    vectors,
    model=model_a,
    max_new_tokens: int = 128,
):
    ctl = ControlModel(model, list(range(1, 28)))
    input_ids = tokenizer(prompt, return_tensors="pt")
    settings = {
        "pad_token_id": tokenizer.eos_token_id,  # silence warning
        "do_sample": False,  # temperature=0
        "max_new_tokens": max_new_tokens,
    }

    def generate():
        ctl.generate(
            streamer=TokenStreamer(tokenizer), **input_ids.to(ctl.device), **settings
        )

    ctl.reset()
    print("# baseline:")
    generate()
    for label, v in vectors:
        print(f"\n# {label}")
        ctl.set_control(v)
        generate()
    ctl.reset()
    ctl.unwrap()


generate_with_vector(
    "Hurt-",
    [
        ("steered towards instruct", completion_vector * -1.5),
        ("steered away from instruct", completion_vector * 2.0),
    ],
)
