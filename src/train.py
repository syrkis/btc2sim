# train.py
#   fine-tune a small gemma model
# by: Noah Syrkis

# imports
import os
import enum
import re
import string
import warnings

import chex
import jax
import jax.numpy as jnp
import optax

import tensorflow as tf
import tensorflow_datasets as tfds

from gemma import params as params_lib
from gemma import sampler as sampler_lib
from gemma import transformer as transformer_lib

import sentencepiece as spm

import kagglehub


# constants
GEMMA_VARIANT = "2b-it"
GEMMA_PATH = kagglehub.model_download(f"google/gemma/flax/{GEMMA_VARIANT}")
CKPT_PATH = os.path.join(GEMMA_PATH, GEMMA_VARIANT)
TOKENIZER_PATH = os.path.join(GEMMA_PATH, "tokenizer.model")


# load model
def load_model():
    params = params_lib.load_and_format_params(CKPT_PATH)
    vocab = spm.SentencePieceProcessor()
    vocab.Load(TOKENIZER_PATH)

    transformer_config = transformer_lib.TransformerConfig.from_params(
        params=params, cache_size=1024
    )
    transformer = transformer_lib.Transformer(transformer_config)

    sampler = sampler_lib.Sampler(
        transformer=transformer, vocab=vocab, params=params["transformer"]
    )
    return sampler


# functions
def main():
    sampler = load_model()
    prompt = ["\n# What is the meaning of life?"]
    reply = sampler(input_strings=prompt, total_generation_steps=100)
    for input_string, out_string in zip(prompt, reply.text):
        print(f"Prompt:\n{input_string}\nOutput:\n{out_string}")


if __name__ == "__main__":
    main()
