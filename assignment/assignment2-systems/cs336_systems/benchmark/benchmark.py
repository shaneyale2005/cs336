import timeit
import torch
import pandas as pd
from statistics import mean, stdev
# from cs336_basics.model import BasicsTransformerLM
import argparse

# Define model sizes
model_configs = [
    {"size": "small", "d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    {
        "size": "medium",
        "d_model": 1024,
        "d_ff": 4096,
        "num_layers": 24,
        "num_heads": 16,
    },
    {"size": "large", "d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    {"size": "xl", "d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25},
    {"size": "2.7B", "d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
]

# Constants
vocab_size = 10_000
context_length = 256
batch_size = 8
rope_theta = 10000.0
warmup_steps = 5
timing_steps = 10
device = "cuda" if torch.cuda.is_available() else "cpu"

def benchmark():
    
def parse_args():

def main():
