import timeit
import torch
import pandas as pd
from statistics import mean, stdev
from cs336_basics.model import BasicsTransformerLM
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

def benchmark(model, x, y, mode):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    def step_forward():
        with torch.no_grad():
            _ = model(x)
    
    def step_forward_backward():
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out.view(-1, vocab_size), y.view(-1))
        loss.backward()
        optimizer.step()
    step_fn = step_forward if mode == "forward" else step_forward_backward

    # Warm up
    for _ in range(warmup_steps):
        step_fn
    
    # Times steps
    times = []
    for _ in range(timing_steps):
        start = timeit.default_timer()
        step_fn()
        torch.cuda.synchronize()
        end = timeit.default_timer()
        times.append(end - start)

    return mean(times), stdev(times)
    
def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Transformer models")
    parser.add_argument("--d_model", type=int, help="Model dimension")
    parser.add_argument("--d_ff", type=int, help="Feedforward dimension")
    parser.add_argument("--num_layers", type=int, help="Number of Transformer layers")
    parser.add_argument("--num_heads", type=int, help="Number of attention heads")
    parser.add_argument("--all", action="store_true", help="Run all predefined configurations")
    parser.add_argument("--context_length", type=int, help="Sequence context length")
    parser.add_argument("--warmup_steps", type=int, help="Number of warmup steps")
    return parser.parse_args

def main():
