from dataclasses import dataclass, field, asdict
from typing import Optional, Callable
from transformers import HfArgumentParser
import torch
from contextlib import nullcontext
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
import numpy as np
import time

from cs336_basics.nn_utils import cross_entropy, clip_gradient
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW

# Example command to run the script:
# uv run python cs336_systems/benchmarking_lm.py \
#     --wandb_run_name "benchmarking_small" \
#     --d_model 768 \
#     --d_ff 3072 \
#     --num_layers 12 \
#     --num_heads 12 \
#     --warmup_iters 5 \
#     --num_runs 5 \
#     --benchmarking_iters 10


# parsing the benchmarking configuration
@dataclass
class BenchMarkingConfig:
    # treatment variables for scaling
    num_layers: int
    d_model: int
    num_heads: int
    d_ff: int
    # optional arguments
    num_runs: Optional[int] = field(default=5)
    benchmarking_iters: Optional[int] = field(default=10)
    warmup_iters: Optional[int] = field(default=1)
    wandb_run_name: Optional[str] = field(default='None')
    mixed_precision: Optional[bool] = field(default=False)
    # use_rms_norm: Optional[bool] = field(default=True)
    rope_theta: Optional[float] = field(default=10000)

    # fixed configs
    wandb_project: str = 'cs336-assignment2-systems'
    context_length: int = 128
    batch_size: int = 16
    vocab_size: int = 10000

    def __post_init__(self):
        self.wandb_logging = self.wandb_run_name != 'None'
        if torch.cuda.is_available():
            self.device = 'cuda'
        elif torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif torch.backends.mps.is_available():
        # MPS uses torch.mps.synchronize()
        torch.mps.synchronize()


# parsing config
parser = HfArgumentParser(BenchMarkingConfig)
config = parser.parse_args_into_dataclasses()[0]
if config.wandb_logging:
    import wandb
    wandb.init(project=config.wandb_project, name=config.wandb_run_name)
logging.info(f'Benchmarking with config: {asdict(config)}')

# generate random dataset for bench marking
x = torch.randint(0, config.vocab_size, (config.batch_size, config.context_length))
x = x.to(config.device)
y = torch.randint(0, config.vocab_size, (config.batch_size, config.context_length))
y = y.to(config.device) 

# initializing a rando model
# Define the specific keys the model architecture needs
model_args = {
    'num_layers': config.num_layers,
    'd_model': config.d_model,
    'num_heads': config.num_heads,
    'd_ff': config.d_ff,
    #'use_rms_norm': config.use_rms_norm,
    'vocab_size': config.vocab_size,
    'context_length': config.context_length,
    'rope_theta': config.rope_theta,
}

model = BasicsTransformerLM(**model_args)
#model = BasicsTransformerLM(**asdict(config))
model = model.to(config.device)
import torch._dynamo
torch._dynamo.config.suppress_errors = True
model = torch.compile(model)

# loading the optimizer
optimizer = AdamW(model.parameters())
# initialize the training context
if config.mixed_precision:
    train_context = torch.amp.autocast(device_type=config.device, dtype=torch.bfloat16)
else:
    train_context = nullcontext()

def forward_pass():
    #torch.cuda.synchronize()
    synchronize()
    logits = model(x)
    loss = cross_entropy(logits, y)
    #torch.cuda.synchronize()
    synchronize()
    return loss

def backward_pass():
    #torch.cuda.synchronize()
    synchronize()
    optimizer.zero_grad()
    loss.backward()
    #torch.cuda.synchronize()
    synchronize()

def timer(run: Callable):
    t1 = time.time()
    result = run()
    t2 = time.time()
    return t2-t1, result

iter_num = 0
# warm up
for _ in range(config.warmup_iters):
    with train_context:
        loss = forward_pass()
        backward_pass()
        clip_gradient(model.parameters(), 1.0)
        optimizer.step()

all_forward_times = []
all_backward_times = []

for run in range(config.num_runs):
    forward_times = np.zeros(config.benchmarking_iters)
    backward_times = np.zeros(config.benchmarking_iters)
    for i in range(config.benchmarking_iters):
        with train_context:
            forward_times[i], loss = timer(forward_pass)
            backward_times[i], _ = timer(backward_pass)
            clip_gradient(model.parameters(), 1.0)
            optimizer.step()
    
    # Store this run's timings
    all_forward_times.extend(forward_times)
    all_backward_times.extend(backward_times)
    
    # Print per-run stats
    print(f'Run {run + 1}/{config.num_runs} - Forward pass: {np.mean(forward_times):.5f}s +- {np.std(forward_times):.5f}s, Backward pass: {np.mean(backward_times):.5f}s +- {np.std(backward_times):.5f}s')


# benchmarking over all runs
print(f'\n--- Overall Benchmarking ({config.num_runs} runs of {config.benchmarking_iters} steps) ---')
print(f'Forward pass time: {np.mean(all_forward_times):.5f}, std: {np.std(all_forward_times):.5f}')
print(f'Backward pass time: {np.mean(all_backward_times):.5f}, std: {np.std(all_backward_times):.5f}')