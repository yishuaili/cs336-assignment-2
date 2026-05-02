import subprocess
import re

configs = {
    "Small": ("--d_model 768 --d_ff 3072 --num_layers 12 --num_heads 12", 4),
    "Medium": ("--d_model 1024 --d_ff 4096 --num_layers 24 --num_heads 16", 4),
    "Medium-Large": ("--d_model 1536 --d_ff 6144 --num_layers 10 --num_heads 24", 4)
}

common_args_template = "--batch_size {bs} --context_length 128 --warmup_iters 2 --num_runs 2 --benchmarking_iters 10"
script_path = "cs336_systems/benchmarking_lm.py"
python_exec = ".venv\\Scripts\\python.exe"

def run_benchmark(name, size_args, bs, mixed=False):
    common_args = common_args_template.format(bs=bs)
    cmd = f"{python_exec} {script_path} {size_args} {common_args}"
    if mixed:
        cmd += " --mixed_precision True"
        
    print(f"Running {name} (BS: {bs}, Mixed Precision: {mixed})...")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    # Parse output
    fwd_match = re.search(r"Forward pass time: ([0-9.]+)", result.stdout)
    bwd_match = re.search(r"Backward pass time: ([0-9.]+)", result.stdout)
    
    if fwd_match and bwd_match:
        fwd = float(fwd_match.group(1))
        bwd = float(bwd_match.group(1))
        return fwd, bwd
    else:
        print(f"Error parsing output for {name} (Mixed: {mixed}):")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        return None, None

results = {}
for size_name, (size_args, bs) in configs.items():
    fwd_fp32, bwd_fp32 = run_benchmark(size_name, size_args, bs, mixed=False)
    fwd_bf16, bwd_bf16 = run_benchmark(size_name, size_args, bs, mixed=True)
    results[size_name] = {
        "FP32": (fwd_fp32, bwd_fp32),
        "BF16": (fwd_bf16, bwd_bf16)
    }

print("\n--- Results ---")
for size, res in results.items():
    print(f"{size} Model:")
    fwd_fp32, bwd_fp32 = res['FP32']
    fwd_bf16, bwd_bf16 = res['BF16']
    
    if fwd_fp32 is not None:
        print(f"  FP32: Forward = {fwd_fp32:.5f}s, Backward = {bwd_fp32:.5f}s")
        print(f"  BF16: Forward = {fwd_bf16:.5f}s, Backward = {bwd_bf16:.5f}s")
        fwd_speedup = fwd_fp32 / fwd_bf16
        bwd_speedup = bwd_fp32 / bwd_bf16
        print(f"  Speedup: Forward = {fwd_speedup:.2f}x, Backward = {bwd_speedup:.2f}x\n")
    else:
        print(f"  Failed to run benchmark for {size}\n")
