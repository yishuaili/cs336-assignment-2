$ErrorActionPreference = "Continue"

uv run nsys profile -o result_small_ctx_128_3 --trace=cuda,nvtx python cs336_systems/benchmarking_lm.py --d_model 768 --d_ff 3072 --num_layers 12 --num_heads 12 --context_length 128 --warmup_iters 5 --num_runs 1 --benchmarking_iters 1

uv run nsys profile -o result_small_ctx_256_3 --trace=cuda,nvtx python cs336_systems/benchmarking_lm.py --d_model 768 --d_ff 3072 --num_layers 12 --num_heads 12 --context_length 256 --warmup_iters 5 --num_runs 1 --benchmarking_iters 1

uv run nsys profile -o result_small_ctx_512_3 --trace=cuda,nvtx python cs336_systems/benchmarking_lm.py --d_model 768 --d_ff 3072 --num_layers 12 --num_heads 12 --context_length 512 --warmup_iters 5 --num_runs 1 --benchmarking_iters 1

uv run nsys profile -o result_small_ctx_1024_3 --trace=cuda,nvtx python cs336_systems/benchmarking_lm.py --d_model 768 --d_ff 3072 --num_layers 12 --num_heads 12 --context_length 1024 --warmup_iters 5 --num_runs 1 --benchmarking_iters 1


uv run nsys profile -o result_med_ctx_128_3 --trace=cuda,nvtx python cs336_systems/benchmarking_lm.py --d_model 1024 --d_ff 4096 --num_layers 24 --num_heads 16 --context_length 128 --warmup_iters 5 --num_runs 1 --benchmarking_iters 1

uv run nsys profile -o result_med_ctx_256_3 --trace=cuda,nvtx python cs336_systems/benchmarking_lm.py --d_model 1024 --d_ff 4096 --num_layers 24 --num_heads 16 --context_length 256 --warmup_iters 5 --num_runs 1 --benchmarking_iters 1

uv run nsys profile -o result_med_ctx_512_3 --trace=cuda,nvtx python cs336_systems/benchmarking_lm.py --d_model 1024 --d_ff 4096 --num_layers 24 --num_heads 16 --context_length 512 --warmup_iters 5 --num_runs 1 --benchmarking_iters 1

uv run nsys profile -o result_med_ctx_1024_3 --trace=cuda,nvtx python cs336_systems/benchmarking_lm.py --d_model 1024 --d_ff 4096 --num_layers 24 --num_heads 16 --context_length 1024 --warmup_iters 5 --num_runs 1 --benchmarking_iters 1


uv run nsys profile -o result_large_ctx_128_3 --trace=cuda,nvtx python cs336_systems/benchmarking_lm.py --d_model 1280 --d_ff 5120 --num_layers 36 --num_heads 20 --context_length 128 --warmup_iters 5 --num_runs 1 --benchmarking_iters 1

uv run nsys profile -o result_large_ctx_256_3 --trace=cuda,nvtx python cs336_systems/benchmarking_lm.py --d_model 1280 --d_ff 5120 --num_layers 36 --num_heads 20 --context_length 256 --warmup_iters 5 --num_runs 1 --benchmarking_iters 1

uv run nsys profile -o result_large_ctx_512_3 --trace=cuda,nvtx python cs336_systems/benchmarking_lm.py --d_model 1280 --d_ff 5120 --num_layers 36 --num_heads 20 --context_length 512 --warmup_iters 5 --num_runs 1 --benchmarking_iters 1

uv run nsys profile -o result_large_ctx_1024_3 --trace=cuda,nvtx python cs336_systems/benchmarking_lm.py --d_model 1280 --d_ff 5120 --num_layers 36 --num_heads 20 --context_length 1024 --warmup_iters 5 --num_runs 1 --benchmarking_iters 1
