$ErrorActionPreference = "Continue"

$OutputDir = "profiles_batch_6"
New-Item -ItemType Directory -Force -Path $OutputDir

uv run nsys profile -o $OutputDir/result_small_ctx_128_batch_6 --trace=cuda,nvtx python cs336_systems/benchmarking_lm.py --batch_size 6 --d_model 768 --d_ff 3072 --num_layers 12 --num_heads 12 --context_length 128 --warmup_iters 5 --num_runs 1 --benchmarking_iters 1 > $OutputDir/output_small_ctx_128.log 2>&1

uv run nsys profile -o $OutputDir/result_small_ctx_256_batch_6 --trace=cuda,nvtx python cs336_systems/benchmarking_lm.py --batch_size 6 --d_model 768 --d_ff 3072 --num_layers 12 --num_heads 12 --context_length 256 --warmup_iters 5 --num_runs 1 --benchmarking_iters 1 > $OutputDir/output_small_ctx_256.log 2>&1

uv run nsys profile -o $OutputDir/result_small_ctx_512_batch_6 --trace=cuda,nvtx python cs336_systems/benchmarking_lm.py --batch_size 6 --d_model 768 --d_ff 3072 --num_layers 12 --num_heads 12 --context_length 512 --warmup_iters 5 --num_runs 1 --benchmarking_iters 1 > $OutputDir/output_small_ctx_512.log 2>&1

uv run nsys profile -o $OutputDir/result_small_ctx_1024_batch_6 --trace=cuda,nvtx python cs336_systems/benchmarking_lm.py --batch_size 6 --d_model 768 --d_ff 3072 --num_layers 12 --num_heads 12 --context_length 1024 --warmup_iters 5 --num_runs 1 --benchmarking_iters 1 > $OutputDir/output_small_ctx_1024.log 2>&1


uv run nsys profile -o $OutputDir/result_med_ctx_128_batch_6 --trace=cuda,nvtx python cs336_systems/benchmarking_lm.py --batch_size 6 --d_model 1024 --d_ff 4096 --num_layers 24 --num_heads 16 --context_length 128 --warmup_iters 5 --num_runs 1 --benchmarking_iters 1 > $OutputDir/output_med_ctx_128.log 2>&1

uv run nsys profile -o $OutputDir/result_med_ctx_256_batch_6 --trace=cuda,nvtx python cs336_systems/benchmarking_lm.py --batch_size 6 --d_model 1024 --d_ff 4096 --num_layers 24 --num_heads 16 --context_length 256 --warmup_iters 5 --num_runs 1 --benchmarking_iters 1 > $OutputDir/output_med_ctx_256.log 2>&1

uv run nsys profile -o $OutputDir/result_med_ctx_512_batch_6 --trace=cuda,nvtx python cs336_systems/benchmarking_lm.py --batch_size 6 --d_model 1024 --d_ff 4096 --num_layers 24 --num_heads 16 --context_length 512 --warmup_iters 5 --num_runs 1 --benchmarking_iters 1 > $OutputDir/output_med_ctx_512.log 2>&1

# uv run nsys profile -o $OutputDir/result_med_ctx_1024_batch_6 --trace=cuda,nvtx python cs336_systems/benchmarking_lm.py --batch_size 6 --d_model 1024 --d_ff 4096 --num_layers 24 --num_heads 16 --context_length 1024 --warmup_iters 5 --num_runs 1 --benchmarking_iters 1 > $OutputDir/output_med_ctx_1024.log 2>&1


uv run nsys profile -o $OutputDir/result_large_ctx_128_batch_6 --trace=cuda,nvtx python cs336_systems/benchmarking_lm.py --batch_size 6 --d_model 1280 --d_ff 5120 --num_layers 36 --num_heads 20 --context_length 128 --warmup_iters 5 --num_runs 1 --benchmarking_iters 1 > $OutputDir/output_large_ctx_128.log 2>&1

# uv run nsys profile -o $OutputDir/result_large_ctx_256_batch_6 --trace=cuda,nvtx python cs336_systems/benchmarking_lm.py --batch_size 6 --d_model 1280 --d_ff 5120 --num_layers 36 --num_heads 20 --context_length 256 --warmup_iters 5 --num_runs 1 --benchmarking_iters 1 > $OutputDir/output_large_ctx_256.log 2>&1

# uv run nsys profile -o $OutputDir/result_large_ctx_512_batch_6 --trace=cuda,nvtx python cs336_systems/benchmarking_lm.py --batch_size 6 --d_model 1280 --d_ff 5120 --num_layers 36 --num_heads 20 --context_length 512 --warmup_iters 5 --num_runs 1 --benchmarking_iters 1 > $OutputDir/output_large_ctx_512.log 2>&1

# uv run nsys profile -o $OutputDir/result_large_ctx_1024_batch_6 --trace=cuda,nvtx python cs336_systems/benchmarking_lm.py --batch_size 6 --d_model 1280 --d_ff 5120 --num_layers 36 --num_heads 20 --context_length 1024 --warmup_iters 5 --num_runs 1 --benchmarking_iters 1 > $OutputDir/output_large_ctx_1024.log 2>&1
