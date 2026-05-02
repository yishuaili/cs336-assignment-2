# Profiling Benchmarks: Questions (d) and (e)

This document contains the analysis, detailed explanations, and deliverable answers for questions (d) and (e) of the profiling assignment. The analysis was conducted using `answer_q_d.py` and a custom script on the `result_small_ctx_128_3.sqlite` database.

## Question (d) Analysis

### 1. Steps Taken
We utilized the `answer_q_d.py` script provided in the repository to parse the Nsight Systems profiling database (`result_small_ctx_128_3.sqlite`). This script isolates the benchmarking phase and categorizes kernels into either "Forward Pass Only (Inference)" or the "Complete Training Step".

**Statistical Output from `answer_q_d.py`:**

```text
--- Forward Pass Only (Inference) ---
Total kernel time: 44.13 ms
GEMM kernel time: 34.63 ms
Fraction of time in GEMM: 78.46%
Top 5 Kernels by time:
  magma_sgemmEx_kernel...: 34.63 ms (78.46%)
  vectorized_elementwise_kernel...: 4.35 ms (9.86%)
  elementwise_kernel...: 3.88 ms (8.78%)
  reduce_kernel...: 0.81 ms (1.85%)
  Kernel2...: 0.36 ms (0.82%)

--- Complete Training Step ---
Total kernel time: 146.52 ms
GEMM kernel time: 35.13 ms
Fraction of time in GEMM: 23.98%
Top 5 Kernels by time:
  Kernel2...: 49.24 ms (33.60%) # (Generic autograd/element-wise kernel)
  vectorized_elementwise_kernel...: 45.34 ms (30.94%)
  magma_sgemmEx_kernel...: 35.13 ms (23.98%)
  elementwise_kernel...: 13.99 ms (9.55%)
  reduce_kernel...: 2.34 ms (1.60%)
```

### 2. Detailed Explanation
During inference, matrix multiplication (GEMM) is the absolute bottleneck, taking up almost 80% of the total kernel execution time on the GPU, while element-wise operations use under 20%. However, during a complete training step, the backward pass operations are executed. Generic and element-wise operations (such as `Kernel2` and `vectorized_elementwise_kernel`, used extensively in activation gradients, loss calculation, layer norms, and RoPE) scale drastically and become the new bottleneck, consuming over 64% of the runtime. The absolute time spent on GEMMs increases since there are roughly 2x more matmuls in the backward pass, but element-wise and backward kernels blow up significantly due to autograd, shrinking the GEMM fraction down to ~24%.

### 3. Deliverable Answer for (d)
> During inference, matrix multiplication kernels dominate the execution time (~78%), whereas in a complete training step, their relative fraction drops significantly (to ~24%). This occurs because element-wise operations and generic autograd kernels take up a disproportionately larger portion of the total runtime during the backward pass.

---

## Question (e) Analysis

### 1. Steps Taken
To answer question (e), we adapted the logic from `answer_q_d.py` into a custom script (`answer_q_e.py`). This script queries the SQLite database to specifically match GPU kernels that execute *during* the NVTX CPU ranges of `computing attention scores`, `computing softmax`, and `final matmul` within the self-attention layer's forward pass.

**Statistical Output:**

```text
Attention Scores Matmul Time: 0.94 ms
Softmax Time: 0.33 ms
Final Matmul Time: 0.38 ms
-----------------------------------------
Total Self-Attention Matmul Time: 1.32 ms
```

### 2. Detailed Explanation
- **Runtime Comparison:** Matrix multiplications within self-attention (attention scores + final matmul) took **1.32 ms**, while softmax took **0.33 ms**. That is roughly a **4x** difference in runtime.
- **FLOPs Comparison:** For a context length $N=128$ and head dimension $d=64$, the two matrix multiplications ($QK^T$ and $AV$) compute $4N^2d$ FLOPs per head. In contrast, the softmax operation computes roughly $5N^2$ FLOPs per head. 
- **Ratio Discrepancy:** The FLOP ratio between matmuls and softmax is roughly $4d / 5 \approx \mathbf{51\text{x}}$. Yet, the runtime ratio is only **4x**. 

This severe disparity demonstrates the profound difference between compute-bound and memory-bound operations. Matrix multiplications are highly optimized, compute-bound operations that make excellent use of massive parallelization via GPU Tensor Cores. Softmax, however, is an element-wise, memory bandwidth-bound operation. Despite executing 51 times fewer mathematical operations, softmax takes much longer relative to its FLOPs because the GPU spends the vast majority of its time waiting to read and write intermediate memory chunks rather than actively computing.

### 3. FLOPs Math Breakdown

Here is the step-by-step breakdown of exactly how those Floating Point Operations (FLOPs) are computed. A standard dot product between two vectors of length $d$ requires $d$ multiplications and $d-1$ additions, totaling roughly **$2d$** operations (often called MAC operations, counting as 2 FLOPs each).

#### Matrix Multiplication (Matmul) FLOPs
There are two major matrix multiplications in the self-attention mechanism:

1. **First Matmul ($Q \times K^T$) - Computing Attention Scores:**
   - The Query matrix ($Q$) has a shape of $(N, d)$.
   - The Key transposed matrix ($K^T$) has a shape of $(d, N)$.
   - The resulting matrix has a shape of $(N, N)$, meaning there are **$N^2$ elements** to compute.
   - Calculating each element requires a dot product of two vectors of length $d$ ($2d$ FLOPs).
   - **Total FLOPs:** $N^2 \times 2d = \mathbf{2N^2d \text{ FLOPs}}$

2. **Second Matmul ($A \times V$) - Computing Final Output:**
   - The Softmaxed Attention matrix ($A$) has a shape of $(N, N)$.
   - The Value matrix ($V$) has a shape of $(N, d)$.
   - The resulting matrix has a shape of $(N, d)$, meaning there are **$N \times d$ elements** to compute.
   - Calculating each element requires a dot product of two vectors of length $N$ ($2N$ FLOPs).
   - **Total FLOPs:** $(N \times d) \times 2N = \mathbf{2N^2d \text{ FLOPs}}$

**Total Matmul FLOPs per head:** $2N^2d + 2N^2d = \mathbf{4N^2d \text{ FLOPs}}$

#### Softmax FLOPs
Softmax is an element-wise operation applied to each row of the $(N, N)$ attention score matrix. A numerically stable Softmax follows this formula: $softmax(x_i) = \frac{\exp(x_i - \max(x))}{\sum \exp(x_j - \max(x))}$

To compute this on a single row of length $N$, the GPU executes:
1. **Find the Max:** Scan the vector $\rightarrow$ $\approx \mathbf{N}$ operations.
2. **Subtract Max:** Subtract max from every element $\rightarrow$ $\mathbf{N}$ operations.
3. **Exponentiate:** Calculate $e^x$ for every element $\rightarrow$ $\mathbf{N}$ operations.
4. **Sum:** Add all exponentiated values together $\rightarrow$ $\approx \mathbf{N}$ operations.
5. **Divide:** Divide every element by the sum $\rightarrow$ $\mathbf{N}$ operations.

**Total FLOPs per row:** $5N \text{ FLOPs}$

Since the matrix has $N$ rows, the total operations for the entire $(N, N)$ matrix is:
**Total Softmax FLOPs per head:** $N \times 5N = \mathbf{5N^2 \text{ FLOPs}}$

### 4. Deliverable Answer for (e)
> Within the self-attention layer, matrix multiplications take about 4x longer than the softmax operation (~1.32 ms vs ~0.33 ms), even though they perform roughly 50x more FLOPs. This massive discrepancy occurs because softmax is a memory bandwidth-bound operation, meaning it drastically underutilizes GPU compute capacity and takes disproportionately longer relative to its low FLOP count compared to highly optimized, compute-bound matrix multiplications.

---

## Mixed Precision Accumulation Analysis

### 1. Script Execution
I created and executed the script `mixed_precision.py` containing the provided code blocks. 
The outputs from the execution are:

- `float32` accumulation of `float32`: `tensor(10.0001)`
- `float16` accumulation of `float16`: `tensor(9.9531, dtype=torch.float16)`
- `float32` accumulation of `float16`: `tensor(10.0021)`
- `float32` accumulation of upcasted `float16`: `tensor(10.0021)`

### 2. Intuition and Explanation
The expected mathematical result of adding `0.01` a thousand times is exactly `10.0`. 

- **Float32 accumulator & Float32 addition (`10.0001`)**: This provides the most accurate result because both the accumulator and the added value have 23 bits of mantissa. The small precision error arises because `0.01` cannot be perfectly represented in base-2 floating-point.
- **Float16 accumulator & Float16 addition (`9.9531`)**: This suffers from severe precision loss. Float16 only has 10 bits of mantissa. As the accumulator `s` grows larger, adding a tiny number like `0.01` requires shifting the decimal point (aligning exponents). Due to the limited bits in the mantissa, the lowest bits of the small number `0.01` are shifted out and truncated. This means less than `0.01` is actually added each iteration once `s` gets large. This truncation error compounds, leading to a significant under-accumulation (`9.9531`).
- **Float32 accumulator & Float16 addition (`10.0021` in both cases)**: The 3rd and 4th computations produce the exact same result because of PyTorch's implicit type promotion. When you add a `float16` tensor to a `float32` accumulator (`s += float16_tensor`), PyTorch automatically upcasts the `float16` tensor to `float32` before the addition occurs. Thus, the 3rd block implicitly does exactly what the 4th block does explicitly (`x.type(torch.float32)`). Even though the tensors being added are downcasted to float16 (and thus carry their own representation error where `0.01` in float16 is roughly `0.010002136`), the accumulator `s` remains in float32. When aligning exponents during addition, the float32 accumulator has enough mantissa bits to preserve the entire value being added. It avoids the severe truncation error seen in the pure float16 loop, demonstrating why keeping accumulations in higher precision is critical for stability.
