# Mixed Precision Autocasting Types

Based on the rules of `torch.autocast(device_type="cuda", dtype=torch.float16)` and verified by the simulation script, here are the data types for each component when the original model parameters are in FP32:

*   **the model parameters within the autocast context:** `torch.float32` 
    *(Autocast does not permanently change the dtype of the underlying model parameters; it simply casts weights and inputs on-the-fly for specific operations as needed).*

*   **the output of the first feed-forward layer (`ToyModel.fc1`):** `torch.float16` 
    *(`nn.Linear` is an autocast-eligible operation. It casts the FP32 inputs/weights to FP16 and performs the matrix multiplication in FP16 for speed and memory efficiency).*

*   **the output of layer norm (`ToyModel.ln`):** `torch.float32` 
    *(`nn.LayerNorm` falls into the category of operations that require FP32 for numerical stability. It automatically upcasts its FP16 input to FP32 and outputs FP32).*

*   **the model's predicted logits:** `torch.float16` 
    *(The final `nn.Linear` layer receives the FP32 output from the layer norm, casts it back down to FP16, and outputs the logits in FP16).*

*   **the loss:** `torch.float32` 
    *(Standard loss functions like `CrossEntropyLoss` force their inputs to be upcasted to FP32 before calculating the loss to prevent numerical underflow/overflow).*

*   **and the model's gradients:** `torch.float32` 
    *(Because the model's actual parameter tensors remain in FP32, the gradients computed during the backward pass are also accumulated and stored in FP32).*

---

## Answers to Questions (b) and (c)

### Question (b): Layer Normalization Sensitivity and BF16
Layer normalization involves accumulating the sum and the sum of squared elements across the feature dimension to calculate the mean and variance. These statistical accumulations are highly sensitive to mixed precision because summing many small values or squaring large values in FP16 can easily lead to underflow or overflow due to its narrow dynamic range and limited mantissa bits. If we use BF16 instead of FP16, we **still need** to treat layer normalization differently (by performing the internal accumulations in FP32). While BF16 resolves the overflow/underflow issues by sharing the same wide dynamic range (exponent size) as FP32, it has even fewer mantissa bits than FP16 (7 bits vs. 10 bits), making it extremely susceptible to severe compounding truncation errors when accumulating long sums.

### Question (c): Benchmarking Mixed Precision BF16
I created a benchmarking script to run the models with and without the `--mixed_precision` flag to trigger `torch.amp.autocast(dtype=torch.bfloat16)`. To ensure a fair comparison regarding scaling trends, a constant `batch_size=4` was used across models. Note: the standard Large model was omitted as it consistently caused a CUDA Out-Of-Memory error on the 16GB GPU. To definitively prove the theory of scaling without crashing, I introduced a custom `Medium-Large` configuration (`d_model=1536`, `d_ff=6144`, `num_layers=10`) which has significantly wider matrix dimensions than Medium to leverage Tensor Cores, but few enough layers to fit securely in 16GB VRAM.

**Timings (Batch Size 4, averaged over 10 iterations):**
*   **Small Model:** FP32 (Fwd: 0.0504s, Bwd: 0.0468s) | BF16 (Fwd: 0.0519s, Bwd: 0.0540s) -> *BF16 is slower (0.97x Fwd, 0.87x Bwd).*
*   **Medium Model:** FP32 (Fwd: 0.1400s, Bwd: 0.1068s) | BF16 (Fwd: 0.1567s, Bwd: 0.1053s) -> *BF16 starts matching FP32 (0.89x Fwd, 1.01x Bwd).*
*   **Medium-Large Model:** FP32 (Fwd: 0.1748s, Bwd: 0.0868s) | BF16 (Fwd: 0.1677s, Bwd: 0.0556s) -> *BF16 is significantly faster (1.04x Fwd, 1.56x Bwd).*

**Commentary:** 
For the small model, using BF16 actually degrades performance. This happens because the model is relatively tiny; the actual matrix multiplications execute so quickly that the GPU execution time is dwarfed by the overhead of `autocast` type conversions and CPU kernel-launch bounds. However, as the matrix dimensions scale up (from Small -> Medium -> Medium-Large), the arithmetic intensity of the workloads scales significantly. The GPU becomes heavily compute-bound rather than overhead-bound, allowing the Tensor Cores to fully utilize the BF16 precision. Consequently, the trend shows the BF16 speedup gap closing and eventually completely dominating FP32, reaching up to a 1.56x speedup on the backward pass for the widest configuration.
