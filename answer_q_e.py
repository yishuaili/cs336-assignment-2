import sqlite3
import sys

def analyze(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT text, start, end FROM NVTX_EVENTS")
    all_nvtx = cursor.fetchall()
    
    warmup_end = 0
    for text, start, end in all_nvtx:
        if text == 'warmup':
            if end > warmup_end:
                warmup_end = end
                
    cursor.execute("""
        SELECT k.start, k.end, s.value AS kernel_name 
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON k.shortName = s.id
    """)
    kernels = cursor.fetchall()
    
    benchmark_kernels = [k for k in kernels if k[0] >= warmup_end]
    
    forward_ranges = [e for e in all_nvtx if e[0] == 'forward pass' and e[1] >= warmup_end]
    
    attn_scores_time = 0
    softmax_time = 0
    final_matmul_time = 0
    
    attn_scores_ranges = [e for e in all_nvtx if e[0] == 'computing attention scores' and e[1] >= warmup_end]
    softmax_ranges = [e for e in all_nvtx if e[0] == 'computing softmax' and e[1] >= warmup_end]
    final_matmul_ranges = [e for e in all_nvtx if e[0] == 'final matmul' and e[1] >= warmup_end]

    for k_start, k_end, k_name in benchmark_kernels:
        # Check if inside forward pass
        in_forward = any(r_start <= k_start and r_end >= k_end for _, r_start, r_end in forward_ranges)
        if not in_forward: continue
        
        duration = k_end - k_start
        if any(r_start <= k_start and r_end >= k_end for _, r_start, r_end in attn_scores_ranges):
            attn_scores_time += duration
        elif any(r_start <= k_start and r_end >= k_end for _, r_start, r_end in softmax_ranges):
            softmax_time += duration
        elif any(r_start <= k_start and r_end >= k_end for _, r_start, r_end in final_matmul_ranges):
            final_matmul_time += duration

    print(f"Attention Scores Matmul Time: {attn_scores_time / 1e6:.2f} ms")
    print(f"Softmax Time: {softmax_time / 1e6:.2f} ms")
    print(f"Final Matmul Time: {final_matmul_time / 1e6:.2f} ms")
    print(f"Total Self-Attention Matmul Time: {(attn_scores_time + final_matmul_time) / 1e6:.2f} ms")

if __name__ == "__main__":
    analyze(sys.argv[1])
