import sqlite3
import sys
from collections import defaultdict

def analyze_profile(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get NVTX events
    cursor.execute("SELECT text, start, end FROM NVTX_EVENTS")
    all_nvtx = cursor.fetchall()
    
    # Find benchmarking boundaries
    # We want to ignore the "warmup" range.
    # Everything after "warmup" is the benchmarking phase.
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
    
    # Filter kernels strictly in the benchmarking phase
    benchmark_kernels = [k for k in kernels if k[0] >= warmup_end]
    
    # To get forward passes ONLY during the benchmarking phase
    forward_ranges = [e for e in all_nvtx if e[0] == 'forward pass' and e[1] >= warmup_end]
    
    forward_kernels = []
    for k_start, k_end, k_name in benchmark_kernels:
        for r_text, r_start, r_end in forward_ranges:
            if k_start >= r_start and k_end <= r_end:
                forward_kernels.append((k_start, k_end, k_name))
                break
                
    def print_stats(name, k_list):
        if not k_list:
            print(f"--- {name} ---")
            print("No kernels found.")
            return

        total_time = 0
        gemm_time = 0
        kernel_times = defaultdict(int)
        
        for k_start, k_end, k_name in k_list:
            duration = k_end - k_start
            total_time += duration
            kernel_times[k_name] += duration
            
            k_lower = k_name.lower()
            if 'gemm' in k_lower or 'matmul' in k_lower or 'dot' in k_lower:
                gemm_time += duration
                
        print(f"--- {name} ---")
        print(f"Total kernel time: {total_time / 1e6:.2f} ms")
        print(f"GEMM kernel time: {gemm_time / 1e6:.2f} ms")
        print(f"Fraction of time in GEMM: {gemm_time / total_time * 100:.2f}%")
        print("Top 5 Kernels by time:")
        
        sorted_kernels = sorted(kernel_times.items(), key=lambda x: x[1], reverse=True)
        for k_name, k_duration in sorted_kernels[:5]:
            print(f"  {k_name[:50]}...: {k_duration / 1e6:.2f} ms ({k_duration / total_time * 100:.2f}%)")
        print()
            
    print_stats("Forward Pass Only (Inference)", forward_kernels)
    print_stats("Complete Training Step", benchmark_kernels)

if __name__ == "__main__":
    analyze_profile(sys.argv[1])
