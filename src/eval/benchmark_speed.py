import time, numpy as np, pandas as pd, torch

@torch.no_grad()
def benchmark(name, model, sizes=((64, 64), (256, 256)), device="cuda", warmup=10, runs=50):
    """Latency (ms), peak GPU memory (MB), parameters and checkpoint size for a given LR input size."""
    model = model.to(device).eval()
    cuda = device == "cuda"
    n_params = sum(p.numel() for p in model.parameters())
    rows = []
    for h, w in sizes:
        x = torch.rand(1, 3, h, w, device=device)
        if cuda:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
        for _ in range(warmup): model(x)
        times = []
        for _ in range(runs):
            if cuda: torch.cuda.synchronize()
            t0 = time.perf_counter(); model(x)
            if cuda: torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
        rows.append({
            "method": name, "device": device, "lr_size": f"{h}x{w}",
            "ms_median": round(float(np.median(times)), 3),
            "ms_std": round(float(np.std(times)), 3),
            "peak_mem_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1) if cuda else float("nan"),
            "params_m": round(n_params / 1e6, 3),
        })
    return pd.DataFrame(rows)