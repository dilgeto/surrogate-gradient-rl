"""Tiempo de forward (rollout) y forward+backward (actualización) del actor según lote/dispositivo."""
import argparse, time, warnings
warnings.simplefilter("ignore")
import torch
from sgrl.snn import SNNActor

ap = argparse.ArgumentParser()
ap.add_argument("--neuron", default="lif")
ap.add_argument("--obs", type=int, default=6)
ap.add_argument("--hidden", type=int, nargs="+", default=[64, 64])
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--batches", type=int, nargs="+", default=[16, 128, 512, 2048, 8192])
a = ap.parse_args()
torch.set_num_threads(a.threads)
for dev in ("cpu", "cuda"):
    if dev == "cuda" and not torch.cuda.is_available():
        continue
    net = SNNActor(a.obs, 2, a.hidden, a.neuron, 16).to(dev)
    for B in a.batches:
        x = torch.randn(B, a.obs, device=dev)
        def sync():
            if dev == "cuda": torch.cuda.synchronize()
        for _ in range(3):
            with torch.no_grad(): net(x)
            net(x)[0].sum().backward()
        sync(); n = 10; t = time.time()
        for _ in range(n):
            with torch.no_grad(): net(x)
        sync(); f = (time.time() - t) / n
        t = time.time()
        for _ in range(n): net(x)[0].sum().backward()
        sync(); fb = (time.time() - t) / n
        print(f"{a.neuron:10s} {dev:4s} thr={a.threads:2d} B={B:5d}  fwd {f*1e3:7.1f} ms ({B/f:9.0f} muestras/s)   fwd+bwd {fb*1e3:7.1f} ms ({B/fb:9.0f} muestras/s)", flush=True)
