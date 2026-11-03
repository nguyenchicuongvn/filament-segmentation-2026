import sys, time, torch
sys.path.insert(0, "src")
from filseg.dl import build_model, device
dev = device()
st = torch.load("outputs/colab/unet1024_effb4/unet1024_effb4.pt", map_location="cpu", weights_only=False)
print(st["meta"])
m = build_model(st["meta"]["encoder"], pretrained=False).to(dev).eval()
m.load_state_dict(st["model"])
for res in (1024, 1536, 2048):
    x = torch.rand(1, 1, res, res, device=dev)
    with torch.no_grad():
        for dt in (torch.float32, torch.float16):
            with torch.autocast("mps", dtype=dt, enabled=dt == torch.float16):
                m(x); torch.mps.synchronize()
                t = time.time()
                for _ in range(3): m(x)
                torch.mps.synchronize()
            print(res, dt, f"{(time.time()-t)/3:.2f} s/forward")
