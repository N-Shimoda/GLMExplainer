# flash_attn_check.py
import math
import time

import torch
from flash_attn import flash_attn_func
from torch.nn import functional as F


def check(B=2, S=256, H=8, D=64, causal=False, dtype=torch.bfloat16):
    assert torch.cuda.is_available()
    dev = "cuda"
    torch.manual_seed(0)

    q = torch.randn(B, S, H, D, device=dev, dtype=dtype).contiguous()
    k = torch.randn(B, S, H, D, device=dev, dtype=dtype).contiguous()
    v = torch.randn(B, S, H, D, device=dev, dtype=dtype).contiguous()

    scale = 1.0 / math.sqrt(D)

    # --- flash-attn ---
    torch.cuda.synchronize()
    t0 = time.time()
    out_fa = flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=scale, causal=causal)
    torch.cuda.synchronize()
    t1 = time.time()

    # --- 参照: PyTorch SDPA (math固定 & 同dtype) ---
    qf = q.reshape(B * H, S, D)
    kf = k.reshape(B * H, S, D)
    vf = v.reshape(B * H, S, D)

    with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
        torch.cuda.synchronize()
        t2 = time.time()
        out_ref = (
            F.scaled_dot_product_attention(qf, kf, vf, attn_mask=None, dropout_p=0.0, is_causal=causal, scale=scale)
            .reshape(B, H, S, D)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        torch.cuda.synchronize()
        t3 = time.time()

    # 誤差評価はfp32に上げて実施（出力のみアップキャスト）
    diff = (out_fa.to(torch.float32) - out_ref.to(torch.float32)).abs()
    print(f"torch: {torch.__version__}, torch.cuda: {torch.version.cuda}")
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"dtype: {dtype}, shape: B={B}, S={S}, H={H}, D={D}, causal={causal}")
    print(f"flash-attn   elapsed: {(t1 - t0) * 1000:.2f} ms")
    print(f"SDPA(math)   elapsed: {(t3 - t2) * 1000:.2f} ms")
    print(f"max|diff|={diff.max().item():.3e}, mean|diff|={diff.mean().item():.3e}")

    tol = 1e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-4
    print("OK ✅" if diff.max().item() < tol else "Mismatch ⚠️")


if __name__ == "__main__":
    # Adaはbf16OK。必要なら dtype=torch.float16 でも試せます
    check(B=2, S=256, H=8, D=64, causal=False, dtype=torch.bfloat16)
