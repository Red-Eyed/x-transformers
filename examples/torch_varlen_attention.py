"""
Demonstration of torch var-len attention (flash_pack_seq).

Compares two approaches on identical weights:
  - padded: sequences zero-padded to the same length, boolean mask applied
  - packed: sequences concatenated end-to-end, cu_seqlens passed via varlen_attn

Both must produce the same output for unmasked token positions.
A benchmark section measures latency and throughput for both approaches
across scenarios with different amounts of padding waste.

Requirements:
  - CUDA device with compute capability >= 8 (Ampere+)
  - PyTorch >= 2.6
"""

import sys
import torch
from packaging import version

# ── guard ────────────────────────────────────────────────────────────────────

def check_requirements():
    if not torch.cuda.is_available():
        sys.exit("CUDA not available.")
    cc = torch.cuda.get_device_capability()
    if cc[0] < 8:
        sys.exit(f"GPU compute capability {cc[0]}.{cc[1]} < 8.0 – flex_attention not supported.")
    if version.parse(torch.__version__) < version.parse("2.6.0"):
        sys.exit(f"PyTorch {torch.__version__} < 2.6.0 – varlen_attn not available.")

# ── data helpers ─────────────────────────────────────────────────────────────

def make_sequences(lengths: list[int], dim: int, device, dtype) -> list[torch.Tensor]:
    """One random (L_i, D) tensor per length."""
    return [torch.randn(L, dim, device=device, dtype=dtype) for L in lengths]

def pack(sequences: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Concatenate sequences into (1, total_tokens, D) and build cu_seqlens (int32).
    cu_seqlens = [0, L0, L0+L1, ..., total] — same convention as flash-attn.
    """
    packed = torch.cat(sequences, dim=0).unsqueeze(0)
    lengths = torch.tensor([s.shape[0] for s in sequences], dtype=torch.int32, device=sequences[0].device)
    cu_seqlens = torch.cat([lengths.new_zeros(1), lengths.cumsum(0)])
    return packed, cu_seqlens

def pad(sequences: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pad sequences into (B, L_max, D) and build boolean mask (B, L_max).
    """
    padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    lengths = torch.tensor([s.shape[0] for s in sequences], device=sequences[0].device)
    mask = torch.arange(padded.shape[1], device=padded.device)[None] < lengths[:, None]
    return padded, mask

def unpad(output: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Flatten (B, L_max, D) -> (total_tokens, D) by removing padding positions."""
    return output[mask]

# ── model helpers ─────────────────────────────────────────────────────────────

def build_model(dim: int, depth: int, heads: int, pack_seq: bool, device):
    from x_transformers.x_transformers import AttentionLayers
    return (
        AttentionLayers(
            dim=dim, depth=depth, heads=heads,
            causal=True,
            rotary_pos_emb=True,   # required by flash_pack_seq
            attn_flash=True,
            attn_flash_pack_seq=pack_seq,
        )
        .to(device)
        .half()
        .eval()
    )

# ── forward helpers ───────────────────────────────────────────────────────────

def forward_padded(model, sequences: list[torch.Tensor]) -> torch.Tensor:
    """Run padded forward; return flat (total_tokens, D) output."""
    padded_x, mask = pad(sequences)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        out = model(padded_x, mask=mask)
    return unpad(out, mask)

def forward_packed(model, sequences: list[torch.Tensor]) -> torch.Tensor:
    """Run packed var-len forward; return flat (total_tokens, D) output."""
    packed_x, cu_seqlens = pack(sequences)
    pack_kwargs = dict(cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        out = model(packed_x, flash_pack_seq_kwargs=pack_kwargs)
    return out.squeeze(0)

# ── benchmark ─────────────────────────────────────────────────────────────────

def measure_ms(fn, warmup: int = 5, iters: int = 20) -> float:
    """Return median latency in milliseconds over `iters` runs."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2]   # median

def benchmark(model_pad, model_pack, sequences: list[torch.Tensor], label: str):
    total_tokens = sum(s.shape[0] for s in sequences)
    padded_x, mask   = pad(sequences)
    packed_x, cu_seq = pack(sequences)
    pack_kwargs = dict(cu_seqlens_q=cu_seq, cu_seqlens_k=cu_seq)
    padding_pct = 100.0 * (padded_x.shape[1] * len(sequences) - total_tokens) / (padded_x.shape[1] * len(sequences))

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        ms_pad  = measure_ms(lambda: model_pad( padded_x, mask=mask))
        ms_pack = measure_ms(lambda: model_pack(packed_x, flash_pack_seq_kwargs=pack_kwargs))

    tps_pad  = total_tokens / (ms_pad  / 1e3)
    tps_pack = total_tokens / (ms_pack / 1e3)

    print(f"\n{label}  (seqs={len(sequences)}, max_len={padded_x.shape[1]}, total={total_tokens} tok, padding={padding_pct:.1f}%)")
    print(f"  {'method':<8}  {'latency':>10}  {'throughput':>16}")
    print(f"  {'padded':<8}  {ms_pad:>8.2f}ms  {tps_pad:>13,.0f} tok/s")
    print(f"  {'packed':<8}  {ms_pack:>8.2f}ms  {tps_pack:>13,.0f} tok/s")
    print(f"  speedup: {ms_pad / ms_pack:.2f}x")

# ── comparison ────────────────────────────────────────────────────────────────

def compare(a: torch.Tensor, b: torch.Tensor, atol: float = 5e-3) -> bool:
    max_diff  = (a - b).abs().max().item()
    mean_diff = (a - b).abs().mean().item()
    print(f"  max  |diff| = {max_diff:.6f}")
    print(f"  mean |diff| = {mean_diff:.6f}")
    ok = max_diff < atol
    print(f"  {'PASS' if ok else 'FAIL'}  (atol={atol})")
    return ok

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    check_requirements()

    torch.manual_seed(0)
    device = torch.device("cuda")

    DIM   = 256
    HEADS = 4
    DEPTH = 4

    # ── correctness check ─────────────────────────────────────────────────────
    LENGTHS = [97, 213, 64, 150]
    print(f"=== correctness  (sequences: {LENGTHS}) ===")
    sequences = make_sequences(LENGTHS, DIM, device, dtype=torch.float16)
    model_pad  = build_model(DIM, DEPTH, HEADS, pack_seq=False, device=device)
    model_pack = build_model(DIM, DEPTH, HEADS, pack_seq=True,  device=device)
    model_pack.load_state_dict(model_pad.state_dict())
    out_pad  = forward_padded(model_pad,  sequences)
    out_pack = forward_packed(model_pack, sequences)
    print(f"  padded output : {out_pad.shape}")
    print(f"  packed output : {out_pack.shape}")
    compare(out_pad, out_pack)

    # ── performance benchmarks ────────────────────────────────────────────────
    # Build models once; reuse across scenarios.
    model_pad  = build_model(DIM, DEPTH, HEADS, pack_seq=False, device=device)
    model_pack = build_model(DIM, DEPTH, HEADS, pack_seq=True,  device=device)
    model_pack.load_state_dict(model_pad.state_dict())

    print(f"\n=== benchmark  dim={DIM}  heads={HEADS}  depth={DEPTH} ===")

    # Low padding waste: all sequences roughly the same length
    uniform = make_sequences([200] * 16, DIM, device, dtype=torch.float16)
    benchmark(model_pad, model_pack, uniform, "uniform lengths (~0 % padding waste)")

    # High padding waste: one long sequence dominates, rest are short
    skewed = make_sequences([512] + [32] * 15, DIM, device, dtype=torch.float16)
    benchmark(model_pad, model_pack, skewed, "skewed lengths (high padding waste)")

    # Many short sequences
    many_short = make_sequences([64] * 32, DIM, device, dtype=torch.float16)
    benchmark(model_pad, model_pack, many_short, "many short sequences")

if __name__ == "__main__":
    main()
