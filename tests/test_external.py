import torch
import pytest
from x_transformers.x_transformers import Attend, Attention, AttentionLayers

param = pytest.mark.parametrize

def reset_exp_det():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def partition(x, num_parts):
    """Split x into num_parts variable-length segments; return splits and cumulative seq lengths."""
    split_points = sorted(torch.randint(1, x.shape[0], (num_parts - 1,)).tolist())
    splits = torch.tensor_split(x, split_points, dim=0)
    lengths = torch.tensor([s.shape[0] for s in splits], dtype=torch.int32, device=x.device)
    cu_seqlens = torch.cat([lengths.new_zeros(1), lengths.cumsum(0)])
    return splits, cu_seqlens

def pad_splits(splits, pad_val=99.0):
    """Pad variable-length splits into a batch; return padded tensor and boolean mask."""
    padded = torch.nn.utils.rnn.pad_sequence(splits, batch_first=True, padding_value=pad_val)
    mask = (padded != pad_val).any(dim=-1)
    return padded, mask

def unpad(output, padded_src, pad_val=99.0):
    """Remove padding tokens from a padded batch output."""
    return torch.cat([o[~(m == pad_val).all(-1)] for o, m in zip(output, padded_src)], dim=0)

def causal_cross_attn_mask(query_mask, seq_len):
    """Build a causal attention mask for packed sequences from a per-token query mask."""
    causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=query_mask.device))
    return (query_mask.unsqueeze(1) & causal.unsqueeze(0)).unsqueeze(1)

def cross_attn_mask(query_mask, key_mask):
    """Build a cross-attention mask from independent query and key masks."""
    return (query_mask.unsqueeze(2) & key_mask.unsqueeze(1)).unsqueeze(1)

@pytest.mark.skipif(
    not torch.cuda.is_available() or \
    torch.cuda.get_device_capability()[0] < 8 or \
    tuple(int(x) for x in torch.__version__.split('.')[:2]) < (2, 5),
    reason="CUDA compute capability must be >= 8 and PyTorch >= 2.5 is required for flex_attention"
)
@param('exp', (
    dict(causal=True, same_partition=True, pos_enc='rotary_pos_emb'),
    dict(causal=False, same_partition=True, pos_enc='rotary_pos_emb'),
    dict(causal=False, same_partition=False, pos_enc='rotary_pos_emb'),
    dict(causal=True, same_partition=True, pos_enc='rotary_xpos'))
)
def test_flash_pack_seq(exp):
    seq_len = 1024
    dim = 256
    n_part = 4
    n_layers = 4
    pad_val = 99.0
    causal = exp['causal']
    same_partition = exp.get('same_partition', False)
    atl_kwargs = {exp['pos_enc']: True}
    mem_len = 128 if not same_partition else seq_len
    x = torch.randn((seq_len, dim)).cuda().half()
    mem = torch.randn((mem_len, dim)).cuda().half()

    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.float16):
            splits, cu_seqlens = partition(x, n_part)
            padded_batch, mask = pad_splits(splits, pad_val)

            if same_partition and causal:
                splits_mem = torch.tensor_split(mem, cu_seqlens[1:-1].tolist(), dim=0)
                cu_seqlens_context = cu_seqlens
            else:
                splits_mem, cu_seqlens_context = partition(mem, n_part)

            padded_mem, context_mask = pad_splits(splits_mem, pad_val)

            if causal:
                attn_mask = causal_cross_attn_mask(mask, padded_batch.shape[1])
            else:
                attn_mask = cross_attn_mask(mask, context_mask)

            # Standard padding
            reset_exp_det()
            atd = Attend(flash=False, flash_pack_seq=False, causal=causal).cuda().eval()
            o_atd = unpad(atd(q=padded_batch[:,None], k=padded_mem[:,None], v=padded_mem[:,None], mask=attn_mask)[0][:,0], padded_batch, pad_val)

            att = Attention(dim=dim, flash=False, causal=causal).cuda().eval()
            o_att = unpad(att(x=padded_batch, context=padded_mem, attn_mask=attn_mask), padded_batch, pad_val)

            atl = AttentionLayers(dim=dim, depth=n_layers, cross_attend=True, causal=causal, attn_flash=True, attn_flash_pack_seq=False, **atl_kwargs).cuda().eval()
            o_atl = unpad(atl(x=padded_batch, context=padded_mem, context_mask=context_mask, mask=mask), padded_batch, pad_val)

            # Block masking (packed sequences via flex_attention)
            reset_exp_det()
            pack_kwargs = dict(cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens)
            pack_kwargs_ctx = dict(cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens_context)

            atd_block = Attend(flash=True, flash_pack_seq=True, causal=causal).cuda().eval()
            o_atd_block = atd_block(x[None,None], mem[None,None], mem[None,None], flash_pack_seq_kwargs=pack_kwargs_ctx)[0][0,0]

            att_block = Attention(dim=dim, flash=True, flash_pack_seq=True, causal=causal).cuda().eval()
            o_att_block = att_block(x=x.unsqueeze(0), context=mem.unsqueeze(0), flash_pack_seq_kwargs=pack_kwargs_ctx)[0]

            atl_block = AttentionLayers(dim=dim, depth=n_layers, cross_attend=True, causal=causal, attn_flash=True, attn_flash_pack_seq=True, **atl_kwargs).cuda().eval()
            o_atl_block = atl_block(x=x.unsqueeze(0), context=mem.unsqueeze(0), flash_pack_seq_kwargs=pack_kwargs, flash_pack_seq_context_kwargs=pack_kwargs_ctx)[0]

            torch.testing.assert_close(o_atd, o_atd_block, atol=5e-3, rtol=5e-3)
            torch.testing.assert_close(o_att, o_att_block, atol=5e-3, rtol=5e-3)
            torch.testing.assert_close(o_atl, o_atl_block, atol=5e-3, rtol=5e-3)
