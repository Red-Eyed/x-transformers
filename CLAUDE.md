# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install
pip install x-transformers

# Install from source (dev)
pip install -e .

# Run all tests
pytest tests/

# Run a single test file
pytest tests/test_x_transformers.py

# Run a specific test
pytest tests/test_x_transformers.py::test_name

# Training examples (use python-fire CLI)
python train_enwik8.py
python train_copy.py
```

## Architecture

This is a PyTorch transformer library with a large number of independently-togglable research features. The design philosophy is composition over subclassing — features are enabled via constructor flags rather than separate classes.

### Core files

- [x_transformers/x_transformers.py](x_transformers/x_transformers.py) — ~4100 lines, the main module. Contains all transformer building blocks, wrappers, and the main entry points.
- [x_transformers/attend.py](x_transformers/attend.py) — The `Attend` class, responsible for the actual attention computation (including flash attention). Isolated here so attention backends can be swapped.
- [x_transformers/autoregressive_wrapper.py](x_transformers/autoregressive_wrapper.py) — `AutoregressiveWrapper` for text generation with caching, sampling strategies, and beam search.
- [x_transformers/continuous.py](x_transformers/continuous.py) — `ContinuousTransformerWrapper` for non-discrete (continuous-valued) inputs.
- [x_transformers/__init__.py](x_transformers/__init__.py) — re-exports everything; start here to see the public API surface.

### Class hierarchy in x_transformers.py

```
AttentionLayers          ← core transformer stack (configurable depth/heads/dim)
  ├── Encoder            ← bidirectional (no causal mask)
  ├── Decoder            ← causal (autoregressive)
  └── PrefixDecoder      ← mixed prefix+causal

TransformerWrapper       ← adds token embeddings + positional encoding around AttentionLayers
XTransformer             ← full encoder-decoder (seq2seq)
ViTransformerWrapper     ← vision transformer (patch embeddings)
```

`AttentionLayers` is the central class. It manages the layer sequence, residuals, norm strategies, and layer-level features. `Attention` and `FeedForward` are its leaf components.

### Tensor dimension conventions

Comments throughout the code use single-letter dimension names:
- `b` — batch, `n` — sequence length, `d` — feature dim, `h` — heads, `i`/`j` — source/target sequence

`einops` (rearrange, repeat, pack, unpack) and `einx` are used heavily for tensor manipulation — prefer these over raw `view`/`permute`.

### Common helper pattern

```python
exists(val)      # val is not None
default(val, d)  # val if exists(val) else d
cast_tuple(val)  # ensure val is a tuple
```

These are defined at module level and used throughout. Use them when adding new features.

### Adding a new feature

New research features follow a consistent pattern:
1. Add a boolean/int constructor flag to `Attention`, `FeedForward`, or `AttentionLayers`
2. Guard the feature behind `exists()` / `if flag:` checks inline
3. Add a test case in `tests/test_x_transformers.py` that constructs a model with the flag enabled and does a forward pass

### Tests

Tests are integration-style: construct a model, run a forward (and sometimes backward) pass, assert shapes or losses. There are no unit tests for individual helper functions. `tests/test_external.py` specifically tests flash attention and requires the `flash-attn` package.
