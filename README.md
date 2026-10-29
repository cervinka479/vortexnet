# VortexNet

Lightweight PyTorch framework for training, sweeping and evaluating small neural networks on
tabular CFD data, built for research on **flow-structure identification from the velocity-gradient
tensor** (shear / residual strain / residual vorticity decomposition in compressible flow).

VortexNet is a from-scratch rewrite of [MathCAS](https://github.com/cervinka479/MathCAS), keeping its
good parts and fixing what made research slow: data re-read every epoch, logic trapped in notebooks,
Windows-only tooling, no tests.

> **Status:** early development. The repository is being built module by module; nothing is
> runnable yet.

## Goals

- **Lightweight and readable** – plain PyTorch, a thin training loop with callbacks, no heavy
  framework on top.
- **Research-practical** – YAML configs with template + override, a CLI for `train`, `evaluate`,
  `sweep`; multi-seed runs and Optuna studies launched from scripts, not notebooks.
- **Fast on tabular data** – dataset cached once in binary form, kept on the GPU, batched by
  indexing instead of `DataLoader` workers.
- **Extensible** – regression and classification tasks, custom (weighted / multitask) losses and
  derived physical features as registered plug-ins, without touching the core.
- **Cross-platform** – Linux and Windows, everything in Python (`pathlib`, multiprocessing-safe).

## Planned layout

```
src/vortexnet/    package: config, data, models, losses, training, evaluation, tracking, cli
configs/          YAML templates and experiment overrides
tests/            pytest
```

## Development

Python 3.12 managed with [uv](https://docs.astral.sh/uv/) on Linux and Windows:

```bash
uv sync                 # creates .venv from uv.lock, downloads Python 3.12 if needed
uv run pytest           # tests
uv run ruff check .     # lint
uv run ruff format .    # format
```

The default PyPI `torch` wheel is CUDA-enabled on Linux; on Windows or CPU-only machines
see the [PyTorch install selector](https://pytorch.org/get-started/locally/) for the matching index.

## License

MIT – see [LICENSE](LICENSE).
