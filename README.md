# Constant Auxiliary Losses Considered Harmful — Code

Anonymous code release for the NeurIPS 2026 submission
"Constant Auxiliary Losses Considered Harmful: A Cautionary Tale in
Belief-Learning Multi-Agent Reinforcement Learning".

This repository contains the `marl_research` Python package, which
implements the VABL test bed and the baseline MARL algorithms
(AERIAL, MAPPO, TarMAC, CommNet, QMIX) used in the paper. The package
is built on JAX / Flax and uses JaxMARL environment wrappers for
Overcooked, SMAX, and MPE.

## Installation

```bash
conda create -n marl python=3.10
conda activate marl
pip install -e marl_research
```

JAX with CUDA is optional; the code falls back to CPU if CUDA is not
available.

## Package layout

```
marl_research/
  algorithms/
    jax/         # JAX implementations of VABL, AERIAL, MAPPO,
                 # TarMAC, CommNet, QMIX, plus training entry points
    base.py      # shared interfaces
  environments/  # JaxMARL environment wrappers
  runners/       # training orchestration
  utils/         # replay buffer, logging, misc
  configs/       # Hydra configs for agents / environments
```

## Training

The unified training entry point exposes each algorithm via `--algo`:

```bash
python -m marl_research.algorithms.jax.train_unified \
    --algo vabl_v2 \
    --layout asymmetric_advantages \
    --episodes 25000 \
    --horizon 400 \
    --n-envs 64 \
    --seed 0 \
    --save results/vabl_aa_seed0.json
```

Key VABL ablation flags (from `train_vabl_vec.py`):
- `--no-attention`: replace MHA with mean pooling
- `--no-aux-loss`: disable auxiliary prediction entirely
- `--aux-lambda <float>`: constant auxiliary weight
- `--aux-anneal-fraction <float>`: linear λ-annealing over first
  fraction of training
- `--stop-gradient-belief`: stop aux gradients from flowing into the
  belief encoder
- `--separate-aux-encoder`: give the aux head its own parallel encoder

Equivalent flags exist on `--algo aerial` for the AERIAL+aux fix-path
experiments.

## License

Code is released under an open-source license; details omitted for
anonymous review.
