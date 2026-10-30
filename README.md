# DeepTOP for Speculative Decoding

DeepTOP learns a threshold policy that selects normal decoding (`k=0`) or
speculative decoding (`k=5`) from serving-system state.

## Setup

Python 3.9 or newer is recommended.

```bash
python -m pip install -r requirements.txt
```

## Train

Run a short smoke test first:

```bash
python main_DeepTOP_spec.py --total_steps 1000
```

Run the default training job:

```bash
python main_DeepTOP_spec.py --total_steps 200000
```

The trained actor is saved as `deeptop_spec_actor.pkl`.

## Evaluate

```bash
python eval_deeptop_spec.py
```

Evaluation writes `deeptop_eval.png`.

## v6: multi-k speculative decoding

The v6 source, simulator settings, workload data, Slurm scripts, and experiment
results are in [`v6/`](v6/). This version learns a threshold policy over
`k = 1..7` with position-dependent draft acceptance.

```bash
cd v6
```

See [the v6 README](v6/README.md) for training, evaluation, and serving commands.
Model checkpoints (including `deeptop_v6_best.pkl`) are excluded from this
repository; train one or supply your own before running checkpoint-based commands.
