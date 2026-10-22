"""Pretrain on synthetic tasks; no real evaluation dataset is seen here."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from miniicl import MiniTabICL
from prior import sample_batch


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--context", type=int, default=32)
    p.add_argument("--query", type=int, default=8)
    p.add_argument("--features", type=int, default=10)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--out", default="artifacts/model.pt")
    args = p.parse_args()
    if min(args.steps, args.batch_size, args.context, args.query,
           args.features, args.width, args.threads) < 1:
        p.error("All numeric controls must be positive")
    if args.features < 3 or args.width % 4:
        p.error("features must be >=3 and width divisible by four")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = MiniTabICL(width=args.width)
    optimizer = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=1e-3)
    start = time.perf_counter()
    history = []
    model.train()
    for step in range(1, args.steps + 1):
        x, y, nf = sample_batch(rng, args.batch_size, args.context,
                                args.query, args.features)
        logits = model(x, y[:, :args.context], nf)
        loss = F.binary_cross_entropy_with_logits(
            logits, y[:, args.context:].float())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == args.steps:
            item = {"step": step, "training_loss": round(loss.item(), 6)}
            history.append(item)
            print(item, flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(),
                "model_config": {"width": args.width},
                "train_config": vars(args)}, out)
    manifest = {"config": vars(args), "wall_seconds": round(time.perf_counter() - start, 2),
                "torch": torch.__version__, "numpy": np.__version__,
                "python": platform.python_version(), "history": history}
    (out.parent / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved {out} after {manifest['wall_seconds']}s", flush=True)


if __name__ == "__main__":
    main()
