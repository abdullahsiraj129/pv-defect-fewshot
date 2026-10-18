"""
Can a solar-cell defect detector be learnt from defect-free examples alone, and how many does it need?

Data: ELPV (Buerhop-Lutz et al., EU PVSEC 2018; Deitsch et al., Solar Energy 2019), 2,624 electroluminescence
images of single solar cells from 44 modules, each annotated with a defect probability of 0, 1/3, 2/3 or 1.
https://github.com/zae-bayern/elpv-dataset  (CC BY-NC-SA 4.0; not redistributed here)

Methods, all on frozen ImageNet features from a Wide ResNet-50-2 (layers 2 and 3), as in PatchCore
(Roth et al., CVPR 2022):
  - Patch memory (normal only): a memory bank of patch features from N defect-free cells; a cell's score is the
    largest distance from any of its patches to the nearest patch in the bank.
  - Global nearest neighbour (normal only): one pooled feature per cell; score = distance to the nearest of the
    N defect-free cells.
  - Few labels (supervised): logistic regression on the pooled features, trained on the same N defect-free cells
    plus k labelled defective cells.
For speed on a CPU, patch features are reduced to 256 dimensions by a fixed random projection and the memory bank
is a random subset of at most 20,000 patches (PatchCore uses greedy coreset selection instead).

Split (fixed, seed 0, stratified by cell type): 70% of defect-free cells form the pool for N; the other 30% are test
normals. Defective = probability 2/3 or 1; 30% of them form the pool of labels for the supervised method and 70% are
test defects. Cells with probability 1/3 are kept out of training and testing and scored separately.

Run:  python run_experiment.py   (writes results/)
"""
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from elpv_dataset.utils import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2

OUT = Path(__file__).parent / "results"
N_VALUES = [10, 25, 50, 100, 250, "all"]
K_LABELS = [10, 50]
SEEDS = [1, 2, 3]
PROJ_DIM = 256
BANK_MAX = 20_000
torch.set_num_threads(max(1, torch.get_num_threads()))


def features(images):
    """Patch features (n, 196, PROJ_DIM) and pooled features (n, 1536) from Wide ResNet-50-2 layers 2 and 3."""
    model = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1).eval()
    taps = {}
    model.layer2.register_forward_hook(lambda m, i, o: taps.__setitem__("l2", o))
    model.layer3.register_forward_hook(lambda m, i, o: taps.__setitem__("l3", o))
    gen = torch.Generator().manual_seed(0)
    projection = torch.randn(512 + 1024, PROJ_DIM, generator=gen) / PROJ_DIM ** 0.5
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    patches, pooled = [], []
    with torch.no_grad():
        for start in range(0, len(images), 32):
            x = torch.from_numpy(images[start:start + 32]).float().div(255).unsqueeze(1).repeat(1, 3, 1, 1)
            x = F.interpolate(x, size=224, mode="bilinear", align_corners=False)
            model((x - mean) / std)
            l2 = F.avg_pool2d(taps["l2"], 3, stride=1, padding=1)               # (b, 512, 28, 28)
            l3 = F.avg_pool2d(taps["l3"], 3, stride=1, padding=1)               # (b, 1024, 14, 14)
            l2 = F.adaptive_avg_pool2d(l2, 14)                                   # align to 14 x 14
            local = torch.cat([l2, l3], dim=1)                                   # (b, 1536, 14, 14)
            pooled.append(local.mean(dim=(2, 3)))
            patches.append((local.flatten(2).transpose(1, 2) @ projection).half())
            print(f"features {min(start + 32, len(images))}/{len(images)}", flush=True)
    return torch.cat(patches).float(), torch.cat(pooled).numpy()


def patch_scores(bank, queries, chunk=64):
    """Per image: max over its patches of the distance to the nearest bank patch."""
    out = []
    for start in range(0, len(queries), chunk):
        q = queries[start:start + chunk]                                         # (c, 196, d)
        d = torch.cdist(q.reshape(-1, q.shape[-1]), bank).min(dim=1).values     # (c*196,)
        out.append(d.view(len(q), -1).max(dim=1).values)
    return torch.cat(out).numpy()


def split(proba, types, rng):
    normal = np.flatnonzero(proba == 0)
    defect = np.flatnonzero(proba >= 0.6)
    pool, test_normal, label_pool, test_defect = [], [], [], []
    for kind in np.unique(types):
        n = rng.permutation(normal[types[normal] == kind])
        d = rng.permutation(defect[types[defect] == kind])
        cut_n, cut_d = int(round(len(n) * 0.7)), int(round(len(d) * 0.3))
        pool += list(n[:cut_n]); test_normal += list(n[cut_n:])
        label_pool += list(d[:cut_d]); test_defect += list(d[cut_d:])
    return (np.array(pool), np.array(test_normal), np.array(label_pool), np.array(test_defect))


def main():
    t0 = time.time()
    OUT.mkdir(exist_ok=True)
    images, proba, types = load_dataset()
    proba = np.round(proba, 3)
    patches, pooled = features(images)

    pool, test_normal, label_pool, test_defect = split(proba, types, np.random.default_rng(0))
    test = np.concatenate([test_normal, test_defect])
    y = np.r_[np.zeros(len(test_normal)), np.ones(len(test_defect))]
    uncertain = np.flatnonzero(proba == 0.333)

    rows, curves = [], {}
    for n in N_VALUES:
        per_seed = {"patch": [], "global": [], **{f"sup{k}": [] for k in K_LABELS}}
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            train = pool if n == "all" else rng.choice(pool, size=n, replace=False)
            bank = patches[train].reshape(-1, PROJ_DIM)
            if len(bank) > BANK_MAX:
                bank = bank[torch.from_numpy(rng.choice(len(bank), BANK_MAX, replace=False))]
            per_seed["patch"].append(roc_auc_score(y, patch_scores(bank, patches[test])))

            scaler = StandardScaler().fit(pooled[train])
            g_train, g_test = scaler.transform(pooled[train]), scaler.transform(pooled[test])
            dist = torch.cdist(torch.from_numpy(g_test).float(), torch.from_numpy(g_train).float()).min(dim=1).values.numpy()
            per_seed["global"].append(roc_auc_score(y, dist))

            for k in K_LABELS:
                labelled = rng.choice(label_pool, size=k, replace=False)
                x = np.concatenate([pooled[train], pooled[labelled]])
                t = np.r_[np.zeros(len(train)), np.ones(k)]
                sc = StandardScaler().fit(x)
                clf = LogisticRegression(max_iter=2000, C=0.1, class_weight="balanced").fit(sc.transform(x), t)
                per_seed[f"sup{k}"].append(roc_auc_score(y, clf.decision_function(sc.transform(pooled[test]))))
        label = str(len(pool)) if n == "all" else str(n)
        curves[label] = {m: (float(np.mean(v)), float(np.std(v))) for m, v in per_seed.items()}
        rows.append(label)
        print(label, {m: round(np.mean(v), 3) for m, v in per_seed.items()}, flush=True)

    # Expert uncertainty: do cells annotated 1/3 score between defect-free and defective? (patch memory, all normals)
    bank = patches[pool].reshape(-1, PROJ_DIM)
    bank = bank[torch.from_numpy(np.random.default_rng(1).choice(len(bank), BANK_MAX, replace=False))]
    scored = np.concatenate([test_normal, uncertain, test_defect])
    s = patch_scores(bank, patches[scored])
    ranks = s.argsort().argsort() / (len(s) - 1)
    groups = {"0 (defect-free)": test_normal, "1/3 (uncertain)": uncertain, "2/3 and 1 (defective)": test_defect}
    offsets = np.cumsum([0] + [len(g) for g in groups.values()])
    percentile = {name: round(float(np.median(ranks[offsets[i]:offsets[i + 1]])) * 100, 1) for i, name in enumerate(groups)}
    by_type = {}
    s_test = s[np.r_[0:len(test_normal), offsets[2]:offsets[3]]]
    for kind in ("mono", "poly"):
        mask = np.r_[types[test_normal] == kind, types[test_defect] == kind]
        by_type[kind] = round(float(roc_auc_score(y[mask], s_test[mask])), 3)

    summary = {
        "dataset": "ELPV", "images": int(len(images)),
        "pool_normal": int(len(pool)), "test_normal": int(len(test_normal)),
        "label_pool_defective": int(len(label_pool)), "test_defective": int(len(test_defect)),
        "uncertain_cells": int(len(uncertain)), "seeds": SEEDS,
        "auroc_by_n": {n: {m: [round(a, 3), round(b, 3)] for m, (a, b) in v.items()} for n, v in curves.items()},
        "patch_memory_auroc_by_cell_type_all_normals": by_type,
        "median_score_percentile_by_annotation": percentile,
        "runtime_seconds": round(time.time() - t0, 1),
    }
    (OUT / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    names = {"patch": "Patch memory (defect-free only)", "global": "Global nearest neighbour (defect-free only)",
             "sup10": "Logistic regression + 10 labelled defects", "sup50": "Logistic regression + 50 labelled defects"}
    table = ["| Defect-free training cells | " + " | ".join(names.values()) + " |", "|---" * 5 + "|"]
    for n in rows:
        table.append(f"| {n} | " + " | ".join(f"{curves[n][m][0]:.3f} ± {curves[n][m][1]:.3f}" for m in names) + " |")
    (OUT / "results.md").write_text("\n".join(table) + "\n", encoding="utf-8")

    fig, ax = plt.subplots(figsize=(7.4, 4.1))
    xs = np.arange(len(rows))
    styles = {"patch": ("#0B5A3A", "-"), "global": ("#5E9C7F", "-"), "sup10": ("#C9A227", "--"), "sup50": ("#1F6FB2", "--")}
    for m, label in names.items():
        mean = np.array([curves[n][m][0] for n in rows])
        sd = np.array([curves[n][m][1] for n in rows])
        color, ls = styles[m]
        ax.plot(xs, mean, marker="o", ms=4, lw=1.9, color=color, ls=ls, label=label)
        ax.fill_between(xs, mean - sd, mean + sd, color=color, alpha=0.12, lw=0)
    ax.set_xticks(xs, rows)
    ax.set_xlabel("Defect-free cells used for training")
    ax.set_ylabel("ROC AUC on held-out cells")
    ax.set_ylim(0.5, 1.0)
    ax.set_title("ELPV: detecting defective solar cells from defect-free examples", fontsize=10)
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "auroc_vs_n.png", dpi=160)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
