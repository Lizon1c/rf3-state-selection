#!/usr/bin/env python3
"""E2-bin: per-bin rho + adjacent-bin-pair AUC for the four main conditions
(sp_no / sp_msa / x_no_msa / x_msa_faesm) — which mutation TYPES gain (or lose)
from MSA. Follows fusion_v3.per_bin_analysis conventions (label quintiles;
within-bin Spearman; adjacent bin-pair AUC) with a sklearn-free rank AUC.
Adds a second stratification: per-position conservation tertiles from
rbd_wh1_uniref30.a3m (196-row MSA) — tests "MSA helps at conserved positions".

Saves per-run val predictions (npz) + aggregated bin tables.
Run: CUDA_VISIBLE_DEVICES=1 python -B e2_bin_analysis.py [cond ...]
Output: /mnt/k/output_heads/rbd/e2_bin_analysis/{results.json,preds_*.npz}
"""
import json, os, time
import numpy as np
import torch, torch.nn as nn
from pathlib import Path
from scipy import stats

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
torch.set_float32_matmul_precision('high')
import sys
sys.path.insert(0, '/mnt/j/conda_envs/foundry/DMS_Project')
from e2_four_condition import (XAttn, load_z_no, load_z_msa, load_faesm,
                               std_per_residue, cross_split, DEVICE, BS, EPOCHS,
                               LR, WD, L)
from fusion_v3 import CrossAttnOrthoConcatFusion, SinglePredictorV2

RBD = Path('/mnt/k/output_heads/rbd')
OUT = RBD / 'e2_bin_analysis'
OUT.mkdir(exist_ok=True)
A3M_DEEP = Path('/mnt/j/conda_envs/foundry/DMS_Project/inputs/GOLD_MSA/rbd_wh1_uniref30.a3m')
t0 = time.time()


def log(m):
    print(f'[{time.time()-t0:7.1f}s] {m}', flush=True)


def auc_rank(y_bin, scores):
    r = stats.rankdata(scores)
    n1 = int(y_bin.sum())
    n0 = len(y_bin) - n1
    if n1 == 0 or n0 == 0:
        return float('nan')
    return float((r[y_bin == 1].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))


def per_bin(preds, true):
    """fusion_v3.per_bin_analysis convention, sklearn-free."""
    bounds = np.percentile(true, np.linspace(0, 100, 6))
    within, aucs = [], []
    for i in range(5):
        m = (true >= bounds[i]) & (true < bounds[i + 1]) if i < 4 else (true >= bounds[i])
        if m.sum() < 5:
            within.append(np.nan)
        else:
            within.append(float(stats.spearmanr(preds[m], true[m])[0]))
    for i in range(4):
        lo = (true >= bounds[i]) & (true < bounds[i + 1])
        hi = (true >= bounds[i + 1]) & (true < bounds[i + 2]) if i < 3 else (true >= bounds[i + 1])
        if lo.sum() < 5 or hi.sum() < 5:
            aucs.append(np.nan)
        else:
            yb = np.concatenate([np.zeros(lo.sum()), np.ones(hi.sum())])
            aucs.append(auc_rank(yb, np.concatenate([preds[lo], preds[hi]])))
    return within, aucs


def load_conservation():
    seqs = [l.strip() for l in open(A3M_DEEP) if l.strip() and not l.startswith('>')]
    arr = np.array([list(s) for s in seqs])
    cons = np.zeros(L)
    for j in range(L):
        vals, cnts = np.unique(arr[:, j], return_counts=True)
        cons[j] = cnts.max() / cnts.sum()
    tert = np.percentile(cons, [100 / 3, 200 / 3])
    pos_group = np.digitize(cons, tert)  # 0=variable, 1=mid, 2=conserved
    return cons, pos_group


# ── data ──
meta = [json.loads(l) for l in open(RBD / 'metadata.jsonl') if l.strip()]
N = len(meta)
y = np.array([m['bind_avg'] for m in meta], np.float32)
pos = np.array([m['site_rbd'] - 1 for m in meta])
cons, pos_group = load_conservation()
log(f'conservation tertiles: {[(pos_group == g).sum() for g in range(3)]}')


def train_pred(model, inputs, tr, va, init):
    torch.manual_seed(init)
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    tr_t, va_t = torch.from_numpy(tr).to(DEVICE), torch.from_numpy(va).to(DEVICE)
    yt = torch.from_numpy(y[tr]).to(DEVICE)
    best, best_pred = -2, None
    ins_tr = [x[tr_t] for x in inputs]
    ins_va = [x[va_t] for x in inputs]

    def fwd(*xs):
        out = model(*xs) if len(xs) > 1 else model(xs[0])
        return out[0] if isinstance(out, tuple) else out

    for ep in range(EPOCHS):
        model.train()
        idx = torch.randperm(len(tr), device=DEVICE)
        for bi in range(0, len(tr), BS):
            b = idx[bi:bi + BS]
            loss = nn.MSELoss()(fwd(*[x[b] for x in ins_tr]), yt[b])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            pv = torch.cat([fwd(*[x[i:i + BS] for x in ins_va])
                            for i in range(0, len(va), BS)]).cpu().numpy()
        r, _ = stats.spearmanr(pv, y[va])
        if not np.isnan(r) and r > best:
            best, best_pred = r, pv
    del model
    torch.cuda.empty_cache()
    return best, best_pred


def main():
    conds = sys.argv[1:] or ['sp_no', 'sp_msa', 'x_no_msa', 'x_msa_faesm']
    log(f'conds: {conds}')
    Z_no = std_per_residue(load_z_no()).to(DEVICE)
    log('z_no ready')
    Z_msa = std_per_residue(load_z_msa()).to(DEVICE)
    log('z_msa ready')
    F_aesm = None

    def inputs_for(c):
        nonlocal F_aesm
        if c == 'sp_no':
            return SinglePredictorV2(128), [Z_no]
        if c == 'sp_msa':
            return SinglePredictorV2(128), [Z_msa]
        if c == 'x_no_msa':
            return XAttn(128, 128), [Z_no, Z_msa]
        if c == 'x_msa_faesm':
            if F_aesm is None:
                F_aesm = std_per_residue(load_faesm()).to(DEVICE)
                log('faesm ready')
            return CrossAttnOrthoConcatFusion(), [Z_msa, F_aesm]
        raise ValueError(c)

    all_runs = []
    for c in conds:
        for sd in [100, 101, 102, 103, 104]:
            tr, va = cross_split(sd)
            for init in [7, 107, 207]:
                model, inputs = inputs_for(c)
                r, pv = train_pred(model, inputs, tr, va, init)
                within, aucs = per_bin(pv, y[va])
                # conservation-tertile rho
                cons_rho = []
                for g in range(3):
                    m = pos_group[pos[va]] == g
                    cons_rho.append(float(stats.spearmanr(pv[m], y[va][m])[0]) if m.sum() >= 5 else float('nan'))
                all_runs.append({'cond': c, 'split': sd, 'init': init, 'val': r,
                                 'bin_rho': within, 'bin_auc': aucs, 'cons_rho': cons_rho})
                np.savez(OUT / f'preds_{c}_{sd}_{init}.npz', va_idx=va, preds=pv, labels=y[va])
                log(f'{c} s{sd} i{init}: val={r:.3f} bins={["%.2f" % b for b in within]}')
                with open(OUT / 'results.json', 'w') as f:
                    json.dump({'runs': all_runs}, f)

    # aggregate
    agg = {}
    for c in conds:
        rs = [r for r in all_runs if r['cond'] == c]
        agg[c] = {
            'val_mean': float(np.mean([r['val'] for r in rs])),
            'bin_rho_mean': [float(np.nanmean([r['bin_rho'][i] for r in rs])) for i in range(5)],
            'bin_auc_mean': [float(np.nanmean([r['bin_auc'][i] for r in rs])) for i in range(4)],
            'cons_rho_mean': [float(np.nanmean([r['cons_rho'][g] for r in rs])) for g in range(3)],
            'cons_rho_std': [float(np.nanstd([r['cons_rho'][g] for r in rs])) for g in range(3)],
        }
        log(f"{c}: val={agg[c]['val_mean']:.4f} cons_rho={['%.3f' % x for x in agg[c]['cons_rho_mean']]}")
    with open(OUT / 'results.json', 'w') as f:
        json.dump({'runs': all_runs, 'agg': agg, 'cons_per_pos': cons.tolist()}, f)
    log('saved')


if __name__ == '__main__':
    main()
