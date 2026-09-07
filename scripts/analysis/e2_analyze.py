#!/usr/bin/env python3
"""E2 analyzer: does MSA shift RF3's distogram preference toward the crystallized state?

For each target × condition (nomsa/msa) × state (A/B):
  - distogram probs from disto.pt (bins: 3.25 + 1.25*k Å, k=0..37, from atomworks convention)
  - experimental Cα distance map from inputs/e2_targets/structs/*.npy
  - metrics: NLL of experimental distances under the distogram (lower = better match),
    and expected-distance MAE; preference = NLL(stateB) - NLL(stateA)
  - distogram entropy (sharpness)
Verdict: for each target, does (pref_msa - pref_nomsa) shift toward the
'crystallized_state' side? Output: /mnt/k/output_heads/e2/e2_results.json + table.
"""
import json
import numpy as np
import torch
from pathlib import Path

PROJ = Path('/mnt/j/conda_envs/foundry/DMS_Project')
TGT_JSON = PROJ / 'inputs' / 'e2_targets' / 'e2_targets.json'
STRUCT_DIR = PROJ / 'inputs' / 'e2_targets' / 'structs'
BASE = Path('/mnt/k/output_heads/e2')
EDGES = np.linspace(2.0, 22.0, 64)  # rf3/metrics/distogram.py bin_distances
BINW = EDGES[1] - EDGES[0]
CENTERS = np.concatenate([[EDGES[0] - BINW / 2],
                          (EDGES[:-1] + EDGES[1:]) / 2,
                          [EDGES[-1] + BINW / 2]])  # 65 bin pseudo-centers


def dist_metrics(disto_logits, dmap, off=0):
    """disto_logits [L,L,65] raw logits; dmap [M,M] experimental distances (NaN-allowed).
    off: offset of dmap's residue 0 within the distogram sequence (0-indexed)."""
    p = torch.softmax(disto_logits.float(), dim=-1).numpy()
    M = dmap.shape[0]
    if off + M > p.shape[0]:
        M = p.shape[0] - off
        dmap = dmap[:M, :M]
    p = p[off:off + M, off:off + M, :]
    d = dmap
    valid = np.isfinite(d)
    if valid.sum() < 50:
        return None
    idx = np.clip(np.digitize(d[valid], EDGES), 0, 64)  # bucketize semantics: 0..64
    pv = p[valid]  # [K, 65]
    probs = pv[np.arange(len(idx)), idx]
    nll = float(-np.log(np.clip(probs, 1e-9, 1)).mean())
    exp_d = (p * CENTERS).sum(-1)
    mae = float(np.abs(exp_d[valid] - d[valid]).mean())
    ent = float(-(p * np.log(np.clip(p, 1e-9, 1))).sum(-1)[valid].mean())
    return {'nll': nll, 'mae': mae, 'entropy': ent, 'n_pairs': int(valid.sum())}


def main():
    targets = json.load(open(TGT_JSON))['targets']
    results = []
    for t in targets:
        tid = t['id']
        row = {'id': tid, 'name': t['name'], 'crystallized': t.get('crystallized_state'),
               'category': t.get('category'),
               'state_a_label': t.get('state_a', {}).get('label', ''),
               'state_b_label': t.get('state_b', {}).get('label', '')}
        for cond in ['nomsa', 'msa']:
            dp = BASE / tid / cond / 'disto.pt'
            if not dp.exists():
                row[cond] = None
                continue
            logits = torch.load(dp, map_location='cpu', weights_only=True)
            per_state = {}
            smeta_all = None
            smp = STRUCT_DIR / f'{tid}.json'
            if smp.exists():
                smeta_all = json.load(open(smp))
            fa = list(Path('/tmp/e2/seqs').glob(f'{tid}_*.fa'))
            fold_seq = open(fa[0]).read().split('\n')[1].strip() if fa else None
            dom_lo = 0
            if fold_seq and len(fold_seq) > 600:
                import re as _re
                m_ = _re.match(r'\s*(\d+)\s*-\s*(\d+)', t.get('construct', {}).get('boundaries', ''))
                if m_:
                    dom_lo = int(m_.group(1)) - 1
                    fold_seq = fold_seq[dom_lo:int(m_.group(2))]
            for sname, sk in [('A', 'stateA'), ('B', 'stateB')]:
                if smeta_all and sk in smeta_all.get('states', {}):
                    sm = smeta_all['states'][sk]
                    dmap = np.load(STRUCT_DIR / sm['npy'])
                    off = fold_seq.find(sm.get('sequence', '')[:40]) if fold_seq else -1
                    if off < 0:
                        rr = sm.get('residue_range_used')
                        off = max(0, (rr[0] - 1) - dom_lo) if rr else 0
                    per_state[sname] = dist_metrics(logits, dmap, off)
                else:
                    per_state[sname] = None
            row[cond] = per_state
        # preference: NLL_B - NLL_A (positive => state A matches better)
        for cond in ['nomsa', 'msa']:
            if row.get(cond) and row[cond].get('A') and row[cond].get('B'):
                row[f'pref_{cond}'] = row[cond]['B']['nll'] - row[cond]['A']['nll']
        if row.get('pref_nomsa') is not None and row.get('pref_msa') is not None:
            row['shift'] = row['pref_msa'] - row['pref_nomsa']
        results.append(row)

    with open(BASE / 'e2_results.json', 'w') as f:
        json.dump(results, f, indent=1)

    print(f"\n{'target':<10}{'category':<16}{'cryst_side':<11}{'pref_no':>8}{'pref_msa':>9}{'shift':>8}  verdict")
    n_toward = n_away = 0
    for r in results:
        if r.get('shift') is None:
            print(f"{r['id']:<10}{r.get('category',''):<16} incomplete")
            continue
        cr = (r['crystallized'] or '').lower()
        al = (r.get('state_a_label') or '').lower()
        bl = (r.get('state_b_label') or '').lower()
        if cr and (cr in al or al in cr):
            cr_side = 'A'
        elif cr and (cr in bl or bl in cr):
            cr_side = 'B'
        else:
            cr_side = 'A'  # default: state_a is usually the PDB-dominant one
        toward = (cr_side == 'A' and r['shift'] > 0) or (cr_side == 'B' and r['shift'] < 0)
        n_toward += toward
        n_away += (not toward)
        verdict = 'TOWARD crystal' if toward else 'away'
        print(f"{r['id']:<10}{r.get('category',''):<16}{cr_side:<11}{r['pref_nomsa']:>8.3f}"
              f"{r['pref_msa']:>9.3f}{r['shift']:>8.3f}  {verdict}")
    print(f"\nSummary: {n_toward} toward / {n_away} away (of {n_toward + n_away} scored)")
    print('saved', BASE / 'e2_results.json')


if __name__ == '__main__':
    main()
