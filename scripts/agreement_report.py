#!/usr/bin/env python3
"""
P1 gate: CP7-agreement comparison between two shadow corpora (baseline ckpt vs
post-BC ckpt), Codex #76 slices. Usage:
  py -3.12 scripts/agreement_report.py --base "E:/.../base/*.jsonl*" --cand "E:/.../cand/*.jsonl*"
Gate: overall +10pp AND non-pass +8pp AND CP7-choice prob/CE improves AND
pass-imitation not inflated.
"""
import argparse, glob, json
import numpy as np
from collections import defaultdict

CONF = ('source_id', 'text', 'text_disambig')


def load(pat):
    rows = []
    for f in glob.glob(pat):
        for line in open(f, encoding='utf-8'):
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    out = []
    for r in rows:
        if r.get('dtype') != 'ACTIVATE_ABILITY_OR_SPELL':
            continue
        if r.get('cp7_idx', -2) < 0:
            continue
        if '{T}: Add' in (r.get('chosen_desc') or ''):
            continue
        out.append(r)
    return out


def stats(rows):
    s = {}
    def agr(sub):
        return (np.mean([1.0 if r['chosen'] == r['cp7_idx'] else 0.0 for r in sub]), len(sub)) if sub else (float('nan'), 0)
    def cp7prob(sub):
        vals = []
        for r in sub:
            p = r.get('probs') or []
            i = r['cp7_idx']
            if i < len(p) and p[i] is not None:
                vals.append(max(1e-6, p[i]))
        return (float(np.mean(vals)), float(np.mean(-np.log(vals)))) if vals else (float('nan'), float('nan'))
    allr = rows
    nonpass = [r for r in rows if r['cp7_idx'] != 0]
    conf = [r for r in rows if r.get('cp7_status') in CONF]
    s['all'] = agr(allr); s['nonpass'] = agr(nonpass); s['conf'] = agr(conf)
    s['prob_all'], s['ce_all'] = cp7prob(allr)
    s['prob_np'], s['ce_np'] = cp7prob(nonpass)
    s['agent_pass_rate'] = np.mean([1.0 if r['chosen'] == 0 else 0.0 for r in allr]) if allr else float('nan')
    s['label_pass_rate'] = np.mean([1.0 if r['cp7_idx'] == 0 else 0.0 for r in allr]) if allr else float('nan')
    bym = defaultdict(list)
    for r in rows:
        bym[r.get('matchup', '?')].append(r)
    s['by_matchup'] = {m: agr(v) for m, v in sorted(bym.items())}
    s['by_matchup_np'] = {m: agr([r for r in v if r['cp7_idx'] != 0]) for m, v in sorted(bym.items())}
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True)
    ap.add_argument('--cand', required=True)
    args = ap.parse_args()
    B, C = stats(load(args.base)), stats(load(args.cand))

    def line(name, b, c):
        (ba, bn), (ca, cn) = b, c
        print(f'  {name:<26} base {ba:.3f} (n={bn})  cand {ca:.3f} (n={cn})  delta {100*(ca-ba):+.1f}pp')

    print('=== CP7 AGREEMENT: baseline vs candidate ===')
    line('all labeled', B['all'], C['all'])
    line('non-pass labels', B['nonpass'], C['nonpass'])
    line('confident (srcid/text)', B['conf'], C['conf'])
    print(f"  cp7-choice prob all      base {B['prob_all']:.3f} cand {C['prob_all']:.3f} | CE {B['ce_all']:.3f} -> {C['ce_all']:.3f}")
    print(f"  cp7-choice prob non-pass base {B['prob_np']:.3f} cand {C['prob_np']:.3f} | CE {B['ce_np']:.3f} -> {C['ce_np']:.3f}")
    print(f"  agent pass-rate          base {B['agent_pass_rate']:.3f} cand {C['agent_pass_rate']:.3f} (label pass-rate {B['label_pass_rate']:.3f}/{C['label_pass_rate']:.3f})")
    # disagreement-row recovery: rows where BASELINE agent disagreed; measured on cand's
    # own rows (distributions differ; report as approximation)
    print('\n=== per matchup (all / non-pass) ===')
    for m in C['by_matchup']:
        b = B['by_matchup'].get(m, (float('nan'), 0)); c = C['by_matchup'][m]
        bn = B['by_matchup_np'].get(m, (float('nan'), 0)); cn = C['by_matchup_np'][m]
        print(f'  {m:<10} all {b[0]:.3f}->{c[0]:.3f} ({100*(c[0]-b[0]):+.1f}pp)   non-pass {bn[0]:.3f}->{cn[0]:.3f} ({100*(cn[0]-bn[0]):+.1f}pp)')
    print('\nGATE: all >= +10pp AND non-pass >= +8pp AND CE improves AND agent pass-rate not inflated vs label pass-rate.')


if __name__ == '__main__':
    main()
