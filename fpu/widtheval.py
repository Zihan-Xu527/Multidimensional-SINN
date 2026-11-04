#!/usr/bin/env python
"""Width re-evaluation on V with the five-family classification set.

Replaces the architecture decision and the success classification that were
originally made on the evaluation ensemble.  Nothing here trains: every
checkpoint is loaded and run forward once at B_gen = 400, matched to the size
of the reference blocks whose median is the denominator.

    r_s^val = L_s^val / L_s^ref,   r^val = max_{s in S_cls} r_s^val
    success  <=>  r^val <= 10

Ratios are loss ratios, matching Section 4.  Because the relative loss is the
squared relative L2 discrepancy, r_s^val is the square of the error ratio
e_s^val / e^val_{s,ref}; the squaring is exact per run, so the family errors
below are reported in both forms.

Checkpoints are selected by their stored cfg, never by filename.  After the
retraining round runs/ holds a mixture: old-tag files that are valid, old-tag
files left behind by the B = 800 sweep, old-tag files trained at batch = 800,
new-tag files, and the 2x40 / 4x25 depth runs that are not in the paper.  Only
B_train == batch == 400 with three layers and the requested sites is kept.

Requires refblocks_d{d}_B400.npz from refblocks.py.

usage:
    python widtheval.py --use-sites 25 75
    python widtheval.py --use-sites 25 50 75
"""
import glob
import os
import re

import numpy as np
import torch

import experiments as E
from toolbox import SINN
from refblocks import stats5, KEYS5

THRESHOLDS = (4.0, 6.25, 9.0, 10.0, 12.25, 16.0, 20.25, 25.0)
B_GEN = 400
THRESH = 10.0


def eligible(c, want_sites):
    """The cfg test that decides whether a checkpoint belongs to this sweep."""
    return (c['B_train'] == 400 and c['batch'] == 400 and c['layers'] == 3
            and list(c['use_sites']) == list(want_sites))


def main():
    cfg = E.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    S = E.Setup(cfg, device)
    d = S.dim

    ref = np.load('refblocks_d%d_B400.npz' % d)
    assert list(ref['keys']) == list(KEYS5), 'family order mismatch'
    e_ref = ref['median_val']
    print('\ndenominator e^val_ref (median of R_1..R_9 vs V):')
    print('   ' + '  '.join('%s %.4f' % (k, v) for k, v in zip(KEYS5, e_ref)))

    ref_V = stats5(S, S.val)

    rows = []
    skipped = {}
    for p in sorted(glob.glob('runs/net_*.pt')):
        ck = torch.load(p, map_location='cpu')
        c = ck['cfg']
        if not eligible(c, cfg.use_sites):
            if list(c['use_sites']) == list(cfg.use_sites):
                key = 'B%d_b%d_%dx%d' % (c['B_train'], c['batch'],
                                         c['layers'], c['hidden'])
                skipped[key] = skipped.get(key, 0) + 1
            continue

        m = re.search(r'_s(\d+)\.pt$', p)
        assert m, 'cannot read a seed index from %s' % p
        seed = int(m.group(1))

        nd = c['noise_dim'] or d
        net = SINN(nd, c['hidden'], c['layers'], d).to(device)
        net.load_state_dict(ck['net'])
        net.eval()

        # Explicit, distinct, reproducible: never the historical val_input,
        # which is the stream the checkpoint itself was selected on.
        torch.manual_seed(900000 + d * 100000 + c['hidden'] * 100 + seed)
        gen = E.generate(net, S, B_GEN)

        st = stats5(S, gen)
        e = np.array([E.rel_l2(st[k], ref_V[k]) for k in KEYS5])
        ratio = (e / e_ref) ** 2          # loss ratio, Section 4 convention
        rows.append(dict(hidden=c['hidden'], seed=seed, e=e, ratio=ratio,
                         r=float(ratio.max()), argmax=KEYS5[int(ratio.argmax())],
                         best_val=float(ck['best']), best_iter=int(ck['best_iter']),
                         path=os.path.basename(p)))
        print('%-46s w%-4d s%-3d  r^val %6.2f  (%s)'
              % (rows[-1]['path'], c['hidden'], seed, rows[-1]['r'],
                 rows[-1]['argmax']), flush=True)

    if skipped:
        print('\nskipped, same sites but wrong configuration:')
        for k in sorted(skipped):
            print('   %-18s %d file(s)' % (k, skipped[k]))

    widths = sorted({r['hidden'] for r in rows})
    print('\nd = %d   sites %s   B_gen = %d   %d checkpoints'
          % (d, S.sites, B_GEN, len(rows)))

    print('\nsuccess counts at r^val <= %g   (loss-ratio convention)' % THRESH)
    print('%-8s %-6s %5s  %10s  %10s   %s'
          % ('width', 'count', 'rate', 'med r^val', 'IQR', 'r^val, sorted'))
    for w in widths:
        rs = sorted(r['r'] for r in rows if r['hidden'] == w)
        n_ok = sum(1 for x in rs if x <= THRESH)
        print('%-8d %-6s %4.0f%%  med %6.2f  IQR %6.2f   %s'
              % (w, '%d/%d' % (n_ok, len(rs)), 100.0 * n_ok / len(rs),
                 np.median(rs), np.percentile(rs, 75) - np.percentile(rs, 25),
                 ' '.join('%.2f' % x for x in rs)))

    print('\nthreshold sensitivity')
    print('%-8s %s' % ('width', ' '.join('%7.1f' % t for t in THRESHOLDS)))
    for w in widths:
        rs = [r['r'] for r in rows if r['hidden'] == w]
        print('%-8d %s' % (w, ' '.join('%7s' % ('%d/%d' % (
            sum(1 for x in rs if x <= t), len(rs))) for t in THRESHOLDS)))

    ok = sorted(r['r'] for r in rows if r['r'] <= THRESH)
    bad = sorted(r['r'] for r in rows if r['r'] > THRESH)
    print('\nboundary: largest successful %s, smallest unsuccessful %s'
          % ('%.2f' % ok[-1] if ok else 'none',
             '%.2f' % bad[0] if bad else 'none'))

    print('\nbinding family (argmax of the ratio), counted over all runs')
    for k in KEYS5:
        n = sum(1 for r in rows if r['argmax'] == k)
        print('   %-6s %d' % (k, n))

    out = 'widtheval_d%d.npz' % d
    np.savez(out, keys=np.array(KEYS5), e_ref=e_ref,
             hidden=np.array([r['hidden'] for r in rows]),
             seed=np.array([r['seed'] for r in rows]),
             err=np.array([r['e'] for r in rows]),
             ratio=np.array([r['ratio'] for r in rows]),
             rval=np.array([r['r'] for r in rows]),
             err_ratio=np.array([r['ratio'] ** 0.5 for r in rows]),
             best_val=np.array([r['best_val'] for r in rows]),
             best_iter=np.array([r['best_iter'] for r in rows]),
             path=np.array([r['path'] for r in rows]),
             convention=np.array('loss_ratio'), thresh=THRESH)
    print('\nsaved %s  (rval is the loss ratio L_s/L_s^ref, threshold %g)'
          % (out, THRESH))

    other = 'widtheval_d%d.npz' % (5 - d)
    if os.path.exists(other):
        cross(out, other)


def cross(a_path, b_path):
    """Cross-target comparison, once both targets have been evaluated."""
    try:
        from scipy.stats import fisher_exact
    except ImportError:
        print('\nscipy not available; skipping the Fisher comparison')
        return
    A, B = np.load(a_path), np.load(b_path)
    da = int(re.search(r'_d(\d)\.npz$', a_path).group(1))
    db = int(re.search(r'_d(\d)\.npz$', b_path).group(1))
    print('\ncross-target comparison, d=%d vs d=%d' % (da, db))
    print('%-8s %-12s %-12s %s' % ('width', 'd=%d' % da, 'd=%d' % db, 'Fisher p'))
    for w in sorted(set(A['hidden']) & set(B['hidden'])):
        ra = A['rval'][A['hidden'] == w]
        rb = B['rval'][B['hidden'] == w]
        ka, kb = int((ra <= THRESH).sum()), int((rb <= THRESH).sum())
        _, p = fisher_exact([[ka, len(ra) - ka], [kb, len(rb) - kb]])
        print('%-8d %-12s %-12s %.3f' % (w, '%d/%d' % (ka, len(ra)),
                                         '%d/%d' % (kb, len(rb)), p))
    print('\nFisher p over the threshold sweep, per width:')
    for w in sorted(set(A['hidden']) & set(B['hidden'])):
        ra = A['rval'][A['hidden'] == w]
        rb = B['rval'][B['hidden'] == w]
        ps = []
        for t in THRESHOLDS:
            ka, kb = int((ra <= t).sum()), int((rb <= t).sum())
            _, p = fisher_exact([[ka, len(ra) - ka], [kb, len(rb) - kb]])
            ps.append(p)
        print('   w%-5d %s' % (w, ' '.join('%6.3f' % p for p in ps)))


if __name__ == '__main__':
    main()
