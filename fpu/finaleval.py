#!/usr/bin/env python
"""Final evaluation against T for the selected architecture.

Items 2-4 of the unified protocol.  Only the runs the validation classification
accepted are evaluated, and every generated ensemble has N = 400 so that it
matches each reference block exactly.

    L_s        = mean( (s(x_hat) - s(x^T))^2 )
    e_s        = ||s(x_hat) - s(x^T)|| / ||s(x^T)||
    L_s,ref    = median_{i=1..9} mean( (s(x^Ri) - s(x^T))^2 )
    r_s^eval   = L_s / L_s,ref = ( e_s / e_s,ref )^2

Loss-ratio convention, matching Section 4.  The two forms agree exactly because
L_s = ||a-b||^2/n and e_s = ||a-b||/||b||, so the constant ||b||^2/n cancels
when numerator and denominator share the reference b, which they do.

Per-family medians across runs also convert exactly, since e_s,ref is constant
across runs and x -> (x/c)^2 is monotone: med_m r_s = (med_m e_s)^2 / e_s,ref^2.
The run-level median does NOT, because max_s is taken within each run first, and
no IQR does, being a difference of quantiles.  Those are computed here from the
per-run squared values, never by squaring a median.

All runs at the selected width are evaluated.  Summaries are conditional on the
validation classification; the rejected runs are kept so the box figure can
overlay them.

--test-data swaps in an independent T while every constant stays as fitted on
the original F: the component scales q_scale, the projection scales, the grids
and the KDE bandwidth all come from the original archive.  F, R_1..R_9 and V
are untouched.  The networks are not retrained and not reselected.

usage:
    python finaleval.py --use-sites 25 75    --hidden 75 \
        --test-data fpu_data/fpu_chain_25_50_75_T2.npz
    python finaleval.py --use-sites 25 50 75 --hidden 75 \
        --test-data fpu_data/fpu_chain_25_50_75_T2.npz
"""
import glob
import json
import os
import re
import sys

import numpy as np
import torch

import experiments as E
from toolbox import SINN
from refblocks import stats5, KEYS5

N_GEN = 400
N_TEST = 12000
THRESH = 10.0

# Physical and solver settings the two archives must agree on.  Only
# master_seed and the trajectory count may differ.
MUST_MATCH = ('N', 'dt', 'delta_t', 'sec_length', 't_total', 'Toss',
              'alpha', 'gamma', 'mag', 'beta', 'sites')


def pop_arg(flag):
    """Take an option out of sys.argv before experiments.parse_args sees it."""
    if flag in sys.argv:
        i = sys.argv.index(flag)
        v = sys.argv[i + 1]
        del sys.argv[i:i + 2]
        return v
    return None


def swap_test(S, path):
    """Replace T by an independent archive, keeping every constant from F."""
    old = np.load(S.cfg.data)
    new = np.load(path)
    mo = json.loads(str(old['meta']))
    mn = json.loads(str(new['meta']))
    for k in MUST_MATCH:
        assert mo[k] == mn[k], 'archives differ in %s: %r vs %r' % (k, mo[k], mn[k])
    assert mo['master_seed'] != mn['master_seed'], \
        'same master seed -- the new T is not an independent realization'

    file_sites = [int(s) for s in new['sites']]
    cols = [file_sites.index(s) for s in S.cfg.use_sites]
    n_all = new['traj'].shape[1]
    assert n_all >= N_TEST, 'new archive holds %d trajectories, need %d' % (n_all, N_TEST)

    # S.q_scale is the ORIGINAL archive's pooled scale, the coordinates the
    # networks were trained in.  The new archive's own q_scale is never used.
    tr = (new['traj'][:S.sec_length, :N_TEST, cols] / S.q_scale).astype(np.float32)
    S.test = torch.from_numpy(tr).to(S.device)
    S.ref = None                      # the stale statistics must not be reused

    rho = np.asarray(new['q_scale'])[cols] / np.asarray(S.q_scale)
    print('\nnew T: %s' % path)
    print('   %d of %d trajectories, master_seed %s (original %s)'
          % (N_TEST, n_all, mn['master_seed'], mo['master_seed']))
    print('   standardized with the ORIGINAL q_scale %s'
          % np.array2string(np.asarray(S.q_scale), precision=5))
    print('   the new archive would have given %s, ratio %s (not used)'
          % (np.array2string(np.asarray(new['q_scale'])[cols], precision=5),
             np.array2string(rho, precision=5)))


def disc(a, b):
    """(sqrt of the mse loss, relative L2 error) of a against b."""
    diff = a - b
    return (float(torch.sqrt((diff ** 2).mean())),
            float(torch.linalg.norm(diff) / torch.linalg.norm(b)))


def iqr(v):
    return float(np.percentile(v, 75) - np.percentile(v, 25))


def main():
    test_data = pop_arg('--test-data')
    cfg = E.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    S = E.Setup(cfg, device)
    d, W = S.dim, cfg.hidden
    if test_data:
        swap_test(S, test_data)
    else:
        print('\nWARNING: using the original T, which the superseded analysis saw.')

    wv = np.load('widtheval_d%d.npz' % d)
    # Accept either convention in the stored file; the per-run conversion
    # between them is exact, so no re-run of widtheval.py is required.
    conv = str(wv['convention']) if 'convention' in wv.files else 'error_ratio'
    rv = wv['rval'] if conv == 'loss_ratio' else wv['rval'] ** 2
    print('\nwidtheval_d%d.npz stored as %s; classifying at r^val <= %g'
          % (d, conv, THRESH))
    at_w = wv['hidden'] == W
    keep = {int(s) for s in wv['seed'][at_w & (rv <= THRESH)]}
    print('validation-accepted at width %d: %d of %d  -> seeds %s'
          % (W, len(keep), int(at_w.sum()),
             ' '.join(str(s) for s in sorted(keep))))

    ref_T = stats5(S, S.test)
    print('\nunit conversion, sqrt(L_s) = e_s * ||s(x^T)||/sqrt(n_s):')
    for k in KEYS5:
        nb = float(torch.linalg.norm(ref_T[k]))
        n = int(ref_T[k].numel())
        print('   %-6s ||s||=%11.3f  n=%8d  factor=%.6f' % (k, nb, n, nb / n ** .5))

    # Reference blocks R_1..R_9 against the new T, at the same N = 400.
    dd = np.load(cfg.data)
    file_sites = [int(s) for s in dd['sites']]
    cols = [file_sites.index(s) for s in cfg.use_sites]
    traj = (dd['traj'][:S.sec_length, :, cols] / S.q_scale).astype(np.float32)
    allt = torch.from_numpy(traj).to(device)
    B = cfg.B_train
    assert cfg.hold_start // B == 10
    bl_s, bl_e = [], []
    for i in range(1, 10):                      # block 0 is F and is excluded
        st = stats5(S, allt[:, i * B:(i + 1) * B, :])
        v = [disc(st[k], ref_T[k]) for k in KEYS5]
        bl_s.append([x[0] for x in v])
        bl_e.append([x[1] for x in v])
    sqrtL_ref = np.median(np.array(bl_s), 0)
    e_ref = np.median(np.array(bl_e), 0)

    rows = []
    for p in sorted(glob.glob('runs/net_*.pt')):
        ck = torch.load(p, map_location='cpu')
        c = ck['cfg']
        if (c['B_train'] != 400 or c['batch'] != 400 or c['layers'] != 3
                or c['hidden'] != W or list(c['use_sites']) != list(cfg.use_sites)):
            continue
        seed = int(re.search(r'_s(\d+)\.pt$', p).group(1))
        net = SINN(c['noise_dim'] or d, W, c['layers'], d).to(device)
        net.load_state_dict(ck['net'])
        net.eval()
        # A third stream: distinct from training and from the validation pass.
        torch.manual_seed(700000 + d * 100000 + W * 100 + seed)
        gen = E.generate(net, S, N_GEN)
        st = stats5(S, gen)
        v = [disc(st[k], ref_T[k]) for k in KEYS5]
        sq = np.array([x[0] for x in v])
        er = np.array([x[1] for x in v])
        rows.append(dict(seed=seed, sqrtL=sq, e=er, r=(er / e_ref) ** 2,
                         accepted=seed in keep,
                         best_val=float(ck['best']), path=os.path.basename(p)))

    rows.sort(key=lambda r: r['seed'])
    acc = [r for r in rows if r['accepted']]
    rej = [r for r in rows if not r['accepted']]
    hdr = '  '.join('%9s' % k for k in KEYS5)
    print('\nevaluated %d runs at width %d: %d accepted, %d rejected (%s)'
          % (len(rows), W, len(acc), len(rej),
             ' '.join('s%d' % r['seed'] for r in rej) or 'none'))

    for name, fmt, key in (('e_s^eval (relative, Eq. 22)', '%9.4f', 'e'),
                           ('sqrt(L_s^eval) (absolute RMS)', '%9.5f', 'sqrtL'),
                           ('r_s^eval', '%9.2f', 'r')):
        print('\nper-run %s          N = %d against T' % (name, N_GEN))
        print('%-6s %s' % ('seed', hdr))
        for r in rows:
            print('%-6d %s   %s' % (r['seed'], '  '.join(fmt % x for x in r[key]),
                                    'acc' if r['accepted'] else 'REJ'))

    print('\nrun-level worst family')
    for r in rows:
        j = int(r['r'].argmax())
        print('   seed %-3d  max_s r_s^eval %7.2f  (%-5s) %s'
              % (r['seed'], r['r'][j], KEYS5[j], 'acc' if r['accepted'] else 'REJ'))

    A = {n: np.array([r[n] for r in acc]) for n in ('sqrtL', 'e', 'r')}
    print('\nsummary over the %d validation-accepted runs' % len(acc))
    print('%-16s %s' % ('', hdr))
    for lab, fmt, key in (('median e', '%9.4f', 'e'), ('IQR e', '%9.4f', 'e'),
                          ('median r', '%9.2f', 'r'), ('IQR r', '%9.2f', 'r'),
                          ('median sqrtL', '%9.5f', 'sqrtL'),
                          ('IQR sqrtL', '%9.5f', 'sqrtL')):
        v = (np.median(A[key], 0) if lab.startswith('median')
             else np.array([iqr(A[key][:, j]) for j in range(5)]))
        print('%-16s %s' % (lab, '  '.join(fmt % x for x in v)))
    print('%-16s %s' % ('ref e', '  '.join('%9.4f' % x for x in e_ref)))
    print('%-16s %s' % ('ref sqrtL', '  '.join('%9.5f' % x for x in sqrtL_ref)))

    mx = A['r'].max(1)
    print('\nrun-level worst-family ratio max_s r_s^eval over accepted runs:')
    print('   median %.2f   IQR %.2f   min %.2f   max %.2f'
          % (np.median(mx), iqr(mx), mx.min(), mx.max()))
    print('   not the maximum of the per-family medians above')

    # med r_s is computed from the per-run squared values.  It equals
    # (med e_s / e_s,ref)^2 only when the number of runs is odd, so that the
    # median is a single order statistic; for an even count the median averages
    # the two central values and (a^2+b^2)/2 > ((a+b)/2)^2 unless a = b.  Both
    # are printed so the gap is visible: the table must report the first, and
    # must not derive its third row from the first two.
    naive = (np.median(A['e'], 0) / e_ref) ** 2
    print('\n--- table rows, d = %d, %d accepted runs ---' % (d, len(acc)))
    print('%-16s %s' % ('', hdr))
    print('%-16s %s' % ('med e^eval', '  '.join('%9.4f' % x
                                                for x in np.median(A['e'], 0))))
    print('%-16s %s' % ('e^eval_ref', '  '.join('%9.4f' % x for x in e_ref)))
    print('%-16s %s' % ('med r^eval', '  '.join('%9.2f' % x
                                                for x in np.median(A['r'], 0))))
    print('%-16s %s' % ('(naive sq.)', '  '.join('%9.2f' % x for x in naive)))
    print('the last row is (med e / e_ref)^2, shown only to document that it is '
          'NOT\nthe reported quantity; n = %d is %s' % (len(acc),
          'odd, so the two rows agree' if len(acc) % 2 else
          'even, so they differ'))

    out = 'finaleval_d%d_w%d.npz' % (d, W)
    np.savez(out, keys=np.array(KEYS5),
             seed=np.array([r['seed'] for r in rows]),
             accepted=np.array([r['accepted'] for r in rows]),
             sqrtL=np.array([r['sqrtL'] for r in rows]),
             e=np.array([r['e'] for r in rows]),
             r=np.array([r['r'] for r in rows]),
             worst=mx, thresh=THRESH, convention=np.array('loss_ratio'),
             sqrtL_ref=sqrtL_ref, e_ref=e_ref,
             sqrtL_blocks=np.array(bl_s), e_blocks=np.array(bl_e),
             best_val=np.array([r['best_val'] for r in rows]),
             n_gen=N_GEN, hidden=W, dim=d,
             test_data=str(test_data), n_test=N_TEST)
    print('\nsaved %s  (e_blocks holds the nine gray-box values per family)' % out)

    # Among the accepted runs only, ordered by retained validation score.
    o = np.argsort([r['best_val'] for r in acc])
    rank = len(acc) // 2 + 1
    sel = acc[o[rank - 1]]
    print('\nrepresentative run, rank %d of %d accepted by retained validation '
          'score: seed %d, best_val %.5f, %s'
          % (rank, len(acc), sel['seed'], sel['best_val'], sel['path']))


if __name__ == '__main__':
    main()
