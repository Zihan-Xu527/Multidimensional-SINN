#!/usr/bin/env python
"""Five-family reference-block errors for the FPU targets.

limits.py computes four families against the final-evaluation ensemble only.
The revised protocol of Section 3.6 needs two denominators:

    e^val_{s,ref}   = median_{i=1..9} e_s( R_i ; V )    -- classification
    e^eval_{s,ref}  = median_{i=1..9} e_s( R_i ; T )    -- final evaluation

and it needs five families, because S_cls now contains ACF2.  This script
computes both, for every disjoint B-trajectory block below hold_start.

Block 0 is the fitting block F.  It is written to the output file so the
inventory is complete, but it is EXCLUDED from every median: the denominator
must be disjoint from the training objective.  Blocks 1..9 are R_1..R_9.

usage:
    python refblocks.py --use-sites 25 75
    python refblocks.py --use-sites 25 50 75
"""
import numpy as np
import torch

import experiments as E
from toolbox import StatLoss

KEYS5 = ('ACF', 'CCF', 'ACF2', 'PDF', 'PROJ')


def stats5(S, x):
    """The four families Setup.stats returns, plus ACF2.

    ACF2 is the ACF of the componentwise squared standardized process.  This
    matches what Setup.terms feeds the 'acf2' loss -- losses['acf2'](x**2) --
    so the reported family and the trained family are the same object.
    """
    c = S.cfg
    with torch.no_grad():
        return dict(
            ACF=StatLoss.acf(x, lags=c.lags),
            CCF=StatLoss.ccf(x, lags=c.lags),
            ACF2=StatLoss.acf(x ** 2, lags=c.lags),
            PDF=StatLoss.gauss_kde(x, -5, 5, c.grid, None, chunk=200000,
                                   bandwidth=S.bw),
            PROJ=StatLoss.gauss_kde_proj(x, S.dirs, -5, 5, c.grid, None,
                                         chunk=200000, bandwidth=S.bw,
                                         scale=S.scales))


def table(name, errs, blocks):
    """errs: (n_blk, 5).  blocks: row indices to summarize over."""
    sub = errs[blocks]
    print('\n%s   (median over blocks %d..%d, F excluded)'
          % (name, blocks[0], blocks[-1]))
    print('%-10s %s' % ('', ' '.join('%9s' % k for k in KEYS5)))
    for lab, v in (('median', np.median(sub, 0)),
                   ('p25', np.percentile(sub, 25, 0)),
                   ('p75', np.percentile(sub, 75, 0)),
                   ('min', sub.min(0)),
                   ('max', sub.max(0))):
        print('%-10s %s' % (lab, ' '.join('%9.4f' % x for x in v)))
    print('%-10s %s' % ('max/min', ' '.join('%9.2f' % x
                                            for x in sub.max(0) / sub.min(0))))
    print('%-10s %s' % ('[F only]', ' '.join('%9.4f' % x for x in errs[0])))


def main():
    cfg = E.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    S = E.Setup(cfg, device)

    # Setup keeps only the first B columns as .train; reload to slice blocks.
    d = np.load(cfg.data)
    file_sites = [int(s) for s in d['sites']]
    cols = [file_sites.index(s) for s in cfg.use_sites]
    traj = (d['traj'][:S.sec_length, :, cols] / S.q_scale).astype(np.float32)
    allt = torch.from_numpy(traj).to(device)

    B = cfg.B_train
    n_blk = cfg.hold_start // B
    assert n_blk == 10, 'expected 10 blocks below hold_start, got %d' % n_blk
    assert cfg.hold_start % B == 0, 'hold_start is not a whole number of blocks'

    ref_V = stats5(S, S.val)
    ref_T = stats5(S, S.test)

    # Setup computed the four-family T statistics independently.  If these
    # disagree, stats5 is not reproducing the reported quantity and every
    # number below is wrong.
    for k in ('ACF', 'CCF', 'PDF', 'PROJ'):
        assert torch.allclose(ref_T[k], S.ref[k]), 'stats5 mismatch on %s' % k

    eV = np.empty((n_blk, len(KEYS5)))
    eT = np.empty((n_blk, len(KEYS5)))
    for i in range(n_blk):
        st = stats5(S, allt[:, i * B:(i + 1) * B, :])
        eV[i] = [E.rel_l2(st[k], ref_V[k]) for k in KEYS5]
        eT[i] = [E.rel_l2(st[k], ref_T[k]) for k in KEYS5]
        print('block %d  [%5d,%5d)   V %s   T %s'
              % (i, i * B, (i + 1) * B,
                 ' '.join('%.4f' % x for x in eV[i]),
                 ' '.join('%.4f' % x for x in eT[i])), flush=True)

    blocks = list(range(1, n_blk))
    print('\nd = %d   sites %s   B = %d   |V| = %d   |T| = %d'
          % (S.dim, S.sites, B, S.val.shape[1], S.test.shape[1]))
    table('e^val_ref   -- blocks vs V  (classification denominator)', eV, blocks)
    table('e^eval_ref  -- blocks vs T  (final-evaluation denominator)', eT, blocks)

    out = 'refblocks_d%d_B%d.npz' % (S.dim, B)
    np.savez(out, keys=np.array(KEYS5), err_val=eV, err_eval=eT,
             sites=np.array(S.sites), B=B,
             median_val=np.median(eV[blocks], 0),
             median_eval=np.median(eT[blocks], 0))
    print('\nsaved %s' % out)
    print('median_val is the denominator of r^val; median_eval is the '
          'denominator of\nthe Table 8 family ratios.  Row 0 of err_* is F '
          'and enters neither.')


if __name__ == '__main__':
    main()
