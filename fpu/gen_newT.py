#!/usr/bin/env python
"""Generate an independent FPU archive to serve as the final-evaluation T.

Self-contained: the same integrator, time axis, burn-in and archive layout as
the notebook that produced the original archive, so the metadata asserts in
finaleval.swap_test pass.  Only --seed and --batch differ from that run.

The output is a NEW file.  It does not touch the original archive, whose F,
R_1..R_9 and V remain the split every network was trained and selected on,
and whose q_scale stays the coordinate system the networks emit in.

    python gen_newT.py --seed 20260930 --batch 12000 \
        --out fpu_data/fpu_chain_25_50_75_T2.npz
"""
import argparse
import json
import math
import os
import time
from multiprocessing import Pool

import numpy as np

# One BLAS thread per process: the integrator is elementwise, so extra
# threads only fight each other for cores.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

# Frozen.  These must match the original archive or swap_test refuses the file.
N = 100
MAG = 1.0            # damping gamma in Eq. (38)
BETA = 1.0
ALPHA = 1.0          # harmonic coefficient
GAMMA = 1.0          # quartic coefficient kappa, and the noise amplitude
DT = 0.01
DELTA_T = 0.5
SEC_LENGTH = 400
TIME = 5 * DELTA_T * SEC_LENGTH        # 1000.0
SITES = [25, 50, 75]
DIM = len(SITES)
DTYPE = 'float32'


def simulate_chunk(job):
    """Integrate n independent FPU chains; save the tagged coordinates."""
    cid, n = job['chunk_id'], job['n_traj']
    length = int(TIME / DT) + 1
    gap = round(DELTA_T / DT)
    n_save = len(range(0, length - 1, gap))
    expected = (n_save + 1, n, DIM)

    path = os.path.join(job['out_dir'], 'chunk_%05d.npy' % cid)
    if os.path.exists(path):
        try:
            if np.load(path, mmap_mode='r').shape == expected:
                return cid, n, 0.0
        except Exception:
            pass
        os.remove(path)

    rng = np.random.default_rng(job['seed'])
    traj = np.zeros((n_save + 1, n, DIM))
    p0 = np.zeros((N, n))
    q0 = np.zeros((N, n))
    noise_scale = np.sqrt(2 * DT * GAMMA / BETA)

    start = time.perf_counter()
    j = 1
    for i in range(1, length):
        q1 = q0 + DT * (p0 - np.roll(p0, 1, axis=0))
        p1 = ((1 - MAG * DT) * p0
              + DT * ALPHA * (np.roll(q0, -1, axis=0) - q0)
              + DT * GAMMA * (np.roll(q0, -1, axis=0) ** 3 - q0 ** 3)
              + noise_scale * rng.standard_normal((N, n)))
        if i % gap == 0:
            for m, site in enumerate(SITES):
                traj[j, :, m] = np.sum(q1[0:site, :], axis=0)
            j += 1
        p0, q0 = p1, q1
    elapsed = time.perf_counter() - start

    tmp = path + '.tmp'
    with open(tmp, 'wb') as fh:
        np.save(fh, traj.astype(DTYPE))
    os.replace(tmp, path)
    return cid, n, elapsed


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, required=True,
                   help='master seed; MUST differ from the original archive')
    p.add_argument('--batch', type=int, default=12000)
    p.add_argument('--out', required=True)
    p.add_argument('--chunk-dir', default='')
    cfg = p.parse_args()

    assert not os.path.exists(cfg.out), \
        '%s already exists; refusing to overwrite an archive' % cfg.out
    os.makedirs(os.path.dirname(cfg.out) or '.', exist_ok=True)
    chunk_dir = cfg.chunk_dir or (os.path.splitext(cfg.out)[0] + '_chunks')
    os.makedirs(chunk_dir, exist_ok=True)

    n_workers = (len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity')
                 else os.cpu_count())
    chunk = min(cfg.batch, max(50, math.ceil(cfg.batch / n_workers)))
    n_chunks = math.ceil(cfg.batch / chunk)

    length = int(TIME / DT) + 1
    gap = round(DELTA_T / DT)
    t = np.linspace(0, TIME, length)
    t_save = t[0:-1:gap]
    toss = int(t_save.size / 4)

    print('total simulated time : %.1f  (%d integrator steps)' % (TIME, length - 1))
    print('saved frames         : %d -> %d after burn-in of %d (%.1f time units)'
          % (t_save.size, t_save.size - toss, toss, toss * DELTA_T))
    print('trajectories         : %d' % cfg.batch)
    print('workers / chunks     : %d / %d (width %d)' % (n_workers, n_chunks, chunk))
    print('chunk cache          : %s' % chunk_dir)
    print('output               : %s' % cfg.out, flush=True)

    seeds = np.random.SeedSequence(cfg.seed).spawn(n_chunks)
    jobs = [dict(chunk_id=c, n_traj=min(chunk, cfg.batch - c * chunk),
                 seed=seeds[c], out_dir=chunk_dir) for c in range(n_chunks)]

    start, done = time.time(), 0
    with Pool(min(n_workers, n_chunks)) as pool:
        for cid, n_traj, el in pool.imap_unordered(simulate_chunk, jobs):
            done += 1
            print('[%3d/%d] chunk %3d  %5d traj  %-9s  wall %.1f min'
                  % (done, n_chunks, cid, n_traj,
                     'cached' if el == 0.0 else '%.1f s' % el,
                     (time.time() - start) / 60), flush=True)
    print('simulation wall time : %.1f min' % ((time.time() - start) / 60))

    traj_full = np.empty((t_save.size + 1, cfg.batch, DIM), dtype=DTYPE)
    col = 0
    for c in range(n_chunks):
        a = np.load(os.path.join(chunk_dir, 'chunk_%05d.npy' % c))
        assert a.shape[0] == t_save.size + 1 and a.shape[2] == DIM, (c, a.shape)
        traj_full[:, col:col + a.shape[1], :] = a
        col += a.shape[1]
    assert col == cfg.batch

    traj = traj_full[toss:-1]
    t_plot = t_save[toss:] - t_save[toss]
    assert traj.shape[0] == t_plot.size, (traj.shape, t_plot.size)

    # This archive's own scale is recorded but is NOT the one used downstream:
    # finaleval standardizes it with the original archive's q_scale.
    q_scale = traj.reshape(-1, DIM).std(axis=0).astype('float64')
    q_mean = traj.reshape(-1, DIM).mean(axis=0).astype('float64')
    print('per-component std  :', np.round(q_scale, 5))
    print('per-component mean :', np.round(q_mean, 5), '(~0 by the q -> -q symmetry)')

    meta = dict(sites=SITES, N=N, mag=MAG, beta=BETA, alpha=ALPHA, gamma=GAMMA,
                dt=DT, delta_t=DELTA_T, t_total=TIME, sec_length=SEC_LENGTH,
                batch=cfg.batch, Toss=int(toss), master_seed=cfg.seed,
                legacy_seed=False, n_chunks=n_chunks, chunk=chunk, dtype=DTYPE,
                created=time.strftime('%Y-%m-%d %H:%M:%S'),
                q_scale=[float(v) for v in q_scale],
                q_mean=[float(v) for v in q_mean])

    np.savez_compressed(cfg.out, traj=traj, t=t_plot.astype('float64'),
                        sites=np.array(SITES), q_scale=q_scale, q_mean=q_mean,
                        meta=json.dumps(meta))
    print('\nsaved %s  traj %s  %.1f MB'
          % (cfg.out, traj.shape, os.path.getsize(cfg.out) / 1e6))
    print(json.dumps(meta, indent=1))


if __name__ == '__main__':
    main()
