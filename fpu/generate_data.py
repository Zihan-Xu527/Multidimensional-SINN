#!/usr/bin/env python
"""Standalone driver: generate FPU-chain data on CPU, in parallel.

Same computation as the notebook, packaged for `srun python generate_data.py`.
Finished chunks are cached on disk, so re-running after a time-out resumes.
"""
import argparse
import json
import math
import os
import time
from multiprocessing import Pool, cpu_count

import numpy as np

from fpu_chain_worker import simulate_chunk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch', type=int, default=4000)
    ap.add_argument('--sites', type=int, nargs='+', default=[25, 50, 75])
    ap.add_argument('--sec-length', type=int, default=400)
    ap.add_argument('--N', type=int, default=100)
    ap.add_argument('--dt', type=float, default=0.01)
    ap.add_argument('--delta-t', type=float, default=0.5)
    ap.add_argument('--mag', type=float, default=1.0)
    ap.add_argument('--beta', type=float, default=1.0)
    ap.add_argument('--alpha', type=float, default=1.0)
    ap.add_argument('--gamma', type=float, default=1.0)
    ap.add_argument('--seed', type=int, default=20260907)
    ap.add_argument('--legacy-seed', action='store_true')
    ap.add_argument('--dtype', default='float32')
    ap.add_argument('--out-dir', default='fpu_data')
    ap.add_argument('--workers', type=int, default=0,
                    help='0 = use every core in this allocation')
    args = ap.parse_args()

    sites = list(args.sites)
    d = len(sites)
    Time = 5 * args.delta_t * args.sec_length
    length = int(Time / args.dt) + 1
    gap = round(args.delta_t / args.dt)
    t = np.linspace(0, Time, length)
    t_save = t[0:-1:gap]
    Toss = int(t_save.size / 4)

    if args.workers > 0:
        n_workers = args.workers
    elif hasattr(os, 'sched_getaffinity'):
        n_workers = len(os.sched_getaffinity(0))
    else:
        n_workers = cpu_count()

    chunk = min(args.batch, max(50, math.ceil(args.batch / n_workers)))
    n_chunks = math.ceil(args.batch / chunk)

    tag = '_'.join(str(s) for s in sites)
    chunk_dir = os.path.join(args.out_dir, 'chunks_' + tag)
    out_file = os.path.join(args.out_dir, 'fpu_chain_' + tag + '.npz')
    os.makedirs(chunk_dir, exist_ok=True)

    print('workers %d | chunks %d x %d traj | %d integrator steps'
          % (n_workers, n_chunks, chunk, length - 1), flush=True)

    seeds = np.random.SeedSequence(args.seed).spawn(n_chunks)
    jobs = [dict(chunk_id=c, n_traj=min(chunk, args.batch - c * chunk),
                 seed=seeds[c], sites=sites, N=args.N, mag=args.mag,
                 beta=args.beta, alpha=args.alpha, gamma=args.gamma,
                 dt=args.dt, delta_t=args.delta_t, t_total=Time,
                 out_dir=chunk_dir, dtype=args.dtype,
                 legacy_seed=args.legacy_seed)
            for c in range(n_chunks)]

    start = time.time()
    done = 0
    with Pool(min(n_workers, n_chunks)) as pool:
        for cid, path, n_traj, elapsed in pool.imap_unordered(simulate_chunk, jobs):
            done += 1
            tag_e = 'cached' if elapsed == 0.0 else '%.1f s' % elapsed
            print('[%4d/%d] chunk %4d  %5d traj  %-8s  elapsed %.1f min'
                  % (done, n_chunks, cid, n_traj, tag_e,
                     (time.time() - start) / 60), flush=True)
    print('simulation wall time: %.1f min' % ((time.time() - start) / 60))

    traj_full = np.empty((t_save.size + 1, args.batch, d), dtype=args.dtype)
    col = 0
    for c in range(n_chunks):
        a = np.load(os.path.join(chunk_dir, 'chunk_%05d.npy' % c))
        traj_full[:, col:col + a.shape[1], :] = a
        col += a.shape[1]
    assert col == args.batch, (col, args.batch)

    traj = traj_full[Toss:-1]
    t_plot = t_save[Toss:] - t_save[Toss]
    flat = traj.reshape(-1, d)
    q_scale = flat.std(axis=0).astype('float64')
    q_mean = flat.mean(axis=0).astype('float64')

    meta = dict(sites=sites, N=args.N, mag=args.mag, beta=args.beta,
                alpha=args.alpha, gamma=args.gamma, dt=args.dt,
                delta_t=args.delta_t, t_total=Time,
                sec_length=args.sec_length, batch=args.batch, Toss=int(Toss),
                master_seed=args.seed, legacy_seed=bool(args.legacy_seed),
                n_chunks=n_chunks, chunk=chunk, dtype=args.dtype,
                created=time.strftime('%Y-%m-%d %H:%M:%S'),
                q_scale=[float(v) for v in q_scale],
                q_mean=[float(v) for v in q_mean])

    np.savez_compressed(out_file, traj=traj, t=t_plot.astype('float64'),
                        sites=np.array(sites), q_scale=q_scale, q_mean=q_mean,
                        meta=json.dumps(meta))

    print('saved', out_file, traj.shape, traj.dtype)
    print('sigma :', np.round(q_scale, 4))
    print('max |value| / sigma :', np.round(np.abs(flat).max(axis=0) / q_scale, 3))


if __name__ == '__main__':
    main()
