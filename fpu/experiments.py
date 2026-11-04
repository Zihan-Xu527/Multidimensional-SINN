#!/usr/bin/env python
"""Production runs for the FPU section of the multidimensional SINN paper.

One invocation trains `--seeds` models for one configuration and scores each of
them against a test split that no loss and no checkpoint decision ever touched.
Results are appended to a JSON-lines file as they finish, so an interrupted job
loses at most the run in flight, and a second invocation with more seeds adds to
the same file instead of replacing it.

The pipeline mirrors FPU_chain_SINN.ipynb: same split, same fixed bandwidth,
same projection scaling, same noise floors.  The notebook is for looking at
things; this is for producing them.

    python experiments.py --use-sites 25 75     --seeds 5 --iters 20000
    python experiments.py --use-sites 25 50 75  --seeds 5 --iters 20000
"""
import argparse
import itertools
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from toolbox import make_loss, StatLoss, SINN

TERMS = ('acf', 'ccf', 'pdf', 'pdf2', 'acf2')
KEYS = ('ACF', 'CCF', 'PDF', 'PROJ')


# --------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='fpu_data/fpu_chain_25_50_75.npz')
    p.add_argument('--use-sites', type=int, nargs='+', default=[25, 50, 75])
    p.add_argument('--mode', default='adaptive',
                   choices=['baseline', 'adaptive'])
    p.add_argument('--seeds', type=int, default=5)
    p.add_argument('--seed0', type=int, default=0,
                   help='first seed index; use it to add seeds to a finished run')
    p.add_argument('--iters', type=int, default=20000)
    p.add_argument('--hidden', type=int, default=75)
    p.add_argument('--layers', type=int, default=3)
    p.add_argument('--n-frames', type=int, default=3)
    p.add_argument('--noise-dim', type=int, default=0, help='0 = same as d')
    p.add_argument('--B-train', type=int, default=400)
    p.add_argument('--hold-start', type=int, default=4000)
    p.add_argument('--n-val', type=int, default=4000)
    p.add_argument('--batch', type=int, default=400)
    p.add_argument('--sec-length', type=int, default=0, help='0 = from the file')
    p.add_argument('--lags', type=int, default=400)
    p.add_argument('--grid', type=int, default=400)
    p.add_argument('--eval-traj', type=int, default=4000)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--sigma-init', type=float, default=2.0)
    p.add_argument('--record-every', type=int, default=50)
    p.add_argument('--out', default='results.jsonl')
    p.add_argument('--outdir', default='runs')
    return p.parse_args()


# --------------------------------------------------------------------------
class Setup:
    """Data, losses and noise floors -- everything a run needs, built once."""

    def __init__(self, cfg, device):
        self.cfg, self.device = cfg, device

        d = np.load(cfg.data)
        meta = json.loads(str(d['meta']))
        file_sites = [int(s) for s in d['sites']]
        cols = [file_sites.index(s) for s in cfg.use_sites]

        self.sites = [file_sites[c] for c in cols]
        self.dim = len(self.sites)
        self.noise_dim = cfg.noise_dim or self.dim
        self.sec_length = cfg.sec_length or meta['sec_length']
        self.pairs = list(itertools.combinations(range(self.dim), 2))
        self.q_scale = d['q_scale'][cols]

        traj = (d['traj'][:self.sec_length, :, cols] / self.q_scale).astype(np.float32)
        allt = torch.from_numpy(traj).to(device)
        n_all = allt.shape[1]
        assert cfg.hold_start + cfg.n_val < n_all, 'test split is empty'

        self.train = allt[:, :cfg.B_train, :]
        self.val = allt[:, cfg.hold_start:cfg.hold_start + cfg.n_val, :]
        self.test = allt[:, cfg.hold_start + cfg.n_val:, :]

        # One bandwidth everywhere, fixed at the reference-ensemble size.
        self.bw = (self.sec_length * cfg.B_train) ** (-1 / 5)
        self.dirs = StatLoss.projection_directions(
            self.dim, n_frames=cfg.n_frames).to(device)
        self.scales = StatLoss.projection_scales(self.train, self.dirs)

        self.L_train = self._losses(self.train)
        self.L_val = self._losses(self.val)
        self.floor = self._floors()

        print('sites %s (d=%d) | train %d val %d test %d | bw %.5f | %d projections'
              % (self.sites, self.dim, self.train.shape[1], self.val.shape[1],
                 self.test.shape[1], self.bw, self.dirs.shape[0]), flush=True)
        print('noise floors: ' + '  '.join('%s %.2e' % (k, self.floor[k])
                                           for k in TERMS), flush=True)

        self.ref = self.stats(self.test)
        self.data_err = {k: rel_l2(v, self.ref[k])
                         for k, v in self.stats(self.train).items()}
        print('information limit at B=%d: %s' % (
            cfg.B_train,
            '  '.join('%s %.4f' % (k, self.data_err[k]) for k in KEYS)), flush=True)

    def _losses(self, data):
        c = self.cfg
        return dict(
            acf=make_loss('acf[fft]', data, lags=c.lags, device=self.device),
            ccf=make_loss('ccf[fft]', data, lags=c.lags, device=self.device),
            acf2=make_loss('acf[fft]', data**2, lags=c.lags, device=self.device),
            pdf=make_loss('pdf', data, lower=-5, upper=5, n=c.grid, bw=None,
                          bandwidth=self.bw, device=self.device),
            pdf2=make_loss('pdfproj', data, lower=-5, upper=5, n=c.grid, bw=None,
                           directions=self.dirs, bandwidth=self.bw,
                           standardize=True, scale=self.scales,
                           device=self.device),
        )

    def terms(self, losses, x):
        return {k: losses[k](x**2 if k == 'acf2' else x) for k in TERMS}

    def _floors(self, n_draws=20):
        draws = {k: [] for k in TERMS}
        with torch.no_grad():
            for _ in range(n_draws):
                idx = torch.randperm(self.val.shape[1],
                                     device=self.device)[:self.cfg.B_train]
                t = self.terms(self.L_train, self.val[:, idx, :])
                for k in TERMS:
                    draws[k].append(float(t[k]))
        return {k: float(np.median(v)) for k, v in draws.items()}

    def stats(self, x):
        c = self.cfg
        with torch.no_grad():
            return dict(
                ACF=StatLoss.acf(x, lags=c.lags),
                CCF=StatLoss.ccf(x, lags=c.lags),
                PDF=StatLoss.gauss_kde(x, -5, 5, c.grid, None, chunk=200000,
                                       bandwidth=self.bw),
                PROJ=StatLoss.gauss_kde_proj(x, self.dirs, -5, 5, c.grid, None,
                                             chunk=200000, bandwidth=self.bw,
                                             scale=self.scales))


def rel_l2(a, b):
    return float(torch.linalg.norm(a - b) / torch.linalg.norm(b))


# --------------------------------------------------------------------------
def train_one(S, seed):
    cfg, device = S.cfg, S.device
    torch.manual_seed(1234 + seed)
    np.random.seed(1234 + seed)
    net = SINN(S.noise_dim, cfg.hidden, cfg.layers, S.dim).to(device)

    opt = optim.Adam(net.parameters(), lr=cfg.lr)
    adaptive = cfg.mode == 'adaptive'
    if adaptive:
        sigma = {k: nn.Parameter(torch.tensor(float(cfg.sigma_init), device=device))
                 for k in TERMS}
        opt_s = optim.Adam(list(sigma.values()), lr=cfg.lr)

    val_input = torch.randn(2 * S.sec_length, cfg.batch, S.noise_dim, device=device)
    hist = {'step': [], 'train': [], 'val': [],
            **{'T_' + k: [] for k in TERMS}, **{'V_' + k: [] for k in TERMS}}
    s_collect = []
    best, best_iter, best_state = float('inf'), -1, None

    start = time.time()
    for step in range(cfg.iters):
        if adaptive:
            s_collect.append([step] + [float(sigma[k]) for k in TERMS])

        opt.zero_grad()
        noise = torch.randn(2 * S.sec_length, cfg.batch, S.noise_dim, device=device)
        pred, _ = net(noise)
        pred = pred[-cfg.lags:, :, :]
        t = S.terms(S.L_train, pred)
        obj = (sum(t[k] / (2. * sigma[k].detach()**2) for k in TERMS) if adaptive
               else sum(t[k] for k in TERMS))
        obj.backward()
        opt.step()

        if adaptive:
            opt_s.zero_grad()
            lb = sum(t[k].detach() / (2. * sigma[k]**2) + torch.log(sigma[k])
                     for k in TERMS)
            lb.backward()
            opt_s.step()
            with torch.no_grad():
                for k in TERMS:
                    sigma[k].clamp_(min=1e-4)

        if step % cfg.record_every == 0:
            with torch.no_grad():
                tv = {k: float(v) for k, v in t.items()}
                vp, _ = net(val_input)
                vv = {k: float(v) for k, v in
                      S.terms(S.L_val, vp[-cfg.lags:, :, :]).items()}
            tt, vt = sum(tv.values()), sum(vv.values())
            hist['step'].append(step)
            hist['train'].append(tt)
            hist['val'].append(vt)
            for k in TERMS:
                hist['T_' + k].append(tv[k])
                hist['V_' + k].append(vv[k])
            if vt < best:
                best, best_iter = vt, step
                best_state = {a: b.detach().cpu().clone()
                              for a, b in net.state_dict().items()}
            if step % (cfg.record_every * 40) == 0:
                print('    %6d  train %.5f  val %.5f' % (step, tt, vt), flush=True)

    elapsed = time.time() - start
    if best_state is not None:
        net.load_state_dict(best_state)
    return net, hist, s_collect, best, best_iter, elapsed


def generate(net, S, n_traj, chunk=2000):
    outs = []
    with torch.no_grad():
        for s in range(0, n_traj, chunk):
            b = min(chunk, n_traj - s)
            p, _ = net(torch.randn(2 * S.sec_length, b, S.noise_dim,
                                   device=S.device))
            outs.append(p[-S.sec_length:].detach())
    return torch.cat(outs, dim=1)


# --------------------------------------------------------------------------
def main():
    cfg = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(cfg.outdir, exist_ok=True)
    print('device:', device, flush=True)

    S = Setup(cfg, device)
    tag = 'd%d_%s_%dx%d_f%d_n%d_B%d_b%d' % (S.dim, cfg.mode, cfg.layers,
                                            cfg.hidden, cfg.n_frames,
                                            S.noise_dim, cfg.B_train,
                                            cfg.batch)

    for seed in range(cfg.seed0, cfg.seed0 + cfg.seeds):
        name = '%s_s%d' % (tag, seed)
        print('\n=== %s ===' % name, flush=True)
        net, hist, s_collect, best, best_iter, elapsed = train_one(S, seed)

        gen = generate(net, S, cfg.eval_traj)
        g = S.stats(gen)
        model_err = {k: rel_l2(g[k], S.ref[k]) for k in KEYS}
        x_floor = {k: hist['V_' + k][-1] / S.floor[k] for k in TERMS}

        print('  best val %.5f at %d | %.1f min' % (best, best_iter, elapsed / 60))
        print('  rel L2 : ' + '  '.join('%s %.4f' % (k, model_err[k]) for k in KEYS))
        print('  x floor: ' + '  '.join('%s %.1f' % (k, x_floor[k]) for k in TERMS),
              flush=True)

        np.savez(os.path.join(cfg.outdir, 'hist_%s.npz' % name),
                 s_collect=np.array(s_collect) if s_collect
                 else np.zeros((0, len(TERMS) + 1)), **hist)
        torch.save({'net': net.state_dict(), 'cfg': vars(cfg),
                    'sites': S.sites, 'best': best, 'best_iter': best_iter},
                   os.path.join(cfg.outdir, 'net_%s.pt' % name))

        row = dict(name=name, tag=tag, sites=S.sites, dim=S.dim, mode=cfg.mode,
                   hidden=cfg.hidden, layers=cfg.layers, n_frames=cfg.n_frames,
                   noise_dim=S.noise_dim, B_train=cfg.B_train, batch=cfg.batch,
                   iters=cfg.iters, seed=seed, best_val=best, best_iter=best_iter,
                   minutes=elapsed / 60,
                   floor={k: S.floor[k] for k in TERMS},
                   x_floor=x_floor, model_err=model_err, data_err=S.data_err)
        with open(cfg.out, 'a') as fh:
            fh.write(json.dumps(row) + '\n')
        print('  appended to %s' % cfg.out, flush=True)

    print('\ndone')


if __name__ == '__main__':
    main()
