"""Worker for the parallel generation of FPU-chain tagged-particle trajectories.

This lives in a file rather than in a notebook cell on purpose: multiprocessing
pickles the worker function by name, and on macOS and Windows the default start
method is 'spawn', where a function defined in a notebook cannot be imported by
the child process.  Writing it out with %%writefile works with both 'fork' and
'spawn'.
"""
import os
import time

import numpy as np

# One BLAS thread per process: the integrator is elementwise, so extra threads
# only fight each other for cores.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


def simulate_chunk(job):
    """Integrate `n_traj` independent FPU chains; save the tagged coordinates.

    Parameters
    ----------
    job: dict
        chunk_id: int
        n_traj: int              number of independent trajectories in this chunk
        seed: SeedSequence | int RNG stream for this chunk
        sites: sequence of int   tagged particle indices
        N, mag, beta, alpha, gamma, dt, delta_t, t_total: model / solver params
        out_dir: str             directory for the chunk file
        dtype: str               'float32' or 'float64' for the saved array
        legacy_seed: bool        reproduce the old np.random.seed((id+1)*100)

    Returns
    -------
    (chunk_id, path, n_traj, elapsed_seconds)
    """
    cid = job["chunk_id"]
    n = job["n_traj"]
    sites = list(job["sites"])
    N = job["N"]
    dt = job["dt"]
    delta_t = job["delta_t"]
    mag, beta, alpha, gamma = job["mag"], job["beta"], job["alpha"], job["gamma"]

    d = len(sites)
    length = int(job["t_total"] / dt) + 1
    gap = round(delta_t / dt)
    n_save = len(range(0, length - 1, gap))       # == t_save.size in the driver
    expected = (n_save + 1, n, d)

    # Resume: reuse a finished chunk, but only if its shape is what this job
    # asks for.  A cached file from a run with a different batch size or a
    # different trajectory length is stale, not a cache hit.
    path = os.path.join(job["out_dir"], "chunk_%05d.npy" % cid)
    if os.path.exists(path):
        stale = True
        try:
            if np.load(path, mmap_mode="r").shape == expected:
                return cid, path, n, 0.0
        except Exception:
            pass
        if stale:
            os.remove(path)

    if job.get("legacy_seed", False):
        np.random.seed((cid + 1) * 100)
        normal = lambda shape: np.random.randn(*shape)   # noqa: E731
    else:
        rng = np.random.default_rng(job["seed"])
        normal = rng.standard_normal

    traj = np.zeros((n_save + 1, n, d))
    p0 = np.zeros((N, n))
    q0 = np.zeros((N, n))
    noise_scale = np.sqrt(2 * dt * gamma / beta)

    start = time.perf_counter()
    j = 1
    for i in range(1, length):
        q1 = q0 + dt * (p0 - np.roll(p0, 1, axis=0))
        p1 = ((1 - mag * dt) * p0
              + dt * alpha * (np.roll(q0, -1, axis=0) - q0)
              + dt * gamma * (np.roll(q0, -1, axis=0) ** 3 - q0 ** 3)
              + noise_scale * normal((N, n)))
        if i % gap == 0:
            for m, site in enumerate(sites):
                traj[j, :, m] = np.sum(q1[0:site, :], axis=0)
            j += 1
        p0 = p1
        q0 = q1
    elapsed = time.perf_counter() - start

    # Write to a temporary name first so an interrupted run never leaves a
    # half-written chunk that the resume logic would then trust.
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:          # np.save(str) would append ".npy"
        np.save(fh, traj.astype(job.get("dtype", "float32")))
    os.replace(tmp, path)

    return cid, path, n, elapsed
