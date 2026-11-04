#!/usr/bin/env python3
"""Convert a SPPARKS text dump of the LLV lattice into a compact frame stack.

A 300-frame text dump of a 100x100 lattice is ~60 MB; the same data as uint8
is 3 MB, which is what you want to move off the cluster.

    in :  dump file written by  `dump 1 text <delta> <file> id i2`
    out:  frames.npz  with
            frames  uint8 (n_frames, L, L)   0 = vacancy, 1 = prey, 2 = predator
            times   float64 (n_frames,)

usage:  python3 dump_to_frames.py dump.llv frames.npz [L]
"""

import sys

import numpy as np


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    src, dst = sys.argv[1], sys.argv[2]
    side = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    n_site = side * side

    times, frames = [], []
    with open(src) as fh:
        line = fh.readline()
        while line:
            if line.startswith("ITEM: TIMESTEP"):
                # SPPARKS writes "<ntimestep> <time>" here. The first field is
                # a snapshot counter, the second is the physical kMC time --
                # take the time when both are present.
                f = fh.readline().split()
                times.append(float(f[1] if len(f) > 1 else f[0]))
            elif line.startswith("ITEM: NUMBER OF"):
                n = int(fh.readline().split()[0])
                if n != n_site:
                    sys.exit("dump has %d sites, expected %d" % (n, n_site))
            elif line.startswith("ITEM: ATOMS"):
                cols = line.split()[2:]
                try:
                    c_id, c_sp = cols.index("id"), cols.index("i2")
                except ValueError:
                    sys.exit("dump columns are %s; need 'id' and 'i2'" % cols)
                grid = np.zeros(n_site, dtype=np.uint8)
                for _ in range(n_site):
                    p = fh.readline().split()
                    grid[int(p[c_id]) - 1] = int(p[c_sp])
                frames.append(grid.reshape(side, side))
            line = fh.readline()

    if not frames:
        sys.exit("no frames parsed -- check the dump format")

    frames = np.asarray(frames, dtype=np.uint8)
    times = np.asarray(times, dtype=np.float64)
    np.savez_compressed(dst, frames=frames, times=times)

    print("parsed %d frames of %dx%d" % (len(frames), side, side))
    print("times  %.1f .. %.1f  (step %.1f)"
          % (times[0], times[-1], times[1] - times[0] if len(times) > 1 else 0))
    print("wrote  %s" % dst)
    occ = np.stack([(frames == k).mean(axis=(1, 2)) for k in (0, 1, 2)], axis=1)
    print("\n  t        vacancy   prey     predator")
    for i in range(0, len(times), max(1, len(times) // 10)):
        print("  %7.1f   %.4f    %.4f   %.4f"
              % (times[i], occ[i, 0], occ[i, 1], occ[i, 2]))


if __name__ == "__main__":
    main()
