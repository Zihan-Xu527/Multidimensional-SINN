# LLV data preparation helpers

`build_llv_archive.py` parses existing SPPARKS `log/log<i>.spparks` files
into `traj`, a float32 array of shape `(1200, 20000, 2)`. It divides species
counts by 10,000 and retains the full pre-transient-removal timeline.
`create_site_file.py` creates the deterministic square-lattice site file.
`dump_to_frames.py` prepares saved lattice frames for the setup illustration.

The two files in `reference_decks/` are reference/candidate input decks.
The actual deck used for the 20,000-run archive has not been confirmed.
They must not be treated as interchangeable bitwise generators: their RNG
consumption differs.

This selected directory intentionally excludes the earlier 400-run launcher,
obsolete text-processing pipeline, independent C cross-check, and the Linux
SPPARKS binary. The custom SPPARKS-fork source/version, build instructions,
and production launch records are still required for simulation from scratch.
Using an existing published archive does not require these optional upstream
steps; provide its download/access information separately.
