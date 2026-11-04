"""Plot a demo's joint densities without training or generating new trajectories."""

from pathlib import Path

import numpy as np


MASS_FRACTIONS = (0.10, 0.30, 0.50, 0.70, 0.90)


def plot_joint_density(z_grid, target_density, sinn_density, output_dir):
    """Overlay matched density levels; leave the notebook backend unchanged."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    z = np.asarray(z_grid, dtype=float)
    target = np.asarray(target_density, dtype=float)
    model = np.asarray(sinn_density, dtype=float)
    if z.ndim != 1 or z.size < 2 or not np.all(np.isfinite(z)):
        raise ValueError("Expected a finite one-dimensional density grid.")
    dz = float(z[1] - z[0])
    if dz <= 0 or not np.allclose(np.diff(z), dz):
        raise ValueError("The density grid must be increasing and uniform.")
    for name, density in (("target", target), ("SINN", model)):
        if density.shape != (z.size, z.size):
            raise ValueError(f"Unexpected {name} joint-density shape.")
        if not np.isfinite(density).all() or np.any(density < 0):
            raise ValueError(f"The {name} density must be finite and nonnegative.")
        if float(density.sum()) <= 0:
            raise ValueError(f"The {name} density has no positive mass.")

    # Same target-mass construction as duffing/fig_joint.py::mass_levels.
    # Importing that module would switch the notebook backend to Agg.
    ordered = np.sort(target.ravel())[::-1]
    cumulative = np.cumsum(ordered) * dz * dz
    cumulative /= cumulative[-1]
    levels = sorted(set(float(ordered[np.searchsorted(cumulative, fraction)])
                        for fraction in MASS_FRACTIONS))

    fig, ax = plt.subplots(figsize=(4.8, 4.5))
    ax.contour(z, z, target.T, levels=levels,
               colors="black", linestyles="solid", linewidths=1.15)
    ax.contour(z, z, model.T, levels=levels,
               colors="red", linestyles="dashed", linewidths=1.15)
    ax.set(xlabel=r"$\widetilde{q}_1$", ylabel=r"$\widetilde{q}_2$",
           xlim=(z[0], z[-1]), ylim=(z[0], z[-1]), title="Joint density")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(handles=[
        Line2D([], [], color="black", ls="-", lw=1.4, label="Target"),
        Line2D([], [], color="red", ls="--", lw=1.4, label="SINN"),
    ], frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(output_dir / f"joint_density_contours.{extension}", dpi=150)
    np.savez_compressed(
        output_dir / "joint_density_contours.npz",
        z_grid=z, target=target, sinn=model, density_levels=np.asarray(levels),
        target_mass_fractions=np.asarray(MASS_FRACTIONS),
    )
    return fig, levels


def main():
    """Replot an existing demo directory using its saved evaluation ensemble."""
    import argparse
    import json
    import sys

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True,
                        help="Existing outputs/demo-* directory.")
    args = parser.parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "duffing"))
    import evaluate as EV

    config = json.loads((run_dir / "demo_config.json").read_text())
    data_dir = (run_dir / "data" if config["regenerate_target"]
                else repo_root / "duffing" / "data")
    tag = f"lam{config['lambda_value']:g}".replace(".", "p")
    ref = EV.load_reference(str(data_dir), tag,
                            str(data_dir / f"traj_{tag}.npy"))
    config_tag = (f"nf{config['noncoordinate_frames']}_lbon_din{config['d_in']}"
                  f"_L{config['layers']}H{config['hidden']}_{tag}")
    report = json.loads(
        (run_dir / "evaluation" / f"report_{config_tag}.json").read_text()
    )
    if report["n_seeds"] != 1 or report["n_gen"] != config["n_eval"]:
        raise ValueError("Expected one model with the saved evaluation size.")
    generated_path = run_dir / "evaluation" / f"gen_{report['display_run']}.npz"
    generated, space = EV.load_generated(str(generated_path), ref, "standardized")
    if generated.shape != (int(ref["l_seq"]), config["n_eval"], 2):
        raise ValueError(f"Unexpected generated shape: {generated.shape}")
    standardized = EV.standardise(generated, ref, space)
    if not np.isfinite(standardized).all():
        raise ValueError("The saved generated ensemble contains non-finite states.")
    model_joint = EV.kde2d(standardized, float(ref["bandwidth_2d"]), ref["z_grid"])
    fig, _ = plot_joint_density(ref["z_grid"], ref["joint"], model_joint, run_dir)
    plt.close(fig)
    print("Saved joint-density contours in:", run_dir)
    print("Reused the saved ensemble; no training or trajectory generation.")


if __name__ == "__main__":
    main()
