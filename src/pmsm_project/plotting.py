"""Automatic figure exports without altering original numerical plot calls."""

import json
from pathlib import Path

_CONTEXT = {}


def set_figure_context(
    *, section: str, source_notebook: str | None, source_cell: int | None
) -> None:
    """Attach the educational chapter and original experiment to later figures.

    Parameters
    ----------
    section : str
        Numbered notebook section, for example '12.9'.
    source_notebook : str or None
        Archived source notebook identifier.
    source_cell : int or None
        Zero-based original source-cell index.

    Returns
    -------
    None
        Updates only figure-export metadata; no scientific state is modified.

    Notes
    -----
    Called outside original timed blocks. Axis units remain those specified
    by the source plot; its section supplies the interpretation and limitations.
    """
    _CONTEXT.update(
        section=section, source_notebook=source_notebook, source_cell=source_cell
    )


def install_figure_export(directory: str | Path, *, notebook: str, run_id: str) -> None:
    """Save every displayed Matplotlib figure and a traceability caption record.

    Parameters
    ----------
    directory : str or pathlib.Path
        Destination created when absent.
    notebook : str
        Source notebook identifier for captions.
    run_id : str
        Unique execution ID; distinct from historical saved figures.

    Returns
    -------
    None
        Replaces pyplot.show with an export-then-display wrapper.

    Raises
    ------
    OSError
        If figure PNG or caption JSON cannot be written.

    Notes
    -----
    This changes only figure persistence. The figure counter and axis labels
    identify each plot; the notebook's ordered code cells identify its experiment.
    """
    import matplotlib.pyplot as plt

    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    original = getattr(plt.show, "_pmsm_original_show", plt.show)
    records = []

    def save_and_show(*args, **kwargs):
        for number in plt.get_fignums():
            fig = plt.figure(number)
            filename = f"figure_{len(records)+1:03d}.png"
            fig.savefig(target / filename, dpi=180, bbox_inches="tight")
            labels = [
                {
                    "title": a.get_title(),
                    "xlabel": a.get_xlabel(),
                    "ylabel": a.get_ylabel(),
                }
                for a in fig.axes
            ]
            records.append(
                {
                    "figure": filename,
                    "notebook": notebook,
                    "run_id": run_id,
                    **_CONTEXT,
                    "axes": labels,
                    "caption": f"Section {_CONTEXT.get('section','setup')}: "
                    + "; ".join(a["title"] for a in labels),
                }
            )
        (target / "figure_captions.json").write_text(json.dumps(records, indent=2))
        return original(*args, **kwargs)

    save_and_show._pmsm_original_show = original
    plt.show = save_and_show
