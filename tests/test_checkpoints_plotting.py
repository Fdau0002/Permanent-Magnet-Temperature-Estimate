"""Prediction evidence and saved-figure traceability checks with tiny fixtures."""

import json
import numpy as np
import pytest
from pmsm_project.checkpoints import load_prediction_checkpoint
from pmsm_project.plotting import install_figure_export, set_figure_context


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "model",
        "profile_id",
        "mode",
        "artifact_sha256",
        "selection_manifest_sha256",
        "source_row",
        "nan",
        "shape",
        "missing",
    ],
)
def test_prediction_checkpoint_identity_and_shape(tmp_path, fault):
    payload = dict(
        model="OLS",
        profile_id=13,
        mode="full",
        artifact_sha256="a" * 64,
        selection_manifest_sha256="b" * 64,
        source_row=np.array([10, 11, 12]),
        prediction=np.array([30.0, 31.0, 32.0]),
    )
    if fault in ["model", "mode", "artifact_sha256", "selection_manifest_sha256"]:
        payload[fault] = "changed"
    elif fault == "profile_id":
        payload[fault] = 16
    elif fault == "source_row":
        payload[fault] = np.array([10, 12, 11])
    elif fault == "nan":
        payload["prediction"][0] = np.nan
    elif fault == "shape":
        payload["prediction"] = payload["prediction"][:2]
    elif fault == "missing":
        del payload["artifact_sha256"]
    path = tmp_path / "profile_013.npz"
    np.savez_compressed(path, **payload)

    def read():
        return load_prediction_checkpoint(
            path,
            model="OLS",
            profile_id=13,
            mode="full",
            source_rows=[10, 11, 12],
            artifact_sha256="a" * 64,
            manifest_sha256="b" * 64,
        )

    if fault is not None:
        with pytest.raises(ValueError):
            read()
    else:
        np.testing.assert_array_equal(read(), [30.0, 31.0, 32.0])


def test_figure_export_keeps_labels_units_and_source_trace(tmp_path, monkeypatch):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    install_figure_export(
        tmp_path, notebook="02_Experimental_Development", run_id="synthetic-test"
    )
    set_figure_context(
        section="12.9", source_notebook="fresh_reconstruction.ipynb", source_cell=149
    )
    fig, ax = plt.subplots()
    ax.plot([0, 1], [30, 31])
    ax.set(
        xlabel="Elapsed time (s)",
        ylabel="Temperature (°C)",
        title="Synthetic traceability check",
    )
    plt.show()
    plt.close(fig)
    assert (tmp_path / "figure_001.png").is_file()
    record = json.loads((tmp_path / "figure_captions.json").read_text())[0]
    assert record["section"] == "12.9" and record["source_cell"] == 149
    assert record["axes"][0]["ylabel"] == "Temperature (°C)"
