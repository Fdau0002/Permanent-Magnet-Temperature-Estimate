"""Source-grounded and numerical checks of the recovered original objectives."""

import ast, copy, json, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scipy.linalg import lstsq
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.exceptions import ConvergenceWarning
from pmsm_project.study import DevelopmentStudy
from pmsm_project.features import make_paper_inputs, make_temporal_features
from pmsm_project.models import make_regressor
from pmsm_project.numerics import compress_profile_ols
from pmsm_project.sampling import nested_sample_order
from pmsm_project.search import MODEL_SPACES, decode_spans, protocol
from pmsm_project.config import LOCKED_DEVELOPMENT_IDS

ROOT = Path(__file__).resolve().parents[1]


def original_node(cell, name):
    notebook = json.loads(
        (ROOT / "provenance/source_notebooks/fresh_reconstruction.ipynb").read_text()
    )
    source = "".join(notebook["cells"][cell]["source"]).replace(
        "%pip -q install scikit-optimize", ""
    )
    return next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def reference_functions(study):
    ns = dict(
        np=np,
        pd=pd,
        time=time,
        warnings=warnings,
        lstsq=lstsq,
        Pipeline=Pipeline,
        TransformedTargetRegressor=TransformedTargetRegressor,
        StandardScaler=StandardScaler,
        LinearRegression=LinearRegression,
        mean_squared_error=mean_squared_error,
        ConvergenceWarning=ConvergenceWarning,
        make_paper_inputs=make_paper_inputs,
        make_temporal_features=make_temporal_features,
        make_regressor=make_regressor,
        compress_profile_ols=compress_profile_ols,
        nested_sample_order=nested_sample_order,
        decode_spans=decode_spans,
        MODEL_SPACES=MODEL_SPACES,
        train_df=study.development,
        profile_cv=study.profile_cv,
        fold_lookup=study.fold_lookup,
        cv_lookup=study.fold_lookup,
        SAMPLES_PER_PROFILE=80,
        TRAIN_SAMPLES=study.train_samples,
        VALIDATION_SAMPLES=640,
    )
    for cell, name in [
        (50, "build_candidate_features"),
        (50, "evaluate_ols_cv"),
        (66, "evaluate_model_cv"),
        (86, "evaluate_full_ols_cv"),
        (100, "evaluate_larger_model_cv"),
    ]:
        exec(
            compile(
                ast.Module(body=[original_node(cell, name)], type_ignores=[]),
                "historical_source",
                "exec",
            ),
            ns,
        )
    return ns


@pytest.fixture
def fixture_study(profile):
    frames = []
    for i, pid in enumerate(LOCKED_DEVELOPMENT_IDS):
        frame = profile.iloc[:80].copy()
        frame.index = np.arange(i * 80, (i + 1) * 80)
        frame["profile_id"] = pid
        frame["pm"] += i / 2
        frame["i_q"] += i / 4
        frames.append(frame)
    cv = pd.DataFrame(
        {
            "profile_id": LOCKED_DEVELOPMENT_IDS,
            "cv_fold": [1] * 19 + [2] * 18 + [3] * 18,
        }
    )
    return DevelopmentStudy(pd.concat(frames), cv, strict=False)


def test_extracted_objective_arithmetic_matches_source():
    mapping = json.loads((ROOT / "provenance/objective_extraction.json").read_text())
    module = ast.parse((ROOT / "src/pmsm_project/study.py").read_text())
    cls = next(node for node in module.body if isinstance(node, ast.ClassDef))
    bindings = {
        "train_df": "development",
        "profile_cv": "profile_cv",
        "fold_lookup": "fold_lookup",
        "cv_lookup": "fold_lookup",
        "SAMPLES_PER_PROFILE": "samples_per_profile",
        "TRAIN_SAMPLES": "train_samples",
        "VALIDATION_SAMPLES": "validation_samples",
    }

    class Normalise(ast.NodeTransformer):
        def visit_Attribute(self, node):
            if isinstance(node.value, ast.Name) and node.value.id == "self":
                return ast.copy_location(ast.Name(id=node.attr, ctx=node.ctx), node)
            return self.generic_visit(node)

        def visit_Name(self, node):
            if isinstance(node.ctx, ast.Load) and node.id in bindings:
                return ast.copy_location(
                    ast.Name(id=bindings[node.id], ctx=node.ctx), node
                )
            return node

    def canonical(node):
        n = copy.deepcopy(node)
        if n.args.args and n.args.args[0].arg == "self":
            n.args.args.pop(0)
        if (
            isinstance(n.body[0], ast.Expr)
            and isinstance(n.body[0].value, ast.Constant)
            and isinstance(n.body[0].value.value, str)
        ):
            n.body.pop(0)
        return ast.dump(Normalise().visit(n), include_attributes=False)

    for name, record in mapping.items():
        actual = next(
            n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name
        )
        assert canonical(actual) == canonical(original_node(record["cell"], name)), name


def test_numerical_candidate_and_ols_objectives_match_original(fixture_study):
    study = fixture_study
    reference = reference_functions(study)
    spans = [10, 50, 200, 1000]
    for a, b in zip(
        study.build_candidate_features(spans),
        reference["build_candidate_features"](spans),
    ):
        np.testing.assert_array_equal(a, b)
    a = study.evaluate_ols_cv(spans)
    b = reference["evaluate_ols_cv"](spans)
    np.testing.assert_allclose(a[1], b[1], rtol=1e-12, atol=1e-10)
    assert a[0] == pytest.approx(b[0], rel=1e-12)


def test_knn_cv_objective_matches_original(fixture_study):
    study = fixture_study
    parameters = [10, 40, 150, 800, 12, "distance"]
    a = study.evaluate_model_cv("k-NN", parameters)
    b = reference_functions(study)["evaluate_model_cv"]("k-NN", parameters)
    np.testing.assert_allclose(
        a["fold_scores"], b["fold_scores"], rtol=1e-12, atol=1e-10
    )
    assert a["spans"] == b["spans"] and a["hyperparameters"] == b["hyperparameters"]


def test_full_qr_objective_matches_source(profile):
    frames = []
    for i, pid in enumerate([2, 3, 4]):
        f = profile.copy()
        f.index += i * len(f)
        f.profile_id = pid
        f["pm"] += i
        frames.append(f)
    cv = pd.DataFrame({"profile_id": [2, 3, 4], "cv_fold": [1, 2, 3]})
    study = DevelopmentStudy(pd.concat(frames), cv, strict=False)
    a = study.evaluate_full_ols_cv([10, 50, 200, 1000])
    b = reference_functions(study)["evaluate_full_ols_cv"]([10, 50, 200, 1000])
    pd.testing.assert_frame_equal(
        a["fold_results"], b["fold_results"], check_exact=True
    )


def test_development_refuses_heldout_or_incompatible_partition(profile):
    cv = pd.DataFrame({"profile_id": [2], "cv_fold": [1]})
    with pytest.raises(ValueError):
        DevelopmentStudy(profile, cv)
    bad = profile.copy()
    bad.profile_id = 13
    with pytest.raises(ValueError, match="Held-out"):
        DevelopmentStudy(
            bad, pd.DataFrame({"profile_id": [13], "cv_fold": [1]}), strict=False
        )


def test_original_budget_and_gp_calls_are_present():
    nb = json.loads((ROOT / "notebooks/02_Experimental_Development.ipynb").read_text())
    lookup = {}
    for cell in nb["cells"]:
        if (
            cell["cell_type"] == "code"
            and cell["metadata"].get("source_notebook") == "fresh_reconstruction.ipynb"
        ):
            key = cell["metadata"]["source_cell"]
            lookup[key] = lookup.get(key, "") + "\n" + "".join(cell["source"])
    for cell in [53, 57, 67, 80, 93, 95, 102]:
        original = json.loads(
            (
                ROOT / "provenance/source_notebooks/fresh_reconstruction.ipynb"
            ).read_text()
        )

        class WithoutFunctionDocs(ast.NodeTransformer):
            def visit_FunctionDef(self, node):
                self.generic_visit(node)
                if (
                    node.body
                    and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)
                ):
                    node.body.pop(0)
                return node

        assert ast.dump(
            WithoutFunctionDocs().visit(ast.parse(lookup[cell])),
            include_attributes=False,
        ) == ast.dump(
            WithoutFunctionDocs().visit(
                ast.parse("".join(original["cells"][cell]["source"]))
            ),
            include_attributes=False,
        )
    assert protocol()["scaled_models"]["train_per_profile"] == {
        "SVR": 320,
        "k-NN": 640,
        "Random Forest": 640,
        "Extra Trees": 640,
        "MLP": 640,
    }
    assert protocol()["stage5"]["repeats"] == 3 and protocol()["stage5"][
        "training_per_profile"
    ] == [80, 160, 320, 640]
    assert "STAGE9_SPANS = tuple(int(_stage9_best" in lookup[123]
    assert "candidate_spans = list(map(int, best_spans))" in lookup[89]


def test_final_source_only_uses_required_handoff():
    nb = json.loads(
        (ROOT / "notebooks/01_Final_Reproducible_Evaluation.ipynb").read_text()
    )
    code = "\n".join(
        "".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"
    )
    assert "frozen_configurations" not in code
    assert "load_manifest(" in code and "configuration_agreement(" in code
    assert code.index("load_manifest(") < code.index("pipe.fit(")
    for line in code.splitlines():
        if "build_pipeline(name" in line:
            assert "SELECTED" in line


def test_larger_knn_objective_matches_original_at_unreduced_budgets(profile):
    """Use three synthetic sessions, retaining 640 fitting/validation rows each."""
    frames = []
    for i, pid in enumerate([2, 3, 4]):
        frame = pd.concat([profile, profile], ignore_index=True)
        frame.index += i * len(frame)
        frame.profile_id = pid
        frame["pm"] += i
        frames.append(frame)
    study = DevelopmentStudy(
        pd.concat(frames),
        pd.DataFrame({"profile_id": [2, 3, 4], "cv_fold": [1, 2, 3]}),
        strict=False,
    )
    parameters = [10, 40, 150, 800, 12, "distance"]
    actual = study.evaluate_larger_model_cv("k-NN", parameters)
    expected = reference_functions(study)["evaluate_larger_model_cv"](
        "k-NN", parameters
    )
    np.testing.assert_allclose(
        actual["fold_scores"], expected["fold_scores"], rtol=1e-12, atol=1e-10
    )
    assert (
        actual["spans"] == expected["spans"]
        and actual["hyperparameters"] == expected["hyperparameters"]
    )
    assert study.train_samples["k-NN"] == study.validation_samples == 640


def test_actual_notebook_nystroem_factory_reconstructs_matched_probe_kernel():
    """Exercise the actual Stage 9 factory on an explicitly synthetic tiny probe."""
    from sklearn.kernel_approximation import Nystroem
    from sklearn.metrics.pairwise import rbf_kernel
    from sklearn.svm import LinearSVR

    notebook = json.loads(
        (ROOT / "notebooks/02_Experimental_Development.ipynb").read_text()
    )
    functions = {}
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        for node in ast.parse("".join(cell["source"])).body:
            if isinstance(node, ast.FunctionDef):
                functions[node.name] = node
    rng = np.random.default_rng(42)
    x = StandardScaler().fit_transform(rng.normal(size=(32, 108)))
    gamma = 1 / (x.shape[1] * x.var())
    namespace = dict(
        Pipeline=Pipeline,
        Nystroem=Nystroem,
        LinearSVR=LinearSVR,
        C_REF=0.2,
        EPSILON_REF=0.1,
        STAGE9_SEED=42,
        STAGE9_GAMMA=gamma,
    )
    for name in ["make_linear_svr", "make_nystroem_svr"]:
        exec(
            compile(
                ast.Module(body=[functions[name]], type_ignores=[]),
                "actual_notebook_factory",
                "exec",
            ),
            namespace,
        )
    pipeline = namespace["make_nystroem_svr"](32)
    mapped = pipeline.named_steps["nystroem"].fit_transform(x)
    exact = rbf_kernel(x, gamma=gamma)
    error = np.linalg.norm(exact - mapped @ mapped.T, "fro") / np.linalg.norm(
        exact, "fro"
    )
    assert error < 1e-5
    assert pipeline.named_steps["linear_svr"].C == 0.2
    assert pipeline.named_steps["linear_svr"].epsilon == 0.1
