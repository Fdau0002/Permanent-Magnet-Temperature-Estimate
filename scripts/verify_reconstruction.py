"""Verify source, dependencies and small numerical identities; never run production BO."""

from __future__ import annotations
import ast
from collections import defaultdict
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def canonical_code(source):
    """Rejoin presentation-only partitions of unchanged run-flag conditions."""
    tree = ast.parse(source)
    statements = []
    for node in tree.body:
        if (
            statements
            and isinstance(node, ast.If)
            and isinstance(statements[-1], ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id.startswith("RUN_STAGE")
            and ast.dump(node.test) == ast.dump(statements[-1].test)
            and not statements[-1].orelse
        ):
            statements[-1].body.extend(node.body)
            statements[-1].orelse = node.orelse
        else:
            statements.append(node)
    tree.body = statements

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

    tree = WithoutFunctionDocs().visit(tree)
    import inspect

    options = {"include_attributes": False}
    if "show_empty" in inspect.signature(ast.dump).parameters:
        options["show_empty"] = True

    return ast.dump(tree, **options)


def main():
    """Write the actual verification outcome and fail on any inconsistent source."""
    import nbformat
    import numpy as np
    from scipy.linalg import qr, lstsq
    from packaging.requirements import Requirement
    from skopt import gp_minimize
    from skopt.space import Real
    from pmsm_project.handoff import software_versions

    result = {
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "source/dependency/synthetic-numerical smoke only; no production optimisation or final fit",
        "python": platform.python_version(),
        "checks": {},
    }
    for name in [
        "data",
        "features",
        "sampling",
        "validation",
        "numerics",
        "models",
        "metrics",
        "search",
        "study",
        "handoff",
        "checkpoints",
        "plotting",
    ]:
        importlib.import_module(f"pmsm_project.{name}")
    result["checks"]["all_production_modules_import"] = True
    versions = {}
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        requirement = Requirement(line)
        actual = importlib.metadata.version(requirement.name)
        if actual not in requirement.specifier:
            raise RuntimeError(
                f"{requirement.name}: require {requirement.specifier}, found {actual}"
            )
        versions[requirement.name] = actual
    result["checks"]["direct_dependency_versions"] = versions
    result["numerical_software"] = software_versions()
    pip = subprocess.run(
        [sys.executable, "-m", "pip", "check"], capture_output=True, text=True
    )
    if pip.returncode:
        raise RuntimeError(pip.stdout + pip.stderr)
    result["checks"]["pip_check"] = pip.stdout.strip()
    source_inventory = json.loads(
        (ROOT / "provenance/source_inventory.json").read_text()
    )
    for record in source_inventory:
        file = ROOT / "provenance/source_notebooks" / record["file"]
        assert hashlib.sha256(file.read_bytes()).hexdigest() == record["sha256"], file
    result["checks"]["historical_sources_unchanged"] = len(source_inventory)
    audit = json.loads((ROOT / "provenance/chapterisation_code_audit.json").read_text())
    details = []
    for name in ["02_Experimental_Development", "01_Final_Reproducible_Evaluation"]:
        nb = nbformat.read(ROOT / "notebooks" / f"{name}.ipynb", as_version=4)
        nbformat.validate(nb)
        groups = defaultdict(list)
        for cell in nb.cells:
            if cell.cell_type != "code":
                continue
            compile(cell.source, f"{name}:{cell.id}", "exec")
            if "parent_cell_id" in cell.metadata:
                groups[cell.metadata.parent_cell_id].append(cell)
            # Output cells are excluded from methodology identity deliberately.
            # This check also supports notebooks saved after the student's run.
            assert isinstance(cell.outputs, list)
        records = [r for r in audit if name.startswith(r["notebook"])]
        assert list(groups) == [r["parent_cell_id"] for r in records]
        for record in records:
            source = "\n".join(c.source for c in groups[record["parent_cell_id"]])
            assert (
                hashlib.sha256(canonical_code(source).encode()).hexdigest()
                == record["canonical_ast_sha256"]
            )
        details.append(
            {
                "notebook": name,
                "cells": len(nb.cells),
                "code_cells": sum(c.cell_type == "code" for c in nb.cells),
                "preserved_parent_code_cells": len(records),
            }
        )
    result["checks"]["notebook_syntax_structure_and_statement_order"] = details
    # Four calls on an analytic scalar quadratic check the installed GP/EI API.
    # This is not a reduced production experiment and cannot export a manifest.
    toy = gp_minimize(
        lambda x: float((x[0] - 0.25) ** 2),
        [Real(-1.0, 1.0)],
        n_calls=4,
        n_initial_points=2,
        acq_func="EI",
        random_state=42,
    )
    assert len(toy.func_vals) == 4 and np.isfinite(toy.func_vals).all()
    result["checks"]["synthetic_gp_dependency_smoke"] = {
        "calls": 4,
        "finite_objectives": True,
        "production_model_selection": False,
    }
    rng = np.random.default_rng(42)
    x = rng.normal(size=(120, 6))
    y = 1.5 + x @ rng.normal(size=6)
    a = np.column_stack([np.ones(len(y)), x])
    r = qr(np.column_stack([a, y]), mode="r")[0]
    compressed = lstsq(r[:, :-1], r[:, -1], lapack_driver="gelsd")[0]
    direct = lstsq(a, y, lapack_driver="gelsd")[0]
    error = float(np.max(np.abs(a @ compressed - a @ direct)))
    assert error < 1e-10
    result["checks"]["independent_synthetic_qr_max_prediction_difference"] = error
    result["status"] = "passed"
    dest = ROOT / "docs/reports/source_verification.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
