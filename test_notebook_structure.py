"""Check educational hierarchy and preservation of all partitioned source code."""

import ast
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re

import nbformat
import pytest

ROOT = Path(__file__).resolve().parents[1]


def canonical_code(source):
    """Rejoin only consecutive partitions of the same invariant run guard."""
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


@pytest.mark.parametrize("number", ["01", "02"])
def test_chapterisation_preserves_every_original_statement(number):
    name = (
        "01_Final_Reproducible_Evaluation"
        if number == "01"
        else "02_Experimental_Development"
    )
    n = nbformat.read(ROOT / "notebooks" / f"{name}.ipynb", as_version=4)
    groups = defaultdict(list)
    for c in n.cells:
        if c.cell_type == "code" and "parent_cell_id" in c.metadata:
            groups[c.metadata.parent_cell_id].append(c)
    audit = json.loads((ROOT / "provenance/chapterisation_code_audit.json").read_text())
    records = [r for r in audit if r["notebook"] == number]
    assert list(groups) == [r["parent_cell_id"] for r in records]
    for record in records:
        cells = groups[record["parent_cell_id"]]
        assert [c.id for c in cells] == record["output_cell_ids"]
        assert [c.metadata.code_part for c in cells] == list(
            range(1, record["parts"] + 1)
        )
        code = "\n".join(c.source for c in cells)
        digest = hashlib.sha256(canonical_code(code).encode()).hexdigest()
        assert digest == record["canonical_ast_sha256"], record["parent_cell_id"]


@pytest.mark.parametrize(
    "name", ["01_Final_Reproducible_Evaluation", "02_Experimental_Development"]
)
def test_numbered_hierarchy_and_five_part_experiment_explanations(name):
    n = nbformat.read(ROOT / "notebooks" / f"{name}.ipynb", as_version=4)
    nbformat.validate(n)
    chapters = []
    sections = {}
    subsections = defaultdict(list)
    active_chapter = None
    for c in n.cells:
        if c.cell_type == "code":
            compile(c.source, str(ROOT / "notebooks" / f"{name}.ipynb"), "exec")
            # Legitimate saved Run All outputs must remain compatible with tests.
            assert isinstance(c.outputs, list)
            assert c.execution_count is None or (
                isinstance(c.execution_count, int) and c.execution_count > 0
            )
            continue
        for line in c.source.splitlines():
            m = re.match(r"^(#{2,4}) (\d+(?:\.\d+)*)(?:\.)? (.+)$", line)
            if not m:
                continue
            level, number, title = m.groups()
            parts = tuple(map(int, number.split(".")))
            if len(level) == 2:
                assert len(parts) == 1
                chapters.append(parts[0])
                active_chapter = parts[0]
            elif len(level) == 3:
                assert len(parts) == 2 and parts[0] == active_chapter
                assert parts not in sections
                sections[parts] = title
            elif len(level) == 4:
                assert len(parts) == 3 and parts[:2] in sections
                subsections[parts[:2]].append(parts[2])
    assert chapters == list(range(1, max(chapters) + 1))
    for key, values in subsections.items():
        assert values == [1, 2, 3, 4, 5], (key, values)
    assert len(subsections) > 20 if name.startswith("02") else len(subsections) > 15


def test_all_final_models_have_mathematics_and_evaluation_sections():
    n = nbformat.read(
        ROOT / "notebooks/01_Final_Reproducible_Evaluation.ipynb", as_version=4
    )
    text = "\n".join(c.source for c in n.cells if c.cell_type == "markdown")
    for i, model in enumerate(
        ["OLS", "k-NN", "Extra Trees", "MLP", "Random Forest", "LinearSVR"], 1
    ):
        assert f"### 6.{i}. {model}\n" in text
        assert f"### 9.{i}. {model} final evidence\n" in text
    for c in n.cells:
        if c.cell_type == "code" and c.metadata.get("presentation_only"):
            assert not any(
                isinstance(a, ast.Call)
                and isinstance(a.func, ast.Attribute)
                and a.func.attr in ["fit", "predict", "tell", "ask"]
                for a in ast.walk(ast.parse(c.source))
            )
