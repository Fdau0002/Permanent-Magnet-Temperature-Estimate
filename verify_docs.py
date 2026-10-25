"""Inspect actual generated pdoc HTML and offline notebook previews."""

from collections import defaultdict
from html.parser import HTMLParser
import json
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


class Page(HTMLParser):
    """Collect local navigation targets, document anchors and MathML counts."""

    def __init__(self, text):
        super().__init__()
        self.links = []
        self.ids = set()
        self.math = 0
        self.titles = []
        self.feed(text)

    def handle_starttag(self, tag, attributes):
        attributes = dict(attributes)
        if "id" in attributes:
            self.ids.add(attributes["id"])
        if tag in ("a", "link", "script", "img"):
            value = attributes.get("href") or attributes.get("src")
            if value:
                self.links.append(value)
        if tag == "math":
            self.math += 1


def main():
    """Fail on missing generated pages, functions or local navigation targets."""
    files = list((ROOT / "docs/html").rglob("*.html")) + list(
        (ROOT / "docs/notebook_previews").glob("*.html")
    )
    if (ROOT/'docs/index.html').is_file():files.append(ROOT/'docs/index.html')
    pages = {path.resolve(): Page(path.read_text()) for path in files}
    errors = []
    checked = 0
    for file, page in pages.items():
        for link in page.links:
            url = urlsplit(link)
            if url.scheme or url.netloc:
                continue
            target = (file.parent / unquote(url.path)).resolve() if url.path else file
            if not target.is_file():
                errors.append({"page": file.name, "missing_target": link})
                continue
            checked += 1
            if (
                url.fragment
                and target in pages
                and unquote(url.fragment) not in pages[target].ids
            ):
                errors.append({"page": file.name, "missing_fragment": link})
    for module in (ROOT / "src/pmsm_project").glob("*.py"):
        expected = (
            ROOT
            / "docs/html"
            / (
                "pmsm_project.html"
                if module.stem == "__init__"
                else f"pmsm_project/{module.stem}.html"
            )
        )
        if not expected.is_file():
            errors.append({"missing_module_page": module.name})
    for module, function in [
        ("handoff", "start_development_run"),
        ("handoff", "load_manifest"),
        ("handoff", "write_handoff_archive"),
        ("checkpoints", "load_prediction_checkpoint"),
        ("study", "evaluate_full_ols_cv"),
        ("search", "protocol"),
    ]:
        text = (ROOT / f"docs/html/pmsm_project/{module}.html").read_text()
        if function not in text:
            errors.append({"missing_function": function})
    math = {
        p.name: page.math
        for p, page in pages.items()
        if p.parent.name == "notebook_previews"
    }
    if len(math) != 2 or not all(n > 50 for n in math.values()):
        errors.append({"mathml_previews_incomplete": math})
    result = {
        "scope": "generated-document structural/navigation/source-content checks",
        "pdoc_html_pages": len(list((ROOT / "docs/html").rglob("*.html"))),
        "notebook_preview_pages": len(math),
        "mathml_equations": math,
        "local_links_checked": checked,
        "errors": errors,
        "status": "passed" if not errors else "failed",
        "browser_pixel_review": "not completed: Chromium executable unavailable and download failed",
    }
    (ROOT / "docs/reports/html_verification.json").write_text(
        json.dumps(result, indent=2)
    )
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
