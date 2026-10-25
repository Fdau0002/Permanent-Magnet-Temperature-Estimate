# Integrate into an existing repository

The supplied ZIP is an integration package, not a clone of your current repository. Its structure cannot certify existing branches, commit history, permissions or publication. No Git operations were performed on your behalf.

1. Extract into a separate review directory. Keep the current repository and its historical originals intact.
2. Compare incoming paths with existing files. Reconcile `README.md`, `.gitignore`, `LICENSE`, `pyproject.toml`, dependency files, `src/pmsm_project/`, `tests/`, `docs/` and similarly named numbered notebooks manually. Do not replace an existing licence without checking ownership/licence compatibility.
3. Keep the two numbered notebooks and matching package source together. A development manifest hashes their code; changing them after optimisation requires a newly generated manifest.
4. Copy the selected reviewed files into your repository. Preserve existing files that are not part of this package. Keep the original CSV, models, caches and large regenerated outputs out of ordinary commits; `.gitignore` is a starting point to merge, not a command to remove files already tracked.
5. Run the documented tests/source checks and inspect generated `docs/html/index.html`. Save the executed notebooks and selected small evidence after your own Run All, with clear run provenance. Clear bulky outputs before committing only if you also retain an accessible executed evidence record.
6. Make your own meaningful commits describing restoration, handoff correctness, educational structure and verification. Review all staged content, paths and licensing before committing. No repository URL or credentials are required by this package.
7. Provide the required accessible repository link and zipped repository clone for Moodle when your own integration is complete. The delivered ZIP alone is not evidence of the required version-control history.

Files most likely to need manual reconciliation: top-level README/licence/dependency specifications; all pre-existing notebook names; an existing package named `pmsm_project`; test configuration; generated documentation paths; citation/AI-use policies. Avoid merging an older historical `config.py` that reintroduces frozen defaults into the corrected production package.
