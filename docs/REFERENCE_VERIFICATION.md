# Bibliographic verification register

No third-party PDF is republished. Metadata are verified against the attached publisher paper and primary publisher, author, university or official software records. A dataset listing is the original depositor's source, not a secondary Kaggle summary.

| Key | Verification and use |
|---|---|
| `kirchgassner2021` | Principal benchmark; checked against all pages of supplied publisher PDF and Paderborn publication listing. This project adapts methods and uses a different public-data selection/partition. |
| `kirgsn_dataset` | Primary public dataset listing checked for provenance, sampling frequency and licence; exact bytes independently hash-verified from attachment. Automated public transfer timed out here. |
| `breiman2001` | Publisher bibliographic record checked; original Random Forest method. |
| `geurts2006` | Publisher record/PDF checked; original Extra Trees method. |
| `smola2004` | Publisher and author/university records checked; epsilon-SVR formulation and regularisation. |
| `williams2001` | Proceedings title/authors and author university record checked. Conference NIPS 2000; bound volume/publication commonly cited 2001. No DOI invented. |
| `snoek2012` | Official proceedings and author university metadata checked; conceptual GP-based Bayesian optimisation, not a claim that every variant in this paper was used. |
| `rumelhart1986` | Publisher and author publication records checked; established backpropagation algorithm. |
| `kingma2015` | Author arXiv record checked; distinguish 2014 preprint from ICLR 2015 conference citation. |
| `roberts2017` | Publisher and author records checked; rationale for blocking dependent observations, not source of this project-specific split. |
| `pandas_ewm` | Versioned official documentation checked for adjust=True, span/alpha definitions. See std page for bias=True. |
| `skopt_gp` | Official 0.10.2 API checked; actual pilot uses EI and four random initial evaluations. |
| `sklearn16` | Versioned official pages checked for stopping, scaling, intercept regularisation and kernel mapping. Runtime uses 1.6.1. |
| `scipy115` | Versioned official QR/lstsq pages checked; actual code uses r-mode QR and gelsd least squares. |
| `pdoc15` | Official documentation checked; no DOI or release year invented. |
| `monash2026brief` | Complete supplied 15-page institutional brief inspected; page-specific audit covers all requirements and rubric descriptors. No public URL/DOI invented. |

The notebook bracket-key mapping is explicit in REFERENCES.md. `fan2008` was additionally checked against the primary JMLR journal record (9(61), 1871–1874, 2008). It identifies the LIBLINEAR framework; regression/intercept specifics are attributed to the pinned scikit-learn LinearSVR implementation documentation.
