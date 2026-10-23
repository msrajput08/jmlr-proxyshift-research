# ProxyShift: coverage certification under imperfect group proxies

Reproducible experiments and research software for the manuscript **Coverage Certification Under Population-Mixture Shift from Imperfect Group Proxies**.

**Research status:** Working JMLR-targeted draft, not peer-reviewed, accepted or submitted. See [research notes](README_RESEARCH.md) and [scientific limitations](docs/RESEARCH_INTEGRITY.md).

## Current repository contents

- `src/`: certificate algorithm and Gaussian / real-data experiment scripts
- `tests/`: structural and numerical validation tests
- `paper/jmlr2e.sty`: official JMLR style file
- `paper/*_table.tex`: measured summary tables
- `.github/workflows/reproduce.yml`: experiment reproduction and testing workflow

The **full manuscript LaTeX/PDF, historical CSV records, and rendered graphics have not yet all been mirrored into this repository**. These are in the separate complete project ZIP supplied in the research conversation; use that as the source of truth until the mirror is complete.

## Reproduce data and tests

Use Python 3.11+:

```bash
python -m pip install -r requirements.txt
python src/experiment.py
python src/real_data.py
python src/known_prior_benchmark.py
python src/generate_tables.py
python src/create_flowchart.py
python -m pytest -q tests/
```

The workflow also runs these commands and archives the generated `results/` and `figures/`.

## Import the remaining manuscript and assets

From a clean local clone, extract the complete project ZIP into the repository root, preserving `paper/`, `results/`, `figures/`, `src/`, and `tests/`:

```bash
git clone https://github.com/msrajput08/jmlr-proxyshift-research.git
cd jmlr-proxyshift-research
# Copy the contents of the unzipped research package into this folder.
git add .
git commit -m "Add manuscript and complete reproducibility assets"
git push origin main
```

Only commit source data and research artifacts; avoid local environments, compiled auxiliary files or private credentials.

## Scientific scope

The finite-sample theorem assumes a known invertible proxy channel, known source and target mixture proportions, and score-distribution invariance within latent groups. The "real data" empirical-population benchmark uses replacement sampling from fixed pools; it is not evidence of field deployment generalization. See `docs/RESEARCH_INTEGRITY.md`.

AI-assisted research drafting and coding are disclosed. The named researcher must verify all claims and follow JMLR submission policies.
