# Coverage Certification Under Population-Mixture Shift from Imperfect Group Proxies

**Version:** 23 October 2026. **Stage:** JMLR-style submission candidate requiring author review and independent novelty validation. **No submission or acceptance claimed.**

The paper is `paper/manuscript.pdf` (14 pages, JMLR official `jmlr2e.sty` with preprint flag to avoid fabricated post-acceptance headers). The human corresponding author details shown are Mohitsing Sanjaysing Patil, `rajputmohit0708@gmail.com`, independent researcher, Pune, Maharashtra, India. The author should supply a postal address through the submission workflow and verify these details.

## Contribution and limits

The paper identifies the shifted latent-group score distribution via known nonsingular proxy channel matrix inversion, develops a stratified DKW uncertainty envelope with a calibration-conditional guarantee, examines singularity and conservatism, and evaluates genuine experiments. The underlying statistical components have important antecedents; *publication-level methodological originality and comparative superiority have not been established*.

## Experiments

- `src/experiment.py`: Gaussian known-prior study (35 repetitions x 4 noise levels x 3 target mixtures, results in `results/monte_carlo.csv`).
- `src/real_data.py`: ordinary real-data train/calibration/test splits, exploratory only (estimated source prior **does not** meet theorem assumptions).
- `src/known_prior_benchmark.py`: four scikit-learn empirical score populations, source and target priors and proxy channel fixed by experimenter, 30 calibration repetitions; with-replacement sampling. This is a conditional-on-fixed-population benchmark, **not** 16,000 unique independent external clinical or deployment examples.
- `src/generate_tables.py`: reproduces three LaTeX tables from CSV. `figures/*.pdf` holds paper vector figures; `src/create_flowchart.py` and `src/experiment.py` regenerate four earlier figures. The known-prior chart is included as source-derived PDF/PNG and can be regenerated from its `results/known_prior_real_summary.csv`.

## Reproduction

Python 3.11+ and `pdflatex`/`latexmk` required. Execute from root:

```bash
python -m pip install -r requirements.txt
pytest -q tests/
python src/experiment.py
python src/real_data.py
python src/known_prior_benchmark.py
python src/generate_tables.py
python src/create_flowchart.py
cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error manuscript.tex
```

## Submission notes

- Official instructions: https://www.jmlr.org/author-info.html
- Official style: https://github.com/JmlrOrg/jmlr-style-file
- Our **preprint** option removes unassigned editor / fake publication fields; only the editor fills these for camera-ready acceptance. Do not submit the old PDF with `Editor: Not assigned`.
- JMLR requires 3–5 suggested action editors and 3–5 reviewers in a cover letter; author must confirm funding, conflicts, overlap, full contact and ethical/originality claims. A fact-checked candidate list is supplied under `docs/`.
- AI assistance was used in coding, reasoning and drafting and must not be falsely described as human-only. The author must personally stand behind every theorem and result.
- This repository does not claim that peer review, professional similarity screening, or independent novelty certification have occurred.