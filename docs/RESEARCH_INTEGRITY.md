# Reproducibility and limitations

This repository contains a **research manuscript draft**, not an accepted or externally peer-reviewed publication.

The core theorem assumes an independently trained fixed score, conditional score-distribution stability within latent groups, a known invertible misclassification channel independent of scores conditional on latent group, and *known* source and target latent mixture weights.

The four public ML dataset evaluations draw **with replacement** from held-out finite empirical score populations; they are conditional-on-fixed-pool experiments rather than independent deployment or clinical validation. Ordinary split experiments estimate the source group distribution and therefore do not satisfy the known-prior assumption of the theorem.

The method may return an infinite threshold (vacuous certificate) as group-proxy noise rises or samples become insufficient.

Source algorithms combine classical matrix inversion and DKW uniform concentration; journal-level novelty and superiority to leading related methods have not been independently certified.

Generative AI assisted research workflow, implementation and drafting. The author must approve scientific claims and the journal-required disclosures before submitting.
