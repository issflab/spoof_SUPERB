# Internal design record

Working documents, not user documentation. Nothing here is needed to install,
reproduce, rebuild or train -- start at [../README.md](../README.md) for that.

They are kept because the reasoning behind several non-obvious choices lives
only here, and a reader who wants to overturn one of those choices should be
able to see what it was weighed against.

| File | What it is | State |
|---|---|---|
| [humanpending.md](humanpending.md) | decisions that were gated on a human, and defects deferred rather than fixed | **all closed** as of 2026-08-12 |
| [PLANNED_CHANGES.md](PLANNED_CHANGES.md) | P1-P19: every structural change, why it was made, and what was measured | complete |
| [REORG_PLAN.md](REORG_PLAN.md) | the Phase-0 audit the current layout came from | historical; names are as they were at audit time |
| [RESULTS_DELTA.md](RESULTS_DELTA.md) | what every published number moved by when the v3 tree replaced the old one, and why | the paper is already in sync; this is the revision record |

## No open defects

`humanpending.md` item 8 -- `best_val_eer = 1` compared against `calculate_EER()`,
which returns a percentage -- was fixed on 2026-08-12. It is now initialised to
`float('inf')` for every architecture, so the first epoch always writes a
checkpoint.

An audit against the published roster found no affected result: `swa.pth` is
written only when at least one epoch passed the same check that writes
`epoch_*.pth`, and all 21 models in the paper's roster have it. No re-run was
needed and no printed number changed. The reasoning and the measured dev EERs
are recorded at the top of `humanpending.md`.
