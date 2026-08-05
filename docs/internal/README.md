# Internal design record

Working documents, not user documentation. Nothing here is needed to install,
reproduce, rebuild or train -- start at [../README.md](../README.md) for that.

They are kept because the reasoning behind several non-obvious choices lives
only here, and a reader who wants to overturn one of those choices should be
able to see what it was weighed against.

| File | What it is | State |
|---|---|---|
| [humanpending.md](humanpending.md) | decisions that were gated on a human, and defects deferred rather than fixed | **5 open**, the rest closed with evidence |
| [PLANNED_CHANGES.md](PLANNED_CHANGES.md) | P1-P19: every structural change, why it was made, and what was measured | complete |
| [REORG_PLAN.md](REORG_PLAN.md) | the Phase-0 audit the current layout came from | historical; names are as they were at audit time |
| [RESULTS_DELTA.md](RESULTS_DELTA.md) | what every published number moved by when the v3 tree replaced the old one, and why | the paper is already in sync; this is the revision record |

## The one live defect

`humanpending.md` item 8: `best_val_eer = 1` in `main.py` is compared against
`calculate_EER()`, which returns a percentage. An SSL model whose dev EER never
drops below 1% therefore trains for every epoch and saves no checkpoint at all
-- neither `epoch_*.pth` nor `swa.pth`. It is corrected for the two non-SSL
baselines and deliberately left in place for the SSL architectures, because
changing it means re-running any model that hit it.

Everything else open is a decision, not a bug.
