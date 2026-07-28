# humanpending.md

## OPEN — 1 of 20 baseline cells blocked by the recurring CUDA fault

**Blocked:** `aasist_raw` x MLAAD only. The other 19 (model, dataset) cells are complete.

The host CUDA fault documented in the closed section below recurred during the aasist_raw
sweep. It took out 6 datasets on the first pass (all recovered on retry), then went down
again for a continuous 60+ minutes during MLAAD, and the orchestrator gave up on that one
dataset. `torch.cuda.is_available()` is currently False while `nvidia-smi` reports three
idle, healthy A100s, which is the same signature as before.

**Nothing further is needed from a human unless the driver stays down.**
`watch_and_run_aasist_mlaad.sh` is armed in the background: it polls CUDA every 5 minutes
and scores the remaining cell automatically the moment the driver returns, then exits. To
run it by hand instead:

```bash
cd /home/alhashim/ASD_SUPERB/spoof_SUPERB
python orchestrate_baselines.py --models aasist_raw --datasets Multilingual
```

Consequence while blocked: the aasist_raw row in the paper has MLAAD, Mean and Pooled
withheld (Mean and Pooled are undefined without all ten datasets). The lfcc_gmm row is
complete.

**Root fix is still the one described in the closed section (needs root).**

---

## OPEN — Non-SSL baselines (LFCC-GMM + standalone AASIST): decisions for review

**Logged:** 2026-07-26. **Nothing here blocks the task**; all items were resolved by
first-principles judgment and execution continued. They are recorded because each is a
methodological choice a reviewer should be able to see and overturn before the numbers
go in the paper.

### 1. Trial lists are taken from the published `linear_head` score files, not re-derived
For all 10 sets, `eval_baselines.py` reads the eval list *and the ground-truth key* from
`/data/.../asd_superb_score_files/linear_head/linear_head_{token}_xls_r_300m.txt`.

Why: several published sets are subsets whose selection rule is recorded nowhere in the
repo — ASV21 DF is 152,955 of 611,829 protocol rows; ASVLD pools a noise x10 / reverb x3 /
resample x4 slice (1,207,509 of 2,065,873); Famous Figures is 346,471 of 348,135; DFEval24
is 1,976 rows matching neither on-disk metadata file (377 / 1,980). Re-deriving from raw
protocols would silently score a *different* trial set, and the baseline EERs could not be
placed in the same table as the SSL models. This also matches the `--restrict_to`
convention already used by `eval_asvld.py` / `eval_mlaad.py`.

**Review point:** if any published subset is considered wrong, the baselines inherit that
same wrongness by construction. That is deliberate — comparability was ranked above
independent re-derivation. Change `--reference_ssl` / `--reference_file` to re-point.

### 2. Famous Figures bonafide path remap (`/-/` -> `/Bonafide/`)
Reference ids encode the empty Source field as a literal `-` directory, but on disk those
files live under `{Speaker}/Bonafide/`. Verified: all 49,945 `/-/` rows are key=bonafide
and all 49,945 resolve after the remap. **Without it the dataset scores zero bonafide
trials and its EER is undefined** — worth a sanity check by someone who knows the dataset.

### 3. LFCC front-end vendored instead of adding `spafe`
`spafe` is not installed in the `spoof_SUPERB` env. The three helpers on the LFCC path
were reimplemented in `lfcc_frontend.py` and verified equal to genuine spafe 0.3.3 (from
the `SER` env) on real ASV19 audio: max |diff| 7.6e-14. Adding a dependency was the
alternative; it would have needed authorization and pins the benchmark to a package for
~40 lines of code.

Note this preserves one spafe quirk verbatim: the filterbank edges are laid on
`linspace(low_freq, high_freq, nfft//2+1)` rather than true rfft bin frequencies, so with
`high_freq=4000` the bank is stretched over the 0-8 kHz axis. **This is what the reference
LFCC-GMM was trained under**, so it was reproduced rather than "fixed". A reviewer may
reasonably want the corrected version — that is a different system, not a bug fix.

### 4. LFCC parameters follow the reference, not the classic ASVspoof baseline
`num_ceps=20`, `nfilts=70`, `low_freq=0`, **`high_freq=4000`**, **`win_len=30 ms`,
`win_hop=15 ms`**, `nfft=1024`, no pre-emphasis, +delta+delta-delta = 60 dims — taken
verbatim from `Rob-ASD/ASD_ML`. The classic ASVspoof2019 LFCC baseline instead uses
20 ms / 10 ms over the full 0-8 kHz band. Reproducing the author's existing system was
ranked higher than matching the literature default. **Review point:** discarding the
4-8 kHz band is a real modelling choice for deepfake detection, where high-band artefacts
are informative.

### 4b. [RESOLVED — recorded because the expected range in the brief was wrong]
The task's sanity band for LFCC-GMM on ASV19 LA eval was "roughly 6-12%", derived from
the classic ASVspoof2019 LFCC-GMM baseline (8.09%). The trained model scores **3.70%**,
outside that band, so it was investigated rather than reported.

Validation (executed): the reference repo ships its own pre-trained GMM from Jan 2024
(`Rob-ASD/ASD_ML/gmm_512_LA_lfcc/`). Scored through this repo's pipeline on the identical
71,237 trials it gives **3.78%** — agreement with our freshly trained model to 0.08 pp.
Score orientation and separation also check out (bonafide mean +3.43 vs spoof -1.60, zero
non-finite values).

Conclusion: 3.70% is correct for THIS LFCC configuration; the 6-12% expectation belonged
to the classic 20 ms / 10 ms full-band baseline, not to the 30 ms / 15 ms, 0-4 kHz
configuration this benchmark's reference implementation uses (see item 4). No action
needed — logged so nobody later "corrects" a number that is right.

### 5. GMM init stride is floored (not a flat every-20th)
The reference's flat stride-20 init assumes its larger pooled training set. On ASV19 LA
train alone it would fit 512 components to 129 bonafide utterances (~55 frames/component),
which collapses components. `effective_init_stride` floors the stride so the init sees
>=1000 utterances: bonafide stride 2 (1,290 utts), spoof stride 20 (1,140 utts). Streaming
EM afterwards still uses **all** utterances, exactly as the reference.

### 6. Pre-existing quirks in the shared training recipe were NOT corrected
AASIST-raw trains through `main.py` unchanged, which means it inherits two oddities that
every SSL model in this benchmark was also trained under:
  - `main.py` builds an Adam at `args.lr` (default 1e-6) and then **overwrites it** with
    `create_optimizer(...)` from `configs/AASIST.conf`, so the effective LR is
    `base_lr = 1e-4`, not `--lr`.
  - the cosine `scheduler` is created but **never stepped**, so the LR is constant.

Both are arguably bugs. They were left alone deliberately: "fixing" them for the baseline
only would mean the baseline was trained under a different recipe than the 24 SSL models
it is meant to be compared against. **Review point:** if these get fixed, they must be
fixed for every model and the whole table re-run.

### 7. AASIST-raw uses gradient accumulation to hold the SSL batch size of 64
The SSL models were trained at `--batch_size 64` (visible in their checkpoint dir names,
`model_weighted_CCE_50_64_linear_head_ASV19_*`). AASIST-raw **cannot** run at batch 64 on a
40 GB A100: at raw-waveform resolution its first residual block holds a
(B, 32, 24, 21490) activation, 4.2 GB at B=64, and the first attempt died with CUDA OOM
after ~38.8 GB. The SSL models avoid this only because their upstream downsamples to ~200
frames before the graph back-end.

Rather than quietly training the baseline at a smaller batch, `main.py` gained a
`--micro_batch` flag (default 0 = disabled, so SSL behaviour is untouched): the loader
runs at 16 and gradients accumulate over 4 chunks, giving the same effective batch of 64,
the same LR and the same number of optimizer steps.

Subtlety worth a reviewer's eye: because the loss is `CrossEntropyLoss(weight=[0.1,0.9])`
with reduction='mean', it normalises by the **sum of sample weights**, not the sample
count — so dividing each micro-batch loss by `accum_steps` is wrong whenever micro-batches
differ in class balance. The implementation accumulates the weighted *sum* and rescales
the group's gradients by the group's total sample weight. `tests/test_grad_accum.py`
pins both properties: accum_steps=1 is bit-identical to the original loop (max|dW| = 0.0),
and micro16 x accum4 matches batch64 to 1.5e-08 (float32 eps ~1.2e-7), with a guard
asserting that un-accumulated batch-16 does NOT match (so the test is not vacuous).

### 8. [NEEDS A GLOBAL FIX] `best_val_eer = 1` silently prevents checkpoint saving
`main.py` initialises `best_val_eer = 1` and compares it against
`calculate_EER(...)`, which returns a **percentage** (`eer_cm * 100`), not a fraction.
So `if dev_eer < best_val_eer` is false for any model whose dev EER is above 1%, and
**nothing is saved** — no `epoch_*.pth`, and because `n_swa_update` stays 0, no `swa.pth`
either. A 50-epoch run can complete and leave an empty checkpoint directory.

Caught in flight: AASIST-raw's epoch 0 gave dev EER 24.53% and wrote no checkpoint.

Fix applied here is deliberately narrow — `best_val_eer = float('inf')` **for the two
non-SSL baselines only** — to respect this task's "do not change existing SSL behaviour"
constraint. **This is not the right long-term fix.**

**Human action wanted:**
  - Correct the comparison globally (either init to `float('inf')` or divide by 100).
  - Check whether any SSL model in the benchmark was affected. Their checkpoint dirs
    under `asd_superb_models/linear_head_models/` contain **only `swa.pth`** and no
    `epoch_*.pth`, which is consistent with either (a) manual cleanup, or (b) those
    models having been trained by different code. If any SSL model's dev EER never fell
    below 1%, its saved weights are not what the training loop intended.

### 8b. AASIST-raw is scored from the best epoch checkpoint, NOT from swa.pth
The SSL models in this benchmark are all scored from `swa.pth`. AASIST-raw deliberately
is not. Measured on clean ASV19 LA dev (`tools_select_aasist_ckpt.py`, dev only -- no
evaluation set influenced the choice):

| checkpoint | dev EER |
|---|---|
| swa.pth | 1.806 % |
| **epoch_44_1.178.pth** | **0.670 %**  <- used |
| epoch_31_1.254.pth | 1.178 % |
| epoch_28_1.413.pth | 1.293 % |

Cause, and it interacts with item 8: `main.py` calls `optimizer_swa.update_swa()` on every
dev-EER improvement. Under the SSL runs' `best_val_eer = 1`, that fires only for
checkpoints already below 1% EER, so their SWA averages a set of already-good models.
The baselines had to initialise to `inf` (or nothing is ever saved), so SWA here averaged
the entire trajectory *starting at epoch 0 with 24.5% dev EER* — producing an average
2.7x worse than the best single checkpoint.

**Review point:** if item 8 is fixed globally, the natural fix is to gate `update_swa()`
on a sensible EER threshold (or a burn-in epoch) rather than on `best_val_eer`'s initial
value. Until then, "score from swa.pth" is only safe for models that actually get below
1% dev EER.

### 9. LFCC-GMM runs on CPU, not on a second GPU
The task asked for the two baselines on two different GPUs. LFCC-GMM is EM over diagonal
GMMs — a BLAS workload with no GPU path in the reference implementation. It runs on CPU
(concurrently with AASIST-raw on GPU 0), so GPU 1 is idle. Reporting it as a "GPU run"
would have been false. Throughput was instead fixed where it actually mattered: pinning
BLAS threads inside the worker pool took scoring from 20 to 1,160 utt/s (58x).

---

## CLOSED for this run — [RECURRING] CUDA driver init failure comes and goes on this host

**Run outcome: all 24 models completed and verified (24/24 PASS) despite two outages.**
The driver recovered on its own again at ~02:12; the parked queue resumed unattended
and every model that the first outage had burned was re-run successfully.

**Still worth a human fix eventually**: the fault recurred twice in 30 minutes and
was never actually repaired, so it will interrupt future long GPU jobs on this host.
Nothing in this task is blocked on it any more.

**Logged:** 2026-07-22 01:35 · briefly cleared 01:38 · recurred 01:56 · cleared 02:12

This is intermittent, not one-off. Timeline:
- 01:35 `cuInit -> 3` host-wide (diagnosis below)
- 01:38 cleared on its own; canary xls_r_300m ran clean on GPU 0 (16m20s, spearman 1.0000)
- 01:56 fault returned mid-batch and took out 6 models in ~3 minutes (all rc=2, no output)
- 02:05 still down; the run is now parked waiting for it

**Impact is contained**: `orchestrate_spoofceleb.py` now gates every launch on a real
CUDA health probe and holds a model on the queue through an outage instead of failing
it (see MAX_ATTEMPTS / CUDA_WAIT_S). The batch resumes by itself the moment the driver
returns and skips already-complete outputs, so no work is lost or repeated. It gives up
only if the driver stays down for 3 continuous hours.

**Human action still wanted**: the underlying driver fault is unfixed and will keep
interrupting long GPU jobs on this host, not just this one. Root fix needs root:

Note it self-cleared once at 01:38 without anyone running the fix, so root cause
stays **unconfirmed** — a transient driver/UVM state rather than a hard fault. The
module reload below is still the standard remedy if it stays down.

<details><summary>Original report (kept for diagnosis)</summary>

**Blocked:** the entire SpoofCeleb scoring run (all 24 models).

### What is wrong
`cuInit()` returns error 3 (`CUDA_ERROR_NOT_INITIALIZED`) for every process on this
host, so `torch.cuda.is_available()` is `False` and no GPU work can start.

### Evidence (all executed)
| Check | Result |
|---|---|
| `nvidia-smi` | healthy: 3x A100-PCIE-40GB, 0 MiB used, 0 ECC errors, compute mode Default |
| `cuInit(0)` via ctypes | **rc=3 "initialization error"**, `cuDeviceGetCount` rc=3 |
| Kernel modules | `nvidia`, `nvidia_uvm`, `nvidia_modeset`, `nvidia_fs` all loaded |
| Driver vs userspace | both 575.57.08 — no version mismatch |
| `/dev/nvidia*`, `/dev/nvidia-uvm` | present, world-rw, majors match `/proc/devices`, open() succeeds |
| Kernel log | no NVRM/Xid errors since boot |
| Clean env (`env -i`) | still rc=3 — not an env var, not `CUDA_VISIBLE_DEVICES` |
| Claude Code sandbox off | still rc=3 — not a sandbox restriction |
| Last known-good GPU run | Jul 21 15:29 (MLAAD logs), same boot — uptime 3 days, no reboot |

So the driver worked yesterday and degraded since, with no reboot and no visible
hardware fault. This is the classic stale-`nvidia_uvm` state.

### Fix required (needs root — `sudo` here demands a password)
```bash
sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm      # try this first
# if rmmod reports the module is in use:
sudo fuser -k /dev/nvidia*        # kills GPU holders — check who else is on the box first
# last resort:
sudo reboot
```
Confirm recovery with:
```bash
/home/alhashim/miniconda3/envs/spoof_SUPERB/bin/python -c "import torch;print(torch.cuda.is_available())"
```

### Why this was logged as human-gated rather than something I resolved
Reloading a kernel module is host-wide and affects any other user's jobs — it is
outside the authorized blast radius for this task (which forbids env changes and
package installs), and it needs root regardless.

### What happens once it is fixed
`watch_and_run_spoofceleb.sh` is armed in the background: it polls CUDA every 60 s
and launches the full 24-model batch automatically the moment the driver recovers.
Nothing further is needed. To run it by hand instead:
```bash
cd /home/alhashim/ASD_SUPERB/spoof_SUPERB
/home/alhashim/miniconda3/envs/spoof_SUPERB/bin/python orchestrate_spoofceleb.py
```

</details>

---

## Resolved / not gated
Everything not requiring a GPU is complete and verified by execution — protocol
parsing, utt_id + label fidelity against the reference (exact match on all 91,130),
end-to-end score-file production, and all three verifier verdict paths. See the
run summary for details.

### 10. [DECIDED] Degenerate LLASA generations are dropped from the LFCC-GMM column
First recorded as "322 Famous Figures files fail to decode". **That was wrong** and is
corrected here. The files decode perfectly. They are 10 to 20 ms long with peak amplitude
around 6e-05 (inaudible), i.e. failed TTS generations, and being shorter than the LFCC
30 ms analysis window they yield zero frames, so the GMM has nothing to score.

**They are not short speech.** Famous Figures overall has a median duration of 7.93 s
(mean 7.83 s, range 0.22 to 32.12 s) and 0.00% of it is under 30 ms. The 322 affected
files are 0.09% of the dataset, and **320 of them are LLASA**:

| speaker | LLASA files | < 30 ms | rate |
|---|---|---|---|
| Joe_Biden | 1,947 | 153 | **7.86%** |
| Kamala_Harris | 4,455 | 63 | 1.41% |
| Vivek_Ramaswamy | 2,368 | 58 | 2.45% |
| Barack_Obama | 2,055 | 29 | 1.41% |
| Donald_Trump | 866 | 13 | 1.50% |
| Elon_Musk / Anthony_Blinken / Tim_Walz | 2,063 | 4 | <0.5% |
| JD_Vance / Mathew_Miller | 602 | 0 | 0% |
| **LLASA total** | **14,356** | **320** | **2.23%** |

Spread across 8 of the 10 speakers, so it is a systematic LLASA failure mode rather than
a per-speaker data problem. Joe_Biden is a 5.6x outlier worth investigating separately.

**Author decision (2026-07-27): drop them.** Tiling them up to a scoreable length was
implemented and then reverted, because it manufactures a score from a fragment carrying
no usable signal. `lfcc_gmm.load_lfcc` therefore applies no padding.

**Consequence to disclose:** the LFCC-GMM column covers 346,149 of 346,471 Famous Figures
trials and 1,040,004 of 1,040,006 MLAAD trials, while the SSL and AASIST columns cover
all of them (their front-ends tile every waveform to 4 s, so these 322 files ARE scored
in every other row, on tiled near-silence). The EER effect is nil at three decimals.

**Wider issue for the paper:** those same 320 LLASA files are being scored as genuine
LLASA samples in all 19 SSL columns. Any per-system analysis that reports LLASA
separately is attributing 2.23% inaudible fragments to LLASA's detectability, and nearly
8% for Joe_Biden. Cleaning that would require excluding them from the FF protocol for
every model and re-scoring the SSL columns.

---

## Reorganisation (2026-07-28) -- deferred items

Raised during the flat-to-package reorganisation. None is a refactor blocker;
each needs a decision that changes results or touches data.

### RP-1  Two conda environments disagree on `soxr` -- BLOCKING for provenance
`soxr` is librosa's resampler, and `verify_mlaad.py` already attributed the
reference-vs-rerun logit offset to exactly this class of drift.

    spoof_SUPERB : librosa 0.11.0  soxr 1.0.0        numpy 2.2.6  torch 2.7.1+cu126
    ASD_SUPERB   : librosa 0.11.0  soxr 0.5.0.post1  numpy 2.2.6  torch 2.7.1+cu126

Before the reorg, `run_asvld_model.sh` and `run_recompression.sh` launched the
ASD_SUPERB interpreter while the MLAAD/SpoofCeleb orchestrators launched
spoof_SUPERB. **Score files for any dataset whose audio is resampled to 16 kHz
were therefore produced by two different resamplers.** SpoofCeleb is natively
16 kHz and unaffected.

The reorg removed the four hardcoded interpreter paths -- everything now uses
`cfg.python`, defaulting to the running interpreter -- so new runs are at least
self-consistent. It does NOT retroactively fix existing score files.

Decision needed: pin one environment, then decide whether any published column
must be re-scored under it. Quantify first by re-scoring one model on one
resampled dataset under each interpreter and comparing.

### RP-2  `create_combined_mlaad_meta.py` still drops rows
`analysis/create_combined_mlaad_meta_all.py` sets `csv.field_size_limit` and
`QUOTE_NONE` with a comment naming the failure it fixes: `ja/kokoro` silently
loses 53 of 1000 rows under default quoting. Its older sibling
`analysis/create_combined_mlaad_meta.py` still uses plain `delimiter="|"` and
still has the bug.

Its output is consumed as the MLAAD protocol by `analysis/organize_tts_scores.py`
and `analysis/verify_tts_protocols.py`.

Decision needed: fixing it could change downstream results, so it wants its own
change with a before/after comparison rather than being folded into a
structural refactor.

### RP-3  Is `Filtering` still meant to be skipped?
The untracked `.asvld_skip` sentinel contained `Filtering`, so every ASVLD run
silently no-op'd that condition. It is now
`scoring/driver.py::DEFAULT_SKIP_CONDITIONS`, visible and test-pinned, with the
same content -- current behaviour preserved exactly.

`Filtering` is excluded from the published ASVLD column, so this is probably
correct. Confirm, and if it should be scored, drop it from the default.

### RP-4  Duplicated aggregation between the two TTS matrix builders
`analysis/compute_far_matrix.py` and `analysis/compute_eer_tts.py` are the same
program over the same 56-system tree; their `pooled_*`, `*_by_tag` and
`build_*_matrix` functions are line-for-line equivalent, differing only in the
inner metric call. FAR itself has no home in `core/metrics.py` -- it is defined
inside the matrix builder.

Not done here: merging them changes figure-producing code, which is a separate
blast radius from the structural reorg. Note also that "Overall Mean" in both is
a mean of per-system means, not a pooled recomputation over utterances.

### RP-5  Score directory `/data/ssl_anti_spoofing/asd_superb_score_files`
Audited but deliberately untouched. 49 GB, 32 top-level entries, 2,509 score
files, zero symlinks; roughly 19 GB is duplicated or regenerable (every "view"
is a physical copy -- `scores_by_acoustic_degradation/Reverberation` is
byte-identical to `scores_by_category_augmented/Reverberation`). `asv19`,
`asv5`, `linear_head` and `scores_by_category` are owned by root or by `adupa`
and must not be moved.

Proposed layout and migration constraints are in the reorg discussion; the work
is deferred by request.
