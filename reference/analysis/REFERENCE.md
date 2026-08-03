# Reference analysis tables

These are the tables `spoof_superb.verification analysis` checks a reproduction against.

- generated: 2026-08-03
- from score tree: `/data/ssl_anti_spoofing/spoof_superb_score_files` (layout `v3`)
- built from outputs: `spoof_superb_outputs`
- repo commit: `c4544ea6196f913494ab2eed55480b4a29543eee`

Reproduce them with:

```bash
python -m spoof_superb.analysis.recompute_main_results
python -m spoof_superb.analysis.acoustic_degradation
python -m spoof_superb.analysis.tts_systems
python -m spoof_superb.verification analysis
```

The score files behind them are indexed in `../manifest.json` with a sha256 each, and fetched with `bin/fetch_scores.sh`.

> Level-2 reference for IEEE Access submission: v3 score tree, 19 SSL + 2 non-SSL rows.
