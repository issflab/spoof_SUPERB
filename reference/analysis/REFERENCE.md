# Reference analysis tables

These are the tables `spoof_superb.verification analysis` checks a reproduction against.

- generated: 2026-08-13
- from score tree: `/data/ssl_anti_spoofing/spoof_superb_score_files`
- built from outputs: `/tmp/claude-114240904/-home-alhashim-ASD-SUPERB/8d041cb2-7b18-468b-8ad2-3f4dace39e7b/scratchpad/refsrc`
- repo commit: `816a898cfe7df5fc52abb8209ac638e9744aeb2a`

Reproduce them with:

```bash
python -m spoof_superb.analysis.recompute_main_results
python -m spoof_superb.analysis.acoustic_degradation
python -m spoof_superb.analysis.tts_systems
python -m spoof_superb.verification analysis
```

The score files behind them are indexed in `../manifest.json` with a sha256 each, and fetched with `bin/fetch_release.sh`.

> degradation/ is now the composition- and coverage-matched analysis; the earlier unmatched module was removed.
