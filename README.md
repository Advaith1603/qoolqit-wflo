# QoolQit contest: wind farm layout optimisation on simulated Rydberg atoms

The deliverable is one self-contained notebook, `wflo_rydberg.ipynb` (paired with `wflo_rydberg.py` via jupytext).
`submission/wflo_rydberg.ipynb` is the executed copy (demo mode, run in a fresh environment) that is submitted.

## Notebook outputs and nbstripout

Notebook outputs are stripped on commit by nbstripout, configured in `.git/info/attributes`. The submitted copy must keep
its outputs, so that file also contains this exemption:

```
submission/*.ipynb -filter -diff
```

`.git/info/attributes` is local to this clone and is **not committed**. On any new clone, run `nbstripout --install`
and re-add the line above before re-committing `submission/wflo_rydberg.ipynb`; otherwise its outputs are stripped
silently. Check with `git check-attr filter -- submission/wflo_rydberg.ipynb` (expected: `filter: unset`).

## Other files

- `PLAN_FILE.md`: project plan and status (the notebook is the source of truth).
- `dev/`: Phase 0 scripts, wake-rule checks, the draft LocalEmulator issue, and `check_placeholders.py`
  (run before every submission build: `python dev/check_placeholders.py wflo_rydberg.ipynb submission/wflo_rydberg.ipynb`).
- `runs/` (gitignored): logs and the incremental output of full runs.
