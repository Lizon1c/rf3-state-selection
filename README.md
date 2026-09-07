# Family-consensus state selection in an AF3-style protein structure network

Companion code, data, and manuscript for the paper:

> **Family-consensus state selection in an AF3-style protein structure network:
> the template channel dominates, the MSA selects, sequence barely matters**

## Headline findings

- Across 13 conformational-switching targets, the MSA in RF3 acts as a **family-consensus state
  selector**, not a crystallographic-bias agent: toward the crystal state on Mad2/XCL1, away from it
  on KaiB/RfaH, sharpening sequence-intrinsic preference on GA88/GB88, null on 8/13 targets.
- The input-channel hierarchy **inverts AF2's**: template (±4–5 log-density units) > MSA (±2.7) >
  sequence (±0.3). Templates win every conflict arm and can flip the GA88/GB88 fold-switch pair
  even against sequence+MSA. (Model-specific caveat: RF3's template track is ground-truth-trained.)
- "MSA ≈ crystallographic template" is falsified at three levels — output preference, representation
  geometry, and DMS function (template features are *harmful* for DMS, MSA beneficial).
- The homodimer cross-chain pair-block amplification by the MSA is **not** inter-chain coevolution:
  an unpaired (row-shuffled) MSA control reproduces the block statistics bit-for-bit.

## Layout

- `paper/` — manuscript (`paper.tex`, compiled `paper.pdf`, all figures regenerated from archived
  JSON by `make_figs.py`) and `TRACEABILITY.md`.
- `scripts/` — target runners/analyzers (E2 thirteen-target sweep, range-matched re-analysis,
  E49/E49b template arms, E3 block decomposition, template filtering).
- `inputs/` — curated target list (`e2_targets.json`), target MSAs, filtered template CIFs.
- `results/` — archived JSON behind every figure/table (range-matched rerun is canonical;
  pre-bugfix numbers are excluded).

## Reproduction notes

Scripts are archived as-run with original absolute paths. Preference scores are log-density
differences between the two conformational states (see Methods); null controls (no-MSA, no-template)
are included for every target.

## License

MIT (see LICENSE).
