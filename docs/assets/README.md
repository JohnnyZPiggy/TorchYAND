# Figures and provenance

These three original SVG figures are documentation assets generated from the
mathematical specification, source layout, and committed experiment records.
They do not reproduce third-party figures.

| File | Grounding | Interpretation |
|---|---|---|
| [yand-architecture.svg](yand-architecture.svg) | [Design specification](../../TorchYAND_MVSK_Design_Spec.md), [solver package](../../torchyand/solver/) | Conceptual solver component flow; not a literal runtime trace |
| [matrix-free-operations.svg](matrix-free-operations.svg) | [MVSK design equations](../../TorchYAND_MVSK_Design_Spec.md), [oracle implementation](../../torchyand/oracle/mvsk.py) | Projection-based analytical derivatives; no dense high-order moment tensors |
| [mvsk-oracle-benchmark.svg](mvsk-oracle-benchmark.svg) | [Issue 21 report](../../reports/sae-issue21/summary.md), [fixed timing protocol](../../experiments/sae_mvsk/README.md) | T4/CPU eager public oracle; T=2048, n=256, batch=256, float32; medians of 7; excludes generation/centering |

**Plot data (milliseconds):** CPU resident 12.019641, T4 resident 1.101796,
T4 transfer-inclusive 2.206174. Relative ratios 10.909 and 5.448.
The SVG uses rounded labels; bar widths are proportional to unrounded median
times (baseline CPU bar width 670 SVG units). The report labels the scientific
pilot as pending independent GPT acceptance.

The historical compiled-PCG 2.38x figure is intentionally **not** plotted here:
the current documentation evidence is a prior README summary, not a committed
raw timing series in this provenance index.

For reproducibility, retain the committed SVG alongside this exact numerical
ledger. No GPU/TPU experiment was run to create these assets.
