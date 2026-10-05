# Research Log: Embedding Normalization Placement

This file is the chronological source of truth for the investigation. Results and interpretations below are limited to the experiments actually run.

## Research question

> What happens to the information and geometry of token embeddings when normalization is incorporated into the embedding representation itself, rather than applied after embedding lookup—and does the placement of normalization affect downstream model behavior?

## Current status

Experiments 0, 1, and 2 are complete. Experiment 0 measured row and pairwise geometry changes on a seeded synthetic embedding matrix. Experiment 1 compared two RMSNorm placements for a fixed table and found numerically equal outputs under the tested convention and tolerance. Experiment 2 compared gradients with respect to a shared trainable table for one controlled loss and found numerically equal gradients. It performed no optimizer updates. Optimization dynamics, learned representation evolution, and Transformer behavior remain untested.

## Investigation flow

```mermaid
flowchart TD
    Q[Original research question]
    E0[Experiment 0: synthetic embedding geometry]
    G[Cosine similarities preserved to numerical precision;<br/>Euclidean distances changed]
    E1[Experiment 1: fixed-table normalization placement]
    F[Fixed-table forward outputs numerically equal<br/>under tested RMSNorm convention and tolerance]
    U[Question shifts from forward computation<br/>to trainable-parameter behavior]
    E2[Experiment 2: gradient comparison<br/>with respect to trainable E (completed)]
    G2[Forward outputs, losses, and gradients<br/>numerically equal for the tested setup]
    U2[Open question: do optimizer updates produce<br/>equivalent parameter evolution and representations?]
    D[After controlled optimization evidence,<br/>decide whether a larger-scale model experiment is justified]

    Q --> E0 --> G --> E1 --> F --> U --> E2 --> G2 --> U2 --> D
```

## Experiment checkpoint table

| Checkpoint | Question | Status and observed result | Commit |
|---|---|---|---|
| [Experiment 0 — embedding geometry](../experiments/00_embedding_geometry/) | What does row-wise RMSNorm do to a toy embedding matrix's magnitudes and geometry? | Complete. Row magnitudes changed, cosine similarities were preserved to numerical precision, and pairwise Euclidean distances changed. | `f3c34aaf5953e66b5980f1615cd7b81f08b46195` — `experiment: measure embedding geometry under RMSNorm` |
| [Experiment 1 — normalization placement](../experiments/01_normalization_placement/) | With a fixed embedding table, do post-lookup RMSNorm and a pre-normalized table produce different outputs? | Complete. The two paths produced equal outputs under `atol=1e-12`, `rtol=1e-12`. | `d8f88daa35ca89060a7890d2ddbc910518c4f3f4` — `experiment: test normalization placement equivalence` |
| [Experiment 2 — trainable embedding gradients](../experiments/02_trainable_embedding_gradients/) | Does fixed-table forward equivalence extend to gradients with respect to a trainable embedding table? | Complete for the controlled setup. Forward outputs, loss, and gradients were numerically equal under the recorded tolerance. No optimizer update was performed. | This documentation/result commit |
| Next question — optimization behavior | Do the gradient results carry through optimizer updates into parameter evolution or resulting representations? | Unanswered; no optimization experiment has been run. | — |

## Experiment 0 checkpoint — embedding geometry

**Purpose.** Measure row-wise RMSNorm effects on a controlled synthetic embedding matrix before comparing normalization placement.

**Setup.** A standard-normal matrix with vocabulary size 128 and embedding dimension 32 was generated with seed 1729. The same rows were used before and after RMSNorm with axis `-1` and epsilon `1e-8`; the convention was `x / max(sqrt(mean(x**2, axis=-1, keepdims=True)), epsilon)`. All 8,128 unique row pairs were measured.

**Observed measurements.** The raw mean row L2 norm was `5.509095` (standard deviation `0.686539`); after RMSNorm the mean was `5.656854` (standard deviation approximately `1.25e-15`). Maximum absolute pairwise cosine change was approximately `2.78e-16`, passing the recorded `1e-12` allclose tolerances. The mean normalized-to-raw pairwise Euclidean distance ratio was `1.030494`.

**What this supports.** For this seeded synthetic matrix, the selected RMSNorm operation changed vector magnitudes and pairwise Euclidean distances while preserving pairwise cosine similarities to numerical precision.

**Limit.** This was a synthetic matrix, not a learned language-model embedding table. It says nothing about training, downstream behavior, or the semantics of learned embeddings.

**Files and artifacts.** [Experiment code and configuration](../experiments/00_embedding_geometry/); [metrics and configuration record](../results/00_embedding_geometry/metrics.json); [saved matrices](../results/00_embedding_geometry/embedding_matrices.npz); [pairwise measurements](../results/00_embedding_geometry/pairwise_geometry.npz); [distance and cosine plots](../figures/00_embedding_geometry/).

**Commit.** `f3c34aaf5953e66b5980f1615cd7b81f08b46195` — `experiment: measure embedding geometry under RMSNorm`.

## Experiment 1 checkpoint — normalization placement with a fixed table

**Question.** For one fixed embedding table, do these computational pathways produce different outputs?

- **A — post-lookup normalization:** `embedding_table[token_ids]` → RMSNorm.
- **B — pre-normalized table:** RMSNorm each row of `embedding_table` → `normalized_table[token_ids]`.

**Setup.** Both conditions used the same seeded `128 × 32` raw table from the Experiment 0 methodology, all token IDs `0` through `127`, the same RMSNorm formula and epsilon (`1e-8`), and the same fixed table. No training or Transformer was used. Equality was evaluated with `atol=1e-12` and `rtol=1e-12`.

**Observed measurements.** The maximum and mean absolute elementwise differences and maximum relative difference were all `0`. Per-token Euclidean differences, pairwise Euclidean distance differences, and pairwise cosine similarity differences were also `0`. The outputs passed the configured numerical equality check.

**Conclusion bounded to the setup.** For a fixed table and the tested RMSNorm convention, these two paths are forward-equivalent in this experiment. This is a fixed-table forward result only.

**Files and artifacts.** [Experiment code and configuration](../experiments/01_normalization_placement/); [metrics and reproducibility record](../results/01_normalization_placement/metrics.json); [saved inputs, outputs, and pairwise measurements](../results/01_normalization_placement/representations.npz).

**Commit.** `d8f88daa35ca89060a7890d2ddbc910518c4f3f4` — `experiment: test normalization placement equivalence`.

## Experiment 2 checkpoint — trainable embedding gradients

**Research question.** Does the fixed-table forward equivalence found in Experiment 1 extend to gradients with respect to the underlying trainable embedding table `E`?

**Why this follows Experiment 1.** Experiment 1 found equal outputs for a fixed table but did not inspect the derivative of those outputs and a loss with respect to `E`. Experiment 2 keeps the placement comparison and initialization controlled, marks the shared table as trainable for autodiff, and measures that derivative without taking optimizer steps.

**Setup and held-constant conditions.** The experiment uses the same seeded initialization methodology and values as Experiments 0 and 1: seed `1729`, vocabulary size `128`, embedding dimension `32`, RMSNorm on axis `-1`, and epsilon `1e-8`. Both paths use the same underlying PyTorch float64 trainable tensor `E`; assertions verify that both conceptual table references are the same object and have exactly equal values. Both paths use token IDs `[3, 17, 42, 91]`, the same deterministic target, identity downstream mapping, and mean-squared-error loss. Equality uses `atol=1e-12` and `rtol=1e-12`. There are no extra trainable parameters, optimizer, parameter updates, Transformer, or external data.

**Path A — post-lookup normalization.** `E[token_ids]` → RMSNorm → representation `A` → mean squared error against the fixed target.

**Path B — pre-normalized table.** RMSNorm each row of `E` → `normalized_E[token_ids]` → representation `B` → mean squared error against the same target.

**Loss construction.** The downstream mapping is identity. The target is a deterministic float64 `linspace` from `-0.5` to `0.5`, reshaped to `(4, 32)`. For each pathway, loss is `mean((representation - target) ** 2)` over all 128 elements. This creates a scalar objective that depends on the selected normalized embedding vectors and yields nonzero gradients for the selected table rows in this run.

**What was measured.** Maximum, mean, and meaningful maximum relative forward element differences; both scalar losses and their absolute difference; maximum, mean, and meaningful maximum relative difference between `d(loss_A)/dE` and `d(loss_B)/dE`; the L2 norm of each gradient and their L2 difference; and numerical equality under the configured tolerance.

**Observed results.** Initial table values were exactly equal and both paths referenced the same tensor object. Maximum and mean absolute forward differences were `0`; maximum relative difference was `0`. Both losses were `1.1457136942185433`, with absolute difference `0`. Gradient maximum and mean absolute differences and maximum relative difference were `0`. Both gradient L2 norms were `0.052245619117913686`; the gradient difference L2 norm was `0`. Forward outputs, losses, and embedding-table gradients all passed the documented tolerance check. The nonzero gradient norms show that the measured equality was not a comparison of two zero gradients.

**Whether the tested results were equivalent.** Forward outputs were numerically equal. Losses were numerically equal. Gradients with respect to the shared `E` were numerically equal under `atol=1e-12`, `rtol=1e-12`.

**What this establishes.** For this seeded table, token subset, RMSNorm convention, deterministic target, and mean-squared-error objective, the two formulations produced numerically equivalent forward outputs, losses, and gradients with respect to `E`.

**What this does not establish.** It does not establish equivalence for other losses, token sequences, dtypes, numerical environments, optimizer updates, optimization dynamics, parameter evolution, learned representations, or Transformer behavior. No optimizer was used and no parameters were updated. Gradient equivalence is not optimization equivalence.

**Files and artifacts.** [Experiment code and configuration](../experiments/02_trainable_embedding_gradients/); [metrics and runtime metadata](../results/02_trainable_embedding_gradients/metrics.json); [initial table, token IDs, target, outputs, losses, and gradients](../results/02_trainable_embedding_gradients/gradients.npz).

## Fixed-table forward equivalence vs. trainable-parameter behavior

These are distinct questions and must not be conflated:

1. **Fixed-table forward equivalence:** Given an unchanged embedding table and the same token IDs, do the two placements compute the same vectors? Experiment 1 tested this and found equal outputs under its stated setup and tolerance.
2. **Trainable-parameter behavior:** Does a loss produce the same gradients with respect to `E`, and, separately, do optimizer updates produce the same parameter evolution or learned representations? Experiment 2 tests gradients only for one fixed controlled loss and finds numerical equality in this run. It does not test updates, optimization, or Transformer behavior.

## Engineering decisions and rationale

| Decision | Rationale |
|---|---|
| Use a seeded synthetic table with small fixed dimensions in the first three experiments. | This isolates the normalization operations and keeps the initial measurements reproducible and inexpensive; it does not substitute for learned or language-model embeddings. |
| Preserve Experiment 0's RMSNorm convention, axis, and epsilon in Experiments 1 and 2. | This keeps the placement and gradient comparisons controlled instead of introducing a formula change as a confound. |
| Reuse the same raw table and token IDs across each comparison's pathways. | Differences should reflect pathway computation, not different initialization or lookup inputs. |
| Measure pairwise geometry in Experiment 0 as well as row norms. | Row magnitudes alone do not characterize pairwise distances or angles. |
| Record configuration, runtime versions, numerical arrays, and metrics. | These records make each result inspectable and support reproducibility. |
| Separate fixed-table forward comparison, gradient comparison, and optimizer training into distinct questions. | Experiment 1 does not answer gradient behavior; Experiment 2 measures gradients but takes no update step. Optimization remains unanswered and should be measured separately before deciding whether a larger-scale model experiment is justified. |
| Use the already available PyTorch autodiff implementation with a single float64 trainable table tensor. | It provides a direct gradient calculation in the existing environment while avoiding additional trainable parameters and optimizer behavior. |
| Do not introduce a Transformer or external dataset in these checkpoints. | Neither is needed to answer the current fixed-table question, and introducing them now would add uncontrolled complexity. |

## Current unanswered question

Under a controlled optimization procedure, do the two normalization parameterizations produce equivalent parameter updates and parameter evolution, and do they lead to equivalent learned representations? The gradient comparison above does not answer this because no optimizer step was performed.

## Planned next experiment

An optimization experiment may compare the parameter updates and resulting embedding representations from the same initialization under a specified task and optimizer. Its objective, update protocol, and measurements must be selected before implementation. Use that evidence to decide whether a larger-scale model experiment is justified. No optimization or Transformer experiment has been implemented.

## Paper documentation checkpoint — 2026-10-05

Created an Overleaf-ready LaTeX report at [paper/main.tex](../paper/main.tex), with an empty [references.bib](../paper/references.bib) because no formal bibliographic sources are present in the repository. The paper documents the research state through Experiments 0--2, states the row-wise algebraic relation between lookup-then-normalize and normalize-table-then-lookup, and distinguishes both from making the normalized representation itself trainable. It includes the existing Experiment 0 figures, copied into [paper/figures/](../paper/figures/), and tables using the checked-in result artifacts. No new experimental results were generated.

Numerical values, formulas, token IDs, environment versions, and experiment commit hashes were checked against the code and result artifacts. The source explicitly limits the gradient result to its one float64 MSE setup and does not claim optimizer or Transformer equivalence. The paper contains no external citations and does not present an Experiment 3 as completed.

Compilation could not be verified in this environment: the built-in LaTeX compiler returned “Unable to find standard directories for platform,” and no local pdfLaTeX, XeLaTeX, LuaLaTeX, or latexmk executable was found. No LaTeX build artifacts were generated. Compile status is therefore unverified; the source is prepared for Overleaf or a configured LaTeX toolchain.
