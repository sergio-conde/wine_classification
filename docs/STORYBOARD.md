# Wine Story — Storyboard & Build Brief

Nov 19, 2026 · @Sergio Conde Ocazionez

## Premise & reader contract

The page opens with a promise and spends every chapter earning it: **"Tell me which wine you like, and I'll tell you which molecule to change."** Chapters 1 and 2 are the road; chapter 3 and the widget keep the promise.

**Stance.** The reader is a co-pilot, not an audience. The message is not "look how much analysis I can do" but "I know how to read this data, so let's explore it together." Every analysis has to earn its place by helping the reader use the widget. Anything that doesn't goes to *Keep exploring*.

**Audience.** Primary: industry hiring managers and data leads. They scan in about 30 seconds and reward rigor they can see without reading code. Secondary: curious non-technical readers who should still be able to play with the widget.

**Honest frame.** The dataset has no taste data, and a cultivar is a grape variety, not a dial you can turn. The counterfactual is a thought experiment: *what would this wine's chemistry need to look like to read as another cultivar?* Say this once, early and lightly, so no enologist can knock the story down.

**Language.** The page is in English.

## Story arc

Eight beats, built backwards from the widget: each one teaches the reader something they need in order to play. Each chapter closes on the question the next one answers.

| Beat | Reader takeaway | Figure | Minimal analysis |
| --- | --- | --- | --- |
| Hook | Pick a wine, get a molecule | Title + teaser of a wine crossing a border | None |
| Ch 1 · The dials | 13 molecules, but some move as one dial | Correlation heatmap with the phenolic block outlined | One correlation + hierarchical clustering on 1 − abs(r) |
| Ch 2 · The map | Chemistry alone draws three territories | LDA map with coloured decision regions | LDA + leave-one-out accuracy; k-means as one sentence |
| Ch 3a · The obvious dial | The model's top molecule isn't always the one *this* wine needs | Path on the map + P(class) curve along the dial | LDA posterior along the phenolic direction |
| Ch 3b · The recipe | The smallest change that crosses the border | Before/after bars per molecule | Counterfactual solver (L2, L1) |
| Ch 3c · Reality check | Cheap on paper ≠ possible in a bottle | Same recipe, constrained | Range bounds, correlated moves, Mahalanobis cost |
| Widget | Your turn | Interactive map + sliders | Exported model only |
| Epilogue | The same idea runs churn, pricing, QC | None | None |

## Chapter 1 — The dials

**The reader learns:** some molecules can't be turned one at a time. Total phenols, flavonoids, OD280/OD315 and proanthocyanins rise and fall as a block, the *phenolic cluster*. This is why one widget slider moves all four together.

**Keeps:**

- One correlation heatmap (pick Pearson or Spearman, not both), with the phenolic block outlined.
- Hierarchical clustering on 1 − abs(r) as the evidence behind the outline, in one sentence or a small dendrogram.

**Moves to Keep exploring:** the Pearson vs Spearman comparison, Bonferroni correction, per-attribute distributions, the parallel-coordinates plot by cultivar.

**Closing question:** *if these dials define a wine, can chemistry alone tell the three cultivars apart?*

## Chapter 2 — The map

**The reader learns:** chemistry alone draws three territories, and how to read the map they will play on in chapter 3. This chapter is the widget's tutorial.

**Keeps:**

- LDA on all 13 standardized attributes, shown as an LD1–LD2 map with coloured decision regions. This is the same canvas the widget uses.
- One number: **98.88% leave-one-out accuracy** (2 of 178 wines misclassified), next to the 98.9% originally published for this dataset.
- k-means as one sentence of validation: without labels it still finds k = 3 and agrees with the cultivars 88% of the time. The territories are real, not imposed by the labels.
- Mahalanobis distances between cultivar means: 1–2 = 5.34 (closest), 2–3 = 5.98, 1–3 = 7.75. This sets up chapter 3: turning a 2 into a 1 should be the cheapest crossing.

**Technical note worth one line on the page:** with three classes LDA has exactly two discriminants, so the LD1–LD2 map holds everything the classifier uses. It is the model, not a lossy projection of it.

**Moves to Keep exploring:** the k sweep (WCSS, silhouette), the contingency table, ARI vs number of PCs, LDA vs k-means head-to-head.

**Closing question:** *if the borders are this clear, what would it take to cross one?*

## Chapter 3 — The heist

The climax runs in three acts on one guided example, turning a typical Cultivar 2 wine into a Cultivar 1. That's the closest pair, so the cheapest heist. Expected outcomes below are hypotheses to verify in code before writing copy.

**3a · The obvious dial fails.** Start from the Cultivar 2 median wine. Turn the phenolic dial (all four molecules together, along their joint direction) across its observed range.

- Figure: the path on the LDA map, plus P(Cultivar 1) along the dial. In LD space the path is straight, but the posterior is not: it either shows a tipping point or stays flat.
- Hypothesis: it stays flat or barely moves, because phenolics mostly separate 1 from 3 on LD1, while 1 vs 2 lives in alcohol, proline and colour intensity on LD2.
- Story line: *the model's most important molecule isn't the one this wine needs.* If the data says otherwise, the beat becomes "here's the tipping point".

**3b · The minimal recipe.** Invert the question: what is the smallest change that makes LDA call this wine a Cultivar 1 with P ≥ 0.9?

- Everything is linear: "target score beats every other score by a margin" is a set of linear constraints. So L2 (smallest overall move, standardized units) is a QP, and L1 (fewest molecules) is an LP. Both are solved exactly.
- The L1 version is the promise kept, literally: *change these 2 molecules by this much.*
- Figure: before/after bars per molecule, in real units (% vol, mg/L).

**3c · Reality check.** Cheap on paper isn't the same as possible in a bottle.

- Constraints: values stay inside the observed range; the phenolic block moves as a unit; cost measured with Mahalanobis on the within-class covariance.
- Figure: the naive, sparse and realistic recipes side by side. Chapter 1's "single dial" pays off here.
- Plausibility test, reused by the widget: Mahalanobis distance to the nearest cultivar beyond the chi-square 99th percentile (13 df) means "no wine in the dataset looks like this".

**Then the hand-off:** "Now you try." The widget opens with the same wine and the same map.

**Bonus level (optional):** 3 → 1, the farthest pair, as the hard mode in the widget.

## Widget spec

One map, a handful of dials, and an honest readout. It runs entirely in the browser from exported JSON, with no backend.

**Canvas**

- LD1–LD2 map with coloured decision regions, the 178 real wines as faint dots, and the reader's wine as a bold dot leaving a trail.
- On mobile the sliders stack under the map.

**Controls**

- Start from: *a typical Cultivar 1 / 2 / 3 wine* (class median). Target: any other cultivar.
- Sliders, shown in real units: **Phenolic block** (one slider, moves four molecules together), **Alcohol**, **Proline**, **Colour intensity**, **Malic acid**. The other molecules stay at the starting wine's values.
- The final slider set follows chapter 3: every molecule an L1 recipe uses gets a slider.
- Buttons: *Show me the minimal recipe* (animates to the precomputed recipe) and *Reset*.

**Readouts**

- P(class) bars for the three cultivars.
- "You crossed the border" when the target's P passes the chosen threshold.
- Plausibility badge: *a real wine* / *stretching it* / *no wine in the dataset looks like this* (Mahalanobis test from 3c).

**Behaviour**

- Guided mode first: the 3a/3b demo plays as the reader scrolls. Free mode unlocks after.
- Slider ranges clamp to the dataset min–max plus a small margin, so readers can stretch but not break the chemistry.
- Maths per frame: standardize, class scores = W·x + b, softmax for posteriors, projection to LD1–LD2. All of it comes from `model.json`.

## Epilogue & Keep exploring

**Epilogue (one short paragraph).** What chapter 3 did has a name in industry: a *counterfactual explanation*. It's the same question as "what would need to change for this customer not to churn?", "for this loan to be approved?" or "for this batch to pass QC?". The wine is the toy; the method is the job.

**Keep exploring (collapsed by default, allowed to get technical):**

- Pearson vs Spearman correlation, with Bonferroni correction
- Hierarchical clustering details (linkage, distance choice)
- Full k-means: k sweep with WCSS and silhouette, contingency table
- ARI vs number of PCA components, and why more PCs dilute k-means
- LDA vs k-means head-to-head
- Later: other classifiers, if they add a point rather than a badge

## Code architecture

MATLAB is the canonical analysis, Python is a verified port, and the widget only consumes JSON. The whole story stands on Sergio's strongest language.

&#91;embedded content: repo architecture · MATLAB canonical, Python port, JS widget\]

The JSON files are the contract between the three parts. If MATLAB and Python both produce them identically, the port is proven.

**Principles**

- One script per chapter. Every figure in the story comes from exactly one function. No figure in the story means no code in v1.
- Toolboxes: Statistics and Machine Learning (`fitcdiscr`, `linkage`) and Optimization (`linprog`, `quadprog`). The Python port also lets readers without a MATLAB licence run it.
- README wording: *"Analysis built in MATLAB, my primary language for 15+ years. A Python port with parity tests is included as part of my ongoing migration."* If the widget is built with AI assistance, say so there too.
- Portfolio tags: `MATLAB` · `Python (port)` · `JavaScript`.

**JSON contract**

| File | Holds | Used by |
| --- | --- | --- |
| `model.json` | Feature names + units, scaler mean/SD, class means, pooled covariance, W and b per class, priors, LD projection (13 × 2), phenolic direction, slider ranges, cultivar names | Widget, parity test |
| `wines.json` | 178 wines: raw values, cultivar, LD1, LD2 | Widget map, parity test |
| `recipes.json` | Per origin → target pair: naive, sparse and realistic recipes, with P(target) and plausibility | Chapter 3 figures, widget's recipe button |

**Repo layout**

```
matlab/   ch1_dials.m · ch2_map.m · ch3_counterfactual.m · export_json.m
python/   port (after v1) · test_parity.py
web/      widget (consumes data/*.json)
data/     wine.data · model.json · wines.json · recipes.json
keep_exploring/   everything cut from the story
```

**Portfolio integration (scondeinsights, Next.js App Router)**

- One folder per project: `app/projects/wine/` holds `page.mdx` (the story) and the components it uses, such as `WineHeist.tsx` marked `'use client'`.
- Prose in MDX through the official `@next/mdx` package: the story reads as text, and components drop in where they belong (`<WineHeist />`).
- Data in `public/data/wine/*.json`, copied from the analysis repo's `data/`. The analysis repo stays the source of truth.
- The Projects grid reads one registry file, `content/projects.ts` (slug, title, one-line hook, tags, status, thumbnail). A new project = one folder + one registry entry.
- Remove the empty root `posts/` and `projects/` folders. Blog posts later follow the same pattern under `app/blog/<slug>/page.mdx`.
- Static by default: no API routes, no database.

## Open decisions & build order

Chapter 3 gets built first in code: the climax decides what chapters 1 and 2 have to teach.

**Open decisions**

- [x] Cultivar names: 1 = Barolo (Nebbiolo grape), 2 = Grignolino, 3 = Barbera. Inferred from the class counts (59 / 71 / 48) and row order of Forina et al. (1986), of which the UCI data is a 13-variable subset, and consistent with the chemistry. A final check against a citable source is still pending (Sergio).
- [x] Guided pair: 2 → 1. The 3a check confirmed the hypothesis: along the whole observed range of the phenolic dial, P(1) never rises above 0.000 for the median cultivar-2 wine.
- [x] Correlation shown in chapter 1: Pearson. Spearman gives the same picture (it differs by at most 0.08 within the phenolic block). The copy should say the block moves together *across* the 178 wines: within each cultivar it is much weaker.
- [x] Starting wine: the class median, so the result speaks for the typical wine rather than for whichever wines the dataset happens to hold. The real wine closest to the median is kept as a robustness check. The same rule applies to all three cultivars in the widget.
- [ ] "Crossed the border" threshold: P ≥ 0.5 or P ≥ 0.9.
- [ ] Final slider set, after the L1 recipes.

**Build order**

1. Freeze this storyboard.
2. MATLAB, chapter 3 first: verify the 3a hypothesis and compute the three recipes for 2 → 1.
3. MATLAB, chapters 1–2 as a clean pipeline + JSON export.
4. Widget prototype: ugly but working, to check that playing with it feels good.
5. Page copy, written against the real numbers.
6. Design polish inside the portfolio; update the old mockup (stack order, tags).
7. Python port + parity tests.
8. Keep exploring appendix.

**Out of scope for v1:** other classifiers, hyperparameter searches, any analysis without a figure in the story.
