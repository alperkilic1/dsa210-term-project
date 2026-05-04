# DSA210 Term Project — Incident Genome

Cloud service outage pattern analysis.

**Course:** DSA210 Introduction to Data Science, Spring 2026
**Author:** Alper Kılıç
**Instructors:** Öznur Taştan, Özgür Asar

## Research question

> Using only features that are observable within the first hour of an incident (service, start-hour, day-of-week, first-hour update count, severity at t=0), can we tell whether an outage will be short (< 60 min) or long (≥ 60 min)?

Repo covers two milestones: the EDA milestone (`milestone1` tag) and the supervised-classifier ML milestone (commits after the tag, head-of-`main`).

## ML milestone summary (5 May)

| Item | Value |
|---|---|
| Models compared | Logistic Regression, Random Forest (baseline), Random Forest (GridSearchCV-tuned) |
| Test accuracy (RF baseline) | 0.7447 |
| Test F1-macro (RF baseline) | 0.7317 |
| 5-fold CV F1-macro (RF baseline) | 0.6412 ± 0.0370 |
| Naive baseline (always-long) | 0.6136 — beaten by +13.1pp absolute |
| Top SHAP feature | `first_hour_updates` (mean \|SHAP\| = 0.103, ~5–6× the next feature) |

Full write-up: [`ml_report.md`](ml_report.md) · notebook: [`ml_baseline.ipynb`](ml_baseline.ipynb) · raw metrics: [`data/ml_results.json`](data/ml_results.json).

## What I did

- Scraped 869 raw incidents from 14 public cloud status pages (GitHub, Cloudflare, OpenAI, Discord, Reddit, Atlassian, Vercel, Netlify, DigitalOcean, Dropbox, Linear, Notion, Twilio, Datadog) via their Statuspage.io API endpoints. Data spans 2019-05-07 to 2026-04-11 (~83 months), though most services only expose their last 12–24 months.
- Cleaned down to 704 resolved incidents with valid duration (dropped 158 unresolved/scheduled-maintenance, 7 with negative duration).
- Flagged 84 outliers with IQR (kept them, didn't drop).
- Built a leakage-free feature `first_hour_updates` (status updates posted within 3600s of incident start) to replace the leaky total `num_updates`.
- 16 figures, 3 hypothesis tests with BH-corrected p-values and effect sizes, bootstrap 95% CI for median duration.

## Key findings

| Question | Result |
|---|---|
| What's a typical outage duration? | median **82.5 min**, bootstrap 95% CI **[73.2, 91.3]** — heavy right tail, mean 480 min |
| Does severity predict first-hour update activity? (H3) | **Yes**. Kruskal-Wallis H=100.6, p≈0, BH-adjusted q≈0, ε²=0.139 (medium effect). Dunn post-hoc: `none` differs from all other groups; `major` vs `minor` also significant. |
| Do business-hours incidents resolve faster? (H1) | **No**. Mann-Whitney U, raw p=0.367, BH q=0.550, Cliff's δ=-0.039 (negligible). |
| Do weekend incidents resolve differently? (H2) | **No**. Mann-Whitney U, raw p=0.878, BH q=0.878, Cliff's δ=-0.015 (negligible). |
| Was the `num_updates`-vs-duration correlation real? | **No**. Leaky version: Spearman ρ≈+0.46. Clean (`first_hour_updates`): Spearman ρ=-0.224 — sign flips and magnitude halves. The original "correlation" was mostly post-resolution updates inflating the count. |
| How imbalanced is the short/long target? | 272 short vs 432 long, 1.59:1 — mild. Stratified split + `class_weight='balanced'` demonstrated in §7a. |

![Day × Hour incident heatmap](figures/temporal_heatmap.png)

![Leakage fix: before and after](figures/leakage_scatter_comparison.png)

![H3: severity vs first-hour update count](figures/h3_severity_updates.png)

![Duration CDF — median 82.5 min with heavy right tail](figures/02d_duration_cdf.png)

![Class imbalance: 272 short vs 432 long (1.59:1)](figures/class_imbalance.png)

![Leakage-free Spearman correlation heatmap](figures/corr_heatmap.png)

## Instructor-feedback addressed

| Proposal concern | How it's addressed |
|---|---|
| Data leakage | `num_updates` replaced by `first_hour_updates` everywhere hypothesis tests and the heatmap use it. Side-by-side comparison in §10. `impact` and `num_components` also flagged as leaky and excluded from the ML feature set. |
| Outliers | IQR flag kept but not dropped (§2). Day-of-week averages (§5e) exclude flagged outliers. Duration figures use log axes so the tail doesn't dominate. |
| Class imbalance | Visualized in §7 with ratio printout. §7a demonstrates stratified 80/20 split and `compute_class_weight('balanced')` output for the ML milestone. |

## Repo map

| Path | What's inside |
|---|---|
| `eda_report.ipynb` | Main notebook, 14 sections, 29 code cells, all outputs committed |
| `collect_data.py` | Fetches raw incidents from Statuspage.io endpoints, writes JSON to `data/raw/` |
| `data/incidents.json` | All 869 parsed incidents |
| `data/incidents_clean.csv` | 704 resolved + feature-enriched rows (target of milestone1 "featurized" rule) |
| `data/raw/*_raw.json` | Per-service scraped payloads |
| `data/stats.json` | Per-service counts from the last `collect_data.py` run |
| `figures/*.png` | 16 EDA figures |
| `proposal.md`, `proposal.pdf` | Original proposal (frozen) |
| `requirements.txt` | Python deps with minimum version pins |

## How to reproduce

```bash
git clone https://github.com/alperkilic1/dsa210-term-project.git
cd dsa210-term-project
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python collect_data.py           # refreshes data/raw/ and data/incidents.json (~3 min)
jupyter nbconvert --to notebook --execute eda_report.ipynb --output eda_report.ipynb
```

Re-running `collect_data.py` will pull whatever is live on the status pages today — so the incident counts will drift slightly from the committed snapshot. The notebook is set up to run end-to-end in about 30 seconds once `data/incidents.json` exists.

## AI assistance

I used ChatGPT and Claude as coding assistants. Per the course guidelines, here is what I delegated and what I kept ownership of:

| Task | Tool | What I asked / received | What I verified |
|---|---|---|---|
| Scraper debugging | ChatGPT | "Why does Statuspage.io return 429 here, and how do I paginate /history.json?" → got `time.sleep` + `/api/v2/incidents/{code}` two-step pattern | Tested manually against 14 services; rate-limit handling adjusted |
| Proposal formatting | ChatGPT | "Convert this draft to half-page single-spaced Markdown" → got Markdown skeleton | Rewrote every claim in my own wording |
| Statistical test selection | Claude | "Duration is right-skewed (skew ≈ 8) — Mann-Whitney vs t-test for H1?" → recommended Mann-Whitney + Cliff's δ | Cross-checked with Tomczak & Tomczak 2014 for ε² and Cliff 1993 for δ |
| ML scaffold | Claude | "Pipeline + StratifiedKFold + GridSearchCV + SHAP TreeExplainer template" → got skeleton | Re-wrote with my feature set, my leakage rules; verified each metric against a manual `classification_report` run |

Ownership: project idea, data source selection, analysis design, the leakage discovery and fix (`first_hour_updates`), feature exclusions (`num_updates`/`impact`/`num_components`), interpretation, and all written prose are my own. No AI-generated text is pasted verbatim into the notebook narrative or this README.

## Submission

- Milestone1 is tagged `milestone1` at commit `0b5147f` (push date 2026-04-14).
- All post-tag commits are FINAL-submission work and are on `main`, not under any milestone tag.
