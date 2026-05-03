# Incident Genome: Predicting Cloud Outage Duration from Early Signals

**DSA210 Introduction to Data Science — Final Report**
**Alper Kilic**
**Instructors:** Oznur Tastan, Ozgur Asar
**Date:** May 2026

---

## Abstract

This project analyzes 704 resolved incidents from 14 public cloud service status pages (GitHub, Cloudflare, OpenAI, Discord, etc.) to predict whether an outage will be short (< 60 min) or long (>= 60 min) using only features observable within the first hour. We address data leakage by replacing total update counts with a time-windowed alternative, conduct three hypothesis tests with BH-corrected p-values, and build a Random Forest classifier as the primary model.

**TODO:**
- Final accuracy/F1 numbers from ML pipeline
- One-sentence SHAP insight summary
- Final word on whether the research question is answered

---

## 1. Introduction & Problem Statement

Cloud service outages cost businesses an estimated $X per minute of downtime. Public status pages (powered by Statuspage.io) expose incident histories as structured JSON, creating an opportunity for data-driven analysis of outage patterns.

**Research Question:**
> Using only features observable within the first hour of an incident (service, start-hour, day-of-week, first-hour update count), can we predict whether an outage will be short (< 60 min) or long (>= 60 min)?

This is a binary classification problem with practical relevance: if early signals reliably predict duration, incident response teams could triage resources more effectively.

**Motivation:**
- No API key required — fully reproducible
- Cross-service comparison reveals industry-wide patterns
- Addresses a gap: most outage studies focus on post-mortems, not early prediction

---

## 2. Data Collection

### 2.1 Sources

14 cloud services with public Statuspage.io endpoints:

| Service | Base URL | Incidents Collected |
|---------|----------|---------------------|
| GitHub | githubstatus.com | 110 |
| DigitalOcean | status.digitalocean.com | 74 |
| Vercel | vercel-status.com | 72 |
| Netlify | netlifystatus.com | 61 |
| Twilio | status.twilio.com | 51 |
| Cloudflare | cloudflarestatus.com | 50 |
| Datadog | status.datadoghq.com | 50 |
| Discord | discordstatus.com | 50 |
| Dropbox | status.dropbox.com | 50 |
| Reddit | redditstatus.com | 50 |
| Atlassian | status.atlassian.com | 31 |
| OpenAI | status.openai.com | 25 |
| Linear | linearstatus.com | 21 |
| Notion | status.notion.so | 9 |

### 2.2 Collection Method

`collect_data.py` fetches incidents via two endpoints per service:
1. `/api/v2/incidents.json` — recent incidents (paginated)
2. `/history.json` — historical incident codes, then fetched individually via `/api/v2/incidents/{code}.json`

Fallback: when individual fetch fails (HTTP 404/429), a minimal record is constructed from history metadata.

### 2.3 Feature Engineering

Each incident is parsed into 21 columns including:
- `duration_minutes` — (resolved_at - created_at) in minutes
- `first_hour_updates` — status updates posted within 3600s of incident start (leakage-free)
- `created_hour`, `created_weekday` — temporal features
- `duration_class` — binary target: "short" (< 60 min) or "long" (>= 60 min)

### 2.4 Dataset Summary

| Metric | Value |
|--------|-------|
| Raw incidents | 869 |
| After cleaning | 704 |
| Date range | 2019-05-07 to 2026-04-11 |
| Services | 14 |
| Features | 21 columns |
| Target balance | 272 short (38.6%) vs 432 long (61.4%) |

---

## 3. Exploratory Data Analysis

### 3.1 Duration Distribution

- Median: **82.5 min**, Bootstrap 95% CI: **[73.2, 91.3] min**
- Mean: 480.2 min (driven by heavy right tail)
- Skewness: 5.8+ (heavily right-skewed)
- Mean/Median ratio: ~5.8x — confirms non-normality, motivating non-parametric tests

### 3.2 Temporal Patterns

- **Peak day:** Tuesday (highest incident count)
- **Peak hour:** 15:00-16:00 UTC
- Business hours (09-17 UTC) account for ~49% of incidents
- No statistically significant weekday/weekend difference in duration (H2, p=0.878)

### 3.3 Service Comparison

Median duration varies dramatically across services:
- Fastest resolution: GitHub (~50 min median)
- Slowest resolution: Atlassian, Dropbox (>300 min median)

### 3.4 Hypothesis Testing Results

| Test | H0 | Result | Effect Size |
|------|----|--------|-------------|
| H1: Business vs off-hours | Same duration distribution | Fail to reject (BH q=0.550) | Cliff's delta = -0.039 (negligible) |
| H2: Weekday vs weekend | Same duration distribution | Fail to reject (BH q=0.878) | Cliff's delta = -0.015 (negligible) |
| H3: Severity vs first-hour updates | Same update distribution | **Reject** (BH q~0, H=100.6) | epsilon-sq = 0.139 (medium) |

### 3.5 Data Leakage Discovery

- `num_updates` (all updates): Spearman rho = **+0.46** with duration
- `first_hour_updates` (time-windowed): Spearman rho = **-0.224**
- Sign flip reveals the original correlation was an artifact of longer incidents accumulating more post-resolution updates

---

## 4. ML Methodology

### 4.1 Feature Set (Leakage-Free)

Only features available at prediction time (t=0..1h):

| Feature | Type | Rationale |
|---------|------|-----------|
| `service` | Categorical (14 levels) | Service identity captures infra maturity |
| `created_hour` | Numeric (0-23) | Time-of-day effect |
| `created_weekday` | Categorical (7 levels) | Day-of-week patterns |
| `first_hour_updates` | Numeric | Communication urgency signal |
| `is_business_hours` | Binary | Derived from created_hour |

**Excluded (leaky):** `num_updates`, `impact` (final severity), `num_components` (cumulative)

### 4.2 Target Variable

Binary: `duration_class` = "short" (< 60 min) | "long" (>= 60 min)

### 4.3 Train/Test Split

- 80/20 stratified split (preserves 38.6%/61.4% ratio)
- Random state = 42 for reproducibility

### 4.4 Models

| Model | Rationale |
|-------|-----------|
| Logistic Regression | Baseline, interpretable coefficients |
| **Random Forest** | Primary — handles categorical features, non-linear interactions, robust to outliers |

### 4.5 Imbalance Handling

- `class_weight='balanced'` (demonstrated in EDA §7a: short=1.291, long=0.816)
- Stratified K-fold cross-validation (k=5)

**TODO:**
- Hyperparameter grid (n_estimators, max_depth, min_samples_leaf)
- Cross-validation results table
- Final model selection rationale

---

## 5. Results

**TODO: Fill from ML pipeline output**

### 5.1 Model Performance

| Model | Accuracy | Precision (long) | Recall (long) | F1 (long) | F1 (short) |
|-------|----------|------------------|---------------|-----------|------------|
| Logistic Regression | ___ | ___ | ___ | ___ | ___ |
| Random Forest | ___ | ___ | ___ | ___ | ___ |

### 5.2 Confusion Matrix

```
              Predicted
              Short    Long
Actual Short  [___]   [___]
Actual Long   [___]   [___]
```

### 5.3 Feature Importance

**TODO:**
- Permutation importance bar chart description
- Top 3 features and their contribution

### 5.4 Cross-Validation

**TODO:**
- 5-fold CV mean +/- std for chosen model
- Comparison with holdout test performance (overfitting check)

---

## 6. Discussion

### 6.1 Answering the Research Question

**TODO:**
- Does the model beat a naive baseline (always predict "long" = 61.4% accuracy)?
- Which early signals are most predictive?

### 6.2 Interpretation

The EDA already established that:
- Severity level drives first-hour communication intensity (H3 significant, medium effect)
- Temporal features (hour, weekday) show no population-level effect on duration (H1, H2 not significant)
- Service identity likely captures unmeasured confounders (infrastructure complexity, team size, SLA pressure)

**TODO:**
- Connect ML feature importance back to hypothesis test findings
- Explain any surprises

### 6.3 Comparison with Baseline

**TODO:**
- Majority-class baseline: 61.4% accuracy
- Model improvement over baseline in absolute % and relative %

---

## 7. Limitations

1. **Sample size (n=704):** Limits statistical power and model generalization. Some services contribute as few as 9 incidents.
2. **Self-reported data:** Status pages are curated by companies — incidents may be underreported, delayed, or mis-labeled.
3. **Impact label leakage:** The `impact` field reflects final (not initial) severity. We excluded it from features, but this removes a potentially useful signal.
4. **Temporal coverage bias:** Most services expose only 12-24 months of history.
5. **No causal claims:** Observational data only.
6. **UTC timestamps only:** No normalization for service timezone.
7. **No text features:** First update body text is collected but not used.

---

## 8. Future Work

- **Text analysis:** NLP on first-update body to extract urgency signals.
- **Multi-class target:** Predict duration quartiles or use regression.
- **Real-time integration:** Lightweight API that takes an incident URL and returns a predicted duration band.
- **Larger dataset:** Expand to 30+ services.
- **Causal analysis:** Pair with deployment logs.

---

## 9. Conclusion

**TODO:** 3-4 sentences summarizing: research question, method, key result, practical implication.

---

## 10. References

1. Statuspage.io API documentation. https://developer.statuspage.io/
2. Tomczak, M. & Tomczak, E. (2014). The need to report effect size estimates revisited. *Trends in Sport Sciences*, 1(21), 19-25.
3. Cliff, N. (1993). Dominance statistics: Ordinal analyses to answer ordinal questions. *Psychological Bulletin*, 114(3), 494-509.
4. Benjamini, Y. & Hochberg, Y. (1995). Controlling the false discovery rate. *JRSS-B*, 57(1), 289-300.
5. Scikit-learn documentation: RandomForestClassifier, class_weight.

---

## AI Assistance Disclosure

I used ChatGPT and Claude for: debugging the scraper, formatting documents, double-checking statistical test choices, and scaffolding the ML pipeline structure. All project ideas, data source selection, analysis design, leakage fix approach, and interpretation were my own. Code and text are my wording; where I received suggestions, I verified against primary references.
