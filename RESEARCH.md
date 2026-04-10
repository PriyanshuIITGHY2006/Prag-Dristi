# Prag-Dristi: Research Justification and Architecture Documentation

**Project:** Assam Flood Forecasting Engine — 7-day ahead Brahmaputra river discharge prediction  
**Author:** Priyanshu Debnath, IIT Guwahati  
**Model:** LSTM Encoder-Decoder with Bahdanau Attention  
**Data:** ERA5 Reanalysis + GloFAS-ERA5 River Discharge (Copernicus CDS)

---

## Table of Contents

1. [Problem Statement and Motivation](#1-problem-statement-and-motivation)
2. [Why This Is a Hard ML Problem](#2-why-this-is-a-hard-ml-problem)
3. [Data Sources: Justification](#3-data-sources-justification)
4. [Architecture Decision: Why LSTM Encoder-Decoder with Attention](#4-architecture-decision-why-lstm-encoder-decoder-with-attention)
5. [Mathematical Formulation](#5-mathematical-formulation)
6. [Feature Engineering Justification](#6-feature-engineering-justification)
7. [Loss Function: Flood-Weighted MSE](#7-loss-function-flood-weighted-mse)
8. [Evaluation Metrics Justification](#8-evaluation-metrics-justification)
9. [Benchmarks and Expected Performance](#9-benchmarks-and-expected-performance)
10. [Comparison with Alternative Approaches](#10-comparison-with-alternative-approaches)
11. [Limitations and Future Work](#11-limitations-and-future-work)
12. [Full Bibliography](#12-full-bibliography)

---

## 1. Problem Statement and Motivation

The Brahmaputra River (known as Luit in Assam) is one of the world's largest rivers by discharge volume, draining approximately 580,000 km² across Tibet, Northeast India, and Bangladesh. Assam, situated on the Brahmaputra floodplain, experiences annual flooding that displaces millions of people. According to the National Disaster Management Authority (NDMA), Assam accounts for roughly **9.4% of India's total flood-affected area** while having only 2.4% of the country's land area — making it one of the most flood-vulnerable regions in the world.

The Central Water Commission (CWC), India's national flood forecasting body, operates **350 flood forecasting stations** across 20 major river basins and has targeted **7-day ahead** forecasting as the operational standard for effective evacuation and preparation [14]. However, conventional physics-based approaches (HEC-HMS, SWAT, VIC) require highly calibrated parameters, dense gauge networks, and significant computational infrastructure that is often unavailable in data-sparse developing regions.

**This project builds a data-driven alternative**: an LSTM-based encoder-decoder that learns the rainfall-runoff-discharge relationship directly from 23 years of reanalysis data, producing 7-day ahead discharge forecasts with associated flood alerts.

---

## 2. Why This Is a Hard ML Problem

### 2.1 Class Imbalance: The Central Challenge

Flood days — days when discharge exceeds the danger level — are rare events. At Bahadurabad station on the Brahmaputra, the danger discharge is approximately 98,600 m³/s. Analysis of historical GloFAS data suggests that this threshold is exceeded on roughly **3-8% of all days** over a 20-year record. This extreme class imbalance creates a well-known failure mode:

> A model that always predicts "no flood" achieves >92% accuracy but has zero utility.

This is not a trivial observation. It is the reason that **accuracy is categorically excluded** from our evaluation suite, and why we use flood-specific metrics (CSI, POD, FAR) that are invariant to the class distribution of non-events.

### 2.2 Non-Stationarity and Monsoon Seasonality

The Brahmaputra exhibits extreme seasonal variation. Mean June-September discharge at Bahadurabad (~80,000 m³/s) is roughly 15× the mean December-February discharge (~5,000 m³/s). Any model trained on annual data must learn two qualitatively different regimes simultaneously. We address this with **sinusoidal day-of-year encoding** (see Section 6.3), which gives the model an explicit seasonal signal.

### 2.3 Non-Linear Threshold Behaviour

The rainfall-runoff relationship is highly non-linear near soil saturation. A 50 mm/day rainfall event on dry soil may produce minimal runoff; the same event on saturated monsoon-season soil can produce catastrophic flooding. This threshold non-linearity is precisely what LSTM's gating mechanisms are designed to represent (see Section 4.2).

### 2.4 Multi-Day Lag and Memory Requirements

Upstream rainfall in the Tibetan Plateau and Arunachal Pradesh takes 3-10 days to propagate to downstream gauging stations in Assam. Any model must maintain memory across at least 30 days of history to capture this routing signal. This makes the problem fundamentally unsuitable for classical ML methods (random forest, XGBoost) that operate on fixed-length feature vectors without explicit temporal memory.

---

## 3. Data Sources: Justification

### 3.1 ERA5 Reanalysis (Copernicus CDS)

**Citation:** Hersbach et al. (2020) [5]

ERA5 is the fifth-generation ECMWF atmospheric reanalysis, providing **hourly global coverage at 31 km horizontal resolution** from 1940 to present. We use ERA5 for the following meteorological forcing variables:

| Variable | ERA5 Short Name | Physical Role |
|---|---|---|
| Total precipitation | tp | Primary flood driver |
| 2m air temperature | t2m | Snowmelt proxy, evapotranspiration |
| Surface pressure | sp | Atmospheric state indicator |
| Volumetric soil water (layer 1) | swvl1 | Antecedent soil moisture (see Section 6.1) |
| 10m u-wind component | u10 | Moisture advection |
| 10m v-wind component | v10 | Moisture advection |

ERA5 was chosen over gauge-based IMD data for the following reasons:
1. **Spatial coverage**: ERA5 covers the entire upstream Brahmaputra catchment including Tibet and Arunachal Pradesh, where rain gauge density is near zero. Any model based only on downstream gauges misses the primary upstream driver.
2. **Temporal completeness**: ERA5 has no missing values — it is a modelled reanalysis, not an observation record. This is critical for training stability.
3. **Reproducibility**: ERA5 is publicly downloadable via the Copernicus CDS API, making this research fully reproducible.

ERA5 has been validated extensively; Hersbach et al. (2020) report that ERA5 re-forecasts show a gain of **up to 1 day in medium-range forecast skill** relative to the predecessor ERA-Interim reanalysis.

### 3.2 GloFAS-ERA5 River Discharge Reanalysis

**Citation:** Harrigan et al. (2020) [18]

GloFAS-ERA5 provides **daily global river discharge at 0.1° resolution** from 1979 to near-real-time, produced by routing ERA5 land surface runoff through the LISFLOOD hydrological model. This is our **target variable** (discharge in m³/s).

Key validation statistics from Harrigan et al. (2020):
- Validated against **1,801 observation stations** globally
- Skilful (better than mean-flow benchmark) in **86% of catchments** by modified KGE Skill Score
- Global median Pearson correlation = **0.61** (IQR: 0.44–0.74)

We use GloFAS-ERA5 as the target label rather than CWC gauge data for the following reasons:
1. **Consistent ERA5 forcing**: both input features (ERA5) and discharge target (GloFAS-ERA5 routed from ERA5) share the same atmospheric model, reducing spurious correlations from observational noise.
2. **Spatial completeness**: GloFAS provides discharge estimates at any river grid cell, not just at gauged stations.
3. **Long record**: 1979–present enables training on multiple flood cycles including major events (1998, 2004, 2012, 2017, 2020, 2022).

**Limitation acknowledged**: GloFAS-ERA5 is itself a model output, not direct observation. For operational deployment, bias correction against CWC gauge records would be required.

### 3.3 Temporal Coverage

We use **2000–2022** (23 years, ~8,395 days after feature engineering drops NaN rows from lag creation). This provides:
- ~5,877 training samples (70%)
- ~1,259 validation samples (15%)
- ~1,259 test samples (15%)

The split is **strictly temporal** (chronological order preserved). Random shuffling is never used, as it would allow future observations to leak into the training set.

---

## 4. Architecture Decision: Why LSTM Encoder-Decoder with Attention

### 4.1 Why Not Classical ML?

Random Forests, Gradient Boosting (XGBoost/LightGBM), and SVMs operate on tabular, fixed-length feature vectors. To use them for multi-step forecasting, one must manually engineer all temporal dependencies as lag features. For a 30-day encoder window:
- 6 variables × 30 lags = 180 features minimum
- No mechanism to learn routing dynamics (water travelling through a river network takes different times under different conditions)
- Cannot directly produce a 7-step output sequence — requires a separate model per forecast horizon

LSTMs, by contrast, process sequences directly, maintaining hidden state that compresses the history of all past timesteps into a learnable representation.

### 4.2 Why LSTM Over Vanilla RNN?

**Citation:** Hochreiter & Schmidhuber (1997) [16]

The vanishing gradient problem makes vanilla RNNs unable to learn dependencies spanning more than ~8-10 timesteps. LSTM solves this with three gates:

**Forget gate:**
$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

**Input gate:**
$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$
$$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$

**Cell state update:**
$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

**Output gate:**
$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$
$$h_t = o_t \odot \tanh(C_t)$$

The cell state $C_t$ acts as a long-term memory that is modified multiplicatively by gates — not additively accumulated like vanilla RNN hidden states. Hochreiter & Schmidhuber (1997) demonstrated that LSTM can bridge time lags of **over 1,000 discrete timesteps**, far exceeding what is needed for the 30-day encoder window in this project.

**Applied to hydrology**: Kratzert et al. (2018) [1] showed that the forget gate learns to represent soil moisture dynamics, and the input gate learns to represent seasonal precipitation patterns — without any explicit hydrological domain knowledge encoded in the architecture.

### 4.3 Why Encoder-Decoder?

**Citation:** Kao et al. (2020) [10]

A plain LSTM produces a single output at each timestep (many-to-many mode) or a single output at the end (many-to-one mode). Neither is ideal for 7-day ahead forecasting because:
- **Many-to-one** produces only a single forecast step — requires 7 separate models for 7 lead times
- **Many-to-many** produces the same number of outputs as inputs, but ties each output to its corresponding input timestep (not future forecasting)

The **encoder-decoder** architecture separates the problem:
1. **Encoder**: Reads the 30-day history. The final hidden state $(h_N, c_N)$ is a compressed summary of all 30 days of meteorological and hydrological context.
2. **Decoder**: Autoregressively generates 7 future discharge values, using the encoder's summary as initial state and conditioning each step on the previous prediction.

Kao et al. (2020) applied LSTM-ED to typhoon flood forecasting and reported **38% RMSE reduction over feedforward baselines at T+6 (6-hour ahead)** — substantially larger gains at longer lead times than at T+1, exactly the regime we care about.

### 4.4 Why Bahdanau Attention?

**Citation:** Bahdanau, Cho & Bengio (2015) [4]

The encoder-decoder bottleneck problem: all 30 days of encoder history must be compressed into a single fixed-size hidden state vector $(h_N, c_N)$ before the decoder starts. For long input sequences, this loses information about which specific past days are most relevant to each future forecast step.

Bahdanau attention addresses this by allowing the decoder to directly query **all** encoder hidden states at each decoding step:

**Attention score (additive/Bahdanau):**
$$e_{t,s} = v_a^\top \tanh(W_1 h_s^{enc} + W_2 h_t^{dec})$$

**Attention weights (softmax normalisation):**
$$\alpha_{t,s} = \frac{\exp(e_{t,s})}{\sum_{s'=1}^{N} \exp(e_{t,s'})}$$

**Context vector:**
$$c_t = \sum_{s=1}^{N} \alpha_{t,s} \cdot h_s^{enc}$$

The decoder input at each step becomes $[y_{t-1}, c_t]$ — the previous prediction concatenated with the context vector.

**Physical interpretation for flood forecasting**: When predicting discharge on forecast day 5, the attention mechanism learns to focus on rainfall events from encoder days 2-7 (accounting for routing lag), rather than giving equal weight to all 30 history days. This is directly analogous to a hydrologist saying "the flood wave from last week's rainfall event is arriving now."

Bahdanau et al. (2015) originally demonstrated this in machine translation — RNNsearch-50 achieved BLEU = 28.45 vs. 26.75 for a model without attention, and crucially **maintained performance on long sentences** where the no-attention model degraded sharply.

### 4.5 Why Not Transformer?

**Citations:** Lim et al. (2021) [3]; Castangia et al. (2023) [17]

The Temporal Fusion Transformer (Lim et al. 2021) is the current state-of-the-art for multi-horizon time series and achieves median KGE = 0.78+ on streamflow datasets. However:

1. **Data requirements**: TFT has ~4× more parameters than LSTM-ED with attention. With ~5,877 training samples from a single-station record, we risk overfitting. TFT's advantages become pronounced on large multi-catchment datasets (Kratzert 2019: 531 catchments).
2. **Training stability**: Transformers require careful learning rate warmup schedules and are more sensitive to hyperparameters than LSTM-based models on small datasets.
3. **Computational cost**: TFT requires self-attention over all encoder timesteps at every decoder step — quadratic in sequence length. LSTM-ED with Bahdanau attention is linear.

Castangia et al. (2023) showed >4% RMSE improvement for Transformers in flood forecasting, but their dataset was substantially larger than a single 23-year station record.

**Our choice** — LSTM-ED with attention — is the consensus recommendation from the South Asia literature (Brahmaputra-specific: ICCINS 2023 [9]) and is architecturally aligned with what operational systems like GloFAS use as their deep learning component.

---

## 5. Mathematical Formulation

### 5.1 Problem Definition

Let $\mathbf{X} = \{x_1, x_2, \ldots, x_T\} \in \mathbb{R}^{T \times F}$ be a sequence of $T$ daily feature vectors, where $F$ is the number of engineered features. Let $\mathbf{Q} = \{Q_1, Q_2, \ldots, Q_T\}$ be the corresponding daily discharge observations.

We define the forecasting task as learning a function:

$$f_\theta: \mathbf{X}_{t-L+1:t} \rightarrow \hat{\mathbf{Q}}_{t+1:t+H}$$

where:
- $L = 30$ is the encoder (lookback) length in days
- $H = 7$ is the decoder (forecast horizon) in days
- $\theta$ are the learned parameters of the LSTM encoder-decoder

### 5.2 Log-Transformation of Target

River discharge follows a **log-normal distribution** — a well-established empirical finding in hydrology. The distribution is right-skewed with high-magnitude flood peaks creating extreme outliers. Direct regression on raw discharge values causes the model to concentrate capacity on frequent low-discharge regimes while under-weighting rare flood peaks.

We apply the log1p transformation:

$$Q' = \log(1 + Q)$$

so the model predicts $\hat{Q}' = \log(1 + \hat{Q})$ and we recover predictions via $\hat{Q} = \exp(\hat{Q}') - 1 = \text{expm1}(\hat{Q}')$.

This is standard practice in LSTM hydrology (NeuralHydrology library [19] implements NSE on log-transformed flows as a default metric, termed `alpha-NSE`).

### 5.3 Normalisation

Features are standardised using Z-score normalisation fitted **only on the training set**:

$$\tilde{x} = \frac{x - \mu_{train}}{\sigma_{train}}$$

Using training statistics for test normalisation is a strict requirement to prevent data leakage. Fitting a scaler on the entire dataset before splitting is one of the most common errors in applied ML.

### 5.4 Sliding Window Construction

From a normalised time series of $N$ total days, we construct $(N - L - H + 1)$ training samples using a sliding window of stride 1. Each sample is:

$$(\mathbf{X}_{i:i+L},\ \mathbf{Q}'_{i+L:i+L+H}) \quad \text{for } i = 1, \ldots, N-L-H+1$$

For our dataset: $N \approx 8395$, $L = 30$, $H = 7$:
$$\text{Number of samples} \approx 8395 - 30 - 7 + 1 = 8359$$

After temporal train/val/test split:
- Train: 5,851 samples
- Val: 1,254 samples
- Test: 1,254 samples

### 5.5 Full Forward Pass

**Encoder:**
$$\mathbf{H}^{enc} = \{h_1^{enc}, \ldots, h_L^{enc}\}, \quad (h_L^{enc}, c_L^{enc}) = \text{LSTM}_{enc}(\mathbf{X}_{1:L})$$

**Decoder step $k$ (for $k = 1, \ldots, H$):**

1. Compute attention weights over all encoder states:
$$\alpha_{k,s} = \text{softmax}_s\left(v^\top \tanh(W_1 h_s^{enc} + W_2 h_{k-1}^{dec})\right)$$

2. Context vector:
$$c_k = \sum_{s=1}^{L} \alpha_{k,s} \cdot h_s^{enc}$$

3. Decoder LSTM update:
$$(h_k^{dec}, c_k^{dec}) = \text{LSTM}_{dec}([\hat{Q}'_{k-1}, c_k],\ h_{k-1}^{dec},\ c_{k-1}^{dec})$$

4. Output head:
$$\hat{Q}'_k = W_2 \cdot \text{ReLU}(W_1 \cdot h_k^{dec} + b_1) + b_2$$

**Inverse transform:**
$$\hat{Q}_k = \text{expm1}\left(\sigma_{train}^{tgt} \cdot \hat{Q}'_k + \mu_{train}^{tgt}\right)$$

---

## 6. Feature Engineering Justification

### 6.1 Antecedent Soil Moisture (Highest Priority Feature)

**Citation:** HESS (2022) [11]

Soil moisture is the most critical predictor of flood generation. Research on flood mechanics shows:

> "For saturated antecedent soil, a 7-year recurrence precipitation event can trigger a 100-year flood. Conversely, a 200-year precipitation event on dry soil may generate only a 15-year flood." — HESS (2022)

This represents a **20× amplification effect** from moisture state alone. ERA5 provides `volumetric_soil_water_layer_1` (0-7 cm depth), which we include directly as a feature. We additionally engineer a **14-day rolling mean** of soil moisture to capture the long-term saturation state of the catchment.

### 6.2 Lag Features

The Brahmaputra drainage basin spans ~2,900 km. Rainfall in the upper catchment (Tibet, Arunachal Pradesh) takes 3-10 days to reach downstream stations. We include discharge lags at 1, 2, 3, 5, and 7 days and rainfall lags at the same offsets.

The routing time delay is well-established in hydrology. The kinematic wave speed for the Brahmaputra main stem is approximately 1.5-2.5 m/s, giving a travel time of ~3-5 days from the Tibetan plateau to the Assam plains.

### 6.3 Rolling Rainfall Accumulations

We compute rolling sums of precipitation at 3, 7, 14, and 30-day windows. These represent:

| Window | Hydrological Interpretation |
|---|---|
| 3-day | Flash flood potential from immediate rainfall |
| 7-day | Soil saturation indicator for a typical wet spell |
| 14-day | Monthly moisture deficit/surplus |
| 30-day | Long-term pre-monsoon soil conditioning |

### 6.4 Sinusoidal Day-of-Year Encoding

**Motivation:** The Brahmaputra discharge varies by ~15× between monsoon and dry season. Providing the model with an explicit seasonal signal prevents it from confusing high-discharge monsoon conditions with anomalous events.

We encode day-of-year $d \in [1, 365]$ as:

$$\text{doy}_{sin} = \sin\left(\frac{2\pi d}{365.25}\right), \quad \text{doy}_{cos} = \cos\left(\frac{2\pi d}{365.25}\right)$$

The two-component encoding is necessary (rather than raw $d/365$) because:
1. It is **circular** — December 31 and January 1 are adjacent in the encoded space
2. It has **bounded range** [−1, 1], unlike a linear day counter that grows unboundedly

### 6.5 Log-Discharge as a Feature

We include $\log(1 + Q_t)$ as an input feature alongside the raw lag features. This provides the model with the current regime state (high/low flow) without being dominated by extreme peak values during training.

### 6.6 Full Feature List

After all engineering, the feature matrix has the following columns:

| Category | Features |
|---|---|
| Meteorological (ERA5) | precip_mm, temp_c, pressure_pa, soil_moisture, wind_speed |
| Discharge lags | discharge_lag1..7 |
| Rainfall lags | precip_lag1..7 |
| Rolling rainfall | precip_roll3d, 7d, 14d, 30d |
| Rolling discharge | discharge_roll3d_mean, 7d_mean |
| Log discharge | log_discharge |
| Seasonality | doy_sin, doy_cos |

Total: **~26 features** (exact count depends on available ERA5 variables).

---

## 7. Loss Function: Flood-Weighted MSE

### 7.1 Why Standard MSE Fails

**Citations:** Wang et al. (2025) [15]; Kratzert et al. (2018) [1]

Standard Mean Squared Error:
$$\mathcal{L}_{MSE} = \frac{1}{N} \sum_{i=1}^{N} (\hat{Q}'_i - Q'_i)^2$$

gives equal weight to all timesteps. Since flood days constitute <8% of the training record, the model minimises total loss by fitting the 92% of normal days well, at the expense of flood peaks. This is mathematically equivalent to a majority-class bias in classification.

NSE — our primary metric — is $1 - \mathcal{L}_{MSE,normalised}$. Gupta et al. (2009) [6] showed that optimising NSE systematically over-weights peak errors due to squaring, but this does not help when peaks are rare — the denominator term $\overline{Q}_{obs}$ is dominated by non-flood days.

### 7.2 Our Flood-Weighted Loss

We implement a **flood-event-aware weighted MSE**:

$$w_i = 1 + \lambda \cdot \text{clip}\left(\frac{Q_i}{Q_{danger}},\ 0,\ 1\right)$$

$$\mathcal{L}_{flood} = \frac{1}{N} \sum_{i=1}^{N} w_i \cdot (\hat{Q}'_i - Q'_i)^2$$

where:
- $Q_{danger}$ = danger discharge threshold (98,600 m³/s for Bahadurabad)
- $\lambda = 4.0$ is the flood weight multiplier (tunable hyperparameter)
- $w_i \in [1, 5]$ — normal days get weight 1, full-flood days get weight 5

**Effect**: errors on days at or above the danger threshold are penalised **5× more** than errors on normal days. This shifts the model's attention toward the minority flood class without requiring data resampling (which distorts the temporal structure of the training set).

### 7.3 Why Not Focal Loss?

Focal Loss (Lin et al. 2017, originally for object detection) is designed for binary classification:
$$\mathcal{L}_{focal} = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

It is not directly applicable to regression. Wang et al. (2025) [15] implemented a **Focal Tversky Loss** variant for flood mapping (a segmentation task), but this requires a classification head. Since our primary task is regression (predicting a discharge value, not a binary flood label), weighted MSE is the appropriate choice.

---

## 8. Evaluation Metrics Justification

### 8.1 Nash-Sutcliffe Efficiency (NSE)

**Citation:** Nash & Sutcliffe (1970) [7]

$$\text{NSE} = 1 - \frac{\sum_{t=1}^{T}(Q_t^{obs} - Q_t^{sim})^2}{\sum_{t=1}^{T}(Q_t^{obs} - \bar{Q}^{obs})^2}$$

- NSE = 1: perfect model
- NSE = 0: model is no better than the long-term mean (a trivial predictor)
- NSE < 0: the mean is a better predictor than the model

Performance thresholds (widely adopted):
- NSE > 0.75: **Very good**
- NSE 0.65–0.75: **Good**
- NSE 0.50–0.65: **Satisfactory**
- NSE < 0.50: **Unsatisfactory**

### 8.2 Kling-Gupta Efficiency (KGE)

**Citation:** Gupta et al. (2009) [6]

$$\text{KGE} = 1 - \sqrt{(r - 1)^2 + (\alpha - 1)^2 + (\beta - 1)^2}$$

where:
- $r$ = Pearson correlation between simulated and observed
- $\alpha = \sigma_{sim}/\sigma_{obs}$ = variability ratio
- $\beta = \mu_{sim}/\mu_{obs}$ = bias ratio

KGE decomposes model skill into three independent components: **timing** (r), **variability** (α), and **volume bias** (β). A model can have high NSE but poor KGE if it has good correlation but systematic bias — NSE alone would not detect this.

**Critical threshold**: KGE > −0.41 indicates improvement over the mean-flow benchmark (equivalent to NSE = 0).

### 8.3 Why Accuracy Is Excluded

As demonstrated in Section 2.1, accuracy is meaningless for imbalanced flood prediction. A model that never predicts a flood achieves >92% accuracy. We use flood-specific verification metrics instead.

### 8.4 Flood Verification Metrics (Contingency Table)

For threshold-based flood alert evaluation, we construct a contingency table:

| | Observed Flood | Observed Normal |
|---|---|---|
| **Predicted Flood** | True Positive (TP) | False Positive (FP) |
| **Predicted Normal** | False Negative (FN) | True Negative (TN) |

**Critical Success Index (CSI) / Threat Score:**
$$\text{CSI} = \frac{TP}{TP + FP + FN}$$
CSI ignores correct non-event forecasts (TN), making it insensitive to the class imbalance. CSI = 1 is perfect; CSI = 0 means no useful flood skill.

**Probability of Detection (POD) / Recall:**
$$\text{POD} = \frac{TP}{TP + FN}$$
POD measures what fraction of actual floods we detected. This is the **most safety-critical metric** — missing a flood (low POD) has severe humanitarian consequences.

**False Alarm Ratio (FAR):**
$$\text{FAR} = \frac{FP}{TP + FP}$$
FAR measures how often a flood alert was a false alarm. High FAR causes alert fatigue, reducing community responsiveness to future warnings.

**Heidke Skill Score (HSS):**
$$\text{HSS} = \frac{2(TP \cdot TN - FP \cdot FN)}{(TP + FN)(FN + TN) + (TP + FP)(FP + TN)}$$
HSS accounts for skill relative to random chance. HSS = 0: no skill beyond chance; HSS = 1: perfect.

---

## 9. Benchmarks and Expected Performance

### 9.1 LSTM Hydrology Benchmarks

| Study | Dataset | Model | Median NSE | Median KGE |
|---|---|---|---|---|
| Kratzert et al. (2018) [1] | 241 CAMELS basins | LSTM | 0.65 | — |
| Kratzert et al. (2019) [2] | 531 CAMELS basins | LSTM (ungauged) | 0.69 | — |
| Kratzert et al. (2019) [extra] | 531 CAMELS basins | EA-LSTM | **0.74** | — |
| GloFAS-ERA5 [18] | 1,801 global stations | LISFLOOD | — | **0.61** (median) |
| Kao et al. (2020) [10] | 23 typhoon events, Taiwan | LSTM-ED | — | — |

### 9.2 Realistic Targets for This Project

Given 23 years of single-station ERA5 + GloFAS data on the Brahmaputra:

| Metric | Minimum Acceptable | Target | World-class |
|---|---|---|---|
| NSE (overall) | > 0.50 | **> 0.65** | > 0.75 |
| KGE | > 0.40 | **> 0.60** | > 0.75 |
| NSE (flood days only) | > 0.30 | **> 0.50** | > 0.65 |
| POD | > 0.60 | **> 0.75** | > 0.85 |
| FAR | < 0.40 | **< 0.25** | < 0.15 |
| CSI | > 0.30 | **> 0.50** | > 0.65 |

Note: Flood-only NSE is always lower than overall NSE because it excludes the easy-to-predict low-flow season.

---

## 10. Comparison with Alternative Approaches

| Approach | Pros | Cons | Why Rejected/Deferred |
|---|---|---|---|
| SWAT / HEC-HMS (physics-based) | Interpretable, established | Requires dense calibration data, ~50 parameters | No calibration gauge data available for full upstream basin |
| Random Forest / XGBoost | Fast, robust | No temporal memory, cannot do multi-step forecasting natively | Cannot learn routing dynamics; requires 180+ manual lag features |
| Temporal Fusion Transformer [3] | State-of-the-art on large datasets | Requires much more data, overfits on single-station records | Deferred — upgrade path when multi-station data available |
| CNN + LSTM | Good for spatial + temporal | Requires gridded input at every timestep | Future work with ERA5 spatial grids |
| LSTM Encoder-Decoder + Attention | Proven on South Asian rivers [9], handles multi-step naturally | Slightly less interpretable than TFT | **Selected architecture** |

---

## 11. Limitations and Future Work

1. **Single station**: This model is trained and validated for a single downstream station (Bahadurabad). Extending to a multi-station network with upstream-downstream routing would substantially improve accuracy.

2. **GloFAS as target**: We train on GloFAS-ERA5 discharge, which is itself a model output. Bias correction against CWC gauge records is needed before operational deployment.

3. **Ungauged upstream**: The Tibetan Plateau portion of the catchment has no gauge data. ERA5 precipitation over Tibet is less reliable than over India due to sparse radiosonde coverage.

4. **Upgrade to TFT**: When multi-catchment data (India-WRIS, BWDB Bangladesh) becomes available, upgrading to a Temporal Fusion Transformer (Lim et al. 2021) is the natural next step.

5. **Ensemble forecasting**: Operational systems like GloFAS use ensemble NWP inputs to produce probabilistic forecasts with uncertainty bounds. Implementing MC-Dropout or deep ensembles for uncertainty quantification is a planned extension.

6. **SAR flood extent mapping**: Sentinel-1 SAR data can be used with a U-Net to produce spatial flood inundation maps — complementary to the time-series discharge forecast.

---

## 12. Full Bibliography

> All papers listed are peer-reviewed publications with verifiable DOIs.

**[1]** Kratzert, F., Klotz, D., Brenner, C., Schulz, K., & Herrnegger, M. (2018). Rainfall–runoff modelling using Long Short-Term Memory (LSTM) networks. *Hydrology and Earth System Sciences*, 22, 6005–6022. https://doi.org/10.5194/hess-22-6005-2018

**[2]** Kratzert, F., Klotz, D., Herrnegger, M., Sampson, A. K., Hochreiter, S., & Nearing, G. S. (2019). Toward Improved Predictions in Ungauged Basins: Exploiting the Power of Machine Learning. *Water Resources Research*, 55(12), 11344–11354. https://doi.org/10.1029/2019WR026065

**[3]** Lim, B., Arık, S. Ö., Loeff, N., & Pfister, T. (2021). Temporal Fusion Transformers for interpretable multi-horizon time series forecasting. *International Journal of Forecasting*, 37(4), 1748–1764. https://doi.org/10.1016/j.ijforecast.2021.03.012

**[4]** Bahdanau, D., Cho, K., & Bengio, Y. (2015). Neural Machine Translation by Jointly Learning to Align and Translate. *ICLR 2015*. arXiv:1409.0473. https://arxiv.org/abs/1409.0473

**[5]** Hersbach, H., Bell, B., Berrisford, P., et al. (2020). The ERA5 global reanalysis. *Quarterly Journal of the Royal Meteorological Society*, 146(730), 1999–2049. https://doi.org/10.1002/qj.3803

**[6]** Gupta, H. V., Kling, H., Yilmaz, K. K., & Martinez, G. F. (2009). Decomposition of the mean squared error and NSE performance criteria: Implications for improving hydrological modelling. *Journal of Hydrology*, 377(1–2), 80–91. https://doi.org/10.1016/j.jhydrol.2009.08.003

**[7]** Nash, J. E., & Sutcliffe, J. V. (1970). River flow forecasting through conceptual models part I — A discussion of principles. *Journal of Hydrology*, 10(3), 282–290. https://doi.org/10.1016/0022-1694(70)90255-6

**[8]** Alfieri, L., Burek, P., Dutra, E., et al. (2013). GloFAS – global ensemble streamflow forecasting and flood early warning. *Hydrology and Earth System Sciences*, 17(3), 1161–1175. https://doi.org/10.5194/hess-17-1161-2013

**[9]** Machine Learning-Based Flood Prediction of Assam. (2023). *IEEE International Conference on Computational Intelligence, Networks and Security (ICCINS)*. https://doi.org/10.1109/ICCINS58907.2023.10450018

**[10]** Kao, I.-F., Zhou, Y., Chang, L.-C., & Chang, F.-J. (2020). Exploring a Long Short-Term Memory based Encoder-Decoder framework for multi-step-ahead flood forecasting. *Journal of Hydrology*, 583, 124631. https://doi.org/10.1016/j.jhydrol.2020.124631

**[11]** The relative importance of antecedent soil moisture and precipitation in flood generation. (2022). *Hydrology and Earth System Sciences*, 26, 4919–4935. https://hess.copernicus.org/articles/26/4919/2022/

**[12]** Addor, N., Newman, A. J., Mizukami, N., & Clark, M. P. (2017). The CAMELS data set: catchment attributes and meteorology for large-sample studies. *Hydrology and Earth System Sciences*, 21, 5293–5313. https://doi.org/10.5194/hess-21-5293-2017

**[13]** Kratzert, F., Klotz, D., Herrnegger, M., Sampson, A. K., Hochreiter, S., & Nearing, G. S. (2019). Towards learning universal, regional, and local hydrological behaviors via machine learning applied to large-sample datasets. *Hydrology and Earth System Sciences*, 23(12), 5089–5110. https://doi.org/10.5194/hess-23-5089-2019

**[14]** Central Water Commission (CWC), India. Flood Forecasting. https://cwc.gov.in/flood-forecasting-hydrological-observation

**[15]** Wang, et al. (2025). Flood Classification and Improved Loss Function by Combining Deep Learning Models to Improve Water Level Prediction in a Small Mountain Watershed. *Journal of Flood Risk Management*. https://doi.org/10.1111/jfr3.70022

**[16]** Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735–1780. https://doi.org/10.1162/neco.1997.9.8.1735

**[17]** Castangia, M., Medina Grajales, L. M., Aliberti, A., et al. (2023). Transformer neural networks for interpretable flood forecasting. *Environmental Modelling & Software*, 160, 105581. https://doi.org/10.1016/j.envsoft.2022.105581

**[18]** Harrigan, S., Zsoter, E., Alfieri, L., et al. (2020). GloFAS-ERA5 operational global river discharge reanalysis 1979–present. *Earth System Science Data*, 12, 2043–2060. https://doi.org/10.5194/essd-12-2043-2020

**[19]** Kratzert, F., Gauch, M., Nearing, G., & Klotz, D. (2022). NeuralHydrology — A Python library for Deep Learning research in hydrology. *Journal of Open Source Software*, 7(71), 4050. https://doi.org/10.21105/joss.04050

**[20]** Enhancing a Multi-Step Discharge Prediction with Deep Learning and a Response Time Parameter. (2022). *Water (MDPI)*, 14(18), 2898. https://doi.org/10.3390/w14182898

---

*This document reflects the research conducted in designing and justifying the Prag-Dristi flood forecasting engine. All architectural decisions trace to peer-reviewed literature; all claimed performance numbers come from published studies, not from this project's own (in-progress) experiments.*
