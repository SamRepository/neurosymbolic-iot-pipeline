python -c "import pandas as pd; df=pd.read_parquet('data/processed/casas_windows.parquet'); print('CASAS labels:', df['label'].nunique()); print(df['label'].value_counts().head(15))"
python -c "import pandas as pd; df=pd.read_parquet('data/processed/sphere_windows.parquet'); print('SPHERE labels:', df['label'].nunique()); print(df['label'].value_counts().head(15))"



## Step 1 — Verify the new CASAS labels in your processed windows

python -c "import pandas as pd; df=pd.read_parquet('data/processed/casas_windows.parquet'); print(df.columns.tolist()); print('n_windows', df['window_id'].nunique()); print('labels', df['label'].nunique()); print(df['label'].value_counts().head(10))"


python -m evaluation.run_experiments --config config/base.yaml --datasets casasIoT Paper\Experimentation\neurosymbolic-iot-pipeline>




## Quick checks to confirm what you actually have on disk

Run these from the repo root.

### PowerShell: find any Torch models

<pre class="overflow-visible! px-0!" data-start="1274" data-end="1359"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"></div></pre>

<pre class="overflow-visible! px-0!" data-start="1274" data-end="1359"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-powershell"><span><span>Get-ChildItem</span><span></span><span>-Recurse</span><span></span><span>-Include</span><span> *.pt,*.pth | </span><span>Select-Object</span><span> FullName
</span></span></code></div></div></pre>

### PowerShell: show scikit models

<pre class="overflow-visible! px-0!" data-start="1396" data-end="1479"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"></div></pre>

<pre class="overflow-visible! px-0!" data-start="1396" data-end="1479"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-powershell"><span><span>Get-ChildItem</span><span></span><span>-Recurse</span><span></span><span>-Include</span><span> *.joblib | </span><span>Select-Object</span><span> FullName
</span></span></code></div></div></pre>

### Python: search for any saved weights

<pre class="overflow-visible! px-0!" data-start="1522" data-end="1726"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"></div></pre>

<pre class="overflow-visible! px-0!" data-start="1522" data-end="1726"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-bash"><span><span>python -c </span><span>"import pathlib; p=pathlib.Path('.'); print('PT/PTH:', [str(x) for x in p.rglob('*.pt')]+[str(x) for x in p.rglob('*.pth')]); print('JOBLIB:', [str(x) for x in p.rglob('*.joblib')])"</span><span>
</span></span></code></div></div></pre>

If you only see `model.joblib` under `outputs/evaluation/...`, then you only trained the Phase-02 scikit baseline.

---


After you paste it, you will be able to run:

* **Activity GRU (multi-class):**

<pre class="overflow-visible! px-0!" data-start="436" data-end="569"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"></div></pre>

<pre class="overflow-visible! px-0!" data-start="436" data-end="569"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-powershell"><span><span>python </span><span>-m</span><span> neurosymbolic_iot.cli.train_neural </span><span>--config</span><span> config/neural_perception.yaml </span><span>--dataset</span><span> casas </span><span>--task</span><span> activity
</span></span></code></div></div></pre>

* **Transition GRU (binary):**

<pre class="overflow-visible! px-0!" data-start="602" data-end="737"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"></div></pre>

<pre class="overflow-visible! px-0!" data-start="602" data-end="737"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-powershell"><span><span>python </span><span>-m</span><span> neurosymbolic_iot.cli.train_neural </span><span>--config</span><span> config/neural_perception.yaml </span><span>--dataset</span><span> casas </span><span>--task</span><span> transition
</span></span></code></div></div></pre>


## Metrics snapshot (baseline)

* **CASAS (LogisticRegression)** test: accuracy ≈ 0.6596, F1-macro ≈ 0.5228, F1-weighted ≈ 0.6705.

**SPHERE (RandomForest)** test: accuracy ≈ 0.6190, F1-macro ≈ 0.4106, F1-weighted ≈ 0.5804.

If you want, I can suggest 1–2 quick baseline tweaks (still Phase 02) to improve SPHERE macro-F1 (class imbalance is strong with very small `a_ascend/t_turn` counts).



# Phase 2 — CASAS (Kyoto ADL Errors) Reproducible Experiments

## Scope (what Phase 2 covered for CASAS)

Phase 2 focused on **Neural Perception** for CASAS with two complementary tasks:

1. **Activity Recognition (multi-class)** on windowed CASAS features (high-dimensional `AD1-*` style features).
2. **Transition Detection (binary)** using a **GRU sequence classifier** (predict whether the current window is a transition between activities).

All runs were executed with a fixed  **seed = 42** .

---

## 1) Dataset: required folder structure

Ensure you are using the **Kyoto ADL Errors** dataset path:

<pre class="overflow-visible! px-0!" data-start="833" data-end="980"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"></div></pre>

<pre class="overflow-visible! px-0!" data-start="833" data-end="980"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre!"><span><span>neurosymbolic-iot-pipeline/
  </span><span>data</span><span>/
    raw/
      casas_kyoto_adl/
        adl_error/
          *.csv
        adl_noerror/
          *.csv
</span></span></code></div></div></pre>

### Important note

If your config points to `data/raw/casas` you might load a different dataset layout (or logs with different parsing assumptions). For  **Kyoto ADL Errors** , always point to:

* `data/raw/casas_kyoto_adl`
* and use file globs for `adl_error/*.csv` and `adl_noerror/*.csv`.

---

## 2) Config: use the correct `datasets.casas` block

In `config/base.yaml`, update the CASAS dataset section from:

<pre class="overflow-visible! px-0!" data-start="1395" data-end="1506"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"></div></pre>

<pre class="overflow-visible! px-0!" data-start="1395" data-end="1506"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-yaml"><span><span>datasets:</span><span>
  </span><span>casas:</span><span>
    </span><span>raw_dir:</span><span></span><span>data/raw/casas</span><span>
    </span><span>file_globs:</span><span> [</span><span>"**/*.txt"</span><span>, </span><span>"**/*.log"</span><span>, </span><span>"**/*.csv"</span><span>]
</span></span></code></div></div></pre>

to the **Kyoto ADL Errors** version:

<pre class="overflow-visible! px-0!" data-start="1546" data-end="1917"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"></div></pre>

<pre class="overflow-visible! px-0!" data-start="1546" data-end="1917"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-yaml"><span><span>datasets:</span><span>
  </span><span>casas:</span><span>
    </span><span>raw_dir:</span><span></span><span>data/raw/casas_kyoto_adl</span><span>

    </span><span># IMPORTANT: use the Kyoto ADL Errors loader</span><span>
    </span><span>format:</span><span></span><span>kyoto_adl_errors</span><span>

    </span><span>file_globs:</span><span>
      </span><span>-</span><span></span><span>"adl_error/*.csv"</span><span>
      </span><span>-</span><span></span><span>"adl_noerror/*.csv"</span><span>

    </span><span>timezone:</span><span></span><span>"UTC"</span><span>

    </span><span>window_minutes:</span><span></span><span>30</span><span>
    </span><span>stride_minutes:</span><span></span><span>5</span><span>
    </span><span>min_events_per_window:</span><span></span><span>1</span><span>

    </span><span>label_mode:</span><span></span><span>majority</span><span>
    </span><span>feature_mode:</span><span></span><span>event_counts</span><span>
</span></span></code></div></div></pre>

This ensures the pipeline uses:

`.../data/raw/casas_kyoto_adl/adl_error` and `.../adl_noerror`, not `.../data/raw/casas`.

---

## 3) Core windowing parameters (CASAS)

We consistently used:

* `window_minutes = 30`
* `stride_minutes = 5`
* `min_events_per_window = 1`
* `label_mode = majority`
* `feature_mode = event_counts`

These settings produce windowed samples (examples: `n_train=284, n_val=61, n_test=61` in one of the best activity runs).

---

## 4) Sanity checks (data + features)

These quick checks helped confirm correctness before training:

### 4.1 Transition label distribution

We checked transition balance on the window dataframe:

<pre class="overflow-visible! px-0!" data-start="2571" data-end="2675"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"></div></pre>

<pre class="overflow-visible! px-0!" data-start="2571" data-end="2675"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-python"><span><span>print</span><span>(df[</span><span>"transition"</span><span>].value_counts())
</span><span>print</span><span>(</span><span>"transition_rate ="</span><span>, df[</span><span>"transition"</span><span>].mean())
</span></span></code></div></div></pre>

Example observed:

* `transition=0: 285`
* `transition=1: 121`
* `transition_rate ≈ 0.298`

During training, the script also logs it, for example:

* `CASAS transition task: transition_rate=0.3705 (n=305)`
* `CASAS transition task: transition_rate=0.3000 (n=400)`

(Exact rates can vary depending on the config used to generate windows and whether timelines were mixed; see warnings below.)

### 4.2 Feature dimensionality

We verified the wide feature space:

* `AD1-* columns: 2512`
* `Total columns: 2566`

This confirmed feature extraction and column construction were working as expected.

---

## 5) Phase 2A — Activity Recognition (best baseline results)

We obtained strong results with a **Logistic Regression** baseline:

**Model:** `LogisticRegression(solver="saga", class_weight="balanced", max_iter=5000, random_state=42)`

**Classes (5):** `Clean, Cook, Eat, PhoneCall, WashHands`

**Validation:**

* accuracy ≈ `0.9180`
* f1_macro ≈ `0.8661`
* f1_weighted ≈ `0.9221`

**Test:**

* accuracy ≈ `0.9508`
* f1_macro ≈ `0.8447`
* f1_weighted ≈ `0.9429`

### Artifacts produced (Activity evaluation)

A typical evaluation run produced:

<pre class="overflow-visible! px-0!" data-start="3818" data-end="4022"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"></div></pre>

<pre class="overflow-visible! px-0!" data-start="3818" data-end="4022"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre!"><span><span>outputs/
  evaluation/
    <TAG>/
      casas/
        model.joblib
        label_encoder.joblib
        feature_cols.json
        metrics.json
        confusion_val.csv
        confusion_test.csv
</span></span></code></div></div></pre>

A run manifest example we saved contained:

* `tag: 20251215_103018`
* `config: config\base.yaml`
* `dataset: casas`
* output dir: `outputs\evaluation\20251215_103018\casas`

### How to verify the run quickly

After the run:

1. Inspect metrics:
   * `outputs/evaluation/<TAG>/casas/metrics.json`
2. Inspect confusion matrices (CSV):
   * `confusion_val.csv`
   * `confusion_test.csv`
3. Confirm feature columns were persisted:
   * `feature_cols.json`

---

## 6) Phase 2B — Transition Detection (GRU, binary)

### Command used

We trained the transition classifier with:

<pre class="overflow-visible! px-0!" data-start="4595" data-end="4711"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"></div></pre>

<pre class="overflow-visible! px-0!" data-start="4595" data-end="4711"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-bash"><span><span>python -m neurosymbolic_iot.cli.train_neural --config config/base.yaml --dataset casas --task transition
</span></span></code></div></div></pre>

### Expected logging (healthy run)

You should see logs similar to:

* Device selection:
  * `Using device: cpu`
* Output directory:
  * `Run output dir: outputs/neural_perception/<TAG>/casas/transition`
* Dataset load:
  * `Loading CASAS files: 100%|...| 26/26 ...`
* Transition info:
  * `CASAS transition task: transition_rate=... (n=...)`
* Training loop:
  * `Epoch 1 | train_loss=... | val_f1_macro=...`
  * …
* Final:
  * `Saved CASAS model to .../model.pth`
  * `Metrics written to: .../metrics.json`

### Artifacts produced (Transition GRU)

A typical run produces:

<pre class="overflow-visible! px-0!" data-start="5287" data-end="5477"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"></div></pre>

<pre class="overflow-visible! px-0!" data-start="5287" data-end="5477"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre!"><span><span>outputs/
  neural_perception/
    <TAG>/
      casas/
        transition/
          model.pth
          metrics.json
          label_map.json
          vocab.json
          task.json
</span></span></code></div></div></pre>

Note: your `/models` folder can remain empty; the neural pipeline saves models under `outputs/neural_perception/...` by design.

---

## 7) Important warnings and what they mean

### 7.1 “No grouping columns found”

You saw:

`CASAS: no grouping columns (e.g., file/session) found. Transition will be computed globally, which may mix independent timelines.`

Meaning:

* Transition labels were computed by comparing each window to the  **previous window globally** .
* If your windows span multiple independent timelines (e.g., different files/participants), this can inflate/deflate transition rate and confuse the classifier.

Recommended improvement (later):

* Ensure your window builder carries a `source_file` / `session_id` column and compute transitions  **per group** .

### 7.2 Config mismatch (`KeyError: 'datasets'`)

This occurred when running with `config/neural_perception.yaml` that did not include a `datasets:` section.

Fix:

* Use `config/base.yaml`, or
* Add `datasets.casas` to `neural_perception.yaml` (same as base) so it is self-contained.

### 7.3 Timezone dtype issues (tz-aware vs tz-naive)

We hit errors like:

* `TypeError: Cannot interpret 'datetime64[ns, UTC]' as a data type`
* `TypeError: Cannot compare tz-naive and tz-aware datetime-like objects`
* `Invalid comparison between dtype=datetime64[ns, UTC] and Timestamp`

Fix applied in code:

* Normalize timestamps consistently, typically:
  * parse as UTC
  * then drop timezone to get UTC-naive `datetime64[ns]`

This prevented failures inside window slicing and event filtering.

---

## 8) Minimal reproducibility checklist (CASAS Phase 2)

1. **Confirm dataset folders**

   * `data/raw/casas_kyoto_adl/adl_error`
   * `data/raw/casas_kyoto_adl/adl_noerror`
2. **Confirm `base.yaml` points to Kyoto ADL Errors**

   * `raw_dir: data/raw/casas_kyoto_adl`
   * `format: kyoto_adl_errors`
   * correct `file_globs`
3. **Run transition GRU**

   <pre class="overflow-visible! px-0!" data-start="7399" data-end="7521"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"></div></pre>
4. <pre class="overflow-visible! px-0!" data-start="7399" data-end="7521"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-bash"><span><span>python -m neurosymbolic_iot.cli.train_neural --config config/base.yaml --dataset casas --task transition
   </span></span></code></div></div></pre>
5. **Confirm outputs exist**

   * `outputs/neural_perception/<TAG>/casas/transition/model.pth`
   * `.../metrics.json`
6. **Sanity check transition balance**

   * Confirm the log line:
     * `CASAS transition task: transition_rate=... (n=...)`

---

## 9) Suggested GitHub “Results” snippet (for CASAS)

You can paste something like:

* **Activity Recognition (CASAS, 5 classes):** accuracy ≈ 0.95 (test), weighted-F1 ≈ 0.94
* **Transition Detection (CASAS, binary):** GRU trained on windowed event sequences; metrics saved under `outputs/neural_perception/<TAG>/casas/transition/metrics.json`.

(You can later replace with the exact transition classifier metrics after selecting the best run.)
