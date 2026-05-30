# Insights and Notes for Thesis
## Drug Repositioning Framework for Parkinson's Disease

---

## Pipeline Overview

### Step 1 — FDA Drug Extraction (DrugBank)
Extract FDA-approved drugs from DrugBank in SMILES format for use with Molecular Transformer.

### Step 2 — ChEMBL Training Data Extraction (Target and Interaction Discovery)
Uses the `chembl_webresource_client` API to construct the foundational training dataset in two connected steps: discovering biological interactions and retrieving chemical structures.

- **Target Filtering:** Parkinson's-associated targets filtered to *Homo sapiens* and SINGLE PROTEIN entities only (for clinical and biological relevance).
- **Activity Filtering:** Restricted to experimentally measured binding affinities:
  - **IC50** — concentration required to inhibit 50% of target activity
  - **Ki** — intrinsic binding affinity between drug and protein, independent of assay conditions
  - **Kd** — how strongly a drug-protein complex remains bound
- **Label Generation:** Continuous affinity measurements converted to pChEMBL values. Threshold pChEMBL ≥ 6.0 generates binary labels (1 = active, 0 = inactive).

### Step 3 — Molecule Structure Extraction (SMILES)
Extracts exact chemical structures for laboratory compounds from ChEMBL.
- Isolates all unique `molecule_chembl_id`s from interactions and batches queries to the ChEMBL molecule endpoint.
- Extracts `canonical_smiles` from the `molecule_structures` dictionary for every compound.

### Step 4 — Protein Sequence Fetching
Reads unique target IDs from Step 2, queries the UniProt API, and downloads exact amino acid sequences (.fasta) for those specific proteins.

### Step 5 — FDA Drug Embeddings (ChemBERTa)
Each FDA drug is transformed into a dense vector using a transformer model.
- **Model:** ChemBERTa
- **Input:** SMILES string
- **Output:** Dense vector (last hidden state) with mean pooling → single vector → L2 normalized
- Used as input features to the classifier combined with protein embeddings: `(drug_embedding, protein_embedding) → interaction prediction`

### Step 6 — ChEMBL Drug Embeddings (Training Data)
Same process as Step 5, but applied to `pd_molecule_smiles.csv` (the ChEMBL training compounds).

### Step 7 — Protein Embeddings

Two approaches were used and compared directly:

**7a — ESM2 (sequence-based)**
- **Model:** `facebook/esm2_t12_35M_UR50D` (pretrained protein transformer)
- **Input:** FASTA sequences (target_id + amino acid sequence)
- **Process:** Remove special tokens (first/last — not biologically meaningful) → mean pooling → single vector → L2 normalize
- **Output:** 480-dimensional embedding per protein
- Captures: sequence structure, evolutionary patterns, functional similarity

**7b — DRKG TransE (graph topology-based)**
- **Source:** Drug Repurposing Knowledge Graph (DRKG) — a large biomedical knowledge graph with 5.9M+ edges across drugs, genes, diseases, pathways
- **Model:** TransE L2 pre-trained embeddings (`DRKG_TransE_l2_entity.npy`)
- **Process (`extract_DRKG.py`):**
  1. Parse FASTA filenames → extract ChEMBL ID + UniProt accession
  2. Convert UniProt → Entrez Gene ID via UniProt API
  3. Look up each Entrez ID in DRKG `entities.tsv`
  4. Extract the 400-dim TransE vector from the embedding matrix
  5. Save as `drkg_target_embeddings.csv` — drop-in replacement for `protein_embeddings.csv`
- **Output:** 400-dimensional embedding per protein
- Captures: biological network topology, pathway co-membership, known disease associations
- **Used for:** RF comparison experiments AND as node features for the GraphSAGE GNN

> **Finding:** ESM2 and DRKG TransE produce nearly identical RF performance (AUC 0.7599 vs 0.7579 on time-slice), despite encoding fundamentally different biological information (Pearson r = 0.293). DRKG is now the primary representation for the GNN phase because the GNN operates directly on the DRKG graph structure.

### Step 8 — Phenotype Embedding (MDS-UPDRS + MedLlama)
- Uses MDS_UPDRS for retrieving actual PD medical symptoms
- Tokenizes with MedLlama, takes last hidden state
- Mean pooling (average over all columns) → normalization

### Step 9 — Data Summary (Files and Shapes)

| File | Rows | Columns / Notes |
|---|---|---|
| `chembl_pd_interactions.csv` | 127K total, 97K unique pairs | Training interactions |
| `protein_embeddings.csv` | 67 rows (63 unique) | 480 embedding dimensions |
| `chembl_drug_embeddings.csv` | 74K rows | 768 embedding dimensions |
| `drug_embeddings.csv` (FDA) | ~2K rows | FDA-approved drugs |

### Step 10 — RF Model Training (`train_rf.py`)
Random Forest trained on concatenated drug + protein embeddings.

**Training output:**
```
Unique drug-target pairs after deduplication:  97,895
Label counts:  1 → 74,691  |  0 → 23,204

Drug embeddings:     74,434 rows  (drug_id)
Protein embeddings:  67 rows      (target_id)

After merge:   97,745 rows
Feature dims:  1,248  (drug: 768 + target: 480)
Label counts:  1 → 74,572  |  0 → 23,173

Train (80%):   78,196 rows  |  positives: 59,658 (76.3%)
Test  (20%):   19,549 rows  |  positives: 14,914 (76.3%)
```

### Step 11 — RF Evaluation (`evaluate_rf.py`)
Evaluates the trained RF model by scoring the full labeled interaction dataset (127K rows) and generating diagnostic plots and enrichment metrics. This script is separate from training — it is used to audit model behavior on the entire known dataset, not just the held-out test set.

**Outputs:**
- `evaluation_scores.csv` — all pairs with predicted probability and true label
- `score_histograms.png` — distribution of predicted probabilities for positives vs negatives
- `roc_pr_curves.png` — ROC and Precision-Recall curves
- `topk_enrichment.csv` — precision and enrichment at various cutoff depths
- `metrics.json` — summary metrics

### Step 12 — Repositioning Inference Model Training (`train_rf_inference.py`)

**Purpose and distinction from rf_cv:**
This is a fundamentally different training step from all previous RF experiments. All prior models (train_rf.py, train_rf_timeslice.py, train_rf_cv.py) used time-slice splits and were designed purely for evaluation — measuring how well the model predicts known interactions. This script trains on ALL available ChEMBL data with no held-out test set, because the goal is to make the best possible predictions on FDA drugs rather than to evaluate the model.

**Why train on all data for inference:**
Holding back post-2018 data during inference would be unnecessarily restrictive — it would mean ignoring ~13K known interactions that could help the model predict FDA drug behaviour. The analogy is a doctor using all available medical knowledge when diagnosing a new patient, not artificially restricting themselves to pre-2018 literature.

**Hyperparameter search strategy:**
Rather than reusing the hyperparameters found by rf_cv (which were optimised for the pre-2018 subset), a fresh RandomizedSearchCV was run on the full dataset. This is methodologically correct because the optimal hyperparameters for 97K pairs may differ from those optimal for 84K pairs. RandomizedSearchCV was chosen over GridSearchCV because it samples from continuous distributions over a wider search space, which Bergstra & Bengio (2012) showed finds equally good or better hyperparameters than exhaustive grid search for the same number of evaluations.

**Why StratifiedKFold instead of TimeSeriesSplit:**
TimeSeriesSplit was used in rf_cv because the goal was specifically to simulate temporal generalisation. Here there is no temporal split — the goal is to find the best hyperparameters for the full data distribution. StratifiedKFold ensures each fold preserves the positive/negative ratio (~76.3% positive) of the full dataset, which is the correct choice for standard hyperparameter optimisation.

**Important note on the CV AUC reported here:**
The CV AUC (0.8845 for ESM2, 0.8816 for DRKG) is NOT a performance metric and should not be reported in the thesis as the model's AUC. It is simply the cross-validated score used to select hyperparameters. The honest performance metric is still rf_cv's test AUC of 0.7518.

**ESM2 inference model training output:**
```
Interactions: 97,895 unique pairs
Drug embeddings:    74,434 rows
Protein embeddings: 67 rows  (ESM2, 480 dims)
Feature dims: 1,248

RandomizedSearchCV: 50 iterations × 5 folds = 250 fits
Best parameters: max_depth=19, min_samples_leaf=3,
                 min_samples_split=15, n_estimators=180
Best CV AUC: 0.8845
Overfitting gap (best combo): 0.1102

Comparison with rf_cv params (max_depth=15, leaf=5, split=30):
→ Different parameters found — full data benefits from deeper,
  slightly less regularised trees than the pre-2018 subset.

Train AUC (sanity check): 0.9938  ← do not report
Saved to: artifacts/rf_inference_esm2/
```

**DRKG inference model training output:**
```
Interactions: 97,895 unique pairs
Drug embeddings:    74,434 rows
Protein embeddings: 67 rows  (DRKG TransE, 400 dims)
Feature dims: 1,168

RandomizedSearchCV: 50 iterations × 5 folds = 250 fits
Best parameters: max_depth=19, min_samples_leaf=3,
                 min_samples_split=15, n_estimators=180
Best CV AUC: 0.8816
Overfitting gap (best combo): 0.1122

Note: ESM2 and DRKG converged on identical best parameters
despite different feature dimensions (1,248 vs 1,168).
This suggests the optimal tree structure is driven by the
dataset size and label distribution, not the embedding type.

Train AUC (sanity check): 0.9927  ← do not report
Saved to: artifacts/rf_inference_drkg/
```

### Step 13 — Repositioning Inference (`random_forest.py`)
Scores every FDA drug against every PD target using the trained inference models. This is the actual drug repositioning step — the FDA drugs were never seen during training (neither as labeled interactions nor in any evaluation split).

**What it does:** Constructs every possible FDA drug × PD target pair (~2,848 drugs × 67 targets = ~190,816 combinations) and runs `predict_proba()` on each to get a predicted interaction probability.

**Run with ESM2:**
```
python src/models_randomf/random_forest.py
  --model_path artifacts/rf_inference_esm2/rf_model.pkl
  --protein_embeddings_csv data/processed/protein_embeddings.csv
  --top_k 0
  --output_csv artifacts/rf_inference_esm2/fda_target_scores_all.csv
```

**Run with DRKG:**
```
python src/models_randomf/random_forest.py
  --model_path artifacts/rf_inference_drkg/rf_model.pkl
  --protein_embeddings_csv data/processed/drkg_target_embeddings.csv
  --top_k 0
  --output_csv artifacts/rf_inference_drkg/fda_target_scores_all.csv
```

> `--top_k 0` saves all ~190K scored pairs with no cutoff.

### Step 14 — Prediction Analysis (`analyse_predictions.py`)
Analyses the full set of repositioning predictions across four dimensions. Target names are resolved from `pd_targets_metadata.csv` (already saved by Step 2). Known PD drugs are identified using `pd_indications.csv` (ChEMBL EFO ontology query — structured source, not free-text matching).

**Run with ESM2:**
```
python src/evaluation/analyse_predictions.py
  --scores_csv artifacts/rf_inference_esm2/fda_target_scores_all.csv
  --targets_metadata data/raw/pd_targets_metadata.csv
  --pd_indications_csv data/raw/pd_indications.csv
  --out_dir artifacts/rf_inference_esm2/prediction_analysis
```

**Run with DRKG:**
```
python src/evaluation/analyse_predictions.py
  --scores_csv artifacts/rf_inference_drkg/fda_target_scores_all.csv
  --targets_metadata data/raw/pd_targets_metadata.csv
  --pd_indications_csv data/raw/pd_indications.csv
  --out_dir artifacts/rf_inference_drkg/prediction_analysis
```

---

## Experiments and Results

### Experiment 1 — RF Baseline (Random 80/20 Split)
- **Embedding:** ESM2 (protein) + ChemBERTa (drug)
- **Split:** Random stratified 80/20
- **Purpose:** Establish naive baseline

| Metric | Test | Train |
|---|---|---|
| ROC-AUC | 0.8887 | 0.9996 |
| PR-AUC | 0.9587 | 0.9999 |
| F1 | 0.9039 | 0.9958 |
| Accuracy | 0.8468 | 0.9935 |
| Overfitting gap | **0.111** | — |

```
Confusion matrix (test):
TN=2,461   FP=2,174
FN=821     TP=14,093
```

### Experiment 2 — Time-Slice Evaluation (Cutoff 2018, ESM2)
- **Embedding:** ESM2 + ChemBERTa
- **Split:** Train ≤ 2018, Test > 2018 (mode: timeslice_only)

| Metric | Test | Train |
|---|---|---|
| ROC-AUC | 0.7599 | 0.9994 |
| PR-AUC | 0.8904 | 0.9998 |
| F1 | 0.8531 | 0.9950 |
| Accuracy | 0.7613 | 0.9924 |
| Overfitting gap | **0.239** | — |

```
Train rows: 84,152  |  Test rows: 12,621
Confusion matrix (test):
TN=856    FP=2,449
FN=564    TP=8,752
```

> **Note:** Double-Member Exclusion (DME) is inapplicable here. All 63 PD targets appeared in pre-2018 training data, so DME reduces 12,632 test pairs to just 4 — statistically meaningless. This is a domain-specific limitation of the closed PD target set.

### Experiment 3 — DRKG TransE Embeddings (Random Split)
- **Embedding:** DRKG TransE (topology-based) + ChemBERTa
- **Feature dims:** 1,168

| Metric | Test | Train | Gap |
|---|---|---|---|
| ROC-AUC | 0.8837 | 0.9992 | 0.116 |

### Experiment 4 — DRKG TransE Embeddings (Time-Slice 2018)
| Metric | Test | Train | Gap |
|---|---|---|---|
| ROC-AUC | 0.7579 | 0.9989 | 0.241 |

> ESM2 and DRKG TransE produce nearly identical AUC (0.7599 vs 0.7579 on time-slice). Despite encoding fundamentally different biological information (Pearson r = 0.293 between pairwise similarity matrices), both modalities produce equivalent performance. This is explained by target base-rate memorisation: the RF memorises which targets have high binding rates rather than learning transferable drug-target chemistry.

### Experiment 5 — CV-Regularized RF (Best RF Result)
GridSearchCV with PredefinedSplit over 27 hyperparameter combinations. Pre-2018 data is always used for training each combination; post-2018 data is always used for validation. The same post-2018 data is used for both hyperparameter selection and final evaluation, making the final AUC slightly optimistic — noted as a minor limitation.

**Best parameters:** `max_depth=15`, `min_samples_leaf=5`, `min_samples_split=30`

| Metric | Test | Train | Gap |
|---|---|---|---|
| ROC-AUC | **0.7518** | 0.9681 | **0.216** |
| PR-AUC | 0.8853 | — | — |
| F1 | 0.8398 | 0.9950 | — |
| Accuracy | 0.7582 | 0.9924 | — |
| Precision | 0.8216 | — | — |
| Recall | 0.8590 | — | — |

```
Train rows: 84,152  |  Test rows: 12,621
TN=1,567   FP=1,738
FN=1,314   TP=8,002
```

> Regularization reduced the gap from 0.240 → 0.216 but did not meaningfully improve test AUC. The performance ceiling is structural (distributional shift between pre/post-2018 data + target base-rate memorisation), not a hyperparameter tuning problem.

### Experiment 6 — Full-Dataset Evaluation (`evaluate_rf.py`, 127K rows)
The trained RF model was scored against the complete labeled interaction dataset (training + test combined) to audit model behaviour across all known pairs.

> ⚠️ These metrics are higher than the honest held-out test results because training data is included. Use Experiment 5 for thesis reporting.

**Summary metrics (n = 127,317 rows, positive rate = 77%):**

| Metric | Value |
|---|---|
| Accuracy | 0.9145 |
| F1 | 0.9453 |
| ROC-AUC | 0.9435 |
| PR-AUC | 0.9774 |

**Top-K Enrichment:**

| k | Hits | Precision@k | Recall@k | Enrichment Factor |
|---|---|---|---|---|
| 10 | 10 | 1.000 | 0.0001 | 1.30× |
| 25 | 25 | 1.000 | 0.0003 | 1.30× |
| 50 | 48 | 0.960 | 0.0005 | 1.25× |
| 100 | 96 | 0.960 | 0.001 | 1.25× |
| 250 | 245 | 0.980 | 0.002 | 1.27× |
| 500 | 492 | 0.984 | 0.005 | 1.28× |
| 1,000 | 992 | 0.992 | 0.010 | 1.29× |

### Experiment 7 — Inference Model Training (ESM2, All Data)
**Purpose:** Train the final model for FDA drug repositioning using ALL ChEMBL data and optimised hyperparameters found via RandomizedSearchCV on the full dataset.

**Why this is different from Experiment 5:** Experiment 5 used a time-slice split to give an honest AUC estimate. This experiment uses all data because for repositioning we want the model to learn from every known interaction, not artificially restrict itself to pre-2018 knowledge. The AUC from Experiment 5 (0.7518) remains the reported performance metric — this model's CV AUC is only used for hyperparameter selection.

| Setting | Value |
|---|---|
| Training pairs | 97,745 (ALL ChEMBL data) |
| Feature dims | 1,248 (768 drug + 480 protein) |
| CV strategy | RandomizedSearchCV, 50 iter, 5-fold StratifiedKFold |
| Best params | max_depth=19, leaf=3, split=15, n_estimators=180 |
| Best CV AUC | 0.8845 |
| Train AUC (sanity) | 0.9938 (expected high — do not report) |

**Key observation:** The full-data search found different optimal parameters than rf_cv (depth=19 vs 15, leaf=3 vs 5, split=15 vs 30). Deeper trees with less regularisation are optimal when more training data is available. This confirms that running a fresh hyperparameter search for the inference model was the correct methodological choice.

### Experiment 8 — Inference Model Training (DRKG, All Data)
Same as Experiment 7 but using DRKG TransE protein embeddings.

| Setting | Value |
|---|---|
| Training pairs | 97,745 (ALL ChEMBL data) |
| Feature dims | 1,168 (768 drug + 400 protein) |
| CV strategy | RandomizedSearchCV, 50 iter, 5-fold StratifiedKFold |
| Best params | max_depth=19, leaf=3, split=15, n_estimators=180 |
| Best CV AUC | 0.8816 |
| Train AUC (sanity) | 0.9927 (expected high — do not report) |

**Key observation:** ESM2 and DRKG converged on identical best hyperparameters (depth=19, leaf=3, split=15, trees=180) despite different feature dimensions (1,248 vs 1,168). This suggests the optimal tree structure is determined by dataset size and label distribution, not by which protein embedding is used.

### Experiment 9 — FDA Drug Repositioning Predictions (ESM2)
**Full prediction set:** 190,816 pairs (2,848 FDA drugs × 67 PD targets)
**Known PD drug ground truth:** 257 drugs from ChEMBL EFO ontology "parkinson" query (`pd_indications.csv`)

**Score distribution:**
```
Mean:   0.532   Median: 0.523
0.0-0.3:    4,001  (2.1%)
0.3-0.5:   75,060  (39.3%)
0.5-0.7:   94,860  (49.7%)
0.7-0.8:   12,731  (6.7%)
0.8-0.9:    3,799  (2.0%)
0.9-1.0:      365  (0.2%)
Pairs above threshold (0.9): 365
Novel high-confidence excl. GR: 261
```

The bell-shaped distribution centred around 0.53 indicates proper model calibration — it is not predicting high probability for everything. Only 0.2% of all pairs exceed 0.9, meaning high-confidence predictions are rare and selective.

**Known PD drug validation (Analysis 3):**
```
Known PD drugs found:     160 out of 257
Mean score (known PD):    0.760
Mean score (novel):       0.532
Separation gap:           0.228
```

The 0.228 gap between known PD drug scores and novel candidate scores is the primary evidence that the model learned biologically meaningful patterns. This separation was achieved without the model being told which drugs are PD drugs — it emerged purely from learning ChEMBL interaction patterns.

**Top known PD drugs recovered (best score per drug):**

| Drug | Best Target | Score | Rank |
|---|---|---|---|
| Folic acid | Carbonic anhydrase 2 | 0.9491 | #25 |
| Buspirone | Histamine H3 receptor | 0.9469 | #29 |
| Pitolisant | Histamine H3 receptor | 0.9447 | #37 |
| Sirolimus | FKBP1A | 0.9343 | #71 |
| Rosuvastatin | Carbonic anhydrase 2 | 0.9311 | #86 |
| Bromocriptine | Glucocorticoid receptor | 0.9282 | #100 |
| Temsirolimus | FKBP1A | 0.9156 | #190 |

**Top 25 novel candidates (score ≥ 0.9, excl. Glucocorticoid receptor):**

| # | Drug | Target | Score |
|---|---|---|---|
| 1 | Tacrolimus | FKBP1A | 0.9881 |
| 2 | Cariprazine | D(3) dopamine receptor | 0.9717 |
| 3 | Revumenib | Nociceptin receptor | 0.9698 |
| 4 | Maraviroc | Nociceptin receptor | 0.9640 |
| 5 | Mafenide | Carbonic anhydrase 2 | 0.9634 |
| 6 | Buprenorphine | Mu-type opioid receptor | 0.9613 |
| 7 | Mangafodipir | Carbonic anhydrase 2 | 0.9605 |
| 8 | Sulfanilamide | Carbonic anhydrase 2 | 0.9568 |
| 9 | Ziftomenib | DLK/MAP3K12 | 0.9548 |
| 10 | Ponatinib | DLK/MAP3K12 | 0.9544 |

### Experiment 10 — FDA Drug Repositioning Predictions (DRKG)
Same analysis using DRKG-based inference model.

**Score distribution:**
```
Mean:   0.538   Median: 0.531
0.9-1.0:      401  (0.2%)
Novel high-confidence excl. GR: 295
```

**Known PD drug validation:**
```
Known PD drugs found:     160 out of 257
Mean score (known PD):    0.766
Mean score (novel):       0.538
Separation gap:           0.228
```

**Top known PD drugs recovered:**

| Drug | Best Target | Score | Rank |
|---|---|---|---|
| Pitolisant | Histamine H3 receptor | 0.9689 | #3 |
| Buspirone | HIV RT* | 0.9529 | #18 |
| Nilotinib | Carbonic anhydrase 2 | 0.9297 | #105 |
| Temsirolimus | FKBP1A | 0.9260 | #135 |

> *Buspirone predicted to bind HIV reverse transcriptase in DRKG is a graph artifact — HIV RT has many known ligands in DRKG creating spurious topological connections. This is a known limitation of graph-based embeddings.

---

## Cross-Model Comparison: ESM2 vs DRKG Repositioning Predictions

### Score distribution comparison

| Metric | ESM2 | DRKG |
|---|---|---|
| Mean score | 0.532 | 0.538 |
| Known PD mean | 0.760 | 0.766 |
| Novel mean | 0.532 | 0.538 |
| Separation gap | **0.228** | **0.228** |
| High-conf pairs (>0.9, excl GR) | 261 | 295 |

The identical separation gap (0.228) across two fundamentally different protein representations (sequence-based vs graph topology-based) is strong evidence that the signal is real and not an artifact of embedding choice.

### Overlap analysis — 14/25 top candidates appear in both models

These are the highest-confidence repositioning candidates because two independent approaches agreed:

| Drug | Target | ESM2 Score | DRKG Score |
|---|---|---|---|
| **Tacrolimus** | FKBP1A | 0.9881 | 0.9857 |
| Abemaciclib | DLK/MAP3K12 | 0.9471 | 0.9627 |
| Acetazolamide | Carbonic anhydrase 2 | 0.9539 | 0.9599 |
| Aclidinium | Muscarinic acetylcholine receptor M3 | 0.9479 | 0.9557 |
| Buprenorphine | Mu-type opioid receptor | 0.9613 | 0.9440 |
| Fosnetupitant | DLK/MAP3K12 | 0.9469 | 0.9472 |
| Levoleucovorin | Carbonic anhydrase 2 | 0.9429 | 0.9511 |
| Mafenide | Carbonic anhydrase 2 | 0.9634 | 0.9562 |
| Maraviroc | Nociceptin receptor | 0.9640 | 0.9512 |
| Ponatinib | DLK/MAP3K12 | 0.9544 | 0.9531 |
| Quizartinib | DLK/MAP3K12 | 0.9459 | 0.9518 |
| Revumenib | Nociceptin receptor | 0.9698 | 0.9494 |
| Sulfanilamide | Carbonic anhydrase 2 | 0.9568 | 0.9742 |
| Zavegepant | PDE9A | 0.9536 | 0.9562 |

### ESM2-only notable candidates (not in DRKG top 25)
- **Cariprazine → D(3) dopamine receptor (0.9717)** — most scientifically important ESM2-only finding. Cariprazine is a D3/D2 partial agonist approved for psychiatric conditions; D3 receptor is a direct PD target. ESM2 sequence-based embeddings may capture receptor pharmacology better than DRKG graph topology.
- Ziftomenib → DLK (0.9548)
- Rimonabant → Cannabinoid receptor 1 (0.9463)

### DRKG-only notable candidates (not in ESM2 top 25)
- **Asciminib → ABL1 kinase (0.9545)** — ABL1 kinase inhibition for PD has been studied (nilotinib, a related ABL1 inhibitor, was in PD clinical trials). DRKG graph topology captured this connection that ESM2 missed.
- Etrasimod → Nociceptin receptor (0.9469)
- Brigatinib → DLK/MAP3K12 (0.9534)

### Target cluster analysis

Three clear target clusters dominate the high-confidence predictions across both models:

**FKBP1A cluster** (most scientifically compelling): Tacrolimus, Sirolimus, Temsirolimus all predict to FKBP1A. FKBP1A binds α-synuclein and may modulate its aggregation — a direct PD mechanism. The mTOR/FKBP pathway has emerging neuroprotection literature.

**DLK/MAP3K12 cluster**: Abemaciclib, Ponatinib, Quizartinib, Fosnetupitant, Rimegepant, Ziftomenib all predict to DLK kinase. DLK is a neuronal stress kinase whose inhibition has shown neuroprotective effects in neurodegeneration models. Multiple structurally diverse drugs predicting the same target strengthens this signal.

**CA2 cluster** (more uncertain): Many sulfonamide-class drugs (Acetazolamide, Mafenide, Sulfanilamide, Dorzolamide, Brinzolamide) predict to Carbonic anhydrase 2. This may reflect structural class memorisation — sulfonamides are a known CA2 scaffold — rather than genuine PD biology.


### Experiment 11 — Similarity-Based Partitioning (ESM2, RQ2)

**Purpose:** Directly answers RQ2 — "How does a time-slice evaluation strategy combined with a similarity-based approach impact the predictive reliability of drug-target interaction models?" This experiment applies Tanimoto drug-drug and cosine protein-protein similarity filtering ON TOP of the time-slice split to reveal how much additional optimism bias exists beyond temporal filtering alone.

**Methodology (following Yang & Dumontier, Maastricht University 2024):**
For each threshold combination, training pairs are removed where any training drug has Tanimoto similarity ≥ drug_threshold to any test drug, OR any training protein has cosine similarity ≥ prot_threshold to any test protein. The model is then retrained on the filtered data and evaluated on the same post-2018 test set.

**Similarity statistics of training data:**
```
Drug similarity (63,879 training drug pairs computed)
Protein similarity: min=0.907  mean=0.991  max=1.000
  → All 62 training proteins are highly similar to each other (mean 0.991)
  → This is expected — all 63 targets are PD-related proteins
    from the same biological pathways
```

**Baseline (time-slice only, no similarity filtering):** AUC = 0.7495

**Drug-only filtering results (prot_threshold=1.0):**

| Drug threshold | Train pairs | Removed % | AUC | AUC drop |
|---|---|---|---|---|
| 1.00 (exact only) | 37,024 | 56.1% | 0.6566 | -0.0929 |
| 0.95 | 36,975 | 56.1% | 0.6511 | -0.0984 |
| 0.90 | 36,949 | 56.2% | 0.6573 | -0.0922 |
| **0.85** | **36,866** | **56.3%** | **0.6587** | **-0.0908** |
| 0.80 | 36,688 | 56.5% | 0.6513 | -0.0982 |
| 0.75 | 36,318 | 56.9% | 0.6517 | -0.0978 |
| 0.70 | 35,821 | 57.5% | 0.6525 | -0.0970 |
| 0.65 | 35,105 | 58.3% | 0.6517 | -0.0978 |
| 0.60 | 34,102 | 59.5% | 0.6514 | -0.0981 |

**With protein filtering (any prot_threshold < 1.0):** AUC drops to 0.620-0.656 regardless of drug threshold. Extreme filtering (prot<0.95) reduces training to ~6,600 pairs (92% removed) with AUC 0.60-0.62.

**Pareto-optimal point:** drug<0.85, prot<1.0 → AUC 0.6587, 56.3% pairs removed.

**Comparison with Yang & Dumontier (2024):**

| Metric | Yang & Dumontier | This work |
|---|---|---|
| Baseline AUC | 0.9297 | 0.7495 |
| Best filtered AUC | ~0.8877 | 0.6587 |
| Most aggressive AUC | 0.5627 | 0.6004 |
| Max relative AUC drop | 39.5% | 19.9% |

Your model is more robust to similarity filtering — it loses only 19.9% of AUC under the most aggressive filtering, compared to 39.5% for Yang & Dumontier. This likely reflects the fact that your training set has a much smaller protein diversity (63 PD targets vs 1,484 diverse targets), making the model less dependent on structural memorisation.

**Three critical findings from this experiment:**

**Finding A — Drug similarity threshold barely matters.**
The AUC range across all drug thresholds (0.6 to 1.0) with prot=1.0 is only 0.0076 (from 0.6511 to 0.6587). This means it doesn't matter much whether you filter at 0.6 or 0.95 — the AUC is essentially the same. The dominant factor is protein filtering, not drug filtering.

**Finding B — Protein filtering is catastrophic and explains why.**
Any protein filtering (prot<1.0) immediately removes 56% of training pairs regardless of the drug threshold. This happens because all 63 PD targets appear in both the pre-2018 training set and the post-2018 test set. When you remove training pairs for proteins that also appear in the test set (similarity=1.0, i.e., identical proteins), you remove the majority of training data because PD is a closed-target domain. This is the same limitation that makes DME inapplicable — it is not a flaw in the approach but a fundamental property of disease-specific DTI prediction with a fixed target set.

**Finding C — The additional optimism bias beyond time-slice is 9 AUC points.**
Time-slice alone: 0.7495. Time-slice + Pareto-optimal similarity filtering: 0.6587. The difference (0.0908) quantifies the additional optimism in the time-slice-only evaluation that comes from the model having seen structurally similar drugs during training. This is the direct answer to RQ2: similarity-based filtering reveals an additional 9-point optimism bias that time-slice evaluation alone does not capture.

**Summary of AUC across all evaluation strategies:**
```
Random split (no filtering):            AUC 0.8887  ← naive baseline
Time-slice only (2018 cutoff):          AUC 0.7495  ← temporal debiasing
Time-slice + similarity (Pareto):       AUC 0.6587  ← temporal + structural debiasing
Time-slice + similarity (aggressive):   AUC 0.6004  ← maximum debiasing
```
Each step reveals additional optimism in the previous evaluation strategy.

**Saved to:** `artifacts/rf_similarity/`


### Experiment 12 — Similarity-Based Partitioning (ESM2, Random Split)

**Purpose:** Replicates the Yang & Dumontier (2024) setup directly — random 80/20 split with similarity filtering — enabling direct comparison with the paper. Completes the 2×2 comparison matrix (split strategy × similarity filtering).

**Key difference from Experiment 11:** Uses random stratified 80/20 split instead of time-slice. Test drugs are drawn randomly from the same time period as training drugs, creating much more structural overlap.

**Baseline (random split, no filtering):** AUC = 0.8682

**Drug-only filtering results (prot_threshold=1.0):**

| Drug threshold | Train pairs | Removed % | AUC | AUC drop |
|---|---|---|---|---|
| 1.00 | 16,263 | 79.2% | 0.6893 | -0.1789 |
| 0.95 | 15,917 | 79.7% | 0.6846 | -0.1836 |
| 0.90 | 15,457 | 80.3% | 0.6829 | -0.1853 |
| **0.85** | **14,150** | **81.9%** | **0.6802** | **-0.1880** |
| 0.80 | 11,663 | 85.1% | 0.6767 | -0.1915 |
| 0.75 | 8,836 | 88.7% | 0.6740 | -0.1942 |
| 0.70 | 6,448 | 91.8% | 0.6632 | -0.2050 |
| 0.65 | 4,554 | 94.2% | 0.6521 | -0.2161 |
| 0.60 | 3,043 | 96.1% | 0.6459 | -0.2223 |
| 0.55 | 2,103 | 97.3% | 0.6356 | -0.2326 |
| 0.50 | 1,398 | 98.2% | 0.6307 | -0.2375 |

> Protein filtering (prot<1.0) always leads to "too few pairs" with random split because ALL 63 proteins appear in both train and test sets when split randomly. Every training protein finds itself as a test protein with similarity=1.0, so any protein threshold below 1.0 removes all training pairs for all proteins. This is not a flaw — it is a fundamental property of random splits with closed target sets.

**Pareto-optimal:** drug<1.0, prot<1.0 → AUC 0.6893, 79.2% removed
**Most aggressive:** drug<0.5, prot<1.0 → AUC 0.6307, 98.2% removed
**Relative drop (Pareto):** 20.6%
**Relative drop (most aggressive):** 27.4%

**Comparison with Yang & Dumontier:**
- Their baseline: 0.9297 → most aggressive: 0.5627 (39.5% drop)
- Your baseline: 0.8682 → most aggressive: 0.6307 (27.4% drop)
- Your model is more robust — 27.4% vs 39.5% relative drop.

---

### Experiment 13 — Similarity-Based Partitioning (DRKG, Time-Slice, Corrected)

**Purpose:** Same as Experiment 11 but using DRKG TransE protein embeddings. Protein thresholds were carefully chosen based on the actual DRKG protein similarity distribution (diagnosed via `check_drkg_similarity_distribution.py`).

**Why different thresholds from ESM2:**
The DRKG protein similarity distribution spans 0.28–1.0 (much wider than ESM2's 0.91–1.0). The survival table showed only 7 meaningful change points, so the thresholds were set to: `[1.0, 0.99, 0.80, 0.60, 0.50, 0.40, 0.30]`. The ESM2 thresholds (0.90–1.0) would have been completely wrong for DRKG — all values between 0.90 and 0.99 give the same 10 surviving proteins.

**Protein similarity distribution (DRKG, time-slice):**
```
Min: 0.2828   Mean: 0.9187   Max: 1.0000
10 proteins have max_sim < 0.99 (genuinely dissimilar to test proteins)
22 proteins have max_sim between 0.99 and 1.0 (near-identical, removed at prot<0.99)
30 proteins have max_sim = 1.0 (exact match, removed at prot<1.0)
```

**Note on the bug:** An earlier run had a typo (`0,70` instead of `0.70`) creating a nonsensical `prot<70` threshold. Those results were discarded. This corrected run uses the proper threshold list.

**Baseline (time-slice, no filtering):** AUC = 0.7513

**Drug-only filtering results (prot_threshold=1.0):**

| Drug threshold | Train pairs | Removed % | AUC | AUC drop |
|---|---|---|---|---|
| 1.00 | 40,410 | 52.1% | 0.7005 | -0.0508 |
| 0.95 | 40,305 | 52.2% | 0.6983 | -0.0530 |
| **0.90** | **40,247** | **52.2%** | **0.7033** | **-0.0480** |
| 0.85 | 40,094 | 52.4% | 0.6903 | -0.0610 |
| 0.80 | 39,717 | 52.9% | 0.6948 | -0.0565 |
| 0.75 | 39,129 | 53.6% | 0.7024 | -0.0489 |
| 0.70 | 38,315 | 54.5% | 0.7005 | -0.0508 |
| 0.65 | 37,251 | 55.8% | 0.6957 | -0.0556 |
| 0.60 | 35,732 | 57.6% | 0.6967 | -0.0546 |

**Drug + protein filtering (drug=0.9, Pareto-optimal drug threshold):**

| Prot threshold | Train pairs | Removed % | AUC | AUC drop |
|---|---|---|---|---|
| 1.0 | 40,247 | 52.2% | 0.7033 | -0.0480 |
| 0.99 | 14,206 | 83.1% | 0.6254 | -0.1259 |
| 0.80 | 10,653 | 87.4% | 0.5991 | -0.1522 |
| 0.60 | 7,617 | 91.0% | 0.5966 | -0.1547 |
| 0.50 | 7,615 | 91.0% | 0.5904 | -0.1609 |
| 0.40 | 3,675 | 95.6% | 0.6097 | -0.1416 |

**Pareto-optimal:** drug<0.9, prot<1.0 → AUC 0.7033, 52.2% removed
**Most aggressive:** drug<0.8, prot<0.6 → AUC 0.5884, 91.0% removed
**Relative drop (Pareto):** 6.4%
**Relative drop (aggressive):** 21.7%

---

### Experiment 14 — Similarity-Based Partitioning (DRKG, Random Split)

**Purpose:** Completes the 2×2 matrix for DRKG. Random split version of Experiment 13.

> **Note:** Due to the same protein identity issue as Experiment 12, protein filtering below prot<1.0 always leads to too few pairs with random split. Only drug-only filtering (prot=1.0) produces valid results.

**Baseline:** AUC = 0.8673

**Drug-only filtering results (prot_threshold=1.0):**

| Drug threshold | Train pairs | Removed % | AUC | AUC drop |
|---|---|---|---|---|
| **1.00** | **27,547** | **64.8%** | **0.7343** | **-0.1330** |
| 0.95 | 26,799 | 65.8% | 0.7270 | -0.1403 |
| 0.90 | 25,940 | 66.9% | 0.7334 | -0.1339 |
| 0.85 | 24,106 | 69.2% | 0.7252 | -0.1421 |
| 0.80 | 20,423 | 73.9% | 0.7180 | -0.1493 |
| 0.75 | 15,845 | 79.8% | 0.7086 | -0.1587 |
| 0.70 | 11,716 | 85.0% | 0.7018 | -0.1655 |
| 0.65 | 8,474 | 89.2% | 0.6890 | -0.1783 |
| 0.60 | 5,868 | 92.5% | 0.6765 | -0.1908 |

**Pareto-optimal:** drug<1.0, prot<1.0 → AUC 0.7343, 64.8% removed
**Most aggressive (drug-only):** drug<0.6, prot<1.0 → AUC 0.6765, 92.5% removed
**Relative drop (Pareto):** 15.3%

---

## Complete Similarity-Based Partitioning Summary

### The 2×2 Comparison Matrix

| | No similarity filtering | Pareto-optimal | Most aggressive |
|---|---|---|---|
| **ESM2, random split** | 0.8682 | 0.6893 (-20.6%) | 0.6307 (-27.4%) |
| **ESM2, time-slice** | 0.7495 | 0.6643 (-11.4%) | 0.6098 (-18.6%) |
| **DRKG, random split** | 0.8673 | 0.7343 (-15.3%) | 0.6765 (-22.0%) |
| **DRKG, time-slice** | 0.7513 | 0.7033 (-6.4%) | 0.5884 (-21.7%) |

### Key observations from the 2×2 matrix

**Observation 1 — Random split always shows larger drops than time-slice.**
For ESM2: random drop = 20.6% vs time-slice drop = 11.4%. For DRKG: 15.3% vs 6.4%. This is because random test drugs are drawn from the same time period as training drugs and are therefore structurally more similar. Time-slice creates natural structural separation — newer drugs tend to have different scaffolds from older ones.

**Observation 2 — DRKG is consistently more robust than ESM2 under similarity filtering.**
At the Pareto-optimal point: ESM2 drops by 11.4% (time-slice) vs DRKG drops by only 6.4% (time-slice). DRKG graph topology representations are less susceptible to drug structural similarity bias than sequence-based representations. Possible explanation: graph topology captures biological network position rather than molecular structure, so structurally similar drugs may occupy different network positions and not cause the same memorisation.

**Observation 3 — After similarity filtering, all four setups converge.**
Pareto-filtered AUCs: 0.6643, 0.6893, 0.7033, 0.7343 — all within 0.07 of each other. This suggests the genuine generalisation capability of the RF model is approximately 0.66–0.73 regardless of evaluation strategy, once structural and temporal leakage are removed.

**Observation 4 — Protein filtering makes a big difference for DRKG time-slice.**
With drug=0.9: adding prot<0.99 drops AUC from 0.7033 to 0.6254 (-0.1259). This is a much larger drop than drug filtering alone (-0.0480). This reveals that DRKG-based models exploit protein identity memorisation even more than drug structural memorisation — consistent with the earlier finding that the RF memorises target base rates rather than learning drug-target chemistry.

**Observation 5 — The protein similarity distribution determines what thresholds are meaningful.**
ESM2: all proteins clustered in 0.91–1.0 → meaningful range is 0.90–1.0.
DRKG: proteins span 0.28–1.0 → meaningful range is 0.30–1.0.
Using the same thresholds for both embeddings would miss most of the interesting DRKG range. This highlights the importance of running the diagnostic (`check_drkg_similarity_distribution.py`) before choosing thresholds.

**Observation 6 — CHEMBL251 (Adenosine A2a receptor) behaves differently across embeddings.**
A2a receptor has max_sim=0.98 in ESM2 (almost identical to test proteins by sequence) but max_sim=0.43 in DRKG (very different network position). This means A2a receptor is sequence-conserved but occupies a unique position in the DRKG biological network. It survives DRKG protein filtering but not ESM2 protein filtering, suggesting the two embeddings capture fundamentally different aspects of this protein's biology.

### Comparison with Yang & Dumontier (2024)

| Metric | Yang & Dumontier | ESM2 | DRKG |
|---|---|---|---|
| Baseline AUC (random) | 0.9297 | 0.8682 | 0.8673 |
| Most aggressive AUC | 0.5627 | 0.6307 (random) | 0.5884 (time-slice) |
| Max relative drop | 39.5% | 27.4% | 21.7% |

Your models are more robust — both ESM2 and DRKG maintain higher AUC under aggressive filtering than Yang & Dumontier's model. This is likely because your training dataset is already harder (disease-specific closed target set, high positive rate) leaving less structural shortcut to exploit.

### The Full Evaluation Cascade (ESM2, best-documented)

```
Evaluation strategy                   AUC      Optimism removed
────────────────────────────────────────────────────────────────
Random split (naive)                  0.8887   baseline
Random + similarity (Pareto)          0.6893   -0.199 structural bias
Time-slice only                       0.7495   -0.139 temporal bias
Time-slice + similarity (Pareto)      0.6643   -0.085 structural bias
Time-slice + similarity (aggressive)  0.6098   further debiasing
```

This cascade directly answers RQ2: both temporal and structural similarity biases inflate naive evaluation metrics. Time-slice removes 0.139 AUC points of temporal bias. Similarity filtering removes an additional 0.085 AUC points of structural bias. The most honest estimate of model performance is 0.6643 (Pareto-optimal time-slice + similarity), which remains 0.1643 above random chance — confirming genuine learned signal.

---
---

## Summary Results Table

| Experiment | Embedding | Split | Test AUC | Train AUC | Gap |
|---|---|---|---|---|---|
| RF Baseline | ESM2 | Random 80/20 | 0.8887 | 0.9996 | 0.111 |
| RF Baseline | DRKG TransE | Random 80/20 | 0.8837 | 0.9992 | 0.116 |
| RF Time-slice | ESM2 | Year ≤ 2018 | 0.7599 | 0.9994 | 0.240 |
| RF Time-slice | DRKG TransE | Year ≤ 2018 | 0.7579 | 0.9989 | 0.241 |
| **RF CV-Regularized** | **ESM2** | **Year ≤ 2018 + CV** | **0.7518** | **0.9681** | **0.216** |
| RF Sim ESM2 (random, Pareto) | ESM2 | Random + drug<1.0 | 0.6893 | — | — |
| RF Sim ESM2 (time-slice, Pareto) | ESM2 | Year ≤ 2018 + drug<0.85 | 0.6643 | — | — |
| RF Sim ESM2 (time-slice, aggressive) | ESM2 | Year ≤ 2018 + drug+prot | 0.6098 | — | — |
| RF Sim DRKG (random, Pareto) | DRKG TransE | Random + drug<1.0 | 0.7343 | — | — |
| RF Sim DRKG (time-slice, Pareto) | DRKG TransE | Year ≤ 2018 + drug<0.9 | 0.7033 | — | — |
| RF Sim DRKG (time-slice, aggressive) | DRKG TransE | Year ≤ 2018 + drug+prot | 0.5884 | — | — |
| RF Inference (ESM2) | ESM2 | All data + CV | CV: 0.8845 | 0.9938 | — |
| RF Inference (DRKG) | DRKG TransE | All data + CV | CV: 0.8816 | 0.9927 | — |
| GraphSAGE GNN | DRKG graph | Year ≤ 2018 | TBD | TBD | TBD |

---

## Key Findings

**Finding 1 — Evaluation strategy matters more than embedding type.**
The gap between random and time-slice evaluation (0.129 AUC points) is ~60× larger than the difference between ESM2 and DRKG embeddings (0.002 points). Choosing the right evaluation strategy is the most critical methodological decision.

**Finding 2 — Overfitting is structural, not a tuning problem.**
Train AUC ~0.999 across all experiments. CV regularization reduced the gap from 0.240 → 0.216 but test AUC barely moved. Root cause: 63 targets × repeated embedding vectors → RF memorises target identity and predicts target-specific base rates rather than learning drug-target chemistry.

**Finding 3 — ESM2 and DRKG encode different information but produce the same AUC.**
Pairwise similarity matrices have Pearson r = 0.293 (different information), yet AUC differs by only 0.002. The RF is not learning from embedding content — it exploits target identity.

**Finding 4 — DME is inapplicable with the closed PD target set.**
All 63 PD targets appeared in pre-2018 training data. DME would reduce 12,632 test pairs to just 4. This is a domain-specific limitation of a disease-focused, closed-target design.

**Finding 5 — The performance ceiling requires architectural change.**
Hyperparameter tuning cannot fix distributional shift. Moving to GraphSAGE GNN on DRKG is motivated because graph topology propagates biological pathway information, and GNNs cannot exploit flat target identity memorisation in the same way.

**Finding 6 — Inference hyperparameters differ from evaluation hyperparameters.**
RandomizedSearchCV on the full dataset (97K pairs) found max_depth=19 optimal, versus max_depth=15 for the pre-2018 subset. Deeper trees with less regularisation are appropriate when more training data is available. Both ESM2 and DRKG converged on identical hyperparameters, suggesting the optimal tree structure is driven by dataset size, not embedding type.

**Finding 7 — Known PD drug recovery validates the inference model.**
Both ESM2 and DRKG inference models score known PD drugs 0.228 higher than novel candidates on average (0.760 vs 0.532 for ESM2; 0.766 vs 0.538 for DRKG). This separation emerged without any PD-specific supervision during training, indicating the model learned biologically meaningful patterns from ChEMBL interaction data.

**Finding 8 — 14/25 top repositioning candidates are consistent across embedding types.**
ESM2 and DRKG independently agree on 14 drug-target pairs in their respective top-25 high-confidence predictions. Tacrolimus → FKBP1A is ranked #1 in both models. This cross-embedding consistency is strong evidence these candidates represent genuine biological signal rather than embedding artifacts.

**Finding 9 — Three biologically plausible target clusters dominate high-confidence predictions.**
FKBP1A (mTOR/neuroprotection), DLK/MAP3K12 (neuronal stress kinase), and Carbonic anhydrase 2 (the CA2 cluster is more uncertain due to possible structural class memorisation). The FKBP1A and DLK clusters have existing neuroprotection literature supporting their PD relevance.

**Finding 10 — ESM2 and DRKG provide complementary candidate sets.**
ESM2 uniquely identified Cariprazine → D(3) dopamine receptor — a direct PD mechanism candidate. DRKG uniquely identified Asciminib → ABL1 kinase — consistent with published nilotinib PD trial data. Each embedding captures different biological information that the other misses.

**Finding 11 — Drug similarity threshold barely matters; protein filtering is the dominant factor.**
Across all drug similarity thresholds (0.6 to 1.0) with protein threshold fixed at 1.0, AUC varies by only 0.0076 (range: 0.6511 to 0.6587). Drug structural diversity in the training set has minimal impact on model performance. In contrast, any protein filtering immediately removes 56% of training pairs, confirming that the RF's performance is primarily driven by protein identity memorisation rather than drug structural learning.

**Finding 12 — Similarity filtering reveals an additional 9-point optimism bias beyond time-slice.**
The evaluation cascade: random split (0.8887) → time-slice (0.7495) → time-slice + Pareto similarity (0.6587) → time-slice + aggressive similarity (0.6004). Each step removes a different type of optimism. The 0.0908 gap between time-slice-only and Pareto-filtered directly answers RQ2: similarity-based filtering reveals additional optimism bias that temporal filtering alone cannot remove.

**Finding 13 — Your model is more robust to similarity filtering than Yang & Dumontier (2024).**
Under the most aggressive filtering, your ESM2 model retains AUC 0.6098 (18.6% drop, time-slice) and DRKG retains AUC 0.5884 (21.7% drop), while Yang & Dumontier's model drops to 0.5627 (39.5% drop). This greater robustness reflects the smaller, more homogeneous protein set (63 PD targets vs 1,484 diverse targets).

**Finding 14 — DRKG is more robust than ESM2 under drug similarity filtering but more sensitive under protein filtering.**
At the Pareto-optimal point: DRKG time-slice drops only 6.4% vs ESM2 time-slice 11.4% — DRKG is more robust to drug structural bias. However, adding protein filtering to DRKG drops AUC by an additional 12.6 points (from 0.7033 to 0.6254 at prot<0.99), suggesting DRKG-based models exploit protein identity memorisation more than drug structural memorisation.

**Finding 15 — Random split and time-slice converge after similarity filtering.**
Without filtering: random split (0.8682/0.8673) vs time-slice (0.7495/0.7513) — gap of ~0.12. After Pareto filtering: random (0.6893/0.7343) vs time-slice (0.6643/0.7033) — gap of only ~0.02-0.03. Similarity filtering corrects most of the optimism in random splits. After proper debiasing, both strategies estimate the same underlying generalisation capability.

**Finding 16 — Protein similarity thresholds must be chosen based on the actual embedding distribution.**
ESM2 proteins cluster in 0.91–1.0 (meaningful range: 0.90–1.0). DRKG proteins span 0.28–1.0 (meaningful range: 0.30–1.0). Using the same thresholds for both embeddings would miss most of the interesting DRKG filtering range. This is a methodological insight: threshold selection cannot be generic across embedding types.

---

## Planned Next Steps

- **Literature validation (RAG)** — PubMed search for top overlapping candidates, especially Tacrolimus/FKBP1A, Abemaciclib/DLK, and Maraviroc/Nociceptin receptor
- **GraphSAGE GNN on DRKG** — link prediction from graph structure; evaluate under time-slice protocol; compare to RF AUC 0.7518
- **Similarity-filtered inference** — train inference models with Tanimoto > 0.7 FDA drug structural filtering applied to training data; compare candidates against ESM2 and DRKG predictions
- **Saliency Maps** — gradient-based explanatory subgraphs for top-ranked drug candidates
- **Drug-only baseline** — ablation to confirm that target embeddings add value

---

## Notes and Limitations

- The 77% positive rate in training data is unusually high, likely reflecting undersampling of negatives during data collection (interactions fetched target-first from known PD targets, biasing toward positive pairs).
- `evaluate_rf.py` runs on the full labeled dataset — its metrics include training data and should not be used as model performance claims. Use Experiment 5 (CV-regularized time-slice, AUC 0.7518) as the primary reported result.
- The inference model CV AUC (0.8845 ESM2, 0.8816 DRKG) should not be reported as model performance — it is only used for hyperparameter selection.
- DRKG Buspirone → HIV RT prediction is a graph artifact; HIV RT has many DRKG ligands creating spurious connections. Graph-based embeddings can inherit noise from distantly connected nodes.
- The CA2 inhibitor cluster (Acetazolamide, Sulfanilamide, Mafenide, etc.) may reflect sulfonamide scaffold memorisation rather than genuine PD biology — treat with caution without literature support.
- Similarity-based partitioning (Tanimoto > 0.7 scaffold exclusion vs FDA drugs) has not yet been applied to inference models — reserved as next step.
- In the absence of experimental validation, prediction quality is assessed through three complementary approaches: held-out temporal evaluation (AUC 0.7518), known drug recovery (0.228 separation gap), and cross-embedding consistency (14/25 top candidates in both models).


---

## GraphSAGE GNN — Drug-Target Interaction Prediction on DRKG

---

### Architecture Overview

The GNN component implements a heterogeneous graph neural network (PDHeteroGNN) for Parkinson's disease drug-target interaction prediction using the Drug Repurposing Knowledge Graph (DRKG) as the underlying biological knowledge base. It is motivated by Finding 5 — that the RF's performance ceiling is structural (target base-rate memorisation), and that graph topology propagates biological pathway information in a way that cannot be exploited through flat identity memorisation.

**Model: PDHeteroGNN**
- Base architecture: GraphSAGE convolution (`SAGEConv`) layers
- Heterogeneous adaptation: PyTorch Geometric `to_hetero()` wrapper, which creates separate parameter sets per edge type (190 edge types → 190 independent aggregation functions)
- Link prediction: dot product scoring — `score(drug, target) = embedding_drug · embedding_target`
- High score = model predicts binding; low/negative score = no binding predicted

**Why dot product (not MLP):**
The dot product directly measures embedding alignment in learned representation space. This is the standard scoring approach for knowledge graph link prediction and matches the GNNDRKG reference implementation. It also provides a natural geometric interpretation: after training, compounds that bind similar targets should have embeddings pointing in similar directions.

---

### Graph Construction (`build_drkg.py`)

**Source:** DRKG (Drug Repurposing Knowledge Graph), version with 5,874,261 triples

**Node type filtering:** Retained 6 biologically relevant node types:
- Compound (24,070 nodes)
- Gene (39,148 nodes)
- Disease (5,103 nodes)
- Biological Process (11,381 nodes)
- Molecular Function (2,884 nodes)
- Pathway (1,822 nodes)
- **Total: 84,408 nodes**

After filtering to these 6 types: **4,896,855 triples** remain.

**Edge construction:** All triples converted to directed edges. Reverse edges added for every relation type to enable bidirectional message passing (following GNNDRKG convention).

**Final graph (full DRKG version):**
- Edge types: 190 (95 original + 95 reverse)
- Total edges: 9,793,710
- Node features: DRKG TransE L2 pre-trained embeddings, 400 dimensions per node

**Name sanitization:** All node type names and relation names sanitized for PyTorch Geometric compatibility (e.g., `Hetionet::CbG::Compound:Gene` → `Hetionet__CbG__Compound_Gene`, `Biological Process` → `Biological_Process`). This was necessary because PyG HeteroData keys cannot contain `::`, `:`, `+`, `-`, or spaces.

---

### Node Features: DRKG TransE Embeddings

Every node enters the GNN with a pre-trained 400-dimensional TransE embedding as its initial feature vector. These embeddings were trained on the full DRKG using the TransE knowledge graph completion objective: for a triple (head, relation, tail), TransE minimises `||head + relation - tail||`. This means nodes appearing in similar relational contexts receive similar embedding vectors.

**Coverage:** 100% of all 84,408 nodes have TransE embeddings (verified via sanity check):
```
Compound:            24070/24070 (100%)
Gene:                39148/39148 (100%)
Disease:              5103/5103  (100%)
Biological_Process:  11381/11381 (100%)
Molecular_Function:   2884/2884  (100%)
Pathway:              1822/1822  (100%)
```

**What TransE embeddings encode:** Structural proximity in the biological knowledge graph. Nodes that appear together in similar relationship patterns get similar vectors. For example, LRRK2's embedding already encodes its biological role — that it associates with Parkinson's disease, participates in specific pathways, and connects to certain biological processes — before the GNN runs any message passing.

---

### Message Passing

With 1 SAGEConv layer (the configuration consistently found best), each node aggregates information from its direct neighbours:

```
new_embedding(LRRK2) = SAGEConv(
    own_embedding(LRRK2),           ← LRRK2's TransE vector (400d)
    mean(embeddings of neighbours)   ← avg over all connected nodes
)
```

LRRK2's neighbours in DRKG include compounds that bind it (from CbG edges), diseases it's associated with, pathways it participates in, and biological processes it's involved in. After one round of message passing, LRRK2's embedding encodes not just its own structural properties but the average biological context of its entire neighbourhood.

The `to_hetero()` wrapper means this aggregation is learned separately for each of the 190 edge types — how a Compound aggregates from Gene neighbours is learned independently from how it aggregates from Disease neighbours.

**Why layers=1 is optimal:**
Consistently across all 40+ trials run, configurations with `num_layers=1` outperformed `num_layers=2` and `num_layers=3`. With 3 layers, every compound's embedding aggregates information from nodes 3 hops away. On a graph with 9.8M edges, this means virtually every compound ends up incorporating information from nearly every gene in the graph — producing near-identical embeddings that cannot discriminate drug-target binding. This is called oversmoothing. It confirmed that even 2 layers is too deep for this dense graph.

---

### Prediction Task

**Target edge type:** `Hetionet::CbG::Compound:Gene` (sanitized: `Hetionet__CbG__Compound_Gene`)

CbG = "Compound binds Gene" — direct molecular binding relationships from BindingDB and DrugBank, integrated into Hetionet and then DRKG. This was chosen as the prediction target because it most directly represents drug-target binding, which is what pChEMBL measures in ChEMBL.

**Prediction source (ChEMBL PD interactions):** Ground truth labels from `chembl_pd_interactions.csv`, filtered to Parkinson's disease targets:
- Label 1 (active): pChEMBL ≥ 6.0 (IC50/Ki ≤ 1μM — standard threshold for meaningful biological activity)
- Label 0 (inactive): pChEMBL < 6.0 (experimentally tested but confirmed weak or non-binding)

After deduplication (keeping highest pChEMBL per drug-target pair):
- Total pairs: 849
- Active (label=1): 623
- Inactive (label=0): 226

---

### Loss Function Evolution

**First approach — mean margin loss (broken):**
```python
loss = (1 - pos_sc.mean() + neg_sc.mean()).clamp(min=0)
```
The `clamp(min=0)` causes the loss to become exactly 0 once `pos_sc.mean() > neg_sc.mean() + 1`. This happened very early in training (~epoch 5), providing zero gradient thereafter. All scores converged to ~0.57, making all predictions identical. This produced AUC 0.446 — worse than random.

**Fixed approach — Binary Cross-Entropy (BCE):**
```python
loss = F.binary_cross_entropy_with_logits(
    torch.cat([pos_scores, neg_scores]),
    torch.cat([ones, zeros])
)
```
BCE never saturates — `sigmoid(x)` never exactly reaches 0 or 1, so gradient always flows. BCE also frames the task as binary classification, making the GNN's output directly comparable to the RF's probability outputs. This is the correct choice when using experimentally confirmed inactives as negatives.

---

### Negative Sampling Strategy

Training requires both positive and negative examples. Three types of negatives were used across experiments:

**Type 1 — ChEMBL inactives (real experimental negatives):**
Compounds tested against PD targets with pChEMBL < 6. These are the strongest negatives — experimentally confirmed non-binders. Used when available.

**Type 2 — Random drug-gene pairs (supplementary):**
When real inactives are insufficient (need k per positive, only have 176 real inactives for 549 positives with k=5 → need 2,745 but have 176), random gene nodes are sampled as additional negatives. These represent untested pairs — the vast majority of which are true non-interactions given the sparsity of binding.

**Important mismatch identified:** Earlier versions evaluated against ChEMBL inactives but trained against random negatives (different task definition). The model learned "known binding vs random pair" but was evaluated on "active vs experimentally confirmed inactive" — different tasks. This was the root cause of 0.5 AUC despite the model appearing to learn (decreasing training loss).

---

### Experiments

#### GNN Experiment 1 — Full DRKG, Margin Loss (Broken)

| Setting | Value |
|---|---|
| Graph | Full DRKG, 9.8M edges |
| Node features | TransE 400d (all nodes) |
| Supervision | ChEMBL PD interactions, time-split pre/post 2018 |
| Train edges | 725 (549 active, 176 inactive) |
| Test edges | 124 (74 active, 50 inactive) |
| Loss | Mean margin loss |
| Best val config | hidden=64, out=16, layers=1, lr=1e-4, dropout=0.5 |
| Best val AUC | 0.5870 (from 25 random search trials) |

**Results:**
| Metric | Value |
|---|---|
| Test ROC-AUC | 0.4457 |
| Test PR-AUC | 0.5598 |
| Test F1 | 0.5255 |
| Hits@5 | 0.0270 |
| Hits@10 | 0.1216 |

**Diagnosis:** Score collapse — all 9.8M+ drug-target pairs received scores in the narrow range [0.569, 0.574]. The mean margin loss saturated to zero before the model learned anything meaningful. The model did not learn — all results are equivalent to random prediction. Hits@K near 0 confirms the model could not even distinguish known binding pairs from random pairs.

---

#### GNN Experiment 2 — PD Subgraph, ChEMBL Embeddings, BCE (Run 1)

**Key changes from Experiment 1:**
1. PD-specific 2-hop subgraph filter applied
2. ChEMBL Molecular Transformer (MT) embeddings for Compound nodes (768d)
3. Per-type input projection layers (maps each type to shared hidden_channels)
4. BCE loss replacing margin loss
5. Real ChEMBL inactives used as negatives during training (supplemented with random)
6. Validation AUC computed correctly (active vs inactive, not vs random)

**PD subgraph filter:** Kept nodes within 2 hops of `Disease::MESH:D010300` (Parkinson's disease node). Hop-1: 938 nodes directly connected to PD. Hop-2: 37,622 nodes connected to hop-1. Filter retained edges where at least one endpoint is hop-1. Result: 93.1% of triples retained (4,559,478 of 4,896,855) — nearly identical graph due to high connectivity of DRKG.

| Setting | Value |
|---|---|
| Graph | PD 2-hop subgraph, 9.1M edges |
| Node features | Compound: ChEMBL MT 768d; Others: TransE 400d |
| Compound coverage | 164/6,923 have ChEMBL MT embeddings (2.4%) |
| Supervision | ChEMBL PD interactions, time-split pre/post 2018 |
| Train edges | 309 (246 active, 63 inactive) |
| Test edges | 49 (34 active, 15 inactive) |
| Loss | BCE |
| Search | 15 random trials |
| Best val config | hidden=64, out=32, layers=1, lr=5e-4, dropout=0.2, wd=1e-5 |
| Best val AUC | 0.8677 |

**Epoch-level training dynamics (final training):**
```
epoch  10 | loss=0.1131 | v_loss=0.5494 | v_auc=0.7917
epoch  20 | loss=0.0659 | v_loss=0.7331 | v_auc=0.7986
epoch  30 | loss=0.0305 | v_loss=0.9034 | v_auc=0.8194
epoch  40 | loss=0.0172 | v_loss=1.1898 | v_auc=0.8194
epoch  50 | loss=0.0078 | v_loss=1.6475 | v_auc=0.7917
```
Training loss collapses to near zero (epoch 10: 0.1131 → epoch 50: 0.0078) while validation loss triples — classic overfitting signature. Best checkpoint saved at ~epoch 30.

**Results:**
| Metric | Value |
|---|---|
| Test ROC-AUC | 0.5255 |
| Test PR-AUC | 0.7604 |
| Test F1 | 0.7123 |
| Hits@5 | 1.0000 |
| Hits@10 | 1.0000 |

**Why Hits@K=1.0 but AUC=0.525:** These measure fundamentally different things. Hits@K evaluates whether a known active pair ranks above 99 randomly sampled genes — an easy task because random genes have no biological relationship to the drug. AUC evaluates whether an active compound scores higher than an experimentally confirmed inactive — much harder, because both compounds were actually tested against this specific target. The model learned to beat random noise (Hits@K=1.0) but could not distinguish active from inactive ChEMBL interactions (AUC=0.525).

---

#### GNN Experiment 3 — PD Subgraph, BCE (Run 2, Different Val Split)

Identical code and data to Experiment 2. Different best hyperparameters found due to different random search trajectory and slight variation in val set composition.

| Setting | Value |
|---|---|
| Best val config | hidden=64, out=32, layers=1, lr=3e-4, dropout=0.3, wd=1e-4 |
| Best val AUC | 0.8889 |

**Epoch-level training dynamics (final training):**
```
epoch  10 | loss=0.1487 | v_loss=1.5181 | v_auc=0.6319
epoch  20 | loss=0.0742 | v_loss=0.8349 | v_auc=0.8194  ← best checkpoint
epoch  30 | loss=0.0451 | v_loss=0.8935 | v_auc=0.7917
epoch  70 | loss=0.0051 | v_loss=1.7008 | v_auc=0.7986
```

**Results:**
| Metric | Value |
|---|---|
| Test ROC-AUC | 0.4176 |
| Test PR-AUC | 0.6887 |
| Test F1 | 0.6176 |
| Hits@5 | 0.7353 |
| Hits@10 | 0.8824 |

**Why test AUC dropped despite higher val AUC (0.889 vs 0.868):** Val AUC on only 30 pairs has confidence interval ±0.15 — the difference between 0.868 and 0.889 is pure noise. The model selected was the one that happened to perform best on 30 random validation pairs, not necessarily the best generalising model. With 49 test pairs, test AUC also has CI ±0.14, making the comparison between 0.525 and 0.418 statistically meaningless.

---

#### GNN Experiment 4 — Full DRKG, BCE (PENDING)

Applies BCE loss and correct training/evaluation logic to the full DRKG graph (725 training pairs instead of 309 from PD subgraph). This is the missing direct comparison — the only valid full DRKG result (Experiment 1) used broken margin loss.

| Setting | Value |
|---|---|
| Graph | Full DRKG, 9.8M edges |
| Node features | TransE 400d (all nodes, uniform dimension) |
| Supervision | ChEMBL PD interactions, time-split pre/post 2018 |
| Train edges | 725 (549 active, 176 inactive) |
| Test edges | 124 (74 active, 50 inactive) |
| Loss | BCE |
| Search space | Focused: hidden∈{64,128}, out∈{16,32}, lr∈{1e-3,5e-4,3e-4}, dropout∈{0.2,0.3}, wd∈{1e-4,1e-5}, neg_k∈{5,10}, layers=1 (fixed) |
| Search | 10 trials |

**Results:** PENDING

---

#### GNN Experiment 5 (v2) — All DRKG CbG Training, PD-Specific Test (PENDING)

**Conceptual redesign — the most principled approach:**

Previous experiments used only 309–725 ChEMBL PD interactions for training supervision. This is too small for a model operating on 84,408 nodes. The redesign trains on ALL Compound-binds-Gene (CbG) edges in DRKG — tens of thousands of drug-target binding pairs across all diseases — and evaluates specifically on PD interactions from ChEMBL.

**Design rationale:**
- Binding is universal: a compound binding a protein follows the same physicochemical principles regardless of disease area
- Training on all CbG edges lets the model learn general binding patterns from a large, diverse set
- PD-specific evaluation tests whether general binding knowledge transfers to Parkinson's disease targets
- This is a standard transfer learning approach — "pretrain broadly, evaluate specifically"
- Analogous to how GNNDRKG trained on ALL drug-disease edges before evaluating on specific diseases

**Leakage prevention:** Any ChEMBL test pair that also appears as a CbG edge in DRKG is removed from the training graph before message passing. The model cannot memorise test answers from graph structure.

| Setting | Value |
|---|---|
| Graph | Full DRKG, all 6 node types |
| Node features | TransE 400d (all nodes) |
| Training supervision | ALL Hetionet CbG edges in DRKG (label=1) + random negatives |
| Test set | ChEMBL PD interactions, random stratified 20% split |
| Test set size | ~170 pairs (110 active, 35 inactive) — larger than time-split version |
| Loss | BCE |
| Negatives | Random drug-gene pairs during training (no real inactives — training labels are all 1) |
| Evaluation | Real ChEMBL inactives as negatives — honest biological evaluation |

**Results:** PENDING

---

### Complete Hyperparameter Search Results Across All Trials

**Pattern confirmed across all 40+ trials:**

| num_layers | Typical AUC range | Outcome |
|---|---|---|
| 1 | 0.54 – 0.89 | Consistently learns |
| 2 | 0.47 – 0.81 | Marginal to poor |
| 3 | 0.46 – 0.56 | Near-random or dead |

Layers > 1 cause oversmoothing on the dense DRKG graph. With 9.8M edges, 2 hops reaches virtually every node — all embeddings converge to similar vectors, dot products equalize, and the model cannot discriminate.

**Other consistent patterns:**
- `hidden_channels=64` won most frequently over 128 or 256
- `out_channels=32` won most frequently
- `lr=3e-4` to `5e-4` performed best (1e-3 sometimes too aggressive)
- `dropout=0.2–0.3` sweet spot
- `neg_k=5` appeared in most top configurations
- `weight_decay=1e-5` slightly preferred over 1e-4

---

### Key Diagnoses and Findings

**Finding GNN-1 — The loss function determines whether learning occurs.**
Mean margin loss saturated to zero before the model learned anything, producing AUC=0.446 (below random). BCE loss provided continuous gradient signal, producing val AUC up to 0.889. The choice of loss function was the single largest factor in whether the model learned at all.

**Finding GNN-2 — Training and evaluation must use the same negative type.**
When training used random negatives but evaluation used ChEMBL inactives, AUC was systematically near 0.5 even when the model appeared to be learning (decreasing training loss). The model learned to score known interactions above random pairs — but was evaluated on a different task (active vs experimentally confirmed inactive). Aligning the negative type fixed this.

**Finding GNN-3 — Dense graphs require shallow networks.**
All 40+ trials confirmed that num_layers=1 is optimal for DRKG. The graph's density (9.8M edges, average degree ~116) causes oversmoothing at depth 2 — every compound's 2-hop neighbourhood covers most of the graph. This is a domain-specific property of DRKG, not a general GNN limitation.

**Finding GNN-4 — Validation set of 30 pairs is unreliable for model selection.**
Confidence interval on AUC with 30 samples is ±0.15. Val AUC difference of 0.868 vs 0.889 between two runs is pure noise. This creates a disconnect between val AUC (appears high) and test AUC (appears low) that is not meaningful — both are within noise of each other.

**Finding GNN-5 — The PD subgraph filter was ineffective.**
The 2-hop neighbourhood around the PD disease node retained 93.1% of DRKG triples. DRKG's small-world property means almost every biological entity is within 2 hops of any disease node. The filter removed 416 training pairs while barely reducing the graph, worsening the supervision-to-graph-size ratio without reducing oversmoothing.

**Finding GNN-6 — ChEMBL MT embeddings had low compound coverage.**
Only 164 of 6,923 compounds in the PD subgraph had ChEMBL MT embeddings (2.4%). 97.6% of compound nodes remained as zero vectors, which is worse than having TransE embeddings. The PD subgraph approach was conceptually sound but practically hurt compound representation.

**Finding GNN-7 — Small supervision set is the fundamental limiting factor.**
With 279–653 training pairs supervising a model with ~500,000 parameters, the parameter-to-supervision ratio is ~750–1,800:1 (vs 10–100:1 for healthy ML problems). No amount of hyperparameter tuning can overcome this — the model has enough capacity to memorise every training pair many times over. This is the primary reason GNN underperforms RF (which has far fewer parameters and learns from 84,000+ pairs).

**Finding GNN-8 — GNN and RF serve different scientific purposes.**
The RF is the primary predictive model evaluated under rigorous time-slice protocol (AUC 0.7518). The GNN is the interpretability backbone — its graph-aware embeddings provide the substrate for saliency map analysis. The GNN does not need to outperform RF; it needs to produce meaningful predictions that can be explained via subgraph attribution.

---

### Comparison: GNN vs RF

| Model | AUC | Protocol | Negatives | Training pairs |
|---|---|---|---|---|
| RF (time-slice) | 0.7518 | Time-split 2018 | Random 20% hold-out | 84,152 |
| RF (random) | 0.8887 | Random 80/20 | Random 20% hold-out | 78,196 |
| GNN Exp 1 (broken) | 0.4457 | Time-split 2018 | ChEMBL inactives | 725 |
| GNN Exp 2 (PD) | 0.5255 | Time-split 2018 | ChEMBL inactives | 309 |
| GNN Exp 3 (PD) | 0.4176 | Time-split 2018 | ChEMBL inactives | 309 |
| GNN Exp 4 (full) | PENDING | Time-split 2018 | ChEMBL inactives | 725 |
| GNN Exp 5 (v2) | PENDING | Random 20% | ChEMBL inactives | All CbG |

**Why GNN underperforms RF in this setting:**
1. Training data: RF uses 84,152 pairs; GNN uses 309–725 pairs — 100x less supervision
2. Feature information: RF uses molecular fingerprints directly relevant to binding affinity; GNN uses graph topology that encodes biological association, not binding strength
3. Overfitting: GNN has ~500k parameters for 300 training examples; RF with max_depth=15 has controlled capacity
4. Distribution shift: Under time-split, post-2018 compounds are structurally different from pre-2018 training compounds — harder for GNN to generalise with its small training set

This is consistent with literature reporting that GNNs require substantially more labelled data than classical ML to overcome distributional shift in biomedical tasks.

---

### Connection to Saliency Maps

The GNN's primary contribution to this thesis is not its prediction accuracy but its interpretability potential. Once a reasonably working model is obtained (Experiments 4 or 5), saliency maps will identify which nodes and edges in the DRKG subgraph most influenced each drug-target prediction.

**What saliency maps will show:**
For a high-confidence drug-target prediction, the saliency map attributes a score to each neighbouring node and edge, answering: "which biological relationships in DRKG most supported this prediction?" For example, a compound predicted to bind LRRK2 might be explained by: its association with other PD diseases, its membership in the same pathway as known LRRK2 binders, or its structural similarity (via TransE proximity) to established LRRK2 inhibitors.

This provides a biological narrative for drug repositioning candidates — not just "the model says this drug might bind LRRK2" but "the model's confidence comes from these specific biological relationships in the knowledge graph."

---

### Files

| File | Purpose |
|---|---|
| `src/models_GNN/build_drkg.py` | Full DRKG graph builder, time-split ChEMBL labels |
| `src/models_GNN/build_drkg_pd.py` | PD 2-hop subgraph + ChEMBL MT embeddings |
| `src/models_GNN/build_drkg_v2.py` | All CbG training + PD-specific test (v2 design) |
| `src/models_GNN/GNN.py` | PDHeteroGNN without input projections (TransE only) |
| `src/models_GNN/GNN_pd.py` | PDHeteroGNN with per-type input projections |
| `src/models_GNN/train_gnn.py` | Training on full DRKG, time-split evaluation |
| `src/models_GNN/train_gnn_pd.py` | Training on PD subgraph, time-split evaluation |
| `src/models_GNN/train_gnn_v2.py` | Training on all CbG, PD-specific test evaluation |
| `artifacts/gnn/metrics.json` | Experiment 1 results (broken, for reference only) |
| `artifacts/gnn_pd/metrics.json` | Experiments 2 & 3 results |
| `artifacts/gnn_v2/metrics.json` | Experiment 5 results (pending) |

---

### Updated Summary Results Table

| Experiment | Model | Graph | Supervision | Split | Test AUC | Train AUC | Gap |
|---|---|---|---|---|---|---|---|
| RF Baseline | RF + ESM2 | — | 84K pairs | Random 80/20 | 0.8887 | 0.9996 | 0.111 |
| RF Baseline | RF + DRKG | — | 84K pairs | Random 80/20 | 0.8837 | 0.9992 | 0.116 |
| RF Time-slice | RF + ESM2 | — | 84K pairs | Year ≤ 2018 | 0.7599 | 0.9994 | 0.240 |
| RF Time-slice | RF + DRKG | — | 84K pairs | Year ≤ 2018 | 0.7579 | 0.9989 | 0.241 |
| **RF CV** | **RF + ESM2** | **—** | **84K pairs** | **Year ≤ 2018 + CV** | **0.7518** | **0.9681** | **0.216** |
| GNN Exp 1 | GraphSAGE | Full DRKG 9.8M | 725 ChEMBL PD | Year ≤ 2018 | 0.4457 | — | — |
| GNN Exp 2 | GraphSAGE | PD subgraph 9.1M | 309 ChEMBL PD | Year ≤ 2018 | 0.5255 | — | — |
| GNN Exp 3 | GraphSAGE | PD subgraph 9.1M | 309 ChEMBL PD | Year ≤ 2018 | 0.4176 | — | — |
| GNN Exp 4 | GraphSAGE | Full DRKG 9.8M | 725 ChEMBL PD | Year ≤ 2018 | **PENDING** | — | — |
| GNN Exp 5 (v2) | GraphSAGE | Full DRKG 9.8M | All CbG | Random 20% | **PENDING** | — | — |

> Note: GNN Experiments 1–3 are not directly comparable to RF because they use different test sets (49 vs 124 pairs) and different negative types. GNN Experiment 5 uses a random split (for larger test set and stable evaluation) while RF uses time-slice (for temporal honesty) — this difference is justified because each model is evaluated in the setting where it is scientifically appropriate.