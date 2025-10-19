# Phase-Aware Polarity Homeostat

**Record → Decode (REST / REPAIR / PRUNE) → Correct chronic depolarization while preserving healthy oscillations.**

This repo implements and evaluates a **phase-aware bioelectric homeostat** for our **Principles of Synthetic Biology (BioE 147/247)** term project.  
We treat tissues as **bioelectric–metabolic dynamical systems** and only intervene on **chronic depolarization** while preserving **physiological pulses** (ERK/ΔΨm rhythms) under **energy** and **geometry** safety rails.

---

## Background & Significance (course-aligned)

- **Biology.** Chronic depolarization covaries with disordered proliferation/migration; short/phasic depolarizations support repair and immune activation.  
- **Winfree logic.** Dynamics live on phase manifolds; control must be **phase-aware**, not amplitude-only.  
- **Engineering goal.** Detect chronic LOW-band occupancy, check oscillation health (PLV), require neighbor consensus (gap-junction domain), and **gate actions by energy** to avoid "zombie forcing".

**Course themes hit:** logic design (rule-tree / tiny MLP), oscillators & phase, IFFL/feedback (safety gates), multicellular communication (gap-junctions), load/fitness (energy costs).

---

## Specific Aims

1) **Aim A — Chronic vs phasic detector & event atlas**  
   Schmitt banding + leaky time-in-LOW + neighbor mismatch → ROC/PR separating chronic vs transient.

2) **Aim B — Phase-aware action gating & safe decoding**  
   Oscillation detector (PLV) with persistence (PLV low for ≥ `min_bad_duration_s`) to avoid suppressing wound waves; neighbor consensus; decoders: rules **and** tiny MLP (WTA + hysteresis + dwell). Compare pulse-preservation vs flicker.

3) **Aim C — Energy safety & microenvironment**  
   Scalar energy store `E(t)` with actuation cost and optional TNT/EV flux; adaptive `E_min` for healing cohorts; viability/safety maps; recovery vs naïve clamps.

---

## Cell Model Assumptions

We simulate a **generic epithelial‑like cell** (not a specific line). Defaults:

- Resting `V_mem` ≈ **−60 mV** (healthy hyperpolarized for many non‑excitable mammalian cells)  
- Initial `V_mem` ≈ **−20 mV** (pathologically depolarized start)  
- Bands (Schmitt): LOW enter/exit = **−15/−8 mV**, HIGH enter/exit = **−55/−45 mV**  
These are illustrative and configurable in `configs/*.yaml`.

---

## Mitigations for Known Failure Cases (what we implement)

We hardened the controller against four common failure patterns:

1) **High energy but sick (mutation–energy disconnect).**  
   Cells can have high ATP/ΔΨm yet remain pathologically depolarized (e.g., KRAS/p53 contexts).  
   **Mitigation (implemented):** enable `features.use_global_v_offset`, set `thresholds.global_v_offset_mV`; optionally enable `features.use_redox_bit`.

2) **Whole‑region pathology (consensus inversion).**  
   If most neighbors are depolarized, the local average looks “normal.”  
   **Mitigation (implemented):** enable `features.use_domain_low_fraction`, set `thresholds.domain_low_fraction`.

3) **Low‑energy but healthy healing.**  
   After injury, E(t) often drops during repair; that’s normal.  
   **Mitigation (implemented):** enable `energy_extras.adaptive_emin` (k, min) and use `actuation.cap_when_lowE` for gentle REPAIR.

4) **Healing waves look chaotic (PLV dips).**  
   Wound/stress waves are messy and temporarily reduce PLV.  
   **Mitigation (implemented):** set `osc_extras.min_bad_duration_s` (PLV persistence); optional multi‑band PLV if enabled later.

---

# MARMIT

> **Minimum Architecture, Requirements & Interfaces**

## M — Minimal Architecture (boxes)

### M1. System Flow (top-level data path)

```mermaid
flowchart LR
  subgraph Tissue["Tissue State"]
    V["Vmem field (grid)"]
    E["Energy E(t) (grid)"]
  end

  V --> REC[Recorder: Schmitt bands; leaky time-in-LOW; neighbor mismatch]
  E --> REC
  V --> OSC[Oscillation Detector: band-pass to analytic phase; PLV vs healthy]
  REC --> FEAT[Feature Vector: V̄, LOW_occ, mismatch, context, E, PLV]
  OSC --> FEAT

  FEAT --> DEC{Decoder: Rules or Tiny MLP; WTA + hysteresis + dwell}
  DEC -->|REST| MON[Monitor]
  DEC -->|REPAIR| GATES[Safety Gates: Energy / Oscillation / Geometry]
  DEC -->|PRUNE| GATES

  GATES -->|allow| ACT[Actuator: bounded hyperpolarizing pulses; amplitude/duty/period/refractory; low-E caps]
  ACT -->|ΔV| V
  ACT -->|cost| E

  FEAT --> LOG[(Event Atlas CSV)]
  DEC --> LOG
```

### M2. Module Wiring (code-level boxes)

```mermaid
flowchart TB
  subgraph Model
    TISSUE[tissue.py: RC grid, Laplacian, leak, noise]
    ENERGY[energy.py: energy dynamics + pump]
  end

  subgraph Sensing
    RECORDER[recorder.py: Schmitt + leaky LOW + mismatch]
    OSCILLATOR[osc.py: analytic phase + PLV]
  end

  subgraph Decoder
    RULES[rules.py: threshold tree]
    MLP[perceptron.py: tiny 6-H-3 + WTA/hyst/dwell]
  end

  subgraph Safety
    GATE[gates.py: energy/osc/geometry gates]
  end

  subgraph Actuation
    PULSES[pulses.py: bounded pulses + caps]
  end

  subgraph Eval
    METRICS[metrics.py: recovery, flicker, PLV retention]
  end

  subgraph Utils
    MATH[math_utils.py: Laplacian, ring buffers, analytic phase]
    IO[io_utils.py: atlas writer, summaries]
  end

  subgraph Experiments
    RUN[run.py: CLI config->modules->loop->logs]
  end

  TISSUE --> RECORDER
  TISSUE --> OSCILLATOR
  ENERGY --> TISSUE
  RECORDER --> RUN
  OSCILLATOR --> RUN
  RUN --> RULES
  RUN --> MLP
  RUN --> GATE
  GATE --> PULSES
  PULSES --> TISSUE
  PULSES --> ENERGY
  RUN --> IO
  RUN --> METRICS
  RUN --> MATH
```

---

## A — Requirements (what must be true)

```mermaid
flowchart LR
  REQS[Requirements]
  R1[Detect chronic LOW; avoid flagging short pulses]
  R2[Exactly one action active; no flicker]
  R3[Block or cap when rhythms healthy; PLV high]
  R4[Block or cap when E &lt; E_min; no zombie forcing]
  R5[Avoid electrical islands; neighbor consensus]
  R6[Reproducible runs; event atlas; summary metrics]

  REQS --> R1
  REQS --> R2
  REQS --> R3
  REQS --> R4
  REQS --> R5
  REQS --> R6
```

**KPIs**

* ROC-AUC ≥ 0.9 (chronic vs transient classification)
* ≈ 0 flicker in steady regimes (hysteresis + dwell)
* Pulse-preservation > naïve clamp baseline
* 0 unsafe actuation when E < E_min

**Acceptance checks (mitigations)**

* Whole‑patch pathology flagged when `domain_low_fraction` ≥ threshold
* False suppression of synthetic repair pulses ↓ ≥80% with PLV persistence
* Faster recovery in low‑E cohort (adaptive `E_min`) without overshoot
* Interventions allowed when `global_v_offset` is high even if E is high

---

## I — Interfaces (configs, CLI, outputs)

### Config (YAML contract)

```yaml
seed: 1337

tissue: { grid: [10,10], dt: 0.005, steps: 3000, EL: -60.0, gL: 0.05, coupling_D: 0.2, noise_rms: 1.0, boundary: periodic }
energy: { E0: 0.7, k_oxphos: 0.2, alpha_actuation_cost: 0.05, beta_tnt_flux: 0.0, gamma_decay: 0.01, Emin: 0.3 }

recorder: { low_enter: -15.0, low_exit: -8.0, tau_low: 600.0, mismatch_threshold: 5.0 }
osc:      { window_seconds: 1200.0, healthy_plv_min: 0.5, downsample: 20 }

decoder:
  type: rules         # rules | perceptron
  hysteresis_margin: 0.05
  decision_dwell: 50
  perceptron: { hidden: 8 }
  rules:      { low_occ_threshold: 200.0, energy_ok: 0.35, mismatch_ok: 5.0 }

actuation:
  amplitude_mV: -10.0
  duty: 0.1
  period_s: 300.0
  refractory_s: 600.0
  cap_when_lowE: { amplitude_mV: -4.0, duty: 0.05 }

safety:  { enable_energy_gate: true, enable_osc_gate: true, enable_geometry_gate: true }
logging: { atlas_stride: 10 }

# --- Mitigation features (new) ---
features:
  use_global_v_offset: true           # catch chronic depolarization even when E is high
  use_domain_low_fraction: true       # detect whole-patch pathology
  use_redox_bit: false                # optional: simple ROS/redox “energy quality” flag

thresholds:
  global_v_offset_mV: 10.0            # long-horizon mean(V) offset that flags chronic drift
  domain_low_fraction: 0.40           # ≥40% cells in LOW ⇒ pathological domain

osc_extras:
  min_bad_duration_s: 600             # PLV must stay < threshold for ≥10 min before acting
  # Optional: multi-band PLV if enabled in code later
  # bands: [[0.0003,0.003],[0.003,0.03]]

energy_extras:
  adaptive_emin:
    enabled: true                     # allow gentle REPAIR during cohort healing
    k: 0.50                           # adaptation strength
    min: 0.20                         # never drop below this

geometry:
  min_coupling_for_consensus: 0.1     # if estimated D < this, reduce consensus weight
```

### CLI (single source of truth)

```bash
python -m polarity_homeostat.experiments.run \
  --config configs/exp_baseline.yaml \
  --out results/exp_baseline
```

**Outputs**

* `event_atlas.csv`: t, action, mean_V, LOW_occ, mismatch, E, PLV, global_v_offset, domain_low_fraction, redox_bit, D_est
* `summary.json`: recovery_time_step, flicker_rate, PLV_retention, final_mean_V, final_mean_E

---


## Limitations & Failure Modes (when the model may not hold)

### ⚠️ 1) Synthetic‑data bias (distribution shift)
**What can happen.** Thresholds and ROC/PR calibration tuned on simulations may not match real rhythm timescales or noise spectra.  
**Why the model misses it.** Mis‑tuned detectors increase false positives/negatives in vivo.  
**Mitigation (planned).** Enable **self‑calibration** (slowly adapt τ_low and PLV thresholds to maintain target error rates) and validate on held‑out scenarios.

---

## Preliminary Results Plan (for the course)

* **Event atlas & ROC/PR** (chronic vs transient).
* **Decision regions** (rules vs tiny MLP) + **flicker rate**.
* **Recovery curves** (bounded pulses vs clamps; pulse‑preservation).
* **Safety envelopes** (E_min, duty, coupling D; TNT/EV scenarios).


---


## Repository Layout (to implement)

```
configs/
src/polarity_homeostat/
  model/        # tissue.py, energy.py
  sensing/      # recorder.py, osc.py
  decoder/      # rules.py, perceptron.py
  safety/       # gates.py
  actuation/    # pulses.py
  eval/         # metrics.py
  utils/        # math_utils.py, io_utils.py
  experiments/  # run.py
tests/          # test_hysteresis.py, test_wta.py, test_plv.py, test_energy_gate.py
results/        # (gitignored)
figures/        # (gitignored)
```

---



## Usage (once code lands)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run baseline experiment
python -m polarity_homeostat.experiments.run \
  --config configs/exp_baseline.yaml \
  --out results/exp_baseline
```

