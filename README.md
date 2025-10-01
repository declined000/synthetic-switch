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
   Oscillation detector (PLV), neighbor consensus; decoders: rules **and** tiny MLP (WTA + hysteresis + dwell). Compare pulse-preservation vs flicker.

3) **Aim C — Energy safety & microenvironment**  
  Scalar energy store `E(t)` with actuation cost and optional TNT/EV flux; viability/safety maps; recovery vs naïve clamps.

---

# MARMIT

> **Minimum Architecture, Requirements, Milestones, Interfaces & Testing**  
> With **spatial, box-and-arrow diagrams** (Mermaid). GitHub renders these.

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
  R4[Block or cap when E &lt; Emin; no zombie forcing]
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
* 0 unsafe actuation when E < Emin

---

## R — Milestones (timeline as Gantt)

```mermaid
gantt
  title MARMIT Milestones
  dateFormat YYYY-MM-DD
  section Skeleton
  Repo layout configs stubs           :m0,  2025-10-01, 1d
  section Sensing
  Tissue and Energy integrators       :m1a, 2025-10-02, 2d
  Recorder Oscillator Atlas           :m1b, 2025-10-04, 2d
  section Decision and Safety
  Rule decoder tests                  :m2a, 2025-10-06, 1d
  Tiny MLP WTA hyst dwell             :m2b, 2025-10-07, 1d
  Energy PLV Geometry gates           :m2c, 2025-10-08, 1d
  section Actuation and Metrics
  Bounded pulses caps                 :m3a, 2025-10-09, 1d
  Recovery flicker PLV summaries      :m3b, 2025-10-10, 1d
  section Experiments
  Sweeps D x Emin x duty TNT flux     :m4,  2025-10-11, 3d
```

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
```

### CLI (single source of truth)

```bash
python -m polarity_homeostat.experiments.run \
  --config configs/exp_baseline.yaml \
  --out results/exp_baseline
```

**Outputs**

* `event_atlas.csv`: t, action, mean_V, LOW_occ, mismatch, E, PLV
* `summary.json`: recovery_time_step, flicker_rate, PLV_retention, final_mean_V, final_mean_E

---

## T — Testing (unit, property, regression)

```mermaid
flowchart TB
  TESTS[Tests]
  U1[Unit Schmitt hysteresis enter exit]
  U2[Unit WTA hysteresis dwell]
  U3[Unit PLV bounds 0 to 1 on sinusoids]
  U4[Unit Energy caps when E &lt; Emin]
  P1[Property D0 uncoupled Laplacian]
  R1[Regression golden seed reproduces summary json]

  TESTS --> U1
  TESTS --> U2
  TESTS --> U3
  TESTS --> U4
  TESTS --> P1

  U1 --> R1
  U2 --> R1
  U3 --> R1
  U4 --> R1
  P1 --> R1
```

**Target coverage**: ≥80% for logic modules (recorder, decoder, gates).  
**Golden run**: one fixed seed & config committed for deterministic regression.

---

## Preliminary Results Plan (for the course)

* **Event atlas & ROC/PR** (chronic vs transient).
* **Decision regions** (rules vs tiny MLP) + **flicker rate**.
* **Recovery curves** (bounded pulses vs clamps; pulse-preservation).
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
tests/          # test_hysteresis.py, test_wta.py, test_plv.py
results/        # (gitignored)
figures/        # (gitignored)
```

---

## Usage (once code lands)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m polarity_homeostat.experiments.run --config configs/exp_baseline.yaml --out results/exp_baseline
```

