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
   Scalar energy store `E(t)` with actuation cost and optional TNT/EV flux; viability/safety maps; recovery vs naive clamps.

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

  V & E --> REC[Recorder<br/>• Schmitt bands (LOW/MID/HIGH)<br/>• Leaky time-in-LOW<br/>• Neighbor mismatch]
  V --> OSC[Oscillation Detector<br/>• Band-pass → analytic phase<br/>• PLV vs healthy]
  REC --> FEAT[Feature Vector<br/>&lt;V̄, LOW_occ, mismatch, context, E, PLV&gt;]
  OSC --> FEAT

  FEAT --> DEC{Decoder<br/>Rules or Tiny MLP<br/>WTA + Hysteresis + Dwell}
  DEC -->|REST| MON[Monitor]
  DEC -->|REPAIR| GATES[Safety Gates<br/>Energy / Oscillation / Geometry]
  DEC -->|PRUNE| GATES

  GATES -->|allow| ACT[Actuator<br/>Bounded hyperpolarizing pulses<br/>(amplitude, duty, period, refractory)<br/>+ low-E caps]
  ACT -->|ΔV| V
  ACT -->|cost| E

  FEAT --> LOG[(Event Atlas CSV)]
  DEC --> LOG
```

### M2. Module Wiring (code-level boxes)

```mermaid
flowchart TB
  subgraph model/
    TISSUE[tissue.py<br/>RC grid + Laplacian + leak + noise]
    ENERGY[energy.py<br/>E dynamics + I_pump(E)]
  end

  subgraph sensing/
    RECORDER[recorder.py<br/>Schmitt + leaky LOW + mismatch]
    OSCILLATOR[osc.py<br/>FFT-Hilbert phase + PLV]
  end

  subgraph decoder/
    RULES[rules.py<br/>threshold tree]
    MLP[perceptron.py<br/>tiny 6→H→3 + WTA/hyst/dwell]
  end

  subgraph safety/
    GATE[gates.py<br/>energy / osc / geometry gates]
  end

  subgraph actuation/
    PULSES[pulses.py<br/>bounded pulses + caps]
  end

  subgraph eval/
    METRICS[metrics.py<br/>recovery, flicker, PLV retention]
  end

  subgraph utils/
    MATH[math_utils.py<br/>Laplacian, ring buffers, analytic phase]
    IO[io_utils.py<br/>atlas writer, summaries]
  end

  subgraph experiments/
    RUN[run.py<br/>CLI: config→modules→loop→logs]
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
  R1[Detect chronic LOW without flagging short pulses]
  R2[Exactly one action active (WTA) with no flicker]
  R3[Block/cap actions when rhythms are healthy (PLV high)]
  R4[Block/cap when E &lt; Emin (no zombie forcing)]
  R5[Avoid electrical islands (neighbor consensus)]
  R6[Reproducible runs + event atlas + summary metrics]
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
  dateFormat  YYYY-MM-DD
  section Skeleton
  M0: Repo layout, configs, stubs         :done,    m0, 2025-10-01, 1d
  section Sensing
  M1a: Tissue + Energy integrators        :active,  m1a, 2025-10-02, 2d
  M1b: Recorder + Oscillator + Atlas      :         m1b, 2025-10-04, 2d
  section Decision and Safety
  M2a: Rule decoder + tests               :         m2a, 2025-10-06, 1d
  M2b: Tiny MLP + WTA/hyst/dwell          :         m2b, 2025-10-07, 1d
  M2c: Energy/PLV/Geometry gates          :         m2c, 2025-10-08, 1d
  section Actuation and Metrics
  M3a: Bounded pulses + caps              :         m3a, 2025-10-09, 1d
  M3b: Recovery/flicker/PLV summaries     :         m3b, 2025-10-10, 1d
  section Experiments
  M4: Sweeps (D x Emin x duty, TNT flux)  :         m4,  2025-10-11, 3d
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
  U1[Unit: Schmitt hysteresis enter/exit correct]
  U2[Unit: WTA + hysteresis + dwell prevents flicker]
  U3[Unit: PLV bounds 0..1 on sinusoids]
  U4[Unit: Energy caps honored when E < Emin]
  P1[Property: D→0 ≡ uncoupled (Laplacian invariants)]
  R1[Regression: golden seed reproduces summary.json]

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

---

### Notes
- The **Mermaid blocks** (flowchart + wiring + sequence/gantt) will render on GitHub.
- If you want **even more “spatial” detail**, I can add:
  - a **sequence diagram** of one simulation step, and
  - a **state diagram** for the decoder’s WTA + hysteresis + dwell.
  

