# The 10-Day Window: Intervention Timing, Parameter Uncertainty, and the Collapse of Epidemic Forecast Reliability

**Author:** Keegan Bryce P. Abeja  
**ORCID:** [0009-0004-2948-6005](https://orcid.org/0009-0004-2948-6005)  
**Affiliation:** Independent Researcher, Quezon National High School  
**Contact:** abejakeeganbryce@gmail.com  
**DOI:** *https://doi.org/10.5281/zenodo.20290318*  
**License:** [MIT](LICENSE) 

***Disclaimer: Currently not peer-reviewed***

---

## Abstract

Epidemic forecasting models are routinely used to guide public health decisions, yet the temporal boundary beyond which parameter uncertainty renders forecasts operationally useless remains poorly characterised. We introduce a formal, computable *predictability horizon*—defined as the first day on which the interquartile range of an ensemble of SEIR trajectories exceeds 20% of the ensemble median cumulative incidence—and demonstrate that this horizon is a stable, reproducible quantity jointly determined by the width of parameter uncertainty and the timing of non-pharmaceutical interventions (NPIs). We simulated 1,000-member ensembles of a standard SEIR model using a literature-calibrated parameter space for a generic respiratory pathogen (R₀ ≈ 2.1–4.9, consistent with pre-variant SARS-CoV-2). Under nominal uncertainty, the predictability horizon is 13 days. Doubling the parameter envelope collapses it to 7–8 days; halving it extends it to 23 days. A 40% reduction in transmission rate (β), analogous to moderate NPI packages, extends the horizon by up to 6 days when applied at Day 0, with diminishing returns through Day 5, and provides no measurable gain when applied after Day 10. A complementary σ-reduction intervention (vaccination-like) produces a smaller absolute horizon gain through a mechanistically distinct pathway. A phase diagram mapping intervention timing × parameter uncertainty to predictability horizon length is presented as a decision-support contribution currently absent from the epidemiological literature.

**Keywords:** SEIR, epidemic forecasting, predictability horizon, ensemble simulation, non-pharmaceutical interventions, parameter uncertainty

---

## Repository Structure

```
seir-predictability-horizon/
|
|-- src/
│   |-- seir_model.py          # Vectorised RK4 SEIR solver (NumPy broadcasting)
│   |-- ensemble.py            # Uniform parameter sampling, configurable uncertainty scale u
│   --- predictability.py     # IQR/median horizon computation and divergence curve utilities
│
|-- figures/
│   |-- ensemble_trajectories.png   # Figure 1 - 1,000-member ensemble trajectories
│   |-- divergence_over_time.png    # Figure 2 - IQR/median divergence curves
│   |-- horizon_vs_intervention.png # Figure 3 - Horizon vs. intervention timing
│   --- phase_diagram.png           # Figure 4 - Phase diagram (central contribution)
│
|-- run_analysis.py            # Reproduces all four figures
|-- requirements.txt           # Dependency list
|-- CITATION.cff               # Citation metadata
|-- LICENSE                    # MIT License
--- README.md                  # This file
```

---

## Installation

Python 3.10 or higher is required. All dependencies are open-source.

```bash
git clone https://github.com/keeganbrycegrey/seir-predictability-horizon.git
cd seir-predictability-horizon
pip install -r requirements.txt
```

### Dependencies

| Package    | Minimum version |
|------------|----------------|
| NumPy      | 1.24+           |
| SciPy      | 1.10+           |
| Matplotlib | 3.7+            |

No proprietary software or licensed data is required.

---

## Reproducing the Analysis

Run the run_analysis.py to regenerate all four figures:

```bash
pip install -r requirements.txt
python run_analysis.py
```

Full runtime is ≤ 120s for the complete analysis, including the phase-diagram grid (n = 1,000 × 104 cells × 2 intervention types). All results are seeded at `np.random.default_rng(42)` and reproduce exactly.

---

## Key Results

| Condition | Predictability Horizon |
|-----------|----------------------|
| Baseline (u = 1×, no intervention) | **13 days** |
| Half uncertainty (u = 0.5×) | 23 days |
| Double uncertainty (u = 2×) | 7–8 days |
| β reduced 40% at Day 0 | 19 days (+6 d) |
| β reduced 40% at Day 5 | 16 days (+3 d) |
| β reduced 40% at Day 10+ | ~13 days (no gain) |
| σ reduced 35% at Day 0 | 15 days (+2 d) |

The critical intervention window closes at approximately **Day 10**. After this point, transmission reductions provide no measurable improvement to forecast reliability under nominal parameter uncertainty.

---

## Citation

If you use this work, please cite it using the metadata in [`CITATION.cff`](CITATION.cff), or use the following APA 7 reference:

> Abeja, K. B. P. (2026). *The 10-day window: Intervention timing, parameter uncertainty, and the collapse of epidemic forecast reliability* https://doi.org/10.5281/zenodo.20290318. Independent Researcher, Quezon National High School. https://orcid.org/0009-0004-2948-6005

---

## License

This project is licensed under the [MIT License](LICENSE)—see the LICENSE file for details.
