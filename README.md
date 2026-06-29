# Graph-Based Energy-Constrained EV Routing on OSM Road Networks

This repository contains the code and selected experimental outputs for the study:

**Graph-Based Energy-Constrained Electric Vehicle Routing on OpenStreetMap Road Networks with Charging-Aware Heuristics and Node-Aware Reinforcement Learning**

The project builds a graph-based EV routing benchmark from OpenStreetMap (OSM) road networks and OSM charging-station locations. It compares distance-only, passive-charging, charging-aware heuristic, full-information planning, and learned node-aware reinforcement-learning policies under battery-feasibility and battery-stress metrics.

## Study scope

The repository is intended to support reproducibility of the paper's main experiments. The work should be interpreted as an **OSM-based, charging-aware, battery-stress-sensitive benchmark**, not as a claim of a new state-of-the-art RL algorithm.

The main benchmark evaluates six routing strategies:

| Method | Role in the benchmark |
|---|---|
| Shortest Path | Distance-only lower baseline |
| Passive Charging | Passive charging baseline |
| Greedy Charging Detour | Simple charging-aware heuristic |
| Energy-aware Charging Detour | Full-information upper-reference online planner |
| Node-aware Q-learning | Main learned policy |
| Node-aware HRL-Q | Exploratory hierarchical ablation |

## Cities and OSM data

The final experiments use OSMnx to reconstruct drivable road networks and OSM charging-station nodes for:

- Amsterdam, Netherlands
- Trondheim, Norway
- Zagreb, Croatia

Charging stations are extracted from OSM using the tag:

```text
amenity=charging_station
```

The raw geographic data are not redistributed because they are publicly available through OpenStreetMap and can be reconstructed with OSMnx using the reported city names and parameters.

## Repository structure

```text
.
├── notebooks/
│   ├── ev_routing_multicity_energy_sensitivity_final.ipynb
│   └── archive/                         # older development notebooks
├── src/
│   └── ev_routing_multicity_energy_sensitivity.py
├── results/
│   └── final_multicity_benchmark/
│       ├── *.csv, *.json, *.txt, *.xlsx
│       ├── cities/
│       │   ├── amsterdam/
│       │   ├── trondheim/
│       │   └── zagreb/
│       └── energy_per_km_sensitivity/
├── figures/                             # selected figures for quick viewing
├── docs/
│   └── reproducibility.md
├── requirements.txt
├── CITATION.cff
└── README.md
```

## Main results included

The repository includes selected outputs from the final multi-city benchmark:

- overall method summaries,
- city-wise results,
- scenario-wise results,
- battery-level sensitivity,
- failure-reason summaries,
- episode-sensitivity analysis,
- energy-consumption sensitivity analysis for `gamma in {2, 5, 10, 15}`,
- selected route visualizations and OSM charging-node maps.

The final run used in the manuscript is stored under:

```text
results/final_multicity_benchmark/
```

## How to run

A typical setup is:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Then open the final notebook:

```text
notebooks/ev_routing_multicity_energy_sensitivity_final.ipynb
```

The notebook can also be run in Google Colab. If running in Colab, mount Google Drive only if you want automatic output persistence. Otherwise, update the output path in the configuration cell.

## Reproducibility notes

- The experiments use OSM-derived data. Results may vary slightly if OSM data are updated after the original run.
- The reported benchmark used a fixed random seed of `42`.
- The learned policies are evaluated as an OD-specific policy-training benchmark, not as city-wide transferable policies.
- The energy model is a normalized road-attribute-aware proxy model, not a calibrated physical EV consumption model.
- The energy-aware detour should be interpreted as a full-information upper-reference planner, not as a learned policy.

## License

This repository is released under the MIT License. See the LICENSE file for details.

## Citation

If you use this repository, please cite the associated paper once publication details are available.
