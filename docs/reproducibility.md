# Reproducibility guide

## Data source

The benchmark reconstructs road-network and charging-station data from OpenStreetMap through OSMnx. The raw data are not stored in this repository because they are publicly available from OSM and can be downloaded again using the city names and OSM tags reported in the paper.

## Final benchmark configuration

- Cities: Amsterdam, Trondheim, Zagreb
- Network type: drivable road network
- Charging-station tag: `amenity=charging_station`
- Battery capacity: 100 normalized units
- Main normalized energy coefficient: `gamma = 10.0`
- Initial battery levels: 40, 60, 80, 100
- OD scenarios: random and battery-stratified challenging
- OD pairs per city and scenario: 20
- RL benchmark episodes: 2000
- Sensitivity gamma values: 2.0, 5.0, 10.0, 15.0
- Sensitivity RL episodes: 1000
- Fixed random seed: 42

## Important interpretation

The methods are compared under different operational roles:

- Energy-aware Charging Detour is a full-information upper-reference planner that scans candidate charging stations and performs online station-to-destination feasibility checks.
- Node-aware Q-learning is the main learned policy and is executed from a learned Q-table after training.
- Node-aware HRL-Q is included as an exploratory hierarchical ablation; high loop counts and timeout behavior should be interpreted as evidence of subgoal instability in the current implementation.

## Output files

The final benchmark outputs are in `results/final_multicity_benchmark/`. Important files include:

- `multi_city_overall_summary_by_method.csv`
- `multi_city_summary_by_city_method.csv`
- `multi_city_summary_by_battery_method.csv`
- `multi_city_benchmark_results_all_modes.csv`
- `episode_sensitivity_summary_both_methods.csv`
- `energy_per_km_sensitivity/energy_sensitivity_overall_by_gamma_method.csv`
- `energy_per_km_sensitivity/energy_sensitivity_paper_table.csv`
- `all_results.xlsx`
