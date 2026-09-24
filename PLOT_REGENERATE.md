Regenerating and controlling plots

This project saves per-execution metadata to `simulations/data.csv` and per-generation fitness histories to `simulations/fitness_history_*.json`.

Regenerate plots (no GA run required)
- Waveforms:  python plot_results.py --waveforms
- Fitness curves: python plot_results.py --fitness
- Summary/comparison plots: python plot_results.py --summary
- All: python plot_results.py

Enable plots during a GA run
- Env var: AUTO_PLOTS=1 python amp_4r_evolution.py
- CLI flag: python amp_4r_evolution.py --plots

Notes
- The GA scripts also accept `--no-plots` to explicitly disable plotting when called as a script.
- If you import the scripts as modules (for programmatic control), set `AUTO_PLOTS = True` on the module before calling `evolve()`.

Examples

Run a short 4R GA with plots enabled:

```bash
AUTO_PLOTS=1 python amp_4r_evolution.py
# or
python amp_4r_evolution.py --plots
```

Regenerate all plots from saved data:

```bash
python plot_results.py
```
