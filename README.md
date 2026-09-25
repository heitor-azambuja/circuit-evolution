# circuit-evolution
Evolve circuits components values with GA.

This code was developed and tested on Ubuntu 20 LTS and Debian 12. It should work on Windows, but installation of Ngspice might be different, check [PySpice documentation](https://pyspice.fabrice-salvaire.fr/releases/v1.4/installation.html#on-windows).

## Installation
Install Ngspice
```bash
sudo apt-get -y install ngspice libngspice0 libngspice0-dev
```

Create virtual environment
```bash
python3 -m venv venv
```

Activate virtual envitonment
```bash
source venv/bin/activate
```

Install python requitements
```bash
python3 -m pip install -r requirements.txt
```

Test PySpice installation
```bash
pyspice-post-installation --check-install
```

## Modules
A brief description of all the modules in the project.

### components.spice
This file contains the Spice models of NPN and PNP transistors that are used in the circuits.

### components.py
This file contains a class that defines a digital potenciomenter.

### circuits.py
Classes to instantiate the circuits that are evolved. Two BJT class A amplifiers, one with 4 resistors and one with 8 resistors, plus `SallenKeyLowPass`, a 4th-order active low-pass filter (two unity-gain Sallen-Key stages with ideal VCVS op-amps).

### evolution_common.py
Shared GA logic used by every run script: the GA loop, failure tracking, CSV/JSON output, and the CLI. Each script only defines a `CircuitSpec` (which circuit, how many resistors, how genes map to resistor values, how quality is measured) and calls `run_cli()`.

Quality measurement lives in an *evaluator*, which is what lets targets in different domains coexist. `TransientGainEvaluator` (here) scores the amplifier in the time domain; `AcResponseEvaluator` scores the filter in the frequency domain.

### amp_4r_evolution.py And amp_8r_evolution.py
Thin per-circuit configs that run the evolution of the amplifier with 4 and 8 resistors respectively, via `evolution_common.py`.

### filter_design.py
Analytical design equations for the Sallen-Key filter: turning an approximation (Butterworth, Chebyshev) into per-stage (ωo, Q) pairs, those into ideal resistors, and those onto digital-pot taps. Pure math, no simulation.

Note the constraint it encodes: a unity-gain Sallen-Key stage can only reach `Q = 0.5·sqrt(C1/C2)`, so **equal capacitors cap Q at 0.5** and put any 4th-order response out of reach. The capacitors must be deliberately unequal, and are chosen per stage and per target.

### filter_evaluation.py
The frequency-domain objective: weighted RMS error in dB between the AC sweep's |H(f)| and the target curve from `scipy.signal`. Passband error is weighted above stopband error, and both curves are floored at −80 dB so the deep stopband — where the ideal response reaches −160 dB — cannot dominate the error.

### filter_sk4_evolution.py
Runs the filter evolution. Holds the per-target table (response, ripple, capacitors) that is the single source of truth for both the evaluator and the circuit setup.

### filter_baseline.py
The control the GA is measured against: designs each stage from the textbook equations, rounds every resistor to the nearest pot tap, and scores it with the *same* evaluator the GA optimises. The gap between the two is the quantization the GA absorbs and the analytical route cannot.

### data_parse.py
Reads and writes run results to/from `simulations/data.csv` (one row per GA run).

### evaluate_sims.py
Summarizes and compares runs from `simulations/data.csv`: error/convergence statistics, success rates, and 4R-vs-8R significance testing, plus the summary/comparison plots.

### plot_results.py
Regenerates waveform, fitness-history, and summary plots from saved data (`simulations/data.csv` and `simulations/fitness_history_*.json`) without re-running the GA. Usage:
```bash
python3 plot_results.py                  # regenerate everything
python3 plot_results.py --waveforms      # waveform plots only
python3 plot_results.py --fitness        # fitness history plots only
python3 plot_results.py --summary        # evaluate_sims summary plots only
```

### examples.py
Standalone PySpice example plotting NPN BJT characteristic curves; not part of the evolution pipeline.

## Running
Evolve class A amplifier with 4 resistors:
```bash
python3 amp_4r_evolution.py
```
Evolve class A amplifier with 8 resistors:
```bash
python3 amp_8r_evolution.py
```

Evolve the 4th-order Sallen-Key low-pass filter, and then its analytical baseline:
```bash
python3 filter_sk4_evolution.py --plots
python3 filter_baseline.py --plots
```
Filter results go to `simulations/filters.csv`, kept separate from the amplifier's `data.csv` because the two have different columns.

All run scripts accept the same CLI flags: `-g/--generations`, `-p/--population`, `-r/--repetitions`, `--seed`, `--exec-counter`, and `--plots` (save PNG plots after each run — off by default). Use `--gain` to run a single amplifier gain, or `--target` to run a single filter response. Run with `--help` for details.

When running, Pyspice might show some warning and error messages about Ngspice not supported version. But it should work fine.

## Running tests
Install dev dependencies and run the unit test suite (no ngspice simulation is required — the tests only cover the pure-Python logic):
```bash
python3 -m pip install -r requirements.txt
pytest
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

<!-- Please make sure to update tests as appropriate. -->

## Resources

- [PySpice](https://pyspice.fabrice-salvaire.fr/releases/v1.4/overview.html)
- [NgSpice](http://ngspice.sourceforge.net/)
- [PyGad](https://pygad.readthedocs.io/en/latest/)
- PySpice introduction [video playlist](https://youtube.com/playlist?list=PL97KTNA1aBe1QXCcVIbZZ76B2f0Sx2Snh&si=VEu1G6qsrAJ8VWXl) by [EngineeringThings](https://www.youtube.com/@engineeringthings)

## Author
- Heitor Teixeira de Azambuja - heitortazamba@gmail.com - [![Linkedin](https://i.stack.imgur.com/gVE0j.png)](https://www.linkedin.com/in/heitor-azambuja/)

## License

[MIT](LICENSE)