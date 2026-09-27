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

Install python requirements
```bash
python3 -m pip install -r requirements-lock.txt
```

`requirements-lock.txt` pins the versions every result in this repository was
produced with. `requirements.txt` lists newer ones that have never been exercised
here — PySpice 1.5 predates numpy 2, which removed aliases it still uses.

On a fresh machine, `./setup_env.sh` does all of the above and then checks it: that
PySpice can reach libngspice and actually simulate, that the tests pass, and how
long one run takes. It writes `environment.txt` recording the host, the ngspice
version and the pinned packages — keep that alongside any results the machine
produces, because ngspice is the solver and apt gives whatever the distribution
carries.

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

Its cutoff metrics report two frequencies, not one: `cutoff_realized_hz` is where the
simulated response crosses −3 dB, and `cutoff_reference_hz` is where the *ideal*
target response crosses it. `cutoff_error_percent` is the difference between them.
Measuring against the nominal `cutoff_hz` instead would be wrong for a Chebyshev I,
whose nominal cutoff is the ripple-band edge — the ideal response's own −3 dB point
sits about 10% above it at order 4 and 0.5 dB ripple, so an exactly correct filter
would read as 10% off.

### filter_targets.py
The filter specification: every target is one (response family, cutoff) pair, and
the cutoff sweep — 1000, 1500, 2000, 3000 Hz — is the filter's analogue of the
amplifier's four gains. The capacitors are fixed across every target and both
variants on purpose, so the sweep tests resistor resolution and nothing else.

### filter_sk4_evolution.py and filter_sk8_evolution.py
The two resistor realizations. 4R puts one X9C103 (10 kΩ) on each filter
resistor, giving a 101 Ω step. 8R puts an X9C103 in series with an X9C102 (1 kΩ)
for a 10.1 Ω step — ten times finer, at the cost of a second part per resistor.
That is the comparison: whether the finer part pays for itself.

### filter_optimum.py
The 4R filter's global optimum, by enumerating all 100 million tap combinations.
The op-amp drives its output as an ideal voltage source, so the second stage does
not load the first and the cascade's response is exactly the product of the two —
which turns a 100⁴ search into 2 × 100² simulations plus arithmetic. Only 4R is
enumerable; 8R has 10¹⁶ combinations.

### filter_tolerance.py
Component tolerance, drawn once per board and fixed for that board's life. Models
what the datasheets state: ±20% end-to-end resistance and 0–100 Ω of wiper on the
X9C, ±10% on E12 capacitors.

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

### campaign.py
The paper's experiment: every (variant, target, seed) cell of one circuit family,
each run in its own worker process. `--shard i/N` splits the job list so several
machines can take disjoint slices. Results are appended to a CSV run by run, so
an interrupted campaign keeps what it finished.

Process isolation is not tidiness: PySpice leaks about 47 kB per simulation and
never returns it, so a run costs a few hundred megabytes that only a process exit
reclaims.

`--variant` and `--population` exist to break one confound. In **both** families the
two variants default to different search budgets — 4R evolves 4 genes with a
population of 20, 8R evolves 8 with a population of 40 — so a plain 4R-vs-8R result
mixes resistor resolution with the number of evaluations each side was allowed.
Re-run one variant at the other's population to separate them:

```bash
RESULTS=results-control FAMILIES=filter \
    ./run_campaign.sh 1/1 --variant 4R --population 40
RESULTS=results-control FAMILIES=amp \
    ./run_campaign.sh 1/1 --variant 4R --population 40
```

Or `FAMILIES="filter amp"` for both in one go — they write separate CSVs.

`--generations` overrides the other half of the budget, and `--target` (repeatable)
narrows the campaign to particular cells, which is what a focused measurement needs:

```bash
python3 campaign.py --family filter --variant 4R --population 40 --generations 800 \
    --target butterworth_1000 --seeds 40 --data-csv gen800.csv --out-dir gen800
```

Every row records the population it ran with, so a control and the original campaign
stay distinguishable inside the CSV. Keep them in separate output directories
though: fitness history filenames are built from circuit, target and seed only, so a
control writes over the histories it is meant to be compared against. `campaign.py`
warns when it is about to, counting only the files the families and variants being
run would actually touch.

### analyze_campaign.py
Statistics for the finished campaign in `simulations/campaign/`, which
`evaluate_sims.py` cannot describe because it is shaped around the amplifier's
time-domain columns. Reads only the CSVs, so every figure quoted in a write-up can
be regenerated and checked:

```bash
python3 analyze_campaign.py
```

It prints an integrity block first (cell coverage, duplicate seeds, evaluation
failures, shard agreement) because a fault there invalidates everything after it,
and it reports the per-variant search budget explicitly: the 4R and 8R specs
default to populations of 20 and 40, so a 4R-vs-8R difference confounds resistor
resolution with search budget until a matched-budget control is run.

Two measurement notes it encodes. The variants are *independent* samples, not
paired — the same seed drives a 4-gene and an 8-gene GA, whose populations are
unrelated — so it uses Mann-Whitney with a Holm correction, not a paired test. And
it re-derives cutoff error rather than reading the CSV's `cutoff_error_percent`,
which lets it analyse runs recorded before that column was corrected. It measures
the ideal response's -3 dB point **on the sweep grid**, the same coarse grid and log
interpolation the realized crossing came from: reading a crossing off 20 points per
decade is biased by up to 0.8%, and only a same-grid comparison cancels it.

It also reports the matched-budget control (`simulations/control/`) when present,
separating what the extra search budget bought from what the extra resistor
resolution bought, and the spread of the analytical design across cells against the
spread of the GA's.

### plot_paper.py
The paper's figures, drawn from the same CSVs `analyze_campaign.py` summarises so a
figure cannot disagree with the table it illustrates. Output lands in
`simulations/figures/` as both PNG and PDF:

```bash
python3 plot_paper.py                  # all six
python3 plot_paper.py --only lottery   # one
```

`lottery` (GA distribution vs the single analytical design vs the enumerated
optimum, per specification), `bestofn` (how many independent runs it takes to beat
the analytical design), `bode` (the response itself, with a passband inset),
`amplifier` (the bimodal error distribution), `asymmetry` (what reordering R1 and R2
recovers for free) and `budget` (search budget separated from resistor resolution).
Four read only CSVs; `bode` and `asymmetry` call ngspice, because they draw
responses rather than summarise recorded runs.

`median_best_of_n` there computes the median of the best of *n* draws exactly, from
the binomial tail, instead of resampling — the figure is the measured distribution
rather than a sample from it.

### filter_optimum.py — all eight cells

Enumerating every cell (`python3 filter_optimum.py --out simulations/optima_4r.json`)
gives the true ceiling for the 4R variant, which is what turns "the GA beat the
analytical design" into a statement with a scale. It also checks itself: for the same
taps, the enumeration computes the cascade as the product of two separately simulated
stages while a GA run simulates the whole netlist at once, and the two agree to about
1e-6 relative — the separability the whole method rests on.

### amp_design.py
The amplifier's analytical design equations, the counterpart of `filter_design`.
The contrast is the point: the filter's equations are exact, while these are
small-signal approximations to a nonlinear device and miss the target gain by
roughly ten percent even with resistors of infinite precision.

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
python3 filter_sk4_evolution.py --plots      # one pot per resistor
python3 filter_sk8_evolution.py --plots      # two in series, ten times finer
python3 filter_baseline.py --plots           # the analytical design, quantized
```
Filter results go to `simulations/filters.csv`, kept separate from the amplifier's `data.csv` because the two have different columns.

### The full campaign

`campaign.py` runs every cell of one family. On a single machine:

```bash
./run_campaign.sh 1/1
```

Split across machines by giving each a disjoint slice — each writes its own CSV,
and the rows carry family, variant, target and seed, so order does not matter:

```bash
./run_campaign.sh 1/2      # machine A
./run_campaign.sh 2/2      # machine B
```

Concatenate when every machine has finished:

```bash
head -1 results/filter_1de2.csv            > filter_all.csv
tail -q -n +2 results/filter_*de*.csv     >> filter_all.csv
```

Early stopping exists (`--patience`) but is off by default and should stay off
for a campaign: at a patience of 50 it halves the generations while leaving 28%
of runs below 99% of the fitness they would otherwise reach.

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