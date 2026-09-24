import json
import uuid
from datetime import datetime
import pygad
import logging
import circuits
import components
import data_parse
import numpy as np
import scipy.spatial
import matplotlib.pyplot as plt
import time

# set non interactive matplotlib backend
import matplotlib
matplotlib.use('Agg')

logger = logging.getLogger()
# logger.setLevel(logging.WARNING)
# logger.setLevel(logging.DEBUG)
logger.setLevel(logging.INFO)

desired_gain_list = [5, 10, 15, 20]
repetitions = 20

pot_100k = components.DigitalPot(100000, 100)
pot_10k = components.DigitalPot(10000, 100)
desired_gain = 20
amp = None  # initialised inside __main__ per run

generations = 400
population = 40
generations_completed = 0

# Coarse step for GA fitness (faster); fine step for final waveform plot
_STEP_TIME_FAST = 0.00005   # 5× coarser → 40 pts per 2ms cycle
_STEP_TIME_FINE = 0.000005  # 10× finer  → 400 pts (for plot only)

exec_counter = 0
circuit_name = 'bjt_class_a_amp_8r'
data_csv = 'simulations/data.csv'
data_json = {}
ga_seed = None


def bjt_amp_fitness_func(ga_instance, solution, solution_idx) -> float:
	try:
		resistances = []
		for i in range(0,7,2):
			resistances.append(pot_100k.get_resistance(solution[i]))
			resistances.append(pot_10k.get_resistance(solution[i + 1]))
		
		amp.configure_resistors(resistances)
		# analysis = amp.transient_analysis(step_time=_STEP_TIME_FAST)
		analysis = amp.transient_analysis(step_time=_STEP_TIME_FINE)
		output = np.array(analysis.out)
		desired = -desired_gain * np.array(analysis['in'])
		n = len(output)
		steady_start = int(0.25 * n)
		result = scipy.spatial.distance.euclidean(output[steady_start:], desired[steady_start:])
		fitness = 1.0 / result if result != 0 else 1e10
		global generations_completed
		current_generation = ga_instance.generations_completed
		if current_generation > generations_completed:
			if logger.isEnabledFor(logging.DEBUG):
				logger.debug(f'Completed generation {current_generation}')
			generations_completed = current_generation
		return fitness
	except Exception as exc:
		if logger.isEnabledFor(logging.DEBUG):
			logger.debug(f'Fitness evaluation failed: {exc}')
		return 1e-10


def compute_metrics(solution) -> dict:
	"""Run simulation for solution and return waveform arrays + error metrics."""
	resistances = []
	for i in range(0, 7, 2):
		resistances.append(pot_100k.get_resistance(solution[i]))
		resistances.append(pot_10k.get_resistance(solution[i + 1]))
	amp.configure_resistors(resistances)
	analysis = amp.transient_analysis(step_time=_STEP_TIME_FINE)
	output = 100 * np.array(analysis.out)
	desired = 100 * (-desired_gain) * np.array(analysis['in'])
	time_us = np.array(analysis.time) * 1000000

	max_error_mv = float(np.max(np.abs(output - desired)))
	max_error_percent = (max_error_mv / desired_gain) * 100
	avg_error_mv = float(np.mean(np.abs(output - desired)))
	avg_error_percent = (avg_error_mv / desired_gain) * 100

	logger.debug(f'Resistances: {resistances}')
	logger.debug(f'Max voltage: {np.max(output):.3f}mV')
	logger.debug(f'Min voltage: {np.min(output):.3f}mV')
	logger.debug(f'Max error: {max_error_mv:.3f}mV => {max_error_percent:.2f}%')
	logger.debug(f'Average error: {avg_error_mv:.3f}mV => {avg_error_percent:.2f}%')

	return {
		'resistances': resistances,
		'output': output,
		'desired': desired,
		'time_us': time_us,
		'max_voltage': float(np.max(output)),
		'min_voltage': float(np.min(output)),
		'max_error': max_error_mv,
		'max_error_percent': max_error_percent,
		'avg_error': avg_error_mv,
		'avg_error_percent': avg_error_percent,
	}


def plot_waveforms(metrics: dict, save_path: str) -> None:
	"""Save waveform comparison plot to save_path."""
	fig, axes = plt.subplots(2, 1, figsize=(10, 7),
							gridspec_kw={'height_ratios': [3, 1]},
							constrained_layout=True)

	ax = axes[0]
	t = metrics['time_us']
	out = metrics['output']
	des = metrics['desired']

	ax.plot(t, des, color='#2ca02c', linewidth=1.8, linestyle='--', label='desired', zorder=3)
	ax.plot(t, out, color='#1f77b4', linewidth=1.8, label='output', zorder=4)
	ax.fill_between(t, out, des, alpha=0.18, color='#d62728', label='error')

	ax.set_xlabel('time [µs]', fontsize=11)
	ax.set_ylabel('voltage [mV]', fontsize=11)
	ax.set_title(
		f'BJT Class A Amplifier (8R) — Gain = {desired_gain}   '
		f'(exec {exec_counter}, seed {ga_seed})',
		fontsize=12, fontweight='bold'
	)
	ax.legend(fontsize=10, loc='upper right')
	ax.grid(True, linestyle=':', alpha=0.6)

	# Annotation box
	res_str = ', '.join(f'{r/1000:.1f}k' for r in metrics['resistances'])
	annotation = (
		f"avg err: {metrics['avg_error_percent']:.2f}%\n"
		f"max err: {metrics['max_error_percent']:.2f}%\n"
		f"R: [{res_str}] Ω"
	)
	ax.text(0.01, 0.97, annotation, transform=ax.transAxes,
			fontsize=9, verticalalignment='top',
			bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8, edgecolor='#aaaaaa'))

	# Error magnitude panel
	axe = axes[1]
	axe.fill_between(t, np.abs(out - des), color='#d62728', alpha=0.5, linewidth=0)
	axe.plot(t, np.abs(out - des), color='#d62728', linewidth=1.0)
	axe.set_xlabel('time [µs]', fontsize=11)
	axe.set_ylabel('|error| [mV]', fontsize=11)
	axe.grid(True, linestyle=':', alpha=0.6)

	fig.savefig(save_path, dpi=150, bbox_inches='tight')
	plt.close(fig)


def plot_solution(solution) -> None:
	"""Compute metrics, store in data_json, and save waveform plot."""
	metrics = compute_metrics(solution)
	data_json['resistors'] = metrics['resistances']
	data_json['max_voltage'] = metrics['max_voltage']
	data_json['min_voltage'] = metrics['min_voltage']
	data_json['max_error'] = metrics['max_error']
	data_json['max_error_percent'] = metrics['max_error_percent']
	data_json['avg_error'] = metrics['avg_error']
	data_json['avg_error_percent'] = metrics['avg_error_percent']
	plot_waveforms(metrics, f'simulations/{circuit_name}_gain{desired_gain}_execution{exec_counter}.png')


_early_stop_patience = generations
_no_improve_count = 0
_best_fitness_seen = -np.inf


def on_generation(ga_instance):
	global _no_improve_count, _best_fitness_seen
	current_best = ga_instance.best_solution()[1]
	if current_best > _best_fitness_seen:
		_best_fitness_seen = current_best
		_no_improve_count = 0
	else:
		_no_improve_count += 1
	if _no_improve_count >= _early_stop_patience:
		return 'stop'


def evolve():
	global _no_improve_count, _best_fitness_seen
	_no_improve_count = 0
	_best_fitness_seen = -np.inf

	start_time = time.perf_counter()
	ga_instance = pygad.GA(num_generations=generations,
						   num_parents_mating=int(population/2),
						   fitness_func=bjt_amp_fitness_func,
						   sol_per_pop=population,
						   num_genes=8,
						   gene_type=int,
						   gene_space=[range(1, 101), range(1, 101), range(1, 101), range(1, 101),
									   range(1, 101), range(1, 101), range(1, 101), range(1, 101)],
						   init_range_low=1,
						   init_range_high=100,
						   parent_selection_type="tournament",
						   keep_parents=1,
						   crossover_type="single_point",
						   mutation_type="random",
						   mutation_probability=0.1,
						   on_generation=on_generation,
						   random_seed=ga_seed)

	ga_instance.run()

	solution, solution_fitness, solution_idx = ga_instance.best_solution()
	runtime_s = time.perf_counter() - start_time

	data_json['solution'] = json.dumps(solution.tolist())
	data_json['solution_fitness'] = solution_fitness
	data_json['best_generation'] = getattr(ga_instance, 'best_solution_generation', None)
	data_json['runtime_s'] = runtime_s

	# Save per-generation fitness history so plots can be regenerated later
	fitness_history_path = f'simulations/fitness_history_{circuit_name}_gain{desired_gain}_execution{exec_counter}.json'
	with open(fitness_history_path, 'w') as fh:
		json.dump({
			'run_id': data_json.get('run_id', ''),
			'ckt_name': circuit_name,
			'desired_gain': desired_gain,
			'exec_counter': exec_counter,
			'seed': ga_seed,
			'best_solutions_fitness': ga_instance.best_solutions_fitness.tolist(),
		}, fh)

	logger.info(f'Parameters of the best solution : {solution}')
	logger.info(f'Fitness value of the best solution = {solution_fitness}')
	
	fitness_path = f'simulations/fitness_{circuit_name}_gain{desired_gain}_execution{exec_counter}.png'
	fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
	ax.plot(ga_instance.best_solutions_fitness, linewidth=2, color='#1f77b4')
	ax.set_xlabel('Generation', fontsize=11)
	ax.set_ylabel('Fitness', fontsize=11)
	ax.set_title(f'Fitness — {circuit_name} gain={desired_gain} exec={exec_counter}', fontsize=11)
	ax.grid(True, linestyle=':', alpha=0.6)
	fig.savefig(fitness_path, dpi=150, bbox_inches='tight')
	plt.close(fig)
	plot_solution(solution)


if __name__ == "__main__":
	for gain in desired_gain_list:
		for i in range(repetitions):
			exec_counter = i + 1
			logger.info(f'Running evolution for gain = {gain} ({exec_counter}/{repetitions})')

			amp = circuits.BJTClassAAmp8R()
			amp.configure_capacitors(47,100,47)
			generations_completed = 0

			desired_gain = gain
			data_json = {}
			data_json['ckt_name'] = circuit_name
			data_json['generations'] = generations
			data_json['population'] = population
			data_json['desired_gain'] = desired_gain
			data_json['exec_counter'] = exec_counter
			ga_seed = int(np.random.randint(0, 2**31 - 1))
			np.random.seed(ga_seed)
			data_json['seed'] = ga_seed
			data_json['run_id'] = str(uuid.uuid4())
			data_json['timestamp'] = datetime.now().isoformat()

			evolve()

			data_parse.dump_json_to_csv(data_csv, data_json)