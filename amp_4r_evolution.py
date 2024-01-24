import pygad
import logging
import circuits
import components
import data_parse
import numpy as np
import scipy.spatial
import matplotlib.pyplot as plt

# set non interactive matplotlib backend
import matplotlib
matplotlib.use('Agg')

logger = logging.getLogger()
# logger.setLevel(logging.INFO)
logger.setLevel(logging.DEBUG)

desired_gain_list = [5, 10, 15, 20]
repetitions = 20

amp = circuits.BJTClassAAmp()
amp.configure_capacitors(47,100,47)
pot_100k = components.DigitalPot(100000, 100)
desired_gain = 10

generations = 400
population = 20
generations_completed = 0

exec_counter = 0
circuit_name = 'bjt_class_a_amp_4r'
data_csv = 'simulations/data.csv'
data_json = {}


def bjt_amp_fitness_func(ga_instance, solution, solution_idx) -> float:
	resistances = []
	for value in solution:
		resistances.append(pot_100k.get_resistance(value))

	amp.configure_resistors(resistances)
	analysis = amp.transient_analysis()
	output = np.array(analysis.out)
	desired = -desired_gain * np.array(analysis['in'])
	result = scipy.spatial.distance.euclidean(output, desired)
	fitness = 1.0 / result
	global generations_completed
	current_generation = ga_instance.generations_completed
	if current_generation > generations_completed:
		logger.debug(f'Completed generation {current_generation}')
		generations_completed = current_generation
	return fitness


def plot_solution(solution) -> None:
	resistances = []
	for value in solution:
		resistances.append(pot_100k.get_resistance(value))
	amp.configure_resistors(resistances)
	analysis = amp.transient_analysis()
	output = 100 * np.array(analysis.out)
	desired = 100 * (-desired_gain) * np.array(analysis['in'])
	time_us = np.array(analysis.time) * 1000000
	
	max_error_mv = np.max(np.abs(output - desired))
	mar_error_percent = max_error_mv / desired_gain * 100
	avg_error_mv = np.mean(np.abs(output - desired))
	avg_error_percent = avg_error_mv / desired_gain * 100
	
	data_json['resistances'] = np.array(resistances)
	data_json['max_voltage'] = np.max(output)
	data_json['min_voltage'] = np.min(output)
	data_json['max_error'] = max_error_mv
	data_json['max_error_percent'] = mar_error_percent
	data_json['avg_error'] = avg_error_mv
	data_json['avg_error_percent'] = avg_error_percent

	logger.debug(f'Resistances: {resistances}')
	logger.debug(f'Max voltage: {np.max(output):.2f}mV')
	logger.debug(f'Min voltage: {np.min(output):.2f}mV')
	logger.debug(f'Max error: {max_error_mv:.2f}mV => {mar_error_percent:.2f}%')
	logger.debug(f'Average error: {avg_error_mv:.2f}mV => {avg_error_percent:.2f}%')
	
	plt.figure(clear=True)
	plt.plot(time_us, output, label='output')
	plt.plot(time_us, desired, label='desired')
	plt.legend()
	plt.grid()
	plt.xlabel('time [us]')
	plt.ylabel('voltage [mV]')
	plt.title(f'BJT Class A Amplifier (Gain = {desired_gain})')
	plt.savefig(f'simulations/{circuit_name}_gain{desired_gain}_execution{exec_counter}.png')
	# plt.show()


def evolve():
	ga_instance = pygad.GA(num_generations=generations,
						   num_parents_mating=4,
						   fitness_func=bjt_amp_fitness_func,
						   sol_per_pop=population,
						   num_genes=4,
						   gene_type=int,
						   gene_space=[range(1, 101), range(1, 101), range(1, 101), range(1, 101)],
						   init_range_low=1,
						   init_range_high=100,
						#    parent_selection_type="sss",
						   parent_selection_type="rws",
						   keep_parents=1,
						   crossover_type="single_point",
						   mutation_type="random",
						   mutation_probability=0.1)
						#    mutation_percent_genes=10,
						#    mutation_num_genes=1)
						#    save_solutions=True)
	
	ga_instance.run()

	solution, solution_fitness, solution_idx = ga_instance.best_solution()

	data_json['solution'] = solution
	data_json['solution_fitness'] = solution_fitness

	logger.info(f'Parameters of the best solution : {solution}')
	logger.info(f'Fitness value of the best solution = {solution_fitness}')
	
	ga_instance.plot_fitness(linewidth=2, color='#1f77b4', font_size=12, save_dir=f'simulations/fitness_{circuit_name}_gain{desired_gain}_execution{exec_counter}.png', label=None)
	plot_solution(solution)


if __name__ == "__main__":
	for gain in desired_gain_list:
		for i in range(repetitions):
			exec_counter = i + 1
			logger.info(f'Running evolution for gain = {gain} ({exec_counter}/{repetitions})')

			amp = circuits.BJTClassAAmp()
			amp.configure_capacitors(47,100,47)
			generations_completed = 0

			desired_gain = gain
			data_json = {}
			data_json['ckt_name'] = circuit_name
			data_json['generations'] = generations
			data_json['population'] = population
			data_json['desired_gain'] = desired_gain
			data_json['exec_counter'] = exec_counter

			evolve()

			data_parse.dump_json_to_csv(data_csv, data_json)