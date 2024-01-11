import pygad
import circuits
import components
import scipy.spatial
import numpy as np
import matplotlib.pyplot as plt

amp = circuits.BJTClassAAmp()
amp.configure_capacitors(47,100,47)
pot_100k = components.DigitalPot(100000, 100)
desired_gain = 10

generations = 400
population = 20
generations_completed = 0


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
		print(f'Completed generation {current_generation}')
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
	print(f'Resistances: {resistances}')
	print(f'Max voltage: {np.max(output):.2f}mV')
	print(f'Min voltage: {np.min(output):.2f}mV')
	max_error_mv = np.max(np.abs(output - desired))
	mar_error_percent = max_error_mv / desired_gain * 100
	print(f'Max error: {max_error_mv:.2f}mV => {mar_error_percent:.2f}%')
	avg_error_mv = np.mean(np.abs(output - desired))
	avg_error_percent = avg_error_mv / desired_gain * 100
	print(f'Average error: {avg_error_mv:.2f}mV => {avg_error_percent:.2f}%')
	plt.plot(time_us, output, label='output')
	plt.plot(time_us, desired, label='desired')
	plt.legend()
	plt.grid()
	plt.xlabel('time [us]')
	plt.ylabel('voltage [mV]')
	plt.title(f'BJT Class A Amplifier (Gain = {desired_gain})')
	plt.show()


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
	print(f'Parameters of the best solution : {solution}')
	print(f'Fitness value of the best solution = {solution_fitness}')
	
	ga_instance.plot_fitness(linewidth=2, color='#1f77b4', font_size=12)  #save_dir='')
	plot_solution(solution)


if __name__ == "__main__":
	evolve()