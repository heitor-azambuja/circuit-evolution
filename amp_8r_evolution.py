import pygad
import circuits
import components
import scipy.spatial
import numpy as np
import matplotlib.pyplot as plt

amp = circuits.BJTClassAAmp8R()
amp.configure_capacitors(47,100,47)
pot_100k = components.DigitalPot(100000, 100)
pot_10k = components.DigitalPot(10000, 100)
desired_gain = 20


def bjt_amp_fitness_func(ga_instance, solution, solution_idx) -> float:
	resistances = []
	for i in range(0,7,2):
		resistances.append(pot_100k.get_resistance(solution[i]))
		resistances.append(pot_10k.get_resistance(solution[i + 1]))
	
	amp.configure_resistors(resistances)
	analysis = amp.transient_analysis()
	output = np.array(analysis.out)
	desired = -desired_gain * np.array(analysis['in'])
	result = scipy.spatial.distance.euclidean(output, desired)
	fitness = 1.0 / result
	print("generations_completed", ga_instance.generations_completed)
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
	print(f'Average error: {np.mean(np.abs(output - desired) * 100):.2f}%')
	print(f'Max error: {np.max(np.abs(output - desired) * 100):.2f}%')
	print(f'Max voltage: {np.max(output)*1000:.2f}mV')
	print(f'Min voltage: {np.min(output)*1000:.2f}mV')
	plt.plot(time_us, output, label='output')
	plt.plot(time_us, desired, label='desired')
	plt.legend()
	plt.grid()
	plt.xlabel('time [us]')
	plt.ylabel('voltage [mV]')
	plt.title(f'BJT Class A Amplifier (Gain = {desired_gain})')
	plt.show()


if __name__ == "__main__":
	ga_instance = pygad.GA(num_generations=100,
						   num_parents_mating=4,
						   fitness_func=bjt_amp_fitness_func,
						   sol_per_pop=20,
						   num_genes=8,
						   gene_type=int,
						   gene_space=[range(1, 101), range(1, 101), range(1, 101), range(1, 101),
					 				   range(1, 101), range(1, 101), range(1, 101), range(1, 101)],
						   init_range_low=1,
						   init_range_high=100,
						#    parent_selection_type="sss",
						   parent_selection_type="rws",
						   keep_parents=1,
						   crossover_type="single_point",
						   mutation_type="random",
						   mutation_percent_genes=10,
						   mutation_num_genes=1)
						#    save_solutions=True)

	ga_instance.run()

	solution, solution_fitness, solution_idx = ga_instance.best_solution()
	print(f'Parameters of the best solution : {solution}')
	print(f'Fitness value of the best solution = {solution_fitness}')
	
	ga_instance.plot_fitness(linewidth=2, color='#1f77b4', font_size=12)  #save_dir='')
	plot_solution(solution)