import data_parse
import numpy as np

if __name__ == "__main__":
    simulation_data = data_parse.load_csv_to_json('simulations/data.csv')

    amp_4r_gain_5_best = np.inf
    amp_4r_gain_5_best_exec = None
    amp_4r_gain_5_max_error = 0
    amp_4r_gain_5_avg = 0
    amp_4r_gain_5_avg_max_error = 0

    amp_4r_gain_10_best = np.inf
    amp_4r_gain_10_best_exec = None
    amp_4r_gain_10_max_error = 0
    amp_4r_gain_10_avg = 0
    amp_4r_gain_10_avg_max_error = 0

    amp_4r_gain_15_best = np.inf
    amp_4r_gain_15_best_exec = None
    amp_4r_gain_15_max_error = 0
    amp_4r_gain_15_avg = 0
    amp_4r_gain_15_avg_max_error = 0

    amp_4r_gain_20_best = np.inf
    amp_4r_gain_20_best_exec = None
    amp_4r_gain_20_max_error = 0
    amp_4r_gain_20_avg = 0
    amp_4r_gain_20_avg_max_error = 0

    amp_8r_gain_5_best = np.inf
    amp_8r_gain_5_best_exec = None
    amp_8r_gain_5_max_error = 0
    amp_8r_gain_5_avg = 0
    amp_8r_gain_5_avg_max_error = 0

    amp_8r_gain_10_best = np.inf
    amp_8r_gain_10_best_exec = None
    amp_8r_gain_10_max_error = 0
    amp_8r_gain_10_avg = 0
    amp_8r_gain_10_avg_max_error = 0

    amp_8r_gain_15_best = np.inf
    amp_8r_gain_15_best_exec = None
    amp_8r_gain_15_max_error = 0
    amp_8r_gain_15_avg = 0
    amp_8r_gain_15_avg_max_error = 0

    amp_8r_gain_20_best = np.inf
    amp_8r_gain_20_best_exec = None
    amp_8r_gain_20_max_error = 0
    amp_8r_gain_20_avg = 0
    amp_8r_gain_20_avg_max_error = 0


    for row in simulation_data:
        row_desired_gain = int(row['desired_gain'])
        row_avg_error_percent = float(row['avg_error_percent'])
        row_max_error_percent = float(row['max_error_percent'])
        if row['ckt_name'] == 'bjt_class_a_amp_4r':
            if row_desired_gain == 5:
                if row_avg_error_percent < amp_4r_gain_5_best:
                    amp_4r_gain_5_best = row_avg_error_percent
                    amp_4r_gain_5_best_exec = row['exec_counter']
                    amp_4r_gain_5_max_error = row_max_error_percent
                amp_4r_gain_5_avg += row_avg_error_percent
                amp_4r_gain_5_avg_max_error += row_max_error_percent
            elif row_desired_gain == 10:
                if row_avg_error_percent < amp_4r_gain_10_best:
                    amp_4r_gain_10_best = row_avg_error_percent
                    amp_4r_gain_10_best_exec = row['exec_counter']
                    amp_4r_gain_10_max_error = row_max_error_percent
                amp_4r_gain_10_avg += row_avg_error_percent
                amp_4r_gain_10_avg_max_error += row_max_error_percent
            elif row_desired_gain == 15:
                if row_avg_error_percent < amp_4r_gain_15_best:
                    amp_4r_gain_15_best = row_avg_error_percent
                    amp_4r_gain_15_best_exec = row['exec_counter']
                    amp_4r_gain_15_max_error = row_max_error_percent
                amp_4r_gain_15_avg += row_avg_error_percent
                amp_4r_gain_15_avg_max_error += row_max_error_percent
            else:
                if row_avg_error_percent < amp_4r_gain_20_best:
                    amp_4r_gain_20_best = row_avg_error_percent
                    amp_4r_gain_20_best_exec = row['exec_counter']
                    amp_4r_gain_20_max_error = row_max_error_percent
                amp_4r_gain_20_avg += row_avg_error_percent
                amp_4r_gain_20_avg_max_error += row_max_error_percent
        elif row['ckt_name'] == 'bjt_class_a_amp_8r':
            if row_desired_gain == 5:
                if row_avg_error_percent < amp_8r_gain_5_best:
                    amp_8r_gain_5_best = row_avg_error_percent
                    amp_8r_gain_5_best_exec = row['exec_counter']
                    amp_8r_gain_5_max_error = row_max_error_percent
                amp_8r_gain_5_avg += row_avg_error_percent
                amp_8r_gain_5_avg_max_error += row_max_error_percent
            elif row_desired_gain == 10:
                if row_avg_error_percent < amp_8r_gain_10_best:
                    amp_8r_gain_10_best = row_avg_error_percent
                    amp_8r_gain_10_best_exec = row['exec_counter']
                    amp_8r_gain_10_max_error = row_max_error_percent
                amp_8r_gain_10_avg += row_avg_error_percent
                amp_8r_gain_10_avg_max_error += row_max_error_percent
            elif row_desired_gain == 15:
                if row_avg_error_percent < amp_8r_gain_15_best:
                    amp_8r_gain_15_best = row_avg_error_percent
                    amp_8r_gain_15_best_exec = row['exec_counter']
                    amp_8r_gain_15_max_error = row_max_error_percent
                amp_8r_gain_15_avg += row_avg_error_percent
                amp_8r_gain_15_avg_max_error += row_max_error_percent
            else:
                if row_avg_error_percent < amp_8r_gain_20_best:
                    amp_8r_gain_20_best = row_avg_error_percent
                    amp_8r_gain_20_best_exec = row['exec_counter']
                    amp_8r_gain_20_max_error = row_max_error_percent
                amp_8r_gain_20_avg += row_avg_error_percent
                amp_8r_gain_20_avg_max_error += row_max_error_percent

    amp_4r_gain_5_avg /= 20
    amp_4r_gain_10_avg /= 20
    amp_4r_gain_15_avg /= 20
    amp_4r_gain_20_avg /= 20
    amp_4r_gain_5_avg_max_error /= 20
    amp_4r_gain_10_avg_max_error /= 20
    amp_4r_gain_15_avg_max_error /= 20
    amp_4r_gain_20_avg_max_error /= 20

    amp_8r_gain_5_avg /= 20
    amp_8r_gain_10_avg /= 20
    amp_8r_gain_15_avg /= 20
    amp_8r_gain_20_avg /= 20
    amp_8r_gain_5_avg_max_error /= 20
    amp_8r_gain_10_avg_max_error /= 20
    amp_8r_gain_15_avg_max_error /= 20
    amp_8r_gain_20_avg_max_error /= 20

    print(f'\namp_4r_gain_5_best: {amp_4r_gain_5_best:.2f}%')
    print(f'amp_4r_gain_5_best_exec: {amp_4r_gain_5_best_exec}')
    print(f'amp_4r_gain_5_max_error: {amp_4r_gain_5_max_error:.2f}%')
    print(f'amp_4r_gain_5_avg: {amp_4r_gain_5_avg:.2f}%')
    print(f'amp_4r_gain_5_avg_max_error: {amp_4r_gain_5_avg_max_error:.2f}%')
    print(f'\namp_4r_gain_10_best: {amp_4r_gain_10_best:.2f}%')
    print(f'amp_4r_gain_10_best_exec: {amp_4r_gain_10_best_exec}')
    print(f'amp_4r_gain_10_max_error: {amp_4r_gain_10_max_error:.2f}%')
    print(f'amp_4r_gain_10_avg: {amp_4r_gain_10_avg:.2f}%')
    print(f'amp_4r_gain_10_avg_max_error: {amp_4r_gain_10_avg_max_error:.2f}%')
    print(f'\namp_4r_gain_15_best: {amp_4r_gain_15_best:.2f}%')
    print(f'amp_4r_gain_15_best_exec: {amp_4r_gain_15_best_exec}')
    print(f'amp_4r_gain_15_max_error: {amp_4r_gain_15_max_error:.2f}%')
    print(f'amp_4r_gain_15_avg: {amp_4r_gain_15_avg:.2f}%')
    print(f'amp_4r_gain_15_avg_max_error: {amp_4r_gain_15_avg_max_error:.2f}%')
    print(f'\namp_4r_gain_20_best: {amp_4r_gain_20_best:.2f}%')
    print(f'amp_4r_gain_20_best_exec: {amp_4r_gain_20_best_exec}')
    print(f'amp_4r_gain_20_max_error: {amp_4r_gain_20_max_error:.2f}%')
    print(f'amp_4r_gain_20_avg: {amp_4r_gain_20_avg:.2f}%')
    print(f'amp_4r_gain_20_avg_max_error: {amp_4r_gain_20_avg_max_error:.2f}%')

    print(f'\namp_8r_gain_5_best: {amp_8r_gain_5_best:.2f}%')
    print(f'amp_8r_gain_5_best_exec: {amp_8r_gain_5_best_exec}')
    print(f'amp_8r_gain_5_max_error: {amp_8r_gain_5_max_error:.2f}%')
    print(f'amp_8r_gain_5_avg: {amp_8r_gain_5_avg:.2f}%')
    print(f'amp_8r_gain_5_avg_max_error: {amp_8r_gain_5_avg_max_error:.2f}%')
    print(f'\namp_8r_gain_10_best: {amp_8r_gain_10_best:.2f}%')
    print(f'amp_8r_gain_10_best_exec: {amp_8r_gain_10_best_exec}')
    print(f'amp_8r_gain_10_max_error: {amp_8r_gain_10_max_error:.2f}%')
    print(f'amp_8r_gain_10_avg: {amp_8r_gain_10_avg:.2f}%')
    print(f'amp_8r_gain_10_avg_max_error: {amp_8r_gain_10_avg_max_error:.2f}%')
    print(f'\namp_8r_gain_15_best: {amp_8r_gain_15_best:.2f}%')
    print(f'amp_8r_gain_15_best_exec: {amp_8r_gain_15_best_exec}')
    print(f'amp_8r_gain_15_max_error: {amp_8r_gain_15_max_error:.2f}%')
    print(f'amp_8r_gain_15_avg: {amp_8r_gain_15_avg:.2f}%')
    print(f'amp_8r_gain_15_avg_max_error: {amp_8r_gain_15_avg_max_error:.2f}%')
    print(f'\namp_8r_gain_20_best: {amp_8r_gain_20_best:.2f}%')
    print(f'amp_8r_gain_20_best_exec: {amp_8r_gain_20_best_exec}')
    print(f'amp_8r_gain_20_max_error: {amp_8r_gain_20_max_error:.2f}%')
    print(f'amp_8r_gain_20_avg: {amp_8r_gain_20_avg:.2f}%')
    print(f'amp_8r_gain_20_avg_max_error: {amp_8r_gain_20_avg_max_error:.2f}%')