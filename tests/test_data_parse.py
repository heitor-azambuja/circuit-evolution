import data_parse


def test_dump_and_load_round_trip(tmp_path):
    csv_file = str(tmp_path / 'data.csv')

    row = {
        'ckt_name': 'bjt_class_a_amp_4r',
        'desired_gain': 10,
        'exec_counter': 1,
        'resistors': [27272.7, 34343.4, 54545.5, 38383.8],
        'solution': [28, 35, 55, 39],
        'solution_fitness': 10.35,
        'generations': 50,
        'population': 10,
        'seed': 1130045385,
        'runtime_s': 1.05,
        'best_generation': 28,
    }

    data_parse.dump_json_to_csv(csv_file, row)
    loaded = data_parse.load_csv_to_json(csv_file)

    assert len(loaded) == 1
    result = loaded[0]
    assert result['resistors'] == [27272.7, 34343.4, 54545.5, 38383.8]
    assert result['ckt_name'] == 'bjt_class_a_amp_4r'
    # optional fields not provided fall back to an empty string
    assert result['avg_error'] == ''


def test_dump_auto_fills_run_id_and_timestamp(tmp_path):
    csv_file = str(tmp_path / 'data.csv')

    data_parse.dump_json_to_csv(csv_file, {'ckt_name': 'x'})
    loaded = data_parse.load_csv_to_json(csv_file)

    assert loaded[0]['run_id']
    assert loaded[0]['timestamp']


def test_dump_appends_multiple_rows(tmp_path):
    csv_file = str(tmp_path / 'data.csv')

    data_parse.dump_json_to_csv(csv_file, {'ckt_name': 'a', 'exec_counter': 1})
    data_parse.dump_json_to_csv(csv_file, {'ckt_name': 'b', 'exec_counter': 2})
    loaded = data_parse.load_csv_to_json(csv_file)

    assert [row['ckt_name'] for row in loaded] == ['a', 'b']
