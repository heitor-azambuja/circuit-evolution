import csv
import json
import logging
import os
import uuid
from datetime import datetime

logger = logging.getLogger(__name__)

RUN_FIELDS = [
    'run_id',
    'timestamp',
    'ckt_name',
    'desired_gain',
    'exec_counter',
    'avg_error',
    'max_error',
    'avg_error_percent',
    'max_error_percent',
    'resistors',
    'solution',
    'solution_fitness',
    'max_voltage',
    'min_voltage',
    'generations',
    'population',
    'seed',
    'runtime_s',
    'best_generation',
]


def _normalize_row_for_dump(json_data):
    row = dict(json_data)

    # Auto-fill audit fields
    if not row.get('run_id'):
        row['run_id'] = str(uuid.uuid4())
    if not row.get('timestamp'):
        row['timestamp'] = datetime.now().isoformat()

    if 'resistors' in row and row['resistors'] is not None and not isinstance(row['resistors'], str):
        row['resistors'] = json.dumps(row['resistors'])

    if 'solution' in row and row['solution'] is not None and not isinstance(row['solution'], str):
        try:
            row['solution'] = json.dumps(list(row['solution']))
        except TypeError:
            row['solution'] = str(row['solution'])

    normalized = {}
    for field in RUN_FIELDS:
        value = row.get(field, '')
        if value is None:
            value = ''
        normalized[field] = value

    for key, value in row.items():
        if key not in normalized and key != 'resistances':
            normalized[key] = value

    return normalized


def _parse_resistors_field(value):
    if value is None:
        return value

    if isinstance(value, list):
        return value

    text = str(value).strip()
    if not text:
        return []

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
        return value
    except (json.JSONDecodeError, TypeError):
        return value

def dump_json_to_csv(csv_file, json_data):
    directory = os.path.dirname(csv_file)
    if directory:
        os.makedirs(directory, exist_ok=True)
    row = _normalize_row_for_dump(json_data)

    # Append against the header already on disk, not against this row's keys.
    # Rows do not all carry the same keys -- fitness_eval_failures only appears
    # when a run had SPICE failures -- and writing each row against its own keys
    # silently produces a ragged file: extra values land past the last column.
    header = None
    if os.path.isfile(csv_file):
        with open(csv_file, newline='') as file:
            header = next(csv.reader(file), None)

    with open(csv_file, 'a', newline='') as file:
        if header is None:
            header = list(row.keys())
            writer = csv.DictWriter(file, fieldnames=header)
            writer.writeheader()
        else:
            writer = csv.DictWriter(file, fieldnames=header)
            dropped = [key for key in row if key not in header]
            if dropped:
                logger.warning(
                    f'{csv_file} has no column for {dropped}; those values are '
                    f'not being recorded. Delete the file to rebuild its header.')

        writer.writerow({key: row.get(key, '') for key in header})


def load_csv_to_json(csv_file):
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        json_data = []
        for row in reader:
            if 'resistors' in row:
                row['resistors'] = _parse_resistors_field(row['resistors'])
            json_data.append(row)

    return json_data