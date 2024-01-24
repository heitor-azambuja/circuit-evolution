import os
import csv

def dump_json_to_csv(csv_file, json_data):
    # Check if the CSV file exists
    file_exists = os.path.isfile(csv_file)

    # Open the CSV file in append mode
    with open(csv_file, 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=json_data.keys())

        # Write headers if the file does not exist
        if not file_exists:
            writer.writeheader()

        # Write the JSON data to the CSV file
        writer.writerow(json_data)


# Load csv file into a json
def load_csv_to_json(csv_file):
    # Open the CSV file for reading
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        json_data = []
        for row in reader:
            json_data.append(row)

    return json_data