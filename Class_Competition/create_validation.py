import sys
import random
import csv
import parse_data

def create_csv(filename, indexes, rna_ids, y_data, x_data):
	with open(filename, "w") as fp:
		file_writer = csv.writer(fp)
		for index in indexes:
			file_writer.writerow([rna_ids[index]] + [y_data[index]] + x_data[index])

def main(argv):
	if len(argv) != 5:
		sys.exit(f"Usage python3 {argv[0]} <file> <percent_validation> <training_filename> <validation_filename>")

	rna_ids, y_data, x_data = parse_data.read_data(argv[1])
	pct_validation = float(argv[2])

	y_counts = {}
	for index, y in enumerate(y_data):
		key = repr(y)
		if key not in y_counts:
			y_counts[key] = []
		y_counts[key].append(index)

	validation_indexes = []
	training_indexes = []
	for key in y_counts:
		random.shuffle(y_counts[key])

		split_point = int(len(y_counts[key]) * pct_validation)
		for index in y_counts[key][:split_point]:
			validation_indexes.append(index)
		for index in y_counts[key][split_point:]:
			training_indexes.append(index)

	create_csv(argv[3], training_indexes, rna_ids, y_data, x_data)
	create_csv(argv[4], validation_indexes, rna_ids, y_data, x_data)

main(sys.argv)
