import csv

def read_data(filename, skip_header=True, delimiter="\t"):
	with open(filename, "r") as csv_file:
		reader = csv.reader(csv_file, delimiter=delimiter)

		if skip_header:
			next(reader, None)

		row_ids = []
		y_data = []
		x_data = []
		for line in reader:
			row_ids.append(line[0])
			y_data.append(float(line[1]))
			x_data.append([float(col) for col in line[2:]])

		return row_ids, y_data, x_data
