def csv_load(filename: str) -> list:
	sheet = []
	with open(filename, 'r') as file:
		for i, row in enumerate(file.read().split('\n')):
			if i % 1000 == 0:
				print(i, row)
			current_row = []
			for item in row.split(','):
				try:
					current_row.append(float(item))
				except ValueError:
					current_row.append(item)
			sheet.append(current_row)
	return sheet

# sote = csv_load('exports.txt')
