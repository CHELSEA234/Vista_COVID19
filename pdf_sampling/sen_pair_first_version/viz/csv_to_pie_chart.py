import matplotlib.pyplot as plt
import csv
import glob

label_list = ['diff_document', 'document', 'consecutive', 'paragraph']
sizes = []

for label in label_list:
	file_list = glob.glob(f'../{label}*')
	sentence_num = 0
	print(f"the strategy is {label}.")
	for cur_file in file_list:
		if '_simi.csv' in cur_file:
			continue
		print(f"working on the file {cur_file}.")
		csv_file = open(cur_file, 'r')
		csv_reader = csv.reader(csv_file, delimiter=',')
		row_count = sum(1 for row in csv_reader)
		csv_file.close()
		sentence_num += row_count
	sizes.append(sentence_num)
	print(f"the total sentence num is: {sentence_num}.")

explode = (0, 0.1, 0, 0)
fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=label_list, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.savefig('pie_chart.png')
