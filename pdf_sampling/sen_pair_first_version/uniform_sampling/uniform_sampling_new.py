import matplotlib.pyplot as plt
import glob
import os
import csv
import numpy as np

label_list = ['diff_document', 'document', 'consecutive', 'paragraph']

FLAT_VALUE = 40000	# max num for each bin.
SAMPLE_VALUE = 1500 	
# samples from each file, take it as 1.5 for consecutive.

# SAMPLE_VALUE_LIST = ([SAMPLE_VALUE] * 3).append(4*SAMPLE_VALUE) 
# SAMPLE_VALUE_LOOP_BREAK = 2000
SAMPLE_VALUE_LOOP_BREAK = 500	# debug mode

STEP_INTERVAL = float(1/20)
min_score, max_score = 0, 1
min_list = np.arange(start=min_score, stop=max_score, step=STEP_INTERVAL)
max_list = min_list + STEP_INTERVAL
counter_dict = dict()

for idx, (min_value, max_value) in enumerate(zip(min_list, max_list)):
	key_value = f'{min_value:.3f}' + f'_{max_value:.3f}'
	counter_dict[key_value] = 0

# doing the csv file.
csv_file_uniform_name = "./uniform_sampling.csv"
csv_file_uniform_handler = open(csv_file_uniform_name, mode='w')
csv_writer = csv.writer(csv_file_uniform_handler, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
csv_writer.writerow(["index", "sim_embed_score", "label", "origin", "file_index"])

def add_into_interval(min_input):
	max_input = min_input + 0.05
	key_str = f'{min_input:.3f}' + f'_{max_input:.3f}'
	if counter_dict[key_str] < FLAT_VALUE:
		counter_dict[key_str] += 1
		return True
	return False

total_sample_index = 0
file_chosen_list = []
for label in label_list:
	file_list = glob.glob(f'../simi_to_csv/{label}*')
	sim_surface = []
	sim_embed = []
	for cur_file in file_list:
		if '_simi.csv' not in cur_file:
			continue
		file_chosen_list.append(cur_file)

def label_retrival(cur_file_name):
	if 'diff_document' in cur_file_name:
		label = "diff_document"
	elif 'document' in cur_file_name:
		label = "document"
	elif 'consecutive' in cur_file_name:
		label = "consecutive"
	elif "paragraph" in cur_file_name:
		label = "paragraph"
	return label


file_chosen_iter = iter(file_chosen_list)
loop_count = 0
break_count = 0
while True:
	loop_count += 1
	try:
		cur_file = next(file_chosen_iter)
	except:
		break_count += 1
		file_chosen_iter = iter(file_chosen_list)
		cur_file = next(file_chosen_iter)

	if loop_count == SAMPLE_VALUE_LOOP_BREAK:
		break

	print(f"working on the {cur_file}...")
	line_count = 0
	label_cur = label_retrival(cur_file)
	sen_file_name = cur_file.replace('_simi.csv', '.csv')
	read_file_handler = open(cur_file, mode='r')
	csv_reader = csv.reader(read_file_handler, delimiter=',')

	# SAMPLE_VALUE_start = break_count * SAMPLE_VALUE if label_cur != "diff_document" else break_count * SAMPLE_VALUE * 4
	# SAMPLE_VALUE_end = (break_count+1) * SAMPLE_VALUE if label_cur != "diff_document" else (break_count+1) * SAMPLE_VALUE * 4
	
	if label_cur in ["consecutive", "paragraph"]:
		SAMPLE_VALUE_start = break_count * int(SAMPLE_VALUE * 1.5)
		SAMPLE_VALUE_end = (break_count+1) * int(SAMPLE_VALUE * 1.5)
	else:
		SAMPLE_VALUE_start = break_count * SAMPLE_VALUE
		SAMPLE_VALUE_end = (break_count+1) * SAMPLE_VALUE

	for row in csv_reader:
		if line_count == 0 or line_count <= SAMPLE_VALUE_start:
			line_count += 1
			continue
		elif line_count > SAMPLE_VALUE_end:
			print(f"sampling in {cur_file} from {SAMPLE_VALUE_start} to {SAMPLE_VALUE_end}.")
			break
		else:
			sim_embed_score = float(row[2])
			sen_file_index = row[0]
			if sim_embed_score >= 0.5:
				if sim_embed_score >= 0.75:
					if sim_embed_score < 0.800:
						write_flag = add_into_interval(0.75)
					elif sim_embed_score >= 0.800 and sim_embed_score < 0.850:
						write_flag = add_into_interval(0.8)
					elif sim_embed_score >= 0.850 and sim_embed_score < 0.900:
						write_flag = add_into_interval(0.85)
					elif sim_embed_score >= 0.900 and sim_embed_score < 0.950:
						write_flag = add_into_interval(0.9)
					elif sim_embed_score >= 0.950 and sim_embed_score < 1:
						write_flag = add_into_interval(0.95)
				elif sim_embed_score < 0.75:
					if sim_embed_score >= 0.70:
						write_flag = add_into_interval(0.70)
					elif sim_embed_score >= 0.65 and sim_embed_score < 0.70:
						write_flag = add_into_interval(0.65)
					elif sim_embed_score >= 0.60 and sim_embed_score < 0.65:
						write_flag = add_into_interval(0.60)
					elif sim_embed_score >= 0.55 and sim_embed_score < 0.60:
						write_flag = add_into_interval(0.55)
					elif sim_embed_score >= 0.5 and sim_embed_score < 0.55:
						write_flag = add_into_interval(0.5)
			else:
				if sim_embed_score >= 0.25:
					if sim_embed_score < 0.3:
						write_flag = add_into_interval(0.25)
					elif sim_embed_score >= 0.300 and sim_embed_score < 0.350:
						write_flag = add_into_interval(0.3)
					elif sim_embed_score >= 0.350 and sim_embed_score < 0.400:
						write_flag = add_into_interval(0.35)
					elif sim_embed_score >= 0.400 and sim_embed_score < 0.450:
						write_flag = add_into_interval(0.4)
					elif sim_embed_score >= 0.450 and sim_embed_score < 0.50:
						write_flag = add_into_interval(0.45)
				elif sim_embed_score < 0.25:
					if sim_embed_score >= 0.20:
						write_flag = add_into_interval(0.20)
					elif sim_embed_score >= 0.15 and sim_embed_score < 0.20:
						write_flag = add_into_interval(0.15)
					elif sim_embed_score >= 0.10 and sim_embed_score < 0.15:
						write_flag = add_into_interval(0.10)
					elif sim_embed_score >= 0.05 and sim_embed_score < 0.10:
						write_flag = add_into_interval(0.05)
					elif sim_embed_score >= 0.0 and sim_embed_score < 0.05:
						write_flag = add_into_interval(0.0)

			line_count += 1
			if write_flag:
				total_sample_index += 1
				write_list = [total_sample_index, sim_embed_score, label_cur, sen_file_name, sen_file_index]
				csv_writer.writerow(write_list)

	read_file_handler.close()
csv_file_uniform_handler.close()
print(f"=========================================")
print(f"the dictionary sampled is: ")
print(counter_dict)

print(f"=========================================")
print(f"Plotting the Pie Chart")
label_list = ['diff_document', 'document', 'consecutive', 'paragraph']
csv_file_uniform_name = "./uniform_sampling.csv"
read_file_handler = open(csv_file_uniform_name, mode='r')
csv_reader = csv.reader(read_file_handler, delimiter=',')
line_count = 0
label_diff = 0; score_diff = []
label_doc = 0; score_doc = []
label_consecutive = 0; score_consecutive = []
label_paragraph = 0; score_paragraph = []
score_overall = []
for row in csv_reader:
	if line_count == 0:
		line_count += 1
		continue
	else:
		cur_score = float(row[1])
		# print(cur_score)
		if row[2] == "consecutive":
			label_consecutive += 1
			score_consecutive.append(cur_score)
		elif row[2] == "diff_document":
			label_diff += 1
			score_diff.append(cur_score)
		elif row[2] == "document":
			label_doc += 1
			score_doc.append(cur_score)
		elif row[2] == "paragraph":
			label_paragraph += 1
			score_paragraph.append(cur_score)
		score_overall.append(cur_score)
		line_count += 1
sizes = [label_diff, label_doc, label_consecutive, label_paragraph]
print(f"the size==> diff: {label_diff}, doc: {label_doc}, consecutive: {label_consecutive}, paragraph: {label_paragraph}.")
read_file_handler.close()
explode = (0, 0.1, 0, 0)
fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=label_list, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.savefig('pie_chart_uniform.png')
