import csv
import os
import matplotlib.pyplot as plt
import numpy as np

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

print(f"=========================================")
print(f"Plotting the Distribution")
folder_name = "score_distribution_embedding_uniform"
os.makedirs(folder_name, exist_ok=True)
label_list.append('overall')
for label_idx, label in enumerate(label_list):
	if label == 'diff_document':
		sim = score_diff
	elif label == "consecutive":
		sim = score_consecutive
	elif label == "document":
		sim = score_doc
	elif label == "paragraph":
		sim = score_paragraph
	elif label == "overall":
		sim = score_overall
	title_cur = label + "_embed_similarity_uniform"
	fig = plt.figure(label_idx)	# important to name a fig
	mean_value = f"{np.mean(sim):.3f}"
	std_value = f"{np.std(sim):.3f}"
	plt.hist(sim, bins=20, color='#607c8e', label=f'mean: {mean_value}\n std: {std_value}')
	plt.title(title_cur)
	plt.legend()
	plt.xlabel('Score similarities')
	plt.ylabel('Frequency')
	# plt.savefig(os.path.join(folder_name, 'score_distribution_'+label+'_uniform.png'))
	fig.savefig(os.path.join(folder_name, 'score_distribution_'+label+'_uniform.png'))
