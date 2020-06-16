import matplotlib.pyplot as plt
import glob
import os
import csv
import numpy as np

def isNaN(num):
    return num != num

bio_embed_flag = True
label_list = ['diff_doc', 'document', 'consecutive', 'paragraph']

if bio_embed_flag:
	folder_name = 'score_distribution_bio_embedding'
	ROW_NUM = 3
else:
	folder_name = 'score_distribution_embedding'
	ROW_NUM = 2

total_sim_embed = []
os.makedirs(folder_name, exist_ok=True)
for label_idx, label in enumerate(label_list):
	print(f"================================")
	file_list = glob.glob(f'../simi_to_csv/{label}*')
	sim_embed = []
	for cur_file in file_list:
		if '_simi.csv' not in cur_file:
			continue
		print(f"working on the {cur_file}...")
		line_count = 0
		read_file_handler = open(cur_file, mode='r')
		csv_reader = csv.reader(read_file_handler, delimiter=',')

		for row in csv_reader:
			if line_count == 0:
				line_count += 1
				continue
			else:
				sim_embed.append(float(row[ROW_NUM]))
				total_sim_embed.append(float(row[ROW_NUM]))
				line_count += 1
		read_file_handler.close()
	print(f"the number of {label} is {len(sim_embed)}.")
	mean_value = f"{np.mean(sim_embed):.3f}"
	std_value = f"{np.std(sim_embed):.3f}"
	# print(f"the mean value is {np.mean(sim_embed):.3f}, std is {np.std(sim_embed):.3f}.")
	sim = sim_embed
	# xmin = min(sim); xmax = max(sim)
	title_cur = label + "_bio_embed_similarity" if bio_embed_flag else label + "_embed_similarity" 
	fig = plt.figure(label_idx) 
	# plt.hist(sim, bins=20, range=[xmin*-1.1, xmax*1.1], color='#607c8e')
	plt.hist(sim, bins=20, color='#607c8e', label=f'mean: {mean_value}\n std: {std_value}')
	plt.legend()
	plt.title(title_cur)
	plt.xlabel('Score similarities')
	plt.ylabel('Frequency')
	# plt.savefig(os.path.join(folder_name, 'score_distribution_'+label+'.png'))    wrong.
	fig.savefig(os.path.join(folder_name, 'score_distribution_'+label+'.png'))

total_sim = total_sim_embed
mean_value = f"{np.mean(total_sim):.3f}"
std_value = f"{np.std(total_sim):.3f}"
overall_title = "overall_bio_embed_similarity" if bio_embed_flag else "overall_embed_similarity" 
fig = plt.figure(len(label_list)+1) 
# xmin = min(total_sim); xmax = max(total_sim)
# plt.hist(total_sim, bins=20, range=[xmin*-1.1, xmax * 1.1], color='#607c8e')
plt.hist(total_sim, bins=20, color='#607c8e', label=f'mean: {mean_value}\n std: {std_value}')
plt.title(overall_title)
plt.legend()
plt.xlabel('Score similarities')
plt.ylabel('Frequency')
fig.savefig(os.path.join(folder_name, 'score_distribution_overall.png'))
