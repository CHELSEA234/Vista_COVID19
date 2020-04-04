#!/usr/bin/env python
# coding: utf-8
# Created by Xiao Guo
'''
Downloading pdf based information in json files.
The file contains precached failed index of json files in biorxiv_medrxiv.
'''

import os
import re
import sys
import json
import tqdm
import glob
import numpy as np
import requests			# package to retrive the pdf

## step 1: retrieve the json files.
json_file_dir = '/nas/vista-ssd02/users/xiaoguo/COVID_research/PDFs/biorxiv_medrxiv'
file_name_list = []
for file in glob.glob(os.path.join(json_file_dir,"*.json")):
	file_name_list.append(file)

missing_count = 0
paper_id_list = []
biorxiv_paper_sign = '10.1101/'		# the sign for the paper id or DOI
biorxiv_paper_failed_sign = 'https://doi.org/10.1101/2020.02.'	# does not have paper number
pre_cached_fail_list = [176, 244, 267, 303, 319, 345, 446, 461, 464, 513, 514, 537, 611, 620, 624, 688, 767]
fail_index_list = []
fail_title_list = []

def update_missing_information(file_idx, pdf_dict):
	'''if current file does not have DOI/hint to the article url.'''
	cur_split = "dummy"
	fail_index_list.append(file_idx)
	fail_title_list.append(pdf_dict['metadata']['title'])
	global missing_count
	missing_count += 1

for file_idx, file_name_cur in enumerate(file_name_list):
	exist_flag = False
	with open(file_name_cur) as json_file:
		pdf_dict = json.load(json_file)
		pdf_str = str(pdf_dict)
		pdf_str_split = pdf_str.split()

		if file_idx in pre_cached_fail_list:
			cur_split = "dummy"
			update_missing_information(file_idx, pdf_dict)
		else:
			for cur_split in pdf_str_split:
				if (biorxiv_paper_sign in cur_split and 
					cur_split != biorxiv_paper_failed_sign):
					exist_flag = True
					if '\'' in cur_split:
						cur_split = re.sub(r'[^\w\.\:\/]', '', cur_split)
						if cur_split[-1] == '.':
							cur_split = cur_split[:-1]
					break	# get the string (DOI).
			if not exist_flag:
				cur_split = "dummy"
				update_missing_information(file_idx, pdf_dict)

		paper_id_list.append(cur_split)

print(f"{missing_count} files are missing out of {file_idx+1} files in total.")
## print out missing pdf information
# for i in range(len(fail_title_list)):
# 	print(fail_index_list[i])
# 	print(fail_title_list[i])

## to retrieve the pdf.
## route: doi/hint ==> article url ==> pdf url
pdf_url_prefix_1 = 'https://www.biorxiv.org/content/'
pdf_url_prefix_2 = 'https://www.medrxiv.org/content/'
pdf_url_suffix = 'v1.full.pdf'
dest_folder = './biorxiv'
os.makedirs(dest_folder, exist_ok=True)

def retrieve_pdf(url_web, pdf_url_prefix, pdf_url_suffix=pdf_url_suffix):
	'''
	to retrieve pdf based on article url.
	return r and retrival_flag.
	'''
	url_full = os.path.join(pdf_url_prefix, url_web+pdf_url_suffix)
	r = requests.get(url_full, stream=True)
	retrieval_flag = (r.status_code != 200 and r.status_code != 503)
	return r, retrieval_flag

for idx, paper_id_cur in tqdem(enumerate(paper_id_list)):
	if idx in fail_index_list:
		continue
	else:
		if 'https://doi.org/' in paper_id_cur:
			url_web = paper_id_cur.replace('https://doi.org/', '')
		elif 'doi.org/' in paper_id_cur:
			url_web = paper_id_cur.replace('doi.org/', '')
		elif 'org/' in paper_id_cur:
			url_web = paper_id_cur.replace('org/', '')
		else:
			url_web = paper_id_cur

		# url_full = os.path.join(pdf_url_prefix_1, url_web+pdf_url_suffix)
		# r = requests.get(url_full, stream=True)

		r, retrieval_flag = retrieve_pdf(url_web, pdf_url_prefix_1)
		if not retrieval_flag:
			r, retrieval_flag = retrieve_pdf(url_web, pdf_url_prefix_2)
			if not retrieval_flag:
				print(f"{idx}th paper id, fail to retrive {paper_id_cur}.")
				continue
		# if r.status_code != 200 and r.status_code != 503:
		# 	r = retrieve_pdf(url_web, pdf_url_prefix_2)
		# 	# url_full = os.path.join(pdf_url_prefix_2, url_web+pdf_url_suffix)
		# 	# r = requests.get(url_full, stream=True)
		# 	if r.status_code !=200 and r.status_code != 503:
		# 		print(f"{idx}th paper id, fail to retrive {paper_id_cur}.")
		# 		continue
		
		new_pdf_name = str(idx).zfill(5) + '_' + url_web.split('/')[-1]+'.pdf'
		new_pdf_name = os.path.join(dest_folder, new_pdf_name)
		with open(new_pdf_name, 'wb') as f:
			f.write(r.content)
