#!/usr/bin/env python
# coding: utf-8
# Created by Xiao Guo
'''
Downloading pdf based information in json files.
The file contains precached failed index of json files in biorxiv_medrxiv.
retrival step: doi/hint ==> article url ==> pdf url; each step has specific sign associated.
'''

import os
import re
import sys
import json
import glob
import argparse
import numpy as np
import requests			# package to retrive the pdf

from tqdm import tqdm

def csv_gen(fail_index_list, fail_title_list):
	'''dump information for missing files.'''
	import csv
	write_list = zip(fail_index_list, fail_title_list)
	with open('missing_file.csv', 'a') as f:
		writer = csv.writer(f, delimiter='\t')
		writer.writerows(["index", "title"])
		writer.writerows(write_list)

def retrieval_helper(url_web, pdf_url_prefix, pdf_url_suffix):
	'''
	to retrieve pdf based on article url.
	return r and retrival_flag.
	'''
	url_full = os.path.join(pdf_url_prefix, url_web+pdf_url_suffix)
	r = requests.get(url_full, stream=True)
	retrieval_flag = (r.status_code != 200 and r.status_code != 503)
	return r, retrieval_flag

def pdf_retrival(dest_folder, paper_id_list, fail_index_list):
	"""to retrieve pdf based on the doi."""
	pdf_url_prefix_1 = 'https://www.biorxiv.org/content/'	# specific title in biorxiv paper
	pdf_url_prefix_2 = 'https://www.medrxiv.org/content/'	# specific title in medrxiv paper
	pdf_url_suffix = 'v1.full.pdf'							# specific format for pdf.
	trim_sign_list = ['https://doi.org/', 'doi.org/', 'org/'] # specific clean json obtained strings	
	os.makedirs(dest_folder, exist_ok=True)	

	for idx, paper_id_cur in tqdm(enumerate(paper_id_list)):
		if idx in fail_index_list:
			continue
		else:
			trim_flag = False
			for trim_sign in trim_sign_list:
				if trim_sign in paper_id_cur:
					url_web = paper_id_cur.replace(trim_sign, '')
					trim_flag = True 	# has been trimmed.
					break
			if not trim_flag:
				url_web = paper_id_cur

		# iterate two specific url prefix.
		r, retrieval_flag = retrieval_helper(url_web, pdf_url_prefix_1, pdf_url_suffix)
		if not retrieval_flag:
			r, retrieval_flag = retrieval_helper(url_web, pdf_url_prefix_2, pdf_url_suffix)
			if not retrieval_flag:
				print(f"{idx}th paper id, fail to retrive {paper_id_cur}.")
				continue

		new_pdf_name = str(idx).zfill(5) + '_' + url_web.split('/')[-1]+'.pdf'
		new_pdf_name = os.path.join(dest_folder, new_pdf_name)
		with open(new_pdf_name, 'wb') as f:
			f.write(r.content)

def valid_doi_retrieval(file_name_list, pre_cached_fail_list):
	'''to retrieve valid doi from each json file.'''
	missing_count = 0
	paper_id_list = []
	# specific sign for DOI in biorxiv
	biorxiv_paper_sign = '10.1101/'			
	# specific sign with no paper #
	biorxiv_paper_failed_sign = 'https://doi.org/10.1101/2020.02.'	
	fail_index_list = []
	fail_title_list = []

	def update_missing_information(file_idx, pdf_dict):
		'''if current file does not have DOI/hint to the article url.'''
		cur_split = "dummy"
		fail_index_list.append(file_idx)
		fail_title_list.append(pdf_dict['metadata']['title'])
		return cur_split

	for file_idx, file_name_cur in enumerate(file_name_list):
		exist_flag = False
		with open(file_name_cur) as json_file:
			pdf_dict = json.load(json_file)
			pdf_str = str(pdf_dict)
			pdf_str_split = pdf_str.split()

			if file_idx in pre_cached_fail_list:
				cur_split = update_missing_information(file_idx, pdf_dict)
				missing_count += 1
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
					cur_split = update_missing_information(file_idx, pdf_dict)
					missing_count += 1
			paper_id_list.append(cur_split)
	print(f"{missing_count} files are missing out of {file_idx+1} files in total.")
	return paper_id_list, fail_index_list, fail_title_list

def main(json_file_dir, debug_mode, pre_cached_fail_list, dest_folder):

	## step 1: retrieve the json files.
	file_name_list = []
	for file in glob.glob(os.path.join(json_file_dir,"*.json")):
		file_name_list.append(file)
	paper_id_list, fail_index_list, fail_title_list = valid_doi_retrieval(file_name_list, pre_cached_fail_list)

	## step 2: record missing files in csv.
	csv_gen(fail_index_list, fail_title_list)

	## step3: to retrieve the pdf.
	pdf_retrival(dest_folder, paper_id_list, fail_index_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Downloading pdfs from three datasets in COVID_19.')
    parser.add_argument('-d', '--json_file_dir', default='/nas/vista-ssd02/users/xiaoguo/COVID_research/Dataset_COVID19/biorxiv_medrxiv',
    					help="the directory that contains json files.")
    parser.add_argument('-de', '--debug_mode', action='store_true', help="debug mode.")
    parser.set_defaults(debug_mode=True)
    parser.add_argument('--pre_cached_fail_list', default=
    					[176, 244, 267, 303, 319, 345, 446, 461, 464, 513, 514, 537, 611, 620, 624, 688, 767])
    parser.add_argument('--dest_folder', default="./biorxiv", help="folder to store pdfs.")
    args = parser.parse_args()
    main(json_file_dir=args.json_file_dir, debug_mode=args.debug_mode, pre_cached_fail_list=args.pre_cached_fail_list,
    	dest_folder=args.dest_folder)
