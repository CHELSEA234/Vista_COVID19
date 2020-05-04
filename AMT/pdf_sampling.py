#!/usr/bin/env python
# coding: utf-8
# Created by Xiao Guo
import os
import sys
import json
import glob
import argparse
import numpy as np
import csv
import nltk.data
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

from pathlib import Path
from tqdm import tqdm

def json_file_retrieval(json_folder_dir, dataset_name_list):
    '''
    return list of json files.
    '''
    json_name_list = []     # json name
    print("loading json file from: ")
    for dataset_name_cur in dataset_name_list:
        json_file_dir = os.path.join(json_folder_dir, dataset_name_cur, 'pdf_json')
        print(f"  {json_file_dir}.")
        for idx, json_file in enumerate(glob.glob(os.path.join(json_file_dir,"*.json"))):
            json_name_list.append(json_file)
    print(f"the total json file number is {len(json_name_list)}.")
    return json_name_list

def sentence_sampling(text_str, requirement, sentence_pair_num=2):
    '''
    sampling the sentence pair based on different requirements.
    '''
    if requirement == "diff_document":
        loop_count = 0
        while True and loop_count < 5:
            first_idx = np.random.randint(len(text_str))
            second_idx = np.random.randint(len(text_str))
            loop_count += 1
            if first_idx != second_idx:
                break
        idx = np.random.randint(sentence_pair_num)
        sentence_pair = [sentence_sampling(text_str[first_idx], requirement='document')[idx],
                        sentence_sampling(text_str[second_idx], requirement='document')[idx]]
        return sentence_pair
    else:
        if requirement == "document":
            text_str = text_str[0]
        text_str_list = tokenizer.tokenize(text_str)
        sentence_pair = []

        if requirement == "consecutive":
            if len(text_str_list) == 1:
                return [[],[]]
            premise_idx = np.random.randint(len(text_str_list)-1)
            hypothesis_idx = premise_idx + 1
        elif requirement in ["paragraph", "document"]:
            loop_count = 0
            while True and loop_count < 5:
                premise_idx = np.random.randint(len(text_str_list))
                hypothesis_idx = np.random.randint(len(text_str_list))
                loop_count += 1
                if premise_idx != hypothesis_idx:
                    break
        else:
            assert False, "Please offer a right requirement."

        sentence_pair.append(text_str_list[premise_idx])
        sentence_pair.append(text_str_list[hypothesis_idx])

        return sentence_pair

def json_text(json_folder_dir, dataset_name_list, percentage, raw_txt_dir, debug_mode, csv_file_handler, requirement):
    '''
    dump text out of json files.
    '''
    csv_writer = csv.writer(csv_file_handler, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(["mode", "premise", "hypothesis"])

    text_num = 0
    json_name_list = json_file_retrieval(json_folder_dir, dataset_name_list)
    exist_flag = os.makedirs(raw_txt_dir, exist_ok=True)
    output_file_train = open(os.path.join(raw_txt_dir, "train.txt"), "w")
    output_file_val = open(os.path.join(raw_txt_dir, "val.txt"), "w")
    sentence_pair_list = []
    different_document_list = []
    for file_idx, file_name_cur in enumerate(json_name_list):
        if debug_mode and file_idx > 15:
            print(f"only check first {15} documents in the debug mode.")
            break

        document_str = []
        with open(file_name_cur) as json_file:
            pdf_dict = json.load(json_file)

            title_abstract_list = pdf_dict['abstract']
            for title_abstract_dict in title_abstract_list:
                text_str = title_abstract_dict['text']
                document_str.append(text_str)

            title_body_txt_list = pdf_dict['body_text']
            for title_body_txt_dict in title_body_txt_list: # some information are duplicate, like copyright things.
                text_str = title_body_txt_dict['text']
                document_str.append(text_str)

                if np.random.randint(3) == 1: # sampling some paragraphs for this.
                    if requirement in ["consecutive", "paragraph"]:
                        write_list = []
                        sentence_pair = sentence_sampling(text_str, requirement)
                        if len(sentence_pair[0]) == 0 or len(sentence_pair[1]) == 1:
                            continue
                        write_list.append(requirement)
                        write_list.extend(sentence_pair)
                        csv_writer.writerow(write_list) 
                        csv_file_handler.flush()  

            if requirement == "document":
                write_list = []
                sentence_pair = sentence_sampling(document_str, requirement)
                if len(sentence_pair[0]) == 0 or len(sentence_pair[1]) == 1:
                    continue
                write_list.append(requirement)
                write_list.extend(sentence_pair)
                csv_writer.writerow(write_list) 
                csv_file_handler.flush()     

            different_document_list.append(document_str)

            if requirement == "diff_document":
                for _ in range(15):
                    write_list = []
                    sentence_pair = sentence_sampling(different_document_list, requirement)
                    if len(sentence_pair[0]) == 0 or len(sentence_pair[1]) == 1:
                        continue
                    write_list.append(requirement)
                    write_list.extend(sentence_pair)
                    csv_writer.writerow(write_list) 
                    csv_file_handler.flush()    

def main(json_folder_dir, debug_mode, dataset_name, percentage, txt_dir):

    if dataset_name == "All":
        dataset_name_list = ['biorxiv_medrxiv', 'comm_use_subset', 'noncomm_use_subset', 'pmc_custom_license']
    else:
        dataset_name_list = [dataset_name]

    ## take text out of json
    requirement = "diff_document"
    # requirement = "document"
    # requirement = "paragraph"
    # requirement = "consecutive"
    csv_file_name = 'demo_' + requirement + '.csv'
    csv_file_handler = open(csv_file_name, mode='w')
    json_text(json_folder_dir, dataset_name_list, percentage, txt_dir, debug_mode, csv_file_handler, requirement=requirement)
    csv_file_handler.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Downloading pdfs from three datasets in COVID_19.')
    parser.add_argument('-d', '--json_folder_dir', default='/nas/medifor/hengameh/COVID19/DATA/Dataset_COVID19',
    					help="the directory that contains json files.")
    parser.add_argument('-de', '--debug_mode', action='store_true', help="debug mode.")
    parser.set_defaults(debug_mode=True)
    parser.add_argument('-data', '--dataset_name', default='biorxiv_medrxiv', 
    					choices=['All', 'biorxiv_medrxiv', "comm_use_subset", "noncomm_use_subset", "pmc_custom_license"],
    					help="to choose dataset from choices.")
    parser.add_argument('-p', '--percentage', default=0.85, help="training data precentage.")
    parser.add_argument('--txt_dir', default='./COVID_19_data')
    args = parser.parse_args()
    main(json_folder_dir=args.json_folder_dir, debug_mode=args.debug_mode, 
        dataset_name=args.dataset_name, percentage=args.percentage, txt_dir=args.txt_dir)
