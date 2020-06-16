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
np.random.seed(0)

from pathlib import Path
from tqdm import tqdm
from csv_loader import *

preprocessing_options = {'spell': False, 'remove_sequences': False, 'punctuations': []}
clean_sentences = SentClean(prep=preprocessing_options).clean_sentences
csv_set = set() # no duplicate sentence pair.
sen_set = set() # for the # of unique sentences.

# use for loading data.
loading_sample, loading_attempt = 0, 0
complete_flag = False # reach the total_number, or exceed the max attempt.

def json_file_retrieval(args, dataset_name_list):
    '''
    return list of json files.
    '''
    json_name_list = []     # json name
    print("loading json file from: ")
    for dataset_name_cur in dataset_name_list:
        json_file_dir = os.path.join(args.json_folder_dir, dataset_name_cur, 'pdf_json')
        print(f"  {json_file_dir}.")
        for idx, json_file in enumerate(glob.glob(os.path.join(json_file_dir,"*.json"))):
            json_name_list.append(json_file)
    print(f"the total json file number is {len(json_name_list)}.")
    return json_name_list

def sentence_sampling(text_str, requirement, sentence_pair_num=2):
    '''
    sampling the sentence pair based on different requirements.
    '''
    EMPTY_SENTENCE_PAIR = [[],[]]
    LOOP_COUNT = 2
    if requirement == "diff_document":
        loop_count = 0
        while True and loop_count < LOOP_COUNT:
            first_idx = np.random.randint(len(text_str))
            second_idx = np.random.randint(len(text_str))
            loop_count += 1
            if first_idx != second_idx:
                break
            if loop_count == LOOP_COUNT:
                return EMPTY_SENTENCE_PAIR
        idx = np.random.randint(sentence_pair_num)
        sentence_pair = [sentence_sampling(text_str[first_idx], requirement='document')[idx],
                        sentence_sampling(text_str[second_idx], requirement='document')[idx]]
        return sentence_pair
    else:
        if requirement == "document":
            text_str = text_str[0] 
        text_str_list = tokenizer.tokenize(text_str)    
        # text_str_list is the entire document in the document requirement.
        # text_str_list is the entire paragraph in the paragraph requirement.
        sentence_pair = []

        if requirement == "consecutive":
            if len(text_str_list) == 1:
                return EMPTY_SENTENCE_PAIR
            premise_idx = np.random.randint(len(text_str_list)-1)   # different to paragraph sampling.
            hypothesis_idx = premise_idx + 1
        elif requirement in ["paragraph", "document"]:
            loop_count = 0
            while True and loop_count < LOOP_COUNT:
                # if two indices are [premise, hypothesis] one time, then [hypothesis, premise] the other.
                # duplicate will be taken later in the csv_omit.
                premise_idx = np.random.randint(len(text_str_list))
                hypothesis_idx = np.random.randint(len(text_str_list))

                # why is here taking much longer time...
                loop_count += 1
                if premise_idx != hypothesis_idx:
                    break
                if loop_count == LOOP_COUNT:
                    return EMPTY_SENTENCE_PAIR
        else:
            assert False, "Please offer a right requirement."

        sentence_pair.append(text_str_list[premise_idx])
        sentence_pair.append(text_str_list[hypothesis_idx])

        return sentence_pair

def csv_omit(input_str, requirement, csv_writer, csv_file_handler, min_len_sent=10, max_num_punctuations=10):
    '''output string into the csv file.'''
    write_list = []
    sentence_pair = sentence_sampling(input_str, requirement)

    sentence_pair.reverse() # the original code reverses the sentence order.
    new_sent_list = clean_sentences(sentence_pair, min_len_sent=min_len_sent, max_num_punctuations=max_num_punctuations)
    
    if len(new_sent_list) != 2:
        return False
    # write_list.append(requirement)
    # print(new_sent_list)
    # print()
    write_list.extend(new_sent_list)

    # check the duplicate sentence pair.
    if str(write_list) in csv_set:
        return False
    elif new_sent_list[0] in sen_set or new_sent_list[1] in sen_set:
        return False
    else:
        csv_set.add(str(write_list))
        sen_set.update(new_sent_list)
        csv_writer.writerow(write_list)
        csv_file_handler.flush()
        return True

def str_generation(target_list, document_str, document_holder, target_str='text'):
    '''
    target_list: list in the raw json file.
    '''
    assert document_str != None, print("Please provide the valid document_str!")
    assert document_holder != None, print("Please provide the valid document_holder!")
    for target_dict in target_list:
        text_str = target_dict[target_str]
        document_holder.append(text_str)
        document_str.append([text_str])
    return document_str, document_holder

def document_generation(json_name_list, args, dataset_name_list, FILE_IDX_BEGUB):
    '''
    iterate json_name_list for gathering dataset.
    '''
    different_document_list = []
    dataset_holder = [] # in which document_holder only contains a piece of text.

    for file_idx, file_name_cur in enumerate(json_name_list):
        if args.debug_mode and file_idx > FILE_IDX_BEGUB:
            print(f"only check first {FILE_IDX_BEGUB} documents in the debug mode.")
            break
        document_str = []
        document_holder = []    # contain one piece of text.
        with open(file_name_cur) as json_file:
            pdf_dict = json.load(json_file)
            document_str, document_holder = str_generation(pdf_dict['abstract'], document_str, document_holder)
            document_str, document_holder = str_generation(pdf_dict['body_text'], document_str, document_holder)

        dataset_holder.append(document_holder)
        different_document_list.append(document_str)

    print(f"====================================================================")
    print(f"loading the information is over.")
    print(f"{dataset_name_list[0]} has {len(different_document_list)} documents.")
    print()
    return different_document_list, dataset_holder

def iteration_check(succeed_flag, args):
    '''simply update the while loop for the sampling.'''
    break_flag = False
    if succeed_flag:
        global loading_sample
        loading_sample += 1
        if loading_sample % args.log_step == 0:
            print(f"finish loading {loading_sample}th samples in requirement {args.requirement}.", flush=True)
        if loading_sample == args.SENTENCE_PAIR_NUM:
            break_flag = True
            args.complete_flag = True
    global loading_attempt 
    loading_attempt += 1
    if loading_attempt == args.MAX_ATTEMPT:
        print(f"exceeding the max loading attempt {args.MAX_ATTEMPT}.")
        break_flag = True
        args.complete_flag = True
    return break_flag

def json_text(args, dataset_name_list, csv_file_handler):
    '''
    dump text out of json files.
    '''
    FILE_IDX_BEGUB = 100
    DOC_SAMPLING_NUM = 20 if not args.debug_mode else 1
    csv_writer = csv.writer(csv_file_handler, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(["text1", "text2"])

    # collect documents
    json_name_list = json_file_retrieval(args, dataset_name_list)
    different_document_list, dataset_holder = document_generation(json_name_list, args, dataset_name_list, FILE_IDX_BEGUB)

    # doing sampling in the one document here.
    # SENTENCE_PAIR_NUM = 10000   # 10,000
    args.SENTENCE_PAIR_NUM = args.total if not args.debug_mode else 10
    args.MAX_ATTEMPT = args.SENTENCE_PAIR_NUM * 20
    args.complete_flag = False

    if args.requirement in ["consecutive", "paragraph"]:
        def sentence_choose():
            document_rand_idx = np.random.randint(len(different_document_list))
            document_rand_chosen = different_document_list[document_rand_idx]
            sentence_rand_idx = np.random.randint(len(document_rand_chosen))
            sentence_rand_chosen = document_rand_chosen[sentence_rand_idx][0]
            return sentence_rand_chosen

        while True:
            sentence_rand_chosen = sentence_choose()
            succeed_flag = csv_omit(sentence_rand_chosen, args.requirement, csv_writer, csv_file_handler)
            if iteration_check(succeed_flag, args):
                break

    elif args.requirement == "document":
        while True and not args.complete_flag:
            document_rand_idx = np.random.randint(len(dataset_holder))
            document_rand_chosen = dataset_holder[document_rand_idx]
            for _ in range(DOC_SAMPLING_NUM):   # sampling each document for 20 times
                succeed_flag = csv_omit(document_rand_chosen, args.requirement, csv_writer, csv_file_handler)
                if iteration_check(succeed_flag, args):
                    break

    elif args.requirement == "diff_document":
        for _ in range(args.MAX_ATTEMPT):
            succeed_flag = csv_omit(dataset_holder, args.requirement, csv_writer, csv_file_handler)
            if iteration_check(succeed_flag, args):
                break
                
    else:
        assert False, print("please offer the valid requirement here.")

def main(args):

    dataset_name_list = [args.dataset_name] if args.dataset_name != "All" else \
                        ['biorxiv_medrxiv', 'comm_use_subset', 'noncomm_use_subset', 'custom_license']
    csv_file_name = 'demo_' + args.requirement + '_' + args.dataset_name + '.csv' if \
                    args.debug_mode else args.requirement + '_' + args.dataset_name + '.csv'

    csv_file_handler = open(csv_file_name, mode='w')
    json_text(args, dataset_name_list, csv_file_handler)
    csv_file_handler.close()
    print(f"the dataset {args.dataset_name} contains {len(sen_set)} unique sentences.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Downloading pdfs from three datasets in COVID_19.')
    parser.add_argument('-d', '--json_folder_dir', default='/nas/medifor/hengameh/COVID19/DATA/Dataset_COVID19',
    					help="the directory that contains json files.")
    parser.add_argument('-debug', '--debug_mode', action='store_true', help="debug mode.")
    parser.add_argument('-data', '--dataset_name', default='All', 
    					choices=['All', 'biorxiv_medrxiv', "comm_use_subset", "noncomm_use_subset", "custom_license"],
    					help="to choose dataset from choices.")
    parser.add_argument('-r', '--requirement', default='consecutive', 
                        choices=['diff_document', 'document', "paragraph", "consecutive"],
                        help="to choose different requirement here.")
    parser.add_argument('--total', type=int, default=75000, help="samples from per dataset per strategy.")
    parser.add_argument('--log_step', type=int, default=1000, help="how many steps to log.")
    parser.add_argument('--txt_dir', default='./COVID_19_data')
    args = parser.parse_args()
    main(args)
