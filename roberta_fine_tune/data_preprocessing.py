#!/usr/bin/env python
# coding: utf-8
# Created by Xiao Guo
'''
The data preprocessing stream based on https://github.com/pytorch/fairseq/tree/master/examples/roberta
The scripts first retrieve text from jsons, and convert them BPE.
'''
import os
import re
import sys
import json
import glob
import argparse
import numpy as np
import csv

from pathlib import Path
from collections import Counter
from fairseq.data.encoders.gpt2_bpe import get_encoder

def json_file_retrieval(json_folder_dir, dataset_name_list):
    '''
    return list of json files.
    '''
    json_name_list = []     # json name
    print("loading json file from: ")
    for dataset_name_cur in dataset_name_list:
        json_file_dir = os.path.join(json_folder_dir, dataset_name_cur)
        print(f"  {json_file_dir}.")
        for idx, json_file in enumerate(glob.glob(os.path.join(json_file_dir,"*.json"))):
            json_name_list.append(json_file)
    print(f"the total json file number is {len(json_name_list)}.")
    return json_name_list

def json_text(json_folder_dir, dataset_name_list, percentage, raw_txt_dir):
    '''
    dump text out of json files.
    '''
    if os.path.isdir(raw_txt_dir):
        print(f"the {raw_txt_dir} already exists.")
        return

    text_num = 0
    json_name_list = json_file_retrieval(json_folder_dir, dataset_name_list)
    exist_flag = os.makedirs(raw_txt_dir, exist_ok=True)
    output_file_train = open(os.path.join(raw_txt_dir, "train.txt"), "w")
    output_file_val = open(os.path.join(raw_txt_dir, "val.txt"), "w")
    text_str_list = []

    for file_idx, file_name_cur in enumerate(json_name_list):
        with open(file_name_cur) as json_file:
            pdf_dict = json.load(json_file)
            title_abstract_list = pdf_dict['abstract']
            for title_abstract_dict in title_abstract_list:
                text_str = title_abstract_dict['text']
                text_str_list.append(text_str + '\n')
                text_num += 1

            title_body_txt_list = pdf_dict['body_text']
            for title_body_txt_dict in title_body_txt_list: # some information are duplicate, like copyright things.
                text_str = title_body_txt_dict['text']
                text_str_list.append(text_str + '\n')
                text_num += 1

    print(f"starting dumping text stream, {percentage} of dataset will be used for training.")
    text_num_partition = int(len(text_str_list) * float(percentage))
    # print("text_num_partition is: ", text_num_partition)
    for text_idx, text_str in enumerate(text_str_list):
        if text_idx <= text_num_partition:
            output_file_train.write(text_str)
        else:
            output_file_val.write(text_str)

    output_file_train.close()
    output_file_val.close()

def main(json_folder_dir, debug_mode, dataset_name, percentage, encoder_json_file_path, vocab_bpe_path):

    if dataset_name == "All":
        dataset_name_list = ['biorxiv_medrxiv', 'comm_use_subset', 'noncomm_use_subset', 'pmc_custom_license']
    else:
        dataset_name_list = [dataset_name]

    ## take text out of json
    txt_dir = "raw_txt_data"
    json_text(json_folder_dir, dataset_name_list, percentage, txt_dir)

    ## BPE on text
    ## To encoder all input arguments here.
    # encoder_json_file_path = 'gpt2_bpe/encoder.json'
    # vocab_bpe_path = 'gpt2_bpe/vocab.bpe'
    bpe = get_encoder(encoder_json_file_path, vocab_bpe_path)


    def encode(line):
        # global bpe
        ids = bpe.encode(line)
        return list(map(str, ids))

    def decode(tokens):
        # global bpe
        return bpe.decode(tokens)


    bpe_dir = 'bpe_data'
    os.makedirs(bpe_dir, exist_ok=True)
    input_file_train = open(os.path.join(txt_dir, "train.txt"), "r", encoding="utf-8")   
    input_file_val = open(os.path.join(txt_dir, "val.txt"), "r", encoding="utf-8")

    output_file_train = open(os.path.join(bpe_dir, "train.bpe"), "w", encoding="utf-8")
    output_file_val = open(os.path.join(bpe_dir, "val.bpe"), "w", encoding="utf-8")

    input_file_list =  [input_file_train, input_file_val]
    output_file_list = [output_file_train, output_file_val]

    for handler_idx, output_handler in enumerate(output_file_list):
        output_text_files = [output_handler]
        enc_lines = []
        for idx,line in enumerate(input_file_list[handler_idx]):
            line = line.strip()
            tokens = encode(line)
            if len(tokens) == 0: continue
            enc_lines.append(" ".join(tokens))
            encoded_lines = [["PASS", enc_lines]]
            for i, (filt, enc_lines) in enumerate(encoded_lines, start=1):
                if filt == "PASS":
                    for enc_line, output_h in zip(enc_lines, output_text_files):
                        print(enc_line, file=output_h)
    print("BPE has been generated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Downloading pdfs from three datasets in COVID_19.')
    parser.add_argument('-d', '--json_folder_dir', default='/nas/vista-ssd02/users/xiaoguo/COVID_research/PDFs/',
    					help="the directory that contains json files.")
    parser.add_argument('-de', '--debug_mode', action='store_true', help="debug mode.")
    parser.set_defaults(debug_mode=False)
    parser.add_argument('-data', '--dataset_name', default="All", 
    					choices=['All', 'biorxiv_medrxiv', "comm_use_subset", "noncomm_use_subset", "pmc_custom_license"],
    					help="to choose dataset from choices.")
    parser.add_argument('-p', '--percentage', default=0.85, help="training data precentage.")
    parser.add_argument('--encoder_json_file_path', default='gpt2_bpe/encoder.json')
    parser.add_argument('--vocab_bpe_path', default='gpt2_bpe/vocab.bpe')
    args = parser.parse_args()
    main(json_folder_dir=args.json_folder_dir, debug_mode=args.debug_mode, dataset_name=args.dataset_name, percentage=args.percentage,
        encoder_json_file_path=args.encoder_json_file_path, vocab_bpe_path=args.vocab_bpe_path)
