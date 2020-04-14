'''
script for submitting jobs to runjob on fine-tune scibert.
'''
from __future__ import print_function
from itertools import product

import os
import argparse

def main(model_name_or_path, train_file_path, eval_file_path, debug):
    ## hardcord python file name and script folder name.
    COMMAND_TEMPLATE = 'python fine_tune_mlm_scibert.py ' if not debug else 'python fine_tune_mlm_scibert.py --debug '
    SCRIPT_FILE = 'scripts'
    file_dir = os.path.dirname(os.path.realpath(__file__))

    ## hyperparameters.
    model_name_or_path = [model_name_or_path]   
    lr_list = [1e-5, 5e-6, 1e-7]
    epoch_list = [2, 3]
    batch_size_list = [4, 8]
    # you can add warmup mechanism here later...

    cnt = 0
    combinations = list(product(*[model_name_or_path, lr_list, epoch_list, batch_size_list]))
    for comb in combinations:
        comb = list(comb)
        output_dir = f'model-SciBERT_lr-{comb[1]}_maxepoch-{comb[2]}_bs-{comb[3]}'
        command = COMMAND_TEMPLATE + (f'--model_name_or_path={comb[0]} '+
                                    f'--learning_rate={comb[1]} '+
                                    f'--num_train_epochs={comb[2]} '+
                                    f'--per_gpu_train_batch_size={comb[3]} '+
                                    f'--output_dir={output_dir} ' +
                                    f'--train_data_file={train_file_path} ' +
                                    f'--eval_data_file={eval_file_path}')

        # print(command)
        os.makedirs(SCRIPT_FILE, exist_ok=True)
        bash_file = os.path.join(file_dir, SCRIPT_FILE, f'{output_dir}.sh')
        dest_file = os.path.join(file_dir, SCRIPT_FILE, f'{output_dir}.out')
        with open( bash_file, 'w' ) as OUT:
            OUT.write('source ~/.bashrc\n')
            OUT.write('conda activate COVID_torch\n')
            OUT.write(f'cd {file_dir}\n')
            OUT.write(command)
        print(command)
        qsub_command = f'qsub -P medifor -q all.q -j y -o {dest_file} -l h_rt=24:00:00,h_vmem=20g,gpu=1 {bash_file}'
        print(qsub_command)
        os.system( qsub_command )
        cnt += 1
        print( 'Submitted #{}'.format(cnt))
        if debug:
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='fine-tune scibert on COVID_19.')
    parser.add_argument('--model_name_or_path', default="../scibert_pertrained/scibert_scivocab_uncased_pytorch", 
                        help="training data precentage.")
    parser.add_argument('--train_file_path', default="../COVID_19_data/abstract_train.txt")
    parser.add_argument('--eval_file_path', default="../COVID_19_data/abstract_val.txt")
    parser.add_argument('--debug', action="store_true", help="debug mode or not.")
    # parser.set_defaults(debug=True)
    args = parser.parse_args()
    main(args.model_name_or_path, args.train_file_path, args.eval_file_path, args.debug)
