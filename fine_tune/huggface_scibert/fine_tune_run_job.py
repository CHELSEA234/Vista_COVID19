'''
script for submitting jobs to runjob on fine-tune scibert.
'''
from __future__ import print_function
import os
from itertools import product

def main(model_name_or_path):
    ## hardcord python file name and script folder name.
    COMMAND_TEMPLATE = 'python fine_tune_mlm_scibert.py '
    SCRIPT_FILE = 'scripts'

    ## hardcord data_path.
    ## TODO: change later.
    train_file_path = '/nas/vista-ssd02/users/xiaoguo/COVID_research/RoBERTa_learning/fine_tune_roberta_Apr_1st_body_txt/raw_txt_data/abstract_train.txt'
    eval_file_path = '/nas/vista-ssd02/users/xiaoguo/COVID_research/RoBERTa_learning/fine_tune_roberta_Apr_1st_body_txt/raw_txt_data/abstract_val.txt'

    ## TODO: later add different models
    model_name_or_path = [model_name_or_path]   
    lr_list = [5e-5, 1e-5, 5e-6]
    epoch_list = [2, 3]
    batch_size_list = [4, 8, 16]

    cnt = 0
    combinations = list(product(*[model_name_or_path, lr_list, epoch_list, batch_size_list]))

    for comb in combinations:
        comb = list(comb)
        output_dir = f'model-SciBERT_lr-{comb[1]}_maxepoch-{comb[2]}_bs-{comb[3]}'
        command = COMMAND_TEMPLATE + f'--model_name_or_path={comb[0]} '+
                                    f'--learning_rate={comb[1]} '+
                                    f'--num_train_epochs={comb[2]} '+
                                    f'--per_gpu_train_batch_size={comb[3]} '+
                                    f'--output_dir={output_dir}' +
                                    f'--train_data_file={train_file_path}' +
                                    f'--eval_data_file={eval_file_path}'

        print(command)
        # os.makedirs(SCRIPT_FILE, exist_ok=True)
        # bash_file = 'scripts/{}.sh'.format(output_dir)
        # with open( bash_file, 'w' ) as OUT:
        #     OUT.write('source ~/.bashrc\n')
        #     OUT.write('conda activate COVID_torch\n')
        #     OUT.write(command)
        # qsub_command = 'qsub -P medifor -q all.q -j y -o {}.out -l h_rt=24:00:00,m_mem_free=20G,gpu=1,h=\!vista13 {}'.format(bash_file, bash_file)
        # os.system( qsub_command )
        # cnt += 1
        # print( 'Submitted #{}'.format(cnt))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='fine-tune scibert on COVID_19.')
    parser.add_argument('--model_name_or_path', default="../scibert_pertrained/scibert_scivocab_uncased_pytorch", 
                        help="training data precentage.")
    args = parser.parse_args()
    main(args.model_name_or_path)
