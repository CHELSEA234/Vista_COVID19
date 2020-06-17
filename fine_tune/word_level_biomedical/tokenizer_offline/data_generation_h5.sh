source ~/.bashrc

conda activate COVID_torch

python tokenizer_offline_h5.py -file val.txt
