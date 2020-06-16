source ~/.bashrc

conda activate COVID_torch

python ../pdf_sampling_balance.py -data biorxiv_medrxiv -r consecutive
