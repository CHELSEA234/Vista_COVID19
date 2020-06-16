source ~/.bashrc

conda activate COVID_torch

python ../pdf_sampling_balance.py -data noncomm_use_subset -r paragraph
