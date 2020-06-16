source ~/.bashrc

conda activate COVID_torch

python ../pdf_sampling_balance.py -data biorxiv_medrxiv -r diff_document
python ../pdf_sampling_balance.py -data comm_use_subset -r diff_document
python ../pdf_sampling_balance.py -data noncomm_use_subset -r diff_document
python ../pdf_sampling_balance.py -data custom_license -r diff_document
