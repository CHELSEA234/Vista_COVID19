source ~/.bashrc

conda activate COVID_torch

python tokenizer_offline_h5.py -file train_COVID.txt -t ../bio_medical_data_h5
python tokenizer_offline_h5.py -file val_COVID.txt -t ../bio_medical_data_h5
