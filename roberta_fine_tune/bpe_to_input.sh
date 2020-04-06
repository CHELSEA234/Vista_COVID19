fairseq-preprocess \
    --only-source \
    --srcdict gpt2_bpe/dict.txt \
    --trainpref bpe_data/abstract_train.bpe \
    --validpref bpe_data/abstract_val.bpe \
    --destdir data-bin/abstract_biorxiv \
    --workers 60
