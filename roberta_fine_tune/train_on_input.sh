TOTAL_UPDATES=20000     # Total number of training steps
WARMUP_UPDATES=1200     # Warmup the learning rate over this many updates
PEAK_LR=5e-5         	# Peak learning rate, adjust as needed
TOKENS_PER_SAMPLE=512   # Max sequence length
MAX_POSITIONS=512       # Num. positional embeddings (usually same as above)
MAX_SENTENCES=16        # Number of sequences per batch (batch size)
UPDATE_FREQ=16          # Increase the batch size 16x

DATA_DIR=data-bin/full_text
TENSORBOARD_DIR=data-bin/full_text/tensorboard_log_$PEAK_LR

fairseq-train --fp16 $DATA_DIR \
	--restore-file pretrained_model/roberta.base/model.pt \
    --task masked_lm --criterion masked_lm \
    --arch roberta_base --sample-break-mode complete --tokens-per-sample $TOKENS_PER_SAMPLE \
    --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_UPDATES \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
    --max-sentences $MAX_SENTENCES --update-freq $UPDATE_FREQ \
    --max-update $TOTAL_UPDATES --log-format simple --log-interval 1 \
    --tensorboard-logdir $TENSORBOARD_DIR
