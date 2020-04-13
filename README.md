## SciBERT fine-tune script:

This script heavily relies but different to `run_language_modeling.py` provided by [hugging transformers](http://github.com).
1. New tensorboard
   1. the progress on `learning rate`
   1. `loss & perplexity` in training w.r.t epoch and iteration
   1. every a few epcoh, evaluation on `loss & perplexity`
1. Multiple Job Sumbission
   1. each experiment owns specific tensorboard and checkpoint folder, no long share the common one (the original script).
1. Data Preprocessing on the Custom text file (*In the Progress*)
1. Different Saving Mechanism
   1. only saves the **best** and the **current** checkpoint
   1. original files saves every a few epoches, runs out of the disk space
1. Logging File
