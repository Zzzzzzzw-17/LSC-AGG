# Aggregating fine-tuned models to capture lexical semantic change

This code accompanies the paper [The Finer they get: On aggregating fine-tuned models to capture lexical
semantic change](), which address LSC by leveraging fine-tuned language models on different tasks. 



##  STEP 1: Extracting token embeddings

- To extract token embeddings of bert-base model, run `python3 code/generate_embeddings_bert.py <PATH_TO_MODEL> <CORPUS> <TARGET_WORDS> <OUTFILE>`

- To extract token embeddings of adapter models (finetuned), run `python3 code/generate_embeddings_adapter.py <PATH_TO_ADAPTER> <PATH_TO_MODEL> <CORPUS> <TARGET_WORDS> <OUTFILE>`

- These scripts produce `npz` archives containing numpy arrays with token embeddings for each target word in a given corpus.
For this work, extracted npz files can be found here: https://1drv.ms/u/s!AhyFFULVgsQqjziU0FPKfWH_JKl1?e=9dTYnn and https://1drv.ms/u/s!AhyFFULVgsQqjzmRxSx6gHoCWpKI?e=Yewqpl

- code are adapted from https://github.com/akutuzov/semeval2020

##  STEP 2: Estimating semantic change
- PRT algorithm: `python3 code/generate_prt_scores.py  --input0=PATH_TO_INPUT0 --input1=PATH_TO_INPUT1 --target=PATH_TO_TARGET_WORDS --output=OUTPUT_PATH`

- APD algorithm: `python3 code/generate_apd_scores.py  PATH_TO_TARGET_WORDS  PATH_TO_INPUT1  PATH_TO_INPUT2  OUTPUT_PATH`

## STEP 3: Evaluating with AUC and correlation
- AUC `python3 code/eval_classification.py`

- Spearmann correlation `python3 code/eval_rank.py`

## Other details
- Please see the lsc.ipynb
- Results of all combination correlation scores can be found here: https://1drv.ms/u/s!AhyFFULVgsQqjzucNHb7AL3PSXDv?e=qXLKXT
- local finetuned models (sst2 and pos) can be found here (codes adapted from https://github.com/ucinlp/null-prompts): https://1drv.ms/u/s!AhyFFULVgsQqj0-btF_FYgdGqbD8?e=NUvdLB