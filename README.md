# Aggregating fine-tuned models to capture lexical semantic change

This code accompanies the paper [The Finer they get: On aggregating fine-tuned models to capture lexical
semantic change](), which addresses lexical semantic change (LSC) by leveraging fine-tuned language models on different tasks. 

We follow [Kutuzov and Giulianellito](https://arxiv.org/abs/2005.00050) to use contextualized embeddings in a pipeline manner. More specifically, following steps are carried out: 

##  STEP 0: Setting up repo
- Clone the project by running `git clone https://github.com/Zzzzzzzw-17/LSC-AGG.git`
- download required library by running `pip install -r requirements.txt`

##  STEP 1: Extracting token embeddings

- You can use the embeddings extracted already [here](https://1drv.ms/u/s!AhyFFULVgsQqj3hCZXco6rGCDiRl?e=pdJbg7).
- If you want to extract embeddings yourself, you can run the following commands: 
    - To extract token embeddings of bert-base model or any local fine-tuned models, run `python3 code/generate_embeddings_bert.py <PATH_TO_MODEL_CONFIG> <CORPUS> <TARGET_WORDS> <OUTFILE>` 

         -  `<PATH_TO_MODEL_CONFIG>` is the path to the model config. The config specifies model name or (finetuned) model path, embedding size and desired last n layer(s) for extraction. In this paper, we simply extracted the embeddings from the top layer (n=1).
         -  `<CORPUS>` is the directory of the corpus. We use the English dataset from [SemEval-2020 Task 1: Unsupervised Lexical Semantic Change Detection](https://competitions.codalab.org/competitions/20948). It can be downloaded from [here]( https://www.ims.uni-stuttgart.de/en/research/resources/corpora/sem-eval-ulscd/).
         - ` <TARGET_WORDS>` is the path of desired words for detection. We use the default 37 words in SemEval-2020 Task 1. It can be found in the `data/target_nopos.txt` in this repo.
         -  `<OUTFILE>` is the path to store your embeddings files. 
         - An example of usage can be: `python3 code/generate_embeddings_bert.py code/model_config  data/corpus1/token data/target_nopos.txt  embeddings/embeddings.npz` (You need to download the corpus file and put them into the data file first.)

    - To extract token embeddings of adapter models, run `python3 code/generate_embeddings_adapter.py <PATH_TO_ADAPTER_CONFIG> <PATH_TO_MODEL_CONFIG> <CORPUS> <TARGET_WORDS> <OUTFILE>` 
        - `<PATH_TO_ADAPTER_CONFIG>` is the path to the adapter config. The file specifies the adapter name and source. e.g. `AdapterHub/bert-base-uncased-pf-cola hf` (Please use the hf version of adapters in order to avoid loading errors).
        - ` <PATH_TO_MODEL_CONFIG>` is the same as describe above. You can also change the model to your own fine-tuned one by specifying the model path.  
        - e.g. `python3 code/generate_embeddings_adapter.py code/adapter_config  code/model_config  data/corpus1/token data/target_nopos.txt  embeddings/embeddings.npz`

    - These scripts produce `npz` archives containing numpy arrays with token embeddings for each target word in a given corpus.

##  STEP 2: Estimating semantic change

To calculate lexical semantic change of each target word, we use PRT and APD algorithm from [Kutuzov and Giulianellito](https://arxiv.org/abs/2005.00050). The results of each fine-tuned models can be found in the` results` folder. To generate them, please run the following command: 
- PRT algorithm: `python3 code/generate_prt_scores.py  <PATH_TO_TARGET_WORDS>  <PATH_TO_INPUT1>  <PATH_TO_INPUT2>  <OUTPUT_PATH>` e.g. `python3 code/generate_prt_scores.py  data/target_nopos.txt embeddings/output1.npz embeddings/output2.npz results` 

- APD algorithm: `python3 code/generate_apd_scores.py  <PATH_TO_TARGET_WORDS>  <PATH_TO_INPUT1>  <PATH_TO_INPUT2>  <OUTPUT_PATH>`  e.g. ` python3 code/generate_apd_scores.py data/target_nopos.txt embeddings/output1.npz embeddings/output2.npz results` 

## STEP 3: Evaluating with AUC and correlation
To calculate AUC/correlation scores and evaluate them against gold standard, please run the following code in the command line 
- AUC `python3 code/eval_classification.py <ModelAnsPath> <TrueAnsPath>` e.g. `code/eval_classification.py results/PRT/bert_base   test_data_truth/task1/english.txt`

- Spearmann correlation `python3 code/eval_rank.py <ModelAnsPath> <TrueAnsPath>` e.g. ` python3 code/eval_ranking.py results/PRT/bert_base   test_data_truth/task2/english.txt`

## Other details
- The dataset can be downloaded here: https://www.ims.uni-stuttgart.de/en/research/resources/corpora/sem-eval-ulscd/
- The lsc.ipynb contains plots for the paper, p-value generation and averaging best combinations.   
- The above code are are adapted from https://github.com/akutuzov/semeval2020
- The correlation scores of all possible combinations of models can be found here: https://1drv.ms/u/s!AhyFFULVgsQqjzucNHb7AL3PSXDv?e=qXLKXT
- local finetuned models (sst2 and pos) can be found here (codes adapted from https://github.com/ucinlp/null-prompts): https://1drv.ms/u/s!AhyFFULVgsQqj0-btF_FYgdGqbD8?e=NUvdLB

## References
[1] Kutuzov, Andrey and Mario Giulianelli. “UiO-UvA at SemEval-2020 Task 1: Contextualised Embeddings for Lexical Semantic Change Detection.” International Workshop on Semantic Evaluation (2020).
[2] Logan IV, Robert L et al. “Cutting Down on Prompts and Parameters: Simple Few-Shot Learning with Language Models.” Findings (2021).

