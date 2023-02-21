# Aggregating fine-tuned models to capture lexical semantic change

This code accompanies the paper [The Finer they get: On aggregating fine-tuned models to capture lexical
semantic change](), which addresses lexical semantic change (LSC) by leveraging fine-tuned language models on different tasks. 

We follow [Kutuzov and Giulianellito](https://arxiv.org/abs/2005.00050) to use contextualized embeddings in a pipeline manner. More specifically, following steps are carried out: 

##  STEP 0: Setting up repo
- Clone the project by running `git clone https://github.com/Zzzzzzzw-17/LSC-AGG.git`
- download required library by running `pip install -r requirements.txt`

##  STEP 1: Extracting token embeddings


- To extract token embeddings of bert-base model, run `python3 code/generate_embeddings_bert.py <PATH_TO_MODEL_CONFIG> <CORPUS> <TARGET_WORDS> <OUTFILE>`

- To extract token embeddings of adapter models (finetuned), run `python3 code/generate_embeddings_adapter.py <PATH_TO_ADAPTER_CONFIG> <PATH_TO_MODEL_CONFIG> <CORPUS> <TARGET_WORDS> <OUTFILE>`

- These scripts produce `npz` archives containing numpy arrays with token embeddings for each target word in a given corpus. You can also download them from here: [corpus1 embeddings](https://1drv.ms/u/s!AhyFFULVgsQqjziU0FPKfWH_JKl1?e=9dTYnn) and [corpus2 embeddings](https://1drv.ms/u/s!AhyFFULVgsQqjzmRxSx6gHoCWpKI?e=Yewqpl).

##  STEP 2: Estimating semantic change

To calculate lexical semantic change of each target word, we use PRT and APD algorithm from [Kutuzov and Giulianellito](https://arxiv.org/abs/2005.00050). The results of each fine-tuned models can be found in the` results` folder. To generate them, please run the following command: 
- PRT algorithm: `python3 code/generate_prt_scores.py  --input0=PATH_TO_INPUT0 --input1=PATH_TO_INPUT1 --target=PATH_TO_TARGET_WORDS --output=OUTPUT_PATH`

- APD algorithm: `python3 code/generate_apd_scores.py  <PATH_TO_TARGET_WORDS>  <PATH_TO_INPUT1>  <PATH_TO_INPUT2>  <OUTPUT_PATH>`

## STEP 3: Evaluating with AUC and correlation
To calculate AUC/correlation scores and evaluate them against gold standard, please run the following code in the command line 
- AUC `python3 code/eval_classification.py <ModelAnsPath> <TrueAnsPath>`

- Spearmann correlation `python3 code/eval_rank.py <ModelAnsPath> <TrueAnsPath>`

## Other details
- The dataset can be downloaded here: https://www.ims.uni-stuttgart.de/en/research/resources/corpora/sem-eval-ulscd/
- The lsc.ipynb contains plots for the paper, p-value generation and averaging best combinations.   
- The above code are are adapted from https://github.com/akutuzov/semeval2020
- The correlation scores of all possible combinations of models can be found here: https://1drv.ms/u/s!AhyFFULVgsQqjzucNHb7AL3PSXDv?e=qXLKXT
- local finetuned models (sst2 and pos) can be found here (codes adapted from https://github.com/ucinlp/null-prompts): https://1drv.ms/u/s!AhyFFULVgsQqj0-btF_FYgdGqbD8?e=NUvdLB

## References
[1] Kutuzov, Andrey and Mario Giulianelli. “UiO-UvA at SemEval-2020 Task 1: Contextualised Embeddings for Lexical Semantic Change Detection.” International Workshop on Semantic Evaluation (2020).
[2] Logan IV, Robert L et al. “Cutting Down on Prompts and Parameters: Simple Few-Shot Learning with Language Models.” Findings (2021).


