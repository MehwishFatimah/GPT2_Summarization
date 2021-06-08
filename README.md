# Environment
- Use enivornment.yml or requirements.txt.
# Description

### Step 1:- Get pretrained model (if want to skip Environment Cache)
- Download pretrained model and tokenizer (GPT-2) in a local folder. Store it in a folder and load it from this location.

### Step 2:- Data processing
- Split data into train/val/test.
- Input to model: "<bos> + text + <sep> + summary + <eos> ". Truncate lengths of text and summary to fit in the design. Total sequence length can be 768 or 1024. 
- Create Datalaoders of train and val.

### Step 3:- GPT2 Tokenizer and Model
- Add special tokens to GPT-2 tokenizer.
- Resize model embeddings for new tokenizer length.
- Fine-tuning model by passing train data and evaluating it on val data during training.
- Store the tokenizer and fine-tuned model.

### Step 4:- Generation/Inference
-  Generate summaries for test set which is not used during fine tune.
-  Simple topk and beam search are used for the generation.

### Step 5:- Evaluation with Rouge
- Compute Rouge scores for test outputs and store it.

### TODO:
- Add argparser (currently all hyperparams are stored in config.py)
- Batch processing (currently working on batch_size = 1)
- Store scores

