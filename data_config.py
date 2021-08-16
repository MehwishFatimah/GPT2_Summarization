#import os

#input setup
data_dir = "/hits/basement/nlp/fatimamh/inputs"
file_name = "wiki_mono.csv"
#model + tokenizer
model_dir = "/hits/basement/nlp/fatimamh/inputs/pretrained_models/gpt2"

#output dir
out_dir = "/hits/basement/nlp/fatimamh/outputs/gpt2_en"
processed_set= "dataset" #split and tokenized
final_model = "fine_tuned"
training_models = "epochs_log"
results = "summaries"

topk_file = "topk_summaries.csv"
beam_file = "beam_summaries.csv"
topk_rouge_file = "topk_rouge_scores.csv"
topk_bleu_file = "topk_bleu_scores.csv"
topk_bert_file = "topk_bert_scores.csv"
topk_bleurt_file = "topk_bleurt_scores.csv"

# text and sum len
max_seq_len = 768 #1024 to do
max_text	= 600 
max_sum	  	= 150
min_sum     = 100 

# split ratio
sr = 0.2

batch_size = 1 # fix it
test_batch = 1 

#training param
epochs = 50
learning_rate = 5e-4
warmup_steps = 1e2
epsilon = 1e-8
gradient_accumulation_steps = 32
max_grad_norm = 1

#generation param
length_penalty=2.0
early_stopping=True
temperature=1 
top_k=10 
top_p=0.5
num_beams=4
gen_type = 0 #0= top_k, 1= beam_search, 2= both



