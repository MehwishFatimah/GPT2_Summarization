#import os

#input setup
data_dir = "/hits/basement/nlp/fatimamh/test_data"
file_name = "en_train_sub.csv"

#tokenizer_dir = "/hits/basement/nlp/fatimamh/hf_pretrained_models/gpt2"
model_dir = "/hits/basement/nlp/fatimamh/hf_pretrained_models/gpt2"

#out dir
out_dir = "/hits/basement/nlp/fatimamh/gpt2b_out"
processed_set= "dataset" #split and tokenized
final_model = "fine_tuned"
training_models = "epochs_log"
generated = "summaries"

topk_file = "topk_summaries.csv"
beam_file = "beam_summaries.csv"
topk_score_file = "topk_scores.csv"

# text and sum len
max_seq_len = 768 #1024 to do
max_text	= 600 
max_sum	  	= 150
min_sum     = 100 

# split ratio
sr = 0.2

batch_size = 1
test_batch = 1 # fix it



#training param
epochs = 100
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
gen_type = 2 #0= top_k, 1= beam_search, 2= both



