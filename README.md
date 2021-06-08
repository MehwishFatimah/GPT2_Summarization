1:- Get pretrained model
	1. Download pretrained model and tokenizer (GPT-2) (outside the code)
	2. store it in a folder and load it from there

2:- Data processing
	1. input to model: <bos> + text + <sep> + sum + <eos> (guess it should be done in loader not in df)
	Truncate lengths of text and sum to fit in the design. 
	Total sequence length = 768 or 1024 
	2. split data
	
3:- Fine-tune tokenizer
	1. add special tokens to GPT-2 tokenizer
	2. fine-tune tokenizer for your data (train+val)

3:- Fine-tuning model

4:- Generation/Inference

5:- Evaluation of system output

