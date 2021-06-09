#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 31 12:30:25 2021

@author: fatimamh

It is neccessary to make train class so that there is no need to pass/return model, loss function and optimizer
TODO: Batch processing
"""

import os
import time
import random
import torch

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = True # for fast training


from transformers import GPT2LMHeadModel, GPT2Config
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.nn import CrossEntropyLoss

from utils.time_mem_utils import format_time
from utils.print_n_plot_utils import *


import data_config as config


class Train(object):
	'''----------------------------------------------------------------
	Initialize train object, check/create final_model and training_models.
	Load configuration and model from the given path. Create optimizer, 
	scheduler and loss_function.
	Args:
		device		 	: torch_device
		model_path	 	: str
		tokenizer_len	: int
		ignore_index 	: int
		train_loader_len: int
	Return: 
		object
	'''
	def __init__(self, device, model_path, tokenizer_len, ignore_index, train_loader_len):

		self.device = device
		
		self.final_model = os.path.join(config.out_dir, config.final_model)
		if not os.path.exists(self.final_model):
			os.makedirs(self.final_model)

		self.training_models = os.path.join(config.out_dir, config.training_models)
		if not os.path.exists(self.training_models):
			os.makedirs(self.training_models)
		
		self.configuration = GPT2Config.from_pretrained(model_path, 
														output_hidden_states=False) #config file
		self.model = GPT2LMHeadModel.from_pretrained(model_path, 
													 config=self.configuration) # instantiate the model
		self.model.resize_token_embeddings(tokenizer_len)
		print('tokenizer len: {}'.format(tokenizer_len))
		self.model.to(self.device)
		
		self.optimizer = AdamW(self.model.parameters(), 
							   lr=config.learning_rate, 
							   eps=config.epsilon)
		self.loss_fct = CrossEntropyLoss(ignore_index=ignore_index) #ignores padding token for loss calculation

		self.gradient_accumulation_steps = config.gradient_accumulation_steps
		self.max_grad_norm = config.max_grad_norm

		# Create the learning rate scheduler. This changes the learning rate as the training loop progresses
		self.scheduler = get_linear_schedule_with_warmup(self.optimizer, 
													num_warmup_steps = config.warmup_steps, 
													num_training_steps = train_loader_len * config.epochs)


	'''----------------------------------------------------------------
	Takes a batch to process for training loop.
	Args:
		batch: tensors
	Return: 
		shift_logits, shift_labels: tensors
	'''
	def process_train_batch(self, batch):         
		inputs, labels = batch['text'].clone().detach(), batch['text'].clone().detach()
		inputs = inputs.to(self.device)
		labels = labels.to(self.device)

		logits = self.model(inputs)[0]
		#outputs = model(  b_input_ids, labels=b_labels,  attention_mask = b_masks, token_type_ids=None)
		#loss = outputs[0]  
		#print('logits: {}'.format(logits.shape))
		
		idx = batch['s_idx'].item() # index of separator token
		print('logits: {}'.format(logits.shape))
		print('idx: {}'.format(idx))
		# only consider loss on reference summary just like seq2seq models
		shift_logits = logits[..., idx:-1, :].contiguous()
		shift_labels = labels[..., idx+1:].contiguous()
		#print('shift_logits: {}'.format(shift_logits.shape))
		#print('shift_labels: {}'.format(shift_labels.shape))

		return shift_logits, shift_labels 

	'''----------------------------------------------------------------
	Takes a batch to process for evaluation loop.
	Args:
		batch: tensors
	Return: 
		shift_logits, shift_labels: tensors
	'''
	def process_eval_batch(self, batch):
		inputs, labels = batch['text'].clone().detach(), batch['text'].clone().detach()
		inputs = inputs.to(self.device)
		labels = labels.to(self.device)
	
		with torch.no_grad():        
			logits = self.model(inputs)[0]
			#print('logits: {}'.format(logits.shape))
			
			idx = batch['s_idx'].item() # index of separator token
			print('logits: {}'.format(logits.shape))
			print('idx: {}'.format(idx))
			# only consider loss on reference summary just like seq2seq models
			shift_logits = logits[..., idx:-1, :].contiguous()
			shift_labels = labels[..., idx+1:].contiguous()
			#print('shift_logits: {}'.format(shift_logits.shape))
			#print('shift_labels: {}'.format(shift_labels.shape))

		return shift_logits, shift_labels

	'''----------------------------------------------------------------
	Takes train dataloaders 
	Args:
		train_dataloader: dataloader
	Return: loss
	'''
	def train_loop(self, train_dataloader):
		self.model.train()
		total_train_loss = 0

		for step, batch in enumerate(train_dataloader):
			self.model.zero_grad()
			shift_logits, shift_labels = self.process_train_batch(batch)	

			loss = self.loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
			loss = loss/self.gradient_accumulation_steps
			loss.backward() 
			torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)       
			self.optimizer.step()
			self.scheduler.step()
			batch_loss = loss.item()
			total_train_loss += batch_loss

		return total_train_loss

	'''----------------------------------------------------------------
	Takes eval dataloaders 
	Args:
		val_dataloader: dataloader
	Return: loss
	'''
	def eval_loop(self, val_dataloader):
		self.model.eval()
		total_eval_loss = 0
		
		for step, batch in enumerate(val_dataloader):
			shift_logits, shift_labels = self.process_eval_batch(batch)	

			loss = self.loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
			batch_loss = loss.item()
			total_eval_loss += batch_loss

		return total_eval_loss

	'''---------------------------------------------------------------- 
	Args:
		total_loss: float
		size	  : int
	Return: 
		average: float
	'''
	def average_loss(self, total_loss, size):
		return total_loss / size

	'''---------------------------------------------------------------- 
	Args:
		e_time: time
		s_time: time
	Return: 
		time: time
	'''
	def elapsed_time(self, e_time, s_time):
		return format_time(e_time - s_time)

	'''----------------------------------------------------------------
	Args: none
	Return: none
	'''
	def save_model(self, path):
		model_to_save = self.model.module if hasattr(self.model, 'module') else self.model  # Take care of distributed/parallel training
		model_to_save.save_pretrained(path)
		# Good practice: save your training arguments together with the trained model
		# torch.save(args, os.path.join(path, 'training_args.bin'))

	'''----------------------------------------------------------------
	Args: none
	Return: none
	'''
	def model_params(self):

		name = "params_" + str(int(time.time())) + ".txt"
		file = os.path.join(config.out_dir, name)
		# Get all of the model's parameters as a list of tuples.
		params = list(self.model.named_parameters())
		f = open(file, "w")

		f.write('The model has {:} different named parameters.\n'.format(len(params)))
		f.write('\n==== Embedding Layer ====\n')
		for p in params[0:2]:
			f.write("{:<55} {:>12}\n".format(p[0], str(tuple(p[1].size()))))

		f.write('\n==== First Transformer ====\n')
		for p in params[2:14]:
			f.write("{:<55} {:>12}\n".format(p[0], str(tuple(p[1].size()))))

		f.write('\n==== Output Layer ====\n')
		for p in params[-2:]:
			f.write("{:<55} {:>12}\n".format(p[0], str(tuple(p[1].size()))))	
		f.close()
	'''----------------------------------------------------------------
	This is the main training/fine_tuning loop
	Takes train and val dataloaders .
	Args:
		train_dataloader: dataloader
		val_dataloader	: dataloader
	Return: none
	'''
	def train_model(self, train_dataloader, val_dataloader):
		'''----------------------------------------------------------------
		'''
		epochs = config.epochs
		total_t0 = time.time()
		time_stamp = str(int(time.time()))
		file = "train_log_" + time_stamp + ".csv"
		print_file = os.path.join(config.out_dir, file)
		
		file = "plot_log_" + time_stamp + ".csv"
		plot_file = os.path.join(config.out_dir, file)

		for epoch_i in range(0, epochs):

			print('======== Epoch {}/{} ========'.format(epoch_i + 1, epochs))
			print('Training...')

			t0 = time.time()
			total_train_loss = self.train_loop(train_dataloader)
			avg_train_loss = self.average_loss(total_train_loss, len(train_dataloader))       
			training_time = self.elapsed_time(time.time(), t0)
			
			print("  Average training loss: {0:.2f}".format(avg_train_loss))
			print("  Training epoch took: {:}".format(training_time))
						
			print("Running Validation...")

			t0 = time.time()

			total_eval_loss = self.eval_loop(val_dataloader)   
			avg_val_loss = self.average_loss(total_eval_loss, len(val_dataloader))
			validation_time = self.elapsed_time(time.time(), t0)    

			print("  Valid. Loss: {0:.2f}".format(avg_val_loss))
			print("  Valid. took: {}".format(validation_time))

			# Record all statistics of this epoch.
			print_row = build_row(epoch_i + 1, avg_train_loss, avg_val_loss, training_time, validation_time)
			print_writer(print_file, print_row)

			plot_row = build_row(epoch_i + 1, avg_train_loss, avg_val_loss)
			plot_writer(plot_file, plot_row)

			if (epoch_i + 1)% 10 == 0: 
				#save model
				print('epoch_i + 1: {}'.format(epoch_i + 1))
				path = os.path.join(self.training_models, str(epoch_i+1))
				self.save_model(self.training_models)

		print("Training complete!")
		print("Total training took {} (h:mm:ss)".format(self.elapsed_time(time.time(), total_t0)))
		

		self.save_model(self.final_model)
		self.model_params()
		plot(plot_file)
		
