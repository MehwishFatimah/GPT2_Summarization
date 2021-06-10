#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 27 14:36:41 2021
TODO: Merge it with data
@author: fatimamh
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, SequentialSampler
from torchvision import transforms, utils

class GPT2Dataset(Dataset):
  def __init__(self, df, gpt2_type="gpt2"):
    self.encodings = df['encodings'].to_list()
    self.sum_idx = df['text_len'].to_list()
  
  def __len__(self):
    return len(self.sum_idx)

  def __getitem__(self, idx):
    text = torch.tensor(self.encodings[idx]['input_ids'])
    attn_mask = torch.tensor(self.encodings[idx]['attention_mask'])
    s_idx = self.sum_idx[idx] + 2 # add bos and cls
    
    out = {'text': text, 'mask': attn_mask, 's_idx': s_idx}
    return out #, self.attn_masks[idx] 

'''----------------------------------------------------------------
'''

def get_gpt2_dataset(train, val): #, test):

  train_dataset = GPT2Dataset(train)
  val_dataset = GPT2Dataset(val)
  #test_dataset = GPT2Dataset(test)

  return train_dataset, val_dataset#, test_dataset