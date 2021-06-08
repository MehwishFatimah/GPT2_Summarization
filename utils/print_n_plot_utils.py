#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 02 14:41:25 2021

@author: fatimamh
"""
import time
import os
from csv import writer
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker

from random import seed
from random import randint
import pandas as pd


'''-----------------------------------------------------------------
Takes file and content to write in a csv file. If file creates first
time, headers will be added
Args:
    file   : str
    content: str
Return: none 
'''
def beam_writer(file, content, beam):
    with open(file, 'a+', newline='') as obj:
        headers = ['text', 'summary']
        sub = ['generated', 'score']
        for i in range(beam):
            headers.extend(sub)
        csv_writer = writer(obj)
        is_empty = os.stat(file).st_size == 0
        if is_empty:
            csv_writer.writerow(headers)
        csv_writer.writerow(content)

'''-----------------------------------------------------------------
Takes file and content to write in a csv file. If file creates first
time, headers will be added
Args:
    file   : str
    content: str
Return: none 
'''
def topk_writer(file, content):
    with open(file, 'a+', newline='') as obj:
        headers = ['text', 'summary', 'generated']
        csv_writer = writer(obj)
        is_empty = os.stat(file).st_size == 0
        if is_empty:
            csv_writer.writerow(headers)
        csv_writer.writerow(content)

'''-----------------------------------------------------------------
Takes file and content to write in a csv file. If file creates first
time, headers will be added
Args:
    file   : str
    content: str
Return: none 
'''
def print_writer(file, content):
    with open(file, 'a+', newline='') as obj:
        headers = ['epoch', 'avg_train_loss', 'avg_val_loss', 'training_time', 'validation_time']
        csv_writer = writer(obj)
        is_empty = os.stat(file).st_size == 0
        if is_empty:
            csv_writer.writerow(headers)
        csv_writer.writerow(content)
'''-----------------------------------------------------------------
Takes file and content to write in a csv file. If file creates first
time, headers will be added. This file is to be used in plot later.
Args:
    file   : str
    content: str
Return: none 
'''
def plot_writer(file, content):
    with open(file, 'a+', newline='') as obj:
        headers = ['epoch', 'train_loss', 'val_loss']
        csv_writer = writer(obj)
        is_empty = os.stat(file).st_size == 0
        if is_empty:
            csv_writer.writerow(headers)
        csv_writer.writerow(content)
'''-----------------------------------------------------------------
Creates content for both print and plot writer.
Args:
    args: args
Return: 
    row: list 

'''
def build_row(*args):
    row =[]
    for i in range(len(args)):
        row.append(args[i])
    return row


'''-----------------------------------------------------------------
Takes file to get the plot data. Plots training and validation loss 
for epochs. Stores the plot file.
Args:
    file   : str
Return: none 

'''
def plot(file): 
    
    df = pd.read_csv(file) 
    #print(df.head())

    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.AutoLocator()#MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    ax.yaxis.get_major_locator().set_params(integer=True)
    #ax.xaxis.set_major_locator(loc)
    ax.xaxis.get_major_locator().set_params(integer=True)
    x = df['epoch']
    y = df['train_loss']
    z = df['val_loss']
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title("Training & Validation Loss")
    plt.plot(x, y, color='blue', linewidth=2, label='Training')
    plt.plot(x, z, color='red', linewidth=2, label='Validation')
    
    plt.legend(loc='best')
    file = os.path.splitext(file)[0] + '.png'
    
    plt.savefig(file)
    print(file)

