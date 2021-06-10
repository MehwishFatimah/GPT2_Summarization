#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 14:48:19 2019
Modified on Tue Nov 26 
@author: fatimamh

FILE:
    DATA_UTILS: WORKS STANDALONE
INPUT:
    FOLDER PATH                      : STR
    JSON FILE NAMES (TRAIN/VAL/TEST) : STR
    CLEAN_TEXT                       : BOOL (default: True)

OUTPUT:
    CSV FILES (TRAIN/VAL/TEST) WITH SAME NAME IN SAME FOLDER
DESCRIPTION:
    THIS CODE TAKES JSON FILES (TRAIN/VAL/TEST) WITH DIFFERENT TAGS IN IT.
        -REMOVE THE TAGS
        -REMOVE INDEX COLUMN
        -UNTOKENIZED
        -CONVERTS FILES TO CSV
"""

'''-----------------------------------------------------------------------
Import libraries and defining command line arguments
-----------------------------------------------------------------------'''
import argparse
import os
import re
import pandas as pd
import time
import resource
#from nltk.tokenize.treebank import TreebankWordDetokenizer as detokenizer
'''-----------------------------------------------------------------------
'''
class GetLastAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string = None):
        if values:
            print(values)
            setattr(namespace, self.dest, values)

parser = argparse.ArgumentParser(description = 'Help for data_utils')

parser.add_argument('--d',   type = str,  default = '/mnt/basement/00_base_codes_data/merged_data_july_2020/base/cross',
                    help = 'Directory path for data in and out.')
parser.add_argument('--f',   nargs = '+', action  = GetLastAction,
                    default = ['en_de_test.json','en_de_val.json','en_de_train.json'],
                    help = 'File names with extension.')
parser.add_argument('--c',   type = bool, default = True,
                    help = 'Clean text: default =True.')
parser.add_argument('--e',   type = str,  default = '.csv',
                    help = 'Output files extension: default =.csv.')


'''----------------------------------------------------------------
'''
def tokenize(text):
    tokens = text.split()
    return tokens

'''-----------------------------------------------------------------------
Takes text of an article containing following tags from dataframe and returns the text
after removing those tags.
  Args:
    text  : str
  Returns:
    text  : str
'''
def replace_tags_a(text):

    text = text.replace('<ARTICLE>', ' ')
    text = text.replace('</ARTICLE>', ' ')
    text = text.replace('<TITLE>', ' ')
    text = text.replace('</TITLE>', ' ')
    text = text.replace('<HEADING>', ' ')
    text = text.replace('</HEADING>', ' ')
    text = text.replace('<SECTION>', ' ')
    text = text.replace('</SECTION>', ' ')
    text = text.replace('<S>', ' ')
    text = text.replace('</S>', ' ')
    text = text.replace('\n', ' ')
    text = re.sub('\s+', ' ', text)

    return text

'''-----------------------------------------------------------------------
Takes text of a summary containing following tags from dataframe and returns the text
after removing those tags.
  Args:
    text   : str
  Returns:
    text   : str
'''
def replace_tags_s(text):

    text = text.replace('<SUMMARY>', ' ')
    text = text.replace('</SUMMARY>', ' ')
    text = text.replace('<S>', ' ')
    text = text.replace('</S>', ' ')
    text = text.replace('\n', ' ')
    text = re.sub('\s+', ' ', text)
    #text = detokenizer().detokenize(text)
    #text = ''.join(text)
    #print(type(text))
    return text


'''-----------------------------------------------------------------------
Takes a dataframe. Print its size, columns and head.
If clean_text is true, then calls cleaning method for article and summary.
If index is stored in file then delete this column.
If short_text is true, then calls short text method for article and summary.
retuns dataframe after specified operations.

  Args:
    df          : dataframe
    clean_text  : bool (default: True)

  Returns:
    df          : dataframe
'''
def clean_data(df, clean_text):

    print(clean_text)
    print('Data before cleaning:\nSize: {}\nColumns: {}\nHead:\n{}'.format(len(df), df.columns, df.head(5)))

    if clean_text:
        print('cleaning text')
        df['text']    = df['text'].apply(lambda x: replace_tags_a(x))
        df['summary'] = df['summary'].apply(lambda x: replace_tags_s(x))
        df['dtext']    = df['dtext'].apply(lambda x: replace_tags_a(x))
        df['dsummary'] = df['dsummary'].apply(lambda x: replace_tags_s(x))

    if 'index' in df.columns:
        del df['index']


    print('Data after cleaning:\nSize: {}\nColumns: {}\nHead:\n{}'.format(len(df), df.columns, df.head(5)))

    return df

'''-----------------------------------------------------------------------
'''
def process_data(files, folder, clean_text, ext):


    for file in files:
        file = os.path.join(folder, file)
        df   = pd.read_json(file, encoding = 'utf-8')
        #print(df.head(2))
        df   = clean_data(df, clean_text)
        print('\n--------------------------------------------')

        file_name  = os.path.splitext(file)[0]
        file       = os.path.join(folder, file_name + ext)
        print(file)
        df.to_csv(file, index = False)

        print('\n======================================================================================')

'''-----------------------------------------------------------------------
'''
if __name__ == "__main__":
    start_time = time.time()
    args = parser.parse_args()
    print('\n\n----------------------Printing all arguments:--------------------\n{}\n----------------------------------------\n'.format(args))
    process_data(args.f, args.d, args.c, args.e)
    print ('\n-------------------Memory and time usage: {:.2f} MBs in {:.2f} seconds.--------------------\n'.format((resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024), (time.time() - start_time)))