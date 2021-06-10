#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 15:13:41 2019
final
@author: fatimamh
"""

import time
import datetime

def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))

'''----------------------------------------------------------------
'''
SIZE_UNITS_MAPPING = [
    (1<<50, (' PB',' PBs')),
    (1<<40, (' TB',' TBs')),
    (1<<30, (' GB',' GBs')),
    (1<<20, (' MB',' MBs')),
    (1<<10, (' KB',' KBs')),
    (1, (' byte', ' bytes')),
]

def get_size(bytes, units = SIZE_UNITS_MAPPING):
    for factor, suffix in units:
        if bytes >= factor:
            break
    amount = bytes/factor 

    if isinstance(suffix, tuple):
        singular, multiple = suffix
        if amount == 1:
            suffix = singular
        else:
            suffix = multiple

    return str(round(amount,3)) + suffix