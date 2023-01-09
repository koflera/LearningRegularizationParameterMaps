#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 10:58:22 2022

@author: kofler01
"""

import torch

def inner_product(t1,t2):
	
	if torch.is_complex(t1):
		
		innerp = torch.sum(t1.flatten() * t2.flatten().conj())
	else:
		innerp = torch.sum(t1.flatten() * t2.flatten())
	return innerp
		