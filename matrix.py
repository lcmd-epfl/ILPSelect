#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 12:25:45 2021

@author: eisenbrand
"""

import numpy as np 
data = np.load("data.npz", allow_pickle=True)

data.files

print(len(data['target_CMs'][2]))
T = data['target_CMs'][2]
Z = [0,2,3] 
S = [2,30,5]

print(T[Z][:,S])


