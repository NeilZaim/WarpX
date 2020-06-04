#! /usr/bin/env python

# Copyright 2020 Luca Fedeli, Neil Zaim
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

## This module is imported by each of the scripts used to analyze the Schwinger tests.

## The pair production rate is calculated using the formula described in
## Bulanov, S. S., et al. Physical review letters 104.22 (2010): 220404.

filename = "diags/diag100001/"

import yt
import numpy as np

data_set = yt.load(filename)

all_data = data_set.all_data()

ele_data = all_data["ele_schwinger",'particle_weight']
pos_data = all_data["pos_schwinger",'particle_weight']

print("number of electrons: ",len(ele_data))
print("number of positrons: ",len(pos_data))
print("total electron weight: ",np.sum(ele_data))
print("total positron weight: ",np.sum(pos_data))

