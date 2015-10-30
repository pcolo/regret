# -*- coding: utf-8 -*-

# Import all libraries needed for the tutorial

# General syntax to import specific functions in a library:
##from (library) import (specific library function)
from tuto_pandas import DataFrame, read_csv

# General syntax to import a library but no functions:
##import (library) as (give the library a nickname/alias)
import matplotlib.pyplot as plt
import tuto_pandas as pd #this is how I usually import pandas
import sys #only needed to determine Python version number



# The inital set of baby names and bith rates
names = ['Bob','Jessica','Mary','John','Mel']
births = [968, 155, 77, 578, 973]

BabyDataSet = zip(names,births)
BabyDataSet

df = pd.DataFrame(data = BabyDataSet, columns=['Names', 'Births'])
df