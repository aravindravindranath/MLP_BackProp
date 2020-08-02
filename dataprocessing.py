# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 18:14:17 2020

@author: I816596
"""

import pandas as pd
import re
import AlgorithmsAndFunctions as AlgoFunc
import numpy as np


#
# Function to convert 2-valued class variable to a binary category variable
# with 0 and 1
#
def convert_class(x,mx,mn):
    if x == mx:
        return 1
    else:
        return 0
#def convert_class(df):
#    class_counts = df["class"].value_counts().reset_index(drop=True)
#    return df["class"].apply( lambda x: x==class_counts[0] and 1 or 0)

#
# Function to scale the feature values to neutralize effects of
# high magnitude features
# Since as per the assignment guideline, the range has been decided to
# be between -1 and +1 for continuous attributes
def scale_column(in_col = pd.Series()):
    # The denominator is chosen as range instead of Standard Deviation
    # to ensure values are truly within -1 and +1. SD leads to numbers 
    # outside this range due to points outside of 1 SD.
    return ( in_col - np.mean(in_col) ) / ( np.max(in_col) - np.min(in_col) )

#
# Function to read the data and apply some pre-processing to make it ready
# for model building. The main steps are to remove columns with no relevance 
# like IDs, sample numbers etc., categorize discrete featues and above all
# removing features which have the same value in all the records ( this does
# not lead to any influence on the classifier )
#
# Parameters
#
# file_name            Name of File containing the data set
# column_names_props   column names along with properties
# drop_row_char        delete records with a featire having a certain value
# replace              replace the data of a particular feature of an instance
# raw_dat              Preprocessed data 

def read_data_pre_process(file_name = "", column_names_props = [],
                          replace = [], drop_row_char = [] ):
    col_names = []
    drop_cols = []
    cols_categorize = []
    cols_scale = []
    for col_name_prop in column_names_props:
        col_names.append(col_name_prop["name"])
        if col_name_prop["drop"]:
            drop_cols.append(col_name_prop["name"])
        if col_name_prop["categorize"]:
            cols_categorize.append(col_name_prop["name"])
        try: 
            if col_name_prop["scale"]:
                cols_scale.append(col_name_prop["name"])
        except:
            cols_scale.clear()
    raw_data = pd.read_csv(file_name, names=col_names)
    raw_data = raw_data.drop(columns=drop_cols, axis=1)
    
    # if there is a replacement of not clear data then chose a replacement
    for i in range(0,len(replace)):
        raw_data.replace(replace[i][0], replace[i][1], inplace=True)
    
    #when some rows needs to be dropped due to unclear data
    for i in range(0, len(drop_row_char)):
        raw_data = raw_data.drop(
        raw_data[raw_data[drop_row_char[i]["name"]]==drop_row_char[i]["char"]].index.tolist()
                ).reset_index(drop=True)    
    
    
    for col in raw_data.columns:
        if col in cols_categorize:
            raw_data = pd.concat([raw_data, pd.get_dummies(raw_data[col], 
                                        prefix=col)], axis=1)
            raw_data = raw_data.drop(col, axis=1)
        if col in cols_scale:
            raw_data[col] = scale_column(raw_data[col])

    # remove attributes from the data set where the value is the same 
    # for all records to simplify the analysis.    
    for col in raw_data.columns:
        values = set(raw_data[col])
        if len(values) == 1:
            raw_data = raw_data.drop(col, axis=1)
            continue     

#   The bias/threshold is represented by x0 with all ones 
#   and applied to all data sets  (b0 in the models)
    raw_data["x0"] = np.ones(len(raw_data))
    return raw_data

