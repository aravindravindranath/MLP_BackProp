# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 18:08:08 2020

@author: I816596
"""

import numpy as np
import pandas as pd
import random
import re
import constants as constant
import AlgorithmsAndFunctions as AlgoFunc
import dataprocessing as dataproc
import glass_data
import iris_data
import smallsoybean
import housevote
import BreastCancerWisconsin

#
# Function which reads the data in pandas dataframe format, does some
# pre-processing and then goes on to apply the Logistic Regression and 
# Adaline algorithm on the 5 data sets. Cross Validation with 5 folds is
# used to train and test the models built
# 
# Parameters
#   *args    The data file to be chosen and the the number of iterations can 
#            be passed in to bypass user prompts.         
def validation_runs(*args):
    
    Eta = 0.001
    
    if len(args) == 0:
        display_prompt = "Choose for which data set you want to apply Logistic Regression "
        display_prompt = display_prompt + "and Adaline learning algorithms on.\n"
        display_prompt = display_prompt + "1: Glass Data\n"
        display_prompt = display_prompt + "2: Iris Data\n" 
        display_prompt = display_prompt + "3: Small Soy Beans Data\n" 
        display_prompt = display_prompt + "4: House Votes\n" 
        display_prompt = display_prompt + "5: Breast Cancer Wisconsin Data\n"
    
        data_set = pd.DataFrame()
        choice = input(display_prompt)
        display_prompt = "Enter the Number of iterations: "
        iterations = input(display_prompt)
    else:
        choice = str(args[0])
        iterations = str(args[1])
            
    # Glass Data Set
    if choice == "1":
        data_set = dataproc.read_data_pre_process("glass.data", 
              glass_data.col_attrs_list )
        class_counts = data_set["class"].value_counts()
        file_name = "glass_output"+iterations+"_"+str(Eta)+"_.txt"
        f = open(file_name, "w")
    
    #Iris Data
    elif choice == "2":
        data_set = dataproc.read_data_pre_process("iris.data", 
              iris_data.col_attrs_list )
        #data_set["class"] = data_set["class"].apply(transform_values)
        class_counts = data_set["class"].value_counts()
        file_name = "iris_output"+iterations+"_"+str(Eta)+"_.txt"
        f = open(file_name, "w")
    
    #Small Soy Bean 
    elif choice == "3":
        data_set = dataproc.read_data_pre_process("soybean-small.data", 
              smallsoybean.col_attrs_list )
        class_counts = data_set["class"].value_counts()
        file_name = "smallsoybean_output"+iterations+"_"+str(Eta)+"_.txt"
        f = open(file_name, "w")
    
    #House Votes 84
    elif choice == "4":
        # There are cetrain records where the congressman or congresswoman
        # decided not to vote ( undecided ). The choice is made to take this
        # as a valid data point to indicate also a neutral non-opinionated 
        # stand on certain issues
        data_set = dataproc.read_data_pre_process("house-votes-84.data", 
              housevote.col_attrs_list, replace = [["?", "u"]]  )
        file_name = "housevotes_output"+iterations+"_"+str(Eta)+"_.txt"
        f = open(file_name, "w")
        
    #Breast Cancer Wisconsim data    
    elif choice == "5":
        # There are certain records which are observed to have a "?"
        # for the feature bare_nuclei. The choice has been made to
        # remove this data from the learning process as there are 
        # only a few entries and there is no best way to guess this.
        data_set = dataproc.read_data_pre_process("breast-cancer-wisconsin.data", 
              BreastCancerWisconsin.col_attrs_list, 
              drop_row_char = [{"name": "bare_nuclei", "char": "?"}] )
        file_name = "bcw_output_adaline"+iterations+"_"+str(Eta)+"_.txt"
        f = open(file_name, "w")

    else:
        print( "Input {} incorrect, enter 1,2,3,4 or 5".format(choice))
    

    # Run both Logistic Regression and Adaline algorithm on the training set and
    # then apply this on the test data and then validate this on the
    # test data and store the results in a file.
    
    file_str = ""
    if choice in("4", "5"):
        
#       For house votes and Breast Cancer data have a 2 class target and hence
#       simple binary classifier is built with single set of weights.
        orig_class = data_set["class"]
#        if choice == "4":
#            data_set["class"] = np.where(orig_class=="democrat", 1, 0)
#        else:
#            data_set["class"] = np.where(orig_class==4, 1, 0)
#     
#        file_str = file_str + "Logistic Regression" + "\n"
#        file_str = file_str + "-------------------" + "\n\n\n"
#        results = AlgoFunc.cross_validation(data_set, "LogisticRegression", 5, Eta, 
#                                            int(iterations), True, 1)
#        for i in range(0,len(results)):
#            file_str = file_str + str(results[i]["model"]) + "\n\n"
#            file_str = file_str + "Accuracy = {}".format(results[i]["Accuracy"])
#            file_str = file_str + "\n" + "X-Entropy = {}".format(results[i]["X-Entropy"])
#            file_str = file_str + "\n\n\n"

        if choice == "4":        
            data_set["class"] = np.where(orig_class=="democrat", 1, -1)
        else:
            data_set["class"] = np.where(orig_class==4, 1, -1)

        file_str = file_str + "Adaline" + "\n"
        file_str = file_str + "-------------------" + "\n\n\n"
        results = AlgoFunc.cross_validation(data_set, "Adaline", 5, Eta, 
                                            int(iterations), True, 1 )
        for i in range(0,len(results)):
            file_str = file_str + str(results[i]["model"]) + "\n\n"
            file_str = file_str + "Accuracy = {}".format(results[i]["Accuracy"])
            file_str = file_str + "\n" + "MSE = {}".format(results[i]["MSE"])
            file_str = file_str + "\n\n\n"
        

    if choice in ("1", "2", "3") :
#       For the other datam, they have more than 2 class targets and hence
#       simple classifier is built with multiple set of weights corresponding
#       to each class label.
        
        file_str = file_str + "Logistic Regression" + "\n"
        file_str = file_str + "-------------------" + "\n\n\n"
        results = AlgoFunc.cross_validation(data_set, "LogisticRegression", 5, Eta, 
                                            int(iterations), False, 1)
        for i in range(0,len(results)):
            file_str = file_str + str(results[i]["model"]) + "\n\n"
            file_str = file_str + "Accuracy = {}".format(results[i]["Accuracy"])
            file_str = file_str + "\n" + "X-Entropy = {}".format(results[i]["X-Entropy"])
            file_str = file_str + "\n\n\n"

        file_str = file_str + "Adaline" + "\n"
        file_str = file_str + "-------------------" + "\n\n\n"
        results = AlgoFunc.cross_validation(data_set, "Adaline", 5, Eta, 
                                            int(iterations), False, 1)
        for i in range(0,len(results)):
            file_str = file_str + str(results[i]["model"]) + "\n\n"
            file_str = file_str + "Accuracy = {}".format(results[i]["Accuracy"])
            file_str = file_str + "\n" + "MSE = {}".format(results[i]["MSE"])
            file_str = file_str + "\n\n\n"        
           
    f.write(file_str)
    f.close()
    print("Open File ", f.name)

def main():
    validation_runs()

if __name__ == "__main__":
    main()

    
