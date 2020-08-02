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


global return_run
return_run = False

def set_return_run(bRun=False):
    global return_run
    return_run = bRun

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
    
    global return_run
    
    Eta = 0.01
    
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
        iterations = int(input(display_prompt))
        display_prompt = "Enter Learning Factor (0,1) : "
        Eta = float(input(display_prompt))
        print("\n")
        print("The number of layers and units within each layer represented in a list\n")
        print("[n,m] - Two Hidden Layers with n units in 1st hidden layer and m units in the 2nd\n")
        print("[] - No hidden Layer\n")
        print("[n] - One hidden Layer with n units\n")
        display_prompt = "Enter Number of hidden layers and neurons as per the format above: "
        str_layer = input(display_prompt)
        layers = list(eval(str_layer))
        
    else:
        choice = str(args[0])
        iterations = int(str(args[1]))
        Eta = float(str(args[2]))
        layers = args[3]
            
    # Glass Data Set
    if choice == "1":
        data_set = dataproc.read_data_pre_process("glass.data", 
              glass_data.col_attrs_list )
        class_counts = data_set["class"].value_counts()
        file_name = "glass_output"+str(iterations)+"_"+str(Eta)+"_"+str(layers)+"_.txt"
        #f = open(file_name, "w")
    
    #Iris Data
    elif choice == "2":
        data_set = dataproc.read_data_pre_process("iris.data", 
              iris_data.col_attrs_list )
        #data_set["class"] = data_set["class"].apply(transform_values)
        class_counts = data_set["class"].value_counts()
        file_name = "iris_output"+str(iterations)+"_"+str(Eta)+"_"+str(layers)+"_.txt"
        #f = open(file_name, "w")
    
    #Small Soy Bean 
    elif choice == "3":
        data_set = dataproc.read_data_pre_process("soybean-small.data", 
              smallsoybean.col_attrs_list )
        class_counts = data_set["class"].value_counts()
        file_name = "smallsoybean_output"+str(iterations)+"_"+str(Eta)+"_"+str(layers)+"_.txt"
        #f = open(file_name, "w")
    
    #House Votes 84
    elif choice == "4":
        # There are cetrain records where the congressman or congresswoman
        # decided not to vote ( undecided ). The choice is made to take this
        # as a valid data point to indicate also a neutral non-opinionated 
        # stand on certain issues
        data_set = dataproc.read_data_pre_process("house-votes-84.data", 
              housevote.col_attrs_list, replace = [["?", "u"]]  )
        file_name = "housevotes_output"+str(iterations)+"_"+str(Eta)+"_"+str(layers)+"_.txt"
        #f = open(file_name, "w")
        
    #Breast Cancer Wisconsim data    
    elif choice == "5":
        # There are certain records which are observed to have a "?"
        # for the feature bare_nuclei. The choice has been made to
        # remove this data from the learning process as there are 
        # only a few entries and there is no best way to guess this.
        data_set = dataproc.read_data_pre_process("breast-cancer-wisconsin.data", 
              BreastCancerWisconsin.col_attrs_list, 
              drop_row_char = [{"name": "bare_nuclei", "char": "?"}] )
        file_name = "breatcancerwisconsin_output"+str(iterations)+"_"+str(Eta)+"_"+str(layers)+"_.txt"
        #f = open(file_name, "w")

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
        if choice == "4":
            data_set["class"] = np.where(orig_class=="democrat", 1, 0)
        else:
            data_set["class"] = np.where(orig_class==4, 1, 0)
     
        results = AlgoFunc.cross_validation( 5, 1, data_set, Eta, iterations,
                                            layers, K=1 )
        for i in range(0,len(results)):
               # run_data = {"Accuracy": [], "X-Entropy": 0.0, "Learning Factor": Eta,
                #"Epochs": max_iter, "Layers": layers }
            file_str = file_str + "Learning Factor = {}".format(results[i]["Learning Factor"])
            file_str = file_str + "\n" + "Epochs = {}".format(results[i]["Epochs"]) + "\n"
            for lay in range(0,len(layers)):
                file_str = file_str + "Hidden Layer = {}  ".format(lay) + \
                           "Number of Neurons: {}".format(layers[lay]) + "\n"
            file_str = file_str + "Accuracy = {}".format(results[i]["Accuracy"])
            file_str = file_str + "\n" + "X-Entropy = {}".format(results[i]["X-Entropy"])
            file_str = file_str + "\n\n\n"
            

    if choice in ("1", "2", "3") :
#       For the other datam, they have more than 2 class targets and hence
#       simple classifier is built with multiple set of weights corresponding
#       to each class label.
        
        results = AlgoFunc.cross_validation( 5, 1, data_set, Eta, iterations,
                                            layers )
        for i in range(0,len(results)):
            file_str = file_str + "Learning Factor = {}".format(results[i]["Learning Factor"])
            file_str = file_str + "\n" + "Epochs = {}".format(results[i]["Epochs"])
            for lay in range(0,len(layers)):
                file_str = file_str + "\n" + "Hidden Layer = {}  ".format(lay) + \
                           "Number of Neurons: {}".format(layers[lay]) + "\n"
            file_str = file_str + "Accuracy = {}".format(results[i]["Accuracy"])
            file_str = file_str + "\n" + "X-Entropy = {}".format(results[i]["X-Entropy"])
            file_str = file_str + "\n\n\n"       
    
    if return_run:
        return results        
    
    f = open(file_name, "w")
    f.write(file_str)
    f.close()
    print("Open File ", f.name)

def main():
    validation_runs()

if __name__ == "__main__":
    main()

    
