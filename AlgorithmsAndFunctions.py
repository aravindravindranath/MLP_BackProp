# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 17:53:22 2020

@author: Aravind Ravindranath
"""

import numpy as np
import pandas as pd
import random
import constants as constant

#
# Function to discretize features with continuous values. Intervals
# are set based on data exploration. With this we can group handful
# of continuous values into a range/bin.
#
# Parameters
#   column_name  Name of the feature
#   data         The Data Set containing the feature
#   interval     Interval in which certain values fall into 
#   min_adj      Adjustment factor to avoid overlaps
#   max_adj      Adjustment factor to avoid overlaps
#
def discretize_columns(column_name, data = pd.DataFrame(), interval = 0.5,
                       min_adj=0, max_adj=0):
    min_val=min(data[column_name])-min_adj
    max_val=max(data[column_name])+max_adj
    ranges = np.linspace(min_val, max_val, round((max_val - min_val)/interval))
    return pd.cut(data[column_name], ranges)

#
# Function to split the data set into training set and validation set.
# 2/3rd of the data will be training set and the rest will be 
# used for validating the model obtained from the training
#
# Parameters
#   data         The Data Set undergoing the split
#   data_trng    Training Data Set
#   data_test    Test Data Set
#   
def split_data_trng_test( data = pd.DataFrame()):
    n = data.shape[0]
    m = round(n*0.67)
    random.seed(constant.RAND_SEED)
    randomlist = random.sample(range(0, n-1), m)
    data_trng = data.iloc[randomlist]
    data_test = data.drop(randomlist)
    return data_trng, data_test


#
# Function which returns the sigmoid of a value
#
# Parameters
#   value      Value for which sigmoid needs to be evaluated
def sigmoid(value=0.0):
    return np.exp(value) / ( 1 + np.exp(value) )  

#
#        
# Function to calculate the weights in the logistic discrimination algorithm
# Gradient descent is used as per the pseudocode listed in Alpaydin textbook.
# The weights of the linear model are returned based on the number of 
# iterations.
#    
# Parameters
#   df       Data Set used for validation
#   Eta      A constant applied to delta which is then added to the weights
#   max_iter The maximum number of iterations executed to change the weight
#            values
#
def calc_wgts_backprop_multi_layers(df= pd.DataFrame(), Eta=0.001, epochs=30,
                                        num_hidden_nodes=[],K=0):
    #df["x0"] = np.ones(len(df))
    #N(h) = df.shape[0] / (ALPHA*(df.shape[1] + 1))
    # Array with all the features excluding the label
    X=np.array(df.drop("class", axis=1))
    #Include the bias
    #X=np.append(X,np.ones([len(X),1]),1)

    
    #Number of nodes in hidden layer    
    Nh =  round(df.shape[0] / (constant.ALPHA*(df.shape[1] + 1)) )

    class_counts = df["class"].value_counts().sort_index()
    if K == 0:         
        K = len(class_counts)
    
#   Zh = np.zeros(Nh)
    layers = list()
    
    #Hidden layer output, input to output layer ( hidden layer to output layer )
   
    Zh = {}
    for i in range(0,len(num_hidden_nodes)):
        layers.append(num_hidden_nodes[i])    
        Zh[i] = np.zeros(layers[i])



    #Define and initialize the weight vectors ( input to hidden layer )
    #W=np.zeros_like(X[1])
    if len(layers) > 0:
#        Wh = np.zeros([layers[0],X.shape[1]])
#        for i in range(0,Wh.shape[0]):
#            for j in range(0,Wh.shape[1]):
#                Wh[i][j] = random.uniform(-0.01,0.01)  
        
        WIh = {}        
        for layer in range(0,len(layers)):
            if layer == 0:
                input_size = X.shape[1]
            else:
                input_size = layers[layer-1]
            WIh[layer] = np.zeros([layers[layer], input_size])
            for i in range(0,WIh[layer].shape[0]):
                for j in range(0,WIh[layer].shape[1]):
                    WIh[layer][i][j] = random.uniform(-0.01,0.01)  
        Wh = None    
    else:
        Wh = np.zeros([K,X.shape[1]])
        for i in range(0,Wh.shape[0]):
            for j in range(0,Wh.shape[1]):
                Wh[i][j] = random.uniform(-0.01,0.01)    
        WIh = None
        
    if len(layers) > 0:
        delta_Wh = {}
        for layer in range(0,len(layers)):
            delta_Wh[layer] = np.zeros(layers[layer])
    else:
        delta_Wh = np.zeros(K)
        
    #Number of classes/labels
    #K = 1 #For 2 class problem
    Yi = np.zeros(K)
    
    #Initialize weight vectors ( hidden layer to hidden/output layer )
    if len(layers) > 0:
        Vi = np.zeros([K,layers[-1]])
        for i in range(0,K):
            for j in range(0,layers[-1]):
                Vi[i][j] = random.uniform(-0.01,0.01)    
        delta_Vi = np.zeros(K)
    else:
        Vi = None
        
        
    lc = 0
    while True and lc < epochs:
        
        if len(layers) == 0:
            for i in range(0, df.shape[0]):           
                if K == 1:
                    #Yi[0] = sigmoid(X[i].dot(Wh[0].T))
                    Yi[0] = X[i].dot(Wh[0].T)
                    for j in range(0,X.shape[1]):
                        Wh[0][j] += Eta*(df.iloc[i]["class"]-Yi[0])*X[i][j] 
                else:
                 
#                    for j in range(0,K):
#                        Oi[j] = X[i].dot(Wh[j].T)
#                    tot_exp = np.sum(np.exp(Oi))    
                    for j in range(0,K):
                        #Yi[j] = np.exp(Oi[j])/tot_exp
                        Yi[j] = X[i].dot(Wh[j].T) 
                        if df.iloc[i]["class"] == class_counts.index[j]:
                            bFound = 1
                        else:
                            bFound = 0
                        for k in range(0,X.shape[1]):
                            Wh[j][k] += Eta*(bFound-Yi[j])*X[i][j] 
           
        else:
                   
            for i in range(0, df.shape[0]):
#               Calculate the outputs at each layer until the output                
                for layer in range(0, len(layers)):
                    if layer == 0:
                        for h in range(0,layers[layer]):
                            Zh[layer][h] = sigmoid(X[i].dot(WIh[0][h].transpose()))                        
                    if layer > 0:
                        #Zh[layer][h] = sigmoid(X[i].dot(Wh[h].transpose()))    
                        for h in range(0,layers[layer]):
                            Zh[layer][h] = sigmoid( Zh[layer-1].dot(WIh[layer][h].transpose()) )
                    if layer == (len(layers) - 1):
                        for j in range(0,K):
                            Yi[j] = Zh[layer].dot(Vi[j].transpose())                       

                error_terms = {}
                for layer in range(len(layers)-1,-1,-1):
                    if layer == (len(layers) -1):
                        if K == 1:
                            bFound = df.iloc[i]["class"]
                            delta_Vi[0] = np.sum(Eta*(bFound-Yi[0])*Zh[layer])
                        else:
                            for j in range(0,K):
                                if ( ( df.iloc[i]["class"] != class_counts.index[j] )
                                     and ( df.iloc[i]["class"] != 1 ) ):
                                    bFound = 0
                                else:
                                    bFound = 1
                                delta_Vi[j] = np.sum(Eta*(bFound-Yi[j])*Zh[layer])

                        error_term = np.zeros([layers[layer],K])            
                    for h in range(0,layers[layer]):
                        
                        if layer == 0:
                            input = X[i]
                        else:
                            input = Zh[layer-1]
                        if layer == (len(layers) -1):
                            if K == 1:
                                bFound = df.iloc[i]["class"]
                                error_term[h][0] = (bFound-Yi[0])*Vi[0][h]
                            else:
                                for j in range(0,K):
                                    if ( ( df.iloc[i]["class"] != class_counts.index[j] )
                                         and ( df.iloc[i]["class"] != 1 ) ):
                                        bFound = 0
                                    else:
                                        bFound = 1                            
                                    error_term[h][j] = (bFound-Yi[j])*Vi[j][h]
                            delta_Wh[layer][h] = Eta*np.sum( np.sum(error_term[h])*Zh[layer][h]*(1-Zh[layer][h])*input )                                                                
                                                                                                      #Check
                            #delta_Wh[layer][h] = np.sum(Eta*inter_term*Zh[layer][h]*(1-Zh[layer][h])*input)
                        
                        else:
#                            for j in range(0,layers[layer+1]):
#                                #check
#                                #inter_term = np.sum((df.iloc[i]["class"]-Zh[layer+1][j])*WIh[layer][j])
#                                inter_term[j] = np.sum( ( Zh[layer+1][j] -   )*WIh[layer][j])
#                                #                                                                        Check      
#                                delta_Wh[layer][h] = np.sum(Eta*inter_term*Zh[layer][h]*(1-Zh[layer][h])*input)
                            error_term = np.zeros([layers[layer], layers[layer+1]])
                            for j in range(0,layers[layer+1]):
                                #check
                                #inter_term = np.sum((df.iloc[i]["class"]-Zh[layer+1][j])*WIh[layer][j])
                                error_term[h][j] = np.sum(error_terms[layer+1][j])*WIh[layer+1][j][h]
                                #                                                                        Check      
                            delta_Wh[layer][h] = Eta*np.sum(np.sum(error_term[h])*Zh[layer][h]*(1-Zh[layer][h])*input)

                    error_terms[layer] = error_term        
                    
                    if layer == (len(layers) -1):
                        for j in range(0,K):
                            Vi[j] = Vi[j] + delta_Vi[j] 

                    for h in range(0,layers[layer]):
                        WIh[layer][h] = WIh[layer][h] + delta_Wh[layer][h]
            
        lc += 1

    return Wh, Vi, layers, WIh, class_counts

#
# Function to verify convergence. Given a data set, the weights are
# getting calculated via gradient descent method. The test data is applied
# on this method in the cross validation run and the accuracy of prediction is
# determined. This is applied for a 2 class logistic regression classifier.
# In addition to the accuracy it also retains the cross-entropy
#    
# Parameters
#   df       Data Set used for validation
#   weights  The weights calculated from the learning
#
def verify_convergence_MLP_backprop(df= pd.DataFrame(), wgts=None,wgts_hdn=None, layers=[], bMClass=True, cls_cnts=None):
    
    columns = list(df.columns)
    answer = None
    accuracy = 0.0
    class_counts = df["class"].value_counts().sort_index()
    probs = []
    pred_counts = class_counts.copy()
    # Get the probability distribution of the target classes based on the
    # data and also prepare data structure to calculate the prediction 
    # probability distribution
    for i in class_counts.index:
        probs.append(class_counts[i]/np.sum(class_counts))      
        pred_counts[i] = 0
    
      
    X=np.array(df.drop("class", axis=1))
    #X=np.append(X,np.ones([len(X),1]),1)
    Zh={}
    for layer in range(0,len(layers)):
        Zh[layer] = np.zeros(layers[layer])
    
    #pred_counts = [0, 0]
    
       
    for i in range(0, df.shape[0]):
        Oi = np.zeros(len(cls_cnts))
        Yi = np.zeros_like(Oi)
        if len(layers) == 0:
            if bMClass:
                for j in range(0,len(cls_cnts)):
                    Oi[j] = X[i].dot(wgts[j].transpose())
                tot_exp = np.sum(np.exp(Oi)) 
                Yi = np.exp(Oi)/tot_exp
                output = cls_cnts.index[np.argmax(Yi)]
                if output == df.iloc[i]["class"]:
                    accuracy = accuracy + 1
                    pred_counts[output] += 1
            else:
                output = sigmoid(X[i].dot(wgts[0].transpose()))         
            
                # The class label is assigned the value 1 
                if output > 0.5:
                    answer = 1
                    pred_counts[1] += 1 
                # The class label is assigned the vaue 0
                else:
                    answer = 0 
                    pred_counts[0] += 1
                if answer ==  df.iloc[i][columns.index("class")]:
                    accuracy = accuracy + 1            
            
            
        else:
            
            for layer in range(0,len(layers)):
                if layer == 0: 
                    Zh[layer] = sigmoid(X[i].dot(wgts_hdn[layer].transpose()))         
                else:
                    Zh[layer] = sigmoid(Zh[layer-1].dot(wgts_hdn[layer].transpose()))  

            if bMClass:
                for j in range(0,len(cls_cnts)):
                    Oi[j] = Zh[len(layers)-1].dot(wgts[j].transpose())
                tot_exp = np.sum(np.exp(Oi)) 
                Yi = np.exp(Oi)/tot_exp
                output = cls_cnts.index[np.argmax(Yi)]
                if output == df.iloc[i]["class"]:
                    accuracy = accuracy + 1
                    pred_counts[output] += 1
            else:
                # The class label is assigned the value 1 
                if sigmoid(Zh[len(layers)-1].dot(wgts.transpose())) > 0.5:
                    answer = 1
                    pred_counts[1] += 1 
                # The class label is assigned the vaue 0
                else:
                    answer = 0 
                    pred_counts[0] += 1
                if answer ==  df.iloc[i][columns.index("class")]:
                    accuracy = accuracy + 1

                    
       
    accuracy = accuracy / df.shape[0]

    # Calculate the probability distribution of the target based on the 
    # passed in data
    if not bMClass:
        for i in class_counts.index:
            probs.append(class_counts[i]/np.sum(class_counts))  
        cross_entropy = - ( pred_counts[0]/np.sum(pred_counts)*np.log2(probs[0]) + 
                            pred_counts[1]/np.sum(pred_counts)*np.log2(probs[1]) )
    else:
        cross_entropy = 0.0
        for i in range(0,len(class_counts.index)):
            cross_entropy = cross_entropy + pred_counts[class_counts.index[i]]/len(class_counts)*np.log2(probs[i]) 
        cross_entropy = -cross_entropy         

    return accuracy, cross_entropy





#
#        
# Function to calculate the weights in the logistic discrimination algorithm
# Gradient descent is used as per the pseudocode listed in Alpaydin textbook.
# The weights of the linear model are returned based on the number of 
# iterations.
#    
# Parameters
#   df       Data Set used for validation
#   Eta      A constant applied to delta which is then added to the weights
#   max_iter The maximum number of iterations executed to change the weight
#            values
#
def calculate_weights_2class_backprop(df= pd.DataFrame(), Eta=0.001, epochs=30):
    #df["x0"] = np.ones(len(df))
    #N(h) = df.shape[0] / (ALPHA*(df.shape[1] + 1))
    # Array with all the features excluding the label
    X=np.array(df.drop("class", axis=1))
    #Include the bias
    #X=np.append(X,np.ones([len(X),1]),1)

    
    #Number of nodes in hidden layer    
    Nh =  round(df.shape[0] / (constant.ALPHA*(df.shape[1] + 1)) )
    #Hidden layer output, input to output layer ( hidden layer to output layer )
    Zh = np.zeros(Nh)
    #Initialize weight vectors ( hidden layer to output layer )

    #Define and initialize the weight vectors ( input to hidden layer )
    #W=np.zeros_like(X[1])
    Wh = np.zeros([Nh,X.shape[1]])
    for i in range(0,Wh.shape[0]):
        for j in range(0,Wh.shape[1]):
            Wh[i][j] = random.uniform(-0.01,0.01)    
    delta_Wh = np.zeros(Nh)
    #Number of classes/labels
    K = 1 #For 2 class problem
    Yi = np.zeros([1,K])
    
    Vi = np.zeros([K,Nh])
    for i in range(0,K):
        for j in range(0,Nh):
            Vi[i][j] = random.uniform(-0.01,0.01)    
    delta_Vi = np.zeros([1,K])
        
    lc = 0
    while True and lc < epochs:

        for i in range(0, df.shape[0]):
            for h in range(0,Nh):
                Zh[h] = sigmoid(X[i].dot(Wh[h].transpose()))
            for j in range(0,K):
                Yi[j] = Zh.dot(Vi[j].transpose())
            for j in range(0,K):
                delta_Vi[j] = np.sum(Eta*(df.iloc[i]["class"]-Yi[j])*Zh)
            for h in range(0,Nh):
                for j in range(0,K):
                    inter_term = np.sum((df.iloc[i]["class"]-Yi[j])*Vi[j])
                delta_Wh[h] = np.sum(Eta*inter_term*Zh[h]*(1-Zh[h])*X[i])
            for j in range(0,K):
                Vi[j] = Vi[j] + delta_Vi[j] 
            for h in range(0,Nh):
                Wh[h] = Wh[h] + delta_Wh[h]
            
        lc += 1

    return Wh, Vi, Nh


#
# Function to verify convergence. Given a data set, the weights are
# getting calculated via gradient descent method. The test data is applied
# on this method in the cross validation run and the accuracy of prediction is
# determined. This is applied for a 2 class logistic regression classifier.
# In addition to the accuracy it also retains the cross-entropy
#    
# Parameters
#   df       Data Set used for validation
#   weights  The weights calculated from the learning
#
def verify_convergence_MLP_2class(df= pd.DataFrame(), wgts=None,wgts_hdn=None, num_nodes=0):
    
    columns = list(df.columns)
    answer = None
    accuracy = 0.0
    class_counts = df["class"].value_counts().sort_index()

    X=np.array(df.drop("class", axis=1))
    #X=np.append(X,np.ones([len(X),1]),1)
    Zh = np.zeros(num_nodes)
    
    pred_counts = [0, 0]
    
       
    for i in range(0, df.shape[0]):

        for h in range(0,num_nodes):
            Zh[h] = sigmoid(X[i].dot(wgts[h].transpose()))         
        
        # The class label is assigned the value 1 
        if sigmoid(Zh.dot(wgts_hdn.transpose())) > 0.5:
            answer = 1
            pred_counts[1] += 1 
        # The class label is assigned the vaue 0
        else:
            answer = 0 
            pred_counts[0] += 1
        if answer ==  df.iloc[i][columns.index("class")]:
            accuracy = accuracy + 1
       
    accuracy = accuracy / df.shape[0]
    probs = []
    # Calculate the probability distribution of the target based on the 
    # passed in data
    for i in class_counts.index:
        probs.append(class_counts[i]/np.sum(class_counts))  
    cross_entropy = - ( pred_counts[0]/np.sum(pred_counts)*np.log2(probs[0]) + 
                        pred_counts[1]/np.sum(pred_counts)*np.log2(probs[1]) )
    return accuracy, cross_entropy




#
# The two function chunk and cross_validation has been borrowed from
# the Data Science course conducted by JHU by Dr Andrew Stewart
#   
# Function breaks a list into n-sublists or folds.
# 
# Parameters
#   xs     input list to be folded
#   n      fold count
#    
def chunk(xs, n):
    k, m = divmod(len(xs), n)
    return [xs[i * k + min(i, m):(i + 1) * k + min(i + 1, m)]
                   for i in range(n)]

#
# Function which does cross validation by folding the data set into
# a training set and testing set. In a 5 fold cross validation every fold
# will become a test set once and the rest of the time be part of the 
# training set. .
# 
# Parameters
#   data          data_set undergoing validation
#   algorithm     Algorithm being tested ( Logistic Regression or Adaline)
#   fold_count    By default set to 5
#   Eta           Laerning factor or constant used for delta calculation     
#   max_iter      Controlling the number of iterations 
#   mClass        Binary target if true else multi class 
#   repetitions   Number of times the cross validation is done. (Redundant) 

def cross_validation(fold_count=5, repetitions=1, data=pd.DataFrame(), Eta = 0.001, max_iter=30,
                     layers = [], K=0 ):
    run_data_list = []
    run_data = {"Accuracy": 0.0, "X-Entropy": 0.0, "Learning Factor": Eta,
                "Epochs": max_iter, "Layers": layers }
    indices = list(range(len( data)))
    for _ in range(repetitions):
        random.shuffle(indices)
        folds = chunk(indices, fold_count)
        for fold in folds:
            test_data = data.iloc[fold]
            train_indices = [idx not in fold for idx in indices]
            train_data = data.iloc[train_indices]
            if K==1:
                bMulti = False
            else:
                bMulti = True
                
            Wh, Vi, layers, WIh, class_counts  = calc_wgts_backprop_multi_layers(train_data, Eta, max_iter,
                                    layers ,K)
            #run_data['model'] = weight_vector
            if len(layers) == 0:
                run_data['Accuracy'], run_data['X-Entropy']  = \
                    verify_convergence_MLP_backprop( test_data, Wh, WIh, layers, bMulti, class_counts)
            else:
                run_data['Accuracy'], run_data['X-Entropy']  = \
                    verify_convergence_MLP_backprop( test_data, Vi, WIh, layers, bMulti, class_counts)
                
            run_data_list.append(dict(run_data))


    return run_data_list

