# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 15:47:23 2020

@author: I816596
"""

from main import validation_runs, set_return_run

import sys

def execute(args):
    
    set_return_run(True)
    file_str = ""
    bLayer1 = bool(eval(args[1]))
    bLayer2 = bool(eval(args[2]))    
    Etas = [0.001, 0.01, 0.05, 0.1]
    iterations = [10,20,30,50,100,200]
    if (not bLayer1) and (not bLayer2):
       file_name = "choice"+args[0]+"_batch"+".txt"
       file_str = file_str + "No Layers\n"
    elif bLayer1 and (not bLayer2):
        file_name = "choice"+args[0]+"_batch_"+"layer1"+".txt"
        file_str = file_str + "One Hidden Layer\n"
    elif bLayer1 and bLayer2:
        file_name = "choice"+args[0]+"_batch_"+"layer1"+"_"+"layer2"+".txt"
        file_str = file_str + "Two Hidden Layers\n"
    for Eta in Etas:
        for iters in iterations:
            if (not bLayer1) and (not bLayer2):
               results = validation_runs(args[0], iters, Eta, [])
               file_str = file_str + str(results) + "\n"
            elif bLayer1 and (not bLayer2):
                for i in range(1,51):
                    results = validation_runs(args[0], iters, Eta, [i])
                    file_str = file_str + str(results) + "\n"
            elif bLayer1 and bLayer2:
                for i in range(1,51,5):
                    for j in range(1,51,5):
                        results = validation_runs(args[0], iters, Eta, [i, j])
                        file_str = file_str + str(results) + "\n"
                        

    f = open(file_name, "w")
    f.write(file_str)
    f.close()
    print("Open File",f.name)           
                
    #print(args[1])
#    for i in range(1,6):
#        for iter in iterations
#        validation_runs(i,i)

#        choice = str(args[0])
#        iterations = int(str(args[1]))
#        Eta = float(str(args[2]))
#        layers = list(eval(args[3]))    

if __name__ == "__main__":
    execute(sys.argv[1:])
