# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 18:08:08 2020

@author: I816596
"""

col_attrs = { "name": "", "drop": False, "categorize": False, "scale": False }
col_attrs_list = []

names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]


for name in names:
    col_attrs["name"] = name
    col_attrs["drop"] = False
    if name == "class":
        col_attrs["scale"] = False
    else:
        col_attrs["scale"] = True
    col_attrs["categorize"] = False   
    col_attrs_list.append(col_attrs)
    col_attrs = {}

