# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 18:08:08 2020

@author: I816596
"""

col_attrs = { "name": "", "drop": False, "categorize": False, "scale": False }
col_attrs_list = []
names = [ "ID", "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "class" ]

for name in names:
    col_attrs["name"] = name
    if name == "ID":
        col_attrs["drop"] = True
    else:
        col_attrs["drop"] = False
    col_attrs["categorize"] = False
    if name == "class":
        col_attrs["scale"] = False
    else:
        col_attrs["scale"] = True
    col_attrs_list.append(col_attrs)
    col_attrs = {}

