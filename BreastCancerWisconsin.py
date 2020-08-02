# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 21:50:06 2020

@author: I816596
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 18:08:08 2020

@author: I816596
"""

col_attrs = { "name": "", "drop": False, "categorize": False, "scale": False }
col_attrs_list = []
names=["sample_code_num", "clump_thickness", "cell_size_uniformity", "cell_shape_uniformity", 
                         "marginal_adhesion", "single_epithelial_cell_size", "bare_nuclei", "bland_chromatin",
                         "normal_nucleoli", "mitoses", "class"]

for name in names:
    col_attrs["name"] = name
    if name == "sample_code_num":
        col_attrs["drop"] = True
    else:
        col_attrs["drop"] = False
    if name == "class":
        col_attrs["categorize"] = False
    else:
        col_attrs["categorize"] = True
    col_attrs["scale"] = False
    col_attrs_list.append(col_attrs)
    col_attrs = {}
