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
names=["date",
"plant-stand",
"precip",
"temp",
"hail",
"crop-hist",
"area-damaged",
"severity",
"seed-tmt",
"germination",
"plant-growth",
"leaves",
"leafspots-halo",
"leafspots-marg",
"leafspot-size",
"leaf-shread",
"leaf-malf",
"leaf-mild",
"stem",
"lodging",      
"stem-cankers",
"canker-lesion",
"fruiting-bodies",
"external decay",
"mycelium",
"int-discolor",
"sclerotia",
"fruit-pods",
"fruit spots",
"seed",
"mold-growth",
"seed-discolor",
"seed-size",
"shriveling",
"roots",
"class"
]

for name in names:
    col_attrs["name"] = name
    col_attrs["drop"] = False
    if name == "class":
        col_attrs["categorize"] = False
    else:
        col_attrs["categorize"] = True
    col_attrs["scale"] = False    
    col_attrs_list.append(col_attrs)
    col_attrs = {}
