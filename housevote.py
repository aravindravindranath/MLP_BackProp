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
names=["class",
"handicapped-infants",
"water-project-cost-sharing",
"adoption-of-the-budget-resolution",
"physician-fee-freeze",
"el-salvador-aid",
"religious-groups-in-schools",
"anti-satellite-test-ban",
"aid-to-nicaraguan-contras",
"mx-missile",
"immigration",
"synfuels-corporation-cutback",
"education-spending",
"superfund-right-to-sue",
"crime",
"duty-free-exports",
"export-administration-act-south-africa"
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
