#!/usr/bin/env python3
# coding: utf-8

# In[1]:


import os
from rdkit import Chem
import numpy as np
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
import warnings
warnings.filterwarnings("ignore", category=Warning)
import pandas as pd
from rdkit.Chem.Draw import IPythonConsole
IPythonConsole.molSize = 400,400
IPythonConsole.drawOptions.addAtomIndices = True
import MDAnalysis as mda
from rdkit.Chem import Draw
from scipy.sparse.csgraph import floyd_warshall
from rdkit.Chem import AllChem
import random
from natsort import natsorted, ns, humansorted
from tqdm import tqdm
from MDAnalysis import transformations
import re


# In[2]:


PATH = '../INIT_at/'
GRO = f'{PATH}min_system.gro'
CG_PATH = '../LHA_test'


# In[3]:


# fragment names
names = np.array(['HS1', 'HS2', 'HS3', 'HS3p', 'HS4', 'HS4p', 'HS6', 'HS6p', 'HS7',
       'HS8', 'HS9', 'HS9p', 'HS11', 'HS12', 'HS12p', 'HS13', 'HS13p',
       'HS14', 'HS14p', 'HS16', 'HS16p', 'HS17', 'HS18', 'HS19', 'HS19p',
       'HS20', 'HS20p', 'HS20fp', 'HS21', 'HS22', 'HS23', 'HS24', 'HS24p',
       'HS25', 'HS25p', 'HS26', 'HS27', 'HS27p', 'HS28', 'HS29', 'HS30',
       'HS30p', 'HS30fp', 'HS32', 'HS32p', 'HS34', 'HS34p', 'HS35',
       'HS35p'], dtype=object)


# In[4]:


# smiles for each fragment
smiles = ['AC(=O)C(O)C(E)=O',
 'ACX=CC(N)=C(O(E))C=CX',
 'AC(CCO)C(E)C(C=CCC([O-])=O)CC(=O)[O-]',
 'AC(CCO)C(E)C(C=CCC(O)=O)CC(=O)O',
 'AC(CO)C(E)C(C=CC([O-])=O)CC([O-])=O',
 'AC(CO)C(E)C(C=CC(O)=O)CC(O)=O',
 'AC(CO)C(C([O-])=O)C(E)CO',
 'AC(CO)C(C(O)=O)C(E)CO',
 'ACX=CC=C(O(E))C(O)=CX(N)',
 'ACX=CC=C(O(E))C(O)=CX',
 'AC(CO)C(C(=O)[O-])C(O)(E)',
 'AC(CO)C(C(=O)O)C(O)(E)',
 'AC(=O)OC(E)=O',
 'ACX=C(C(N)=C(CY=CXC(=C(C(O)OY)(C([O-])=O)))(O(E)))O',
 'ACX=C(C(N)=C(CY=CXC(=C(C(O)OY)(C(O)=O)))(O(E)))O',
 'AC(=O)C(E)C(O)C(O)C(C([O-])=O)O',
 'AC(=O)C(E)C(O)C(O)C(C(O)=O)O',
 'ACX=C(O)C(C([O-])=O)=C(O)C(C(E)=O)=CX(O)',
 'ACX=C(O)C(C(O)=O)=C(O)C(C(E)=O)=CX(O)',
 'ACX=C(O)C(S)=C(C([O-])=O)C(O(E))=CX(O)',
 'ACX=C(O)C(S)=C(C(O)=O)C(O(E))=CX(O)',
 'ACCCC(E)',
 'ACC(NC(E))=O',
 'ACC(C([O-])=O)C(E)',
 'ACC(C(O)=O)C(E)',
 'ACX=C(C([O-])=O)C=C(E)C(C([O-])=O)=CX',
 'ACX=C(C([O-])=O)C=C(E)C(C(O)=O)=CX',
 'ACX=C(C(O)=O)C=C(E)C(C(O)=O)=CX',
 'ACX=C(O)C(E)=C(O)C(C(OC)=O)=CX',
 'ACX=C(OC)C=C(E)C(OC)=CX(OC)',
 'ACC(C(E))=O',
 'ACC(NC(E)C([O-])=O)=O',
 'ACC(NC(E)C(O)=O)=O',
 'AC(C([O-])=O)C(O)CO(E)',
 'AC(C(O)=O)C(O)CO(E)',
 'ACC(O)C(E)',
 'ACX=CC(C([O-])=O)=C(O)C(E)=CX(O)',
 'ACX=CC(C(O)=O)=C(O)C(E)=CX(O)',
 'ACX=CC(=O)C=C(E)CX(=O)',
 'ACX=C(C=C(CY=CXC(=C(C=CY))O)(E))O',
 'ACX=C(C([O-])=O)C=C(C([O-])=O)C(E)=CX(C([O-])=O)',
 'ACX=C(C([O-])=O)C=C(C(O)=O)C(E)=CX(C([O-])=O)',
 'ACX=C(C(O)=O)C=C(C(O)=O)C(E)=CX(C(O)=O)',
 'ACX=CC(C([O-])=O)=CC(O)=CX(E)',
 'ACX=CC(C(O)=O)=CC(O)=CX(E)',
 'ACX=CYOCZ=C(O)C(NC(E))=CC(C([O-])=O)=CZCY=C(C([O-])=O)C=CX',
 'ACX=CYOCZ=C(O)C(NC(E))=CC(C(O)=O)=CZCY=C(C(O)=O)C=CX',
 'ACC(NC(C([O-])=O)C(CC(E))C)=O',
 'ACC(NC(C(O)=O)C(CC(E))C)=O']


# In[5]:


fragments_mapping = {
    'HS1': [[1, 2], [3, 4, 5], [6, 7]],
    'HS2': [[1, 6, 7], [2, 3, 4, 5], [12, 13], [8, 9, 10, 11]],
    'HS3': [[1, 2, 3, 4, 5], [9, 10, 11, 12], [13, 14, 15, 16], [6, 7, 8, 17]],
    'HS3p': [[1, 2, 3, 4, 5], [9, 10, 11, 12, 13], [14, 15, 16, 17, 18], [6, 7, 8, 19]],
    'HS4': [[1, 2, 3, 4, 15], [5, 6, 7, 11], [8, 9, 10], [12, 13, 14]],
    'HS4p': [[1, 2, 3, 4, 17], [5, 6, 7, 12], [8, 9, 10, 11], [13, 14, 15, 16]],
    'HS6': [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
    'HS6p': [[1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13]],
    'HS7': [[1, 6, 7, 8, 9], [2, 3, 4, 5], [13, 14], [10, 11, 12]],
    'HS8': [[1, 6, 7], [2, 3, 4, 5], [11, 12], [8, 9, 10]],
    'HS9': [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11]],
    'HS9p': [[1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12]],
    'HS11': [[1, 2, 3, 4, 5]],
    'HS12': [[1, 2, 3, 4], [10, 11, 12], [13, 14, 15], [16, 17, 18, 19], [21, 22], [9, 20], [5, 6, 7, 8]],
    'HS12p': [[1, 2, 3, 4], [10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20], [22, 23], [9, 21], [5, 6, 7, 8]],
    'HS13': [[1, 2, 15], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14]],
    'HS13p': [[1, 2, 16], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14, 15]],
    'HS14': [[1, 9, 10, 11], [2, 3, 4, 5], [6, 7, 8], [12, 13, 14, 15], [16, 17]],
    'HS14p': [[1, 10, 11, 12], [2, 3, 4, 5], [6, 7, 8, 9], [13, 14, 15, 16], [17, 18]],
    'HS16': [[1, 2, 3, 4], [5, 6, 7, 11], [12, 13, 14], [15, 16], [8, 9, 10]],
    'HS16p': [[1, 2, 3, 4], [5, 6, 7, 11], [12, 13, 14, 15], [16, 17], [8, 9, 10]],
    'HS17': [[1, 2, 3, 4]],
    'HS18': [[1, 2, 3], [4, 5, 6]],
    'HS19': [[1, 2, 6], [3, 4, 5]],
    'HS19p': [[1, 2, 7], [3, 4, 5, 6]],
    'HS20': [[1, 2], [6, 7, 8], [9, 10, 11], [3, 4, 5], [12, 13, 14]],
    'HS20p': [[1, 2], [6, 7, 8], [9, 10, 11, 12], [3, 4, 5], [13, 14, 15]],
    'HS20fp': [[1, 2], [7, 8, 9], [10, 11, 12, 13], [3, 4, 5, 6], [14, 15, 16]],
    'HS21': [[1, 9, 10, 11], [2, 3, 4], [5, 6, 7, 8], [12, 13, 14, 15]],
    'HS22': [[1, 5], [6, 7], [2, 8], [3, 4], [9, 10], [11, 12, 13]],
    'HS23': [[1, 2, 3, 4]],
    'HS24': [[1, 2, 3], [6, 7, 8], [4, 5, 9]],
    'HS24p': [[1, 2, 3], [6, 7, 8, 9], [4, 5, 10]],
    'HS25': [[1, 5, 6, 7], [2, 3, 4], [8, 9]],
    'HS25p': [[1, 6, 7, 8], [2, 3, 4, 5], [9, 10]],
    'HS26': [[1, 2, 3, 4, 5]],
    'HS27': [[1, 8, 9, 10], [2, 3, 4], [5, 6, 7], [11, 12, 13, 14]],
    'HS27p': [[1, 9, 10, 11], [2, 3, 4], [5, 6, 7, 8], [12, 13, 14, 15]],
    'HS28': [[1, 6, 7], [2, 3], [4, 5, 10], [8, 9]],
    'HS29': [[1, 2, 3, 4], [7, 8, 9, 10, 11], [12, 13, 14, 15], [16, 17, 18], [5, 6]],
    'HS30': [[1, 2], [3, 4, 5], [13, 14, 15], [10, 11, 12], [7, 8, 9], [6, 16]],
    'HS30p': [[1, 2], [3, 4, 5], [14, 15, 16], [11, 12, 13], [7, 8, 9, 10], [6, 17]],
    'HS30fp': [[1, 2], [3, 4, 5, 6], [15, 16, 17, 18], [12, 13, 14], [8, 9, 10, 11], [7, 19]],
    'HS32': [[1, 2, 3], [4, 8, 9], [5, 6, 7], [10, 11, 12, 13]],
    'HS32p': [[1, 2, 3], [4, 9, 10], [5, 6, 7, 8], [11, 12, 13, 14]],
    'HS34': [[1, 2, 3], [4, 5, 6], [15, 19, 20], [21, 22, 23, 24], [7, 8, 9], [16, 17, 18], [25, 26, 27], [10, 11], [13, 14], [12]],
    'HS34p': [[1, 2, 3], [4, 5, 6], [16, 21, 22], [23, 24, 25, 26], [7, 8, 9, 10], [17, 18, 19, 20], [27, 28, 29], [11, 12], [14, 15], [13]],
    'HS35': [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12, 13]],
    'HS35p': [[1, 2, 3], [4, 5, 6], [7, 8, 9, 10], [11, 12, 13, 14]]
}


# In[6]:


fragments_connections = {
    'HS1': [[0, 1], [1, 2]],
    'HS2': [[0, 1], [0, 2], [1, 2]],
    'HS3': [[0, 3], [1, 3], [2, 3]],
    'HS3p': [[0, 3], [1, 3], [2, 3]],
    'HS4': [[0, 1], [1, 2], [1, 3]],
    'HS4p': [[0, 1], [1, 2], [1, 3]],
    'HS6': [[0, 1], [1, 2]],
    'HS6p': [[0, 1], [1, 2]],
    'HS7': [[0, 1], [0, 2], [1, 2]],
    'HS8': [[0, 1], [0, 2], [1, 2]],
    'HS9': [[0, 1], [1, 2]],
    'HS9p': [[0, 1], [1, 2]],
    'HS11': [],
    'HS12': [[0, 1], [0, 3], [0, 4], [1, 2], [1, 3], [3, 4]],
    'HS12p': [[0, 1], [0, 3], [0, 4], [1, 2], [1, 3], [3, 4]],
    'HS13': [[0, 1], [1, 2], [2, 3], [3, 4]],
    'HS13p': [[0, 1], [1, 2], [2, 3], [3, 4]],
    'HS14': [[0, 1], [0, 3], [1, 2], [1, 3], [3, 4]],
    'HS14p': [[0, 1], [0, 3], [1, 2], [1, 3], [3, 4]],
    'HS16': [[0, 1], [0, 3], [1, 2], [1, 3]],
    'HS16p': [[0, 1], [0, 3], [1, 2], [1, 3]],
    'HS17': [],
    'HS18': [[0, 1]],
    'HS19': [[0, 1]],
    'HS19p': [[0, 1]],
    'HS20': [[0, 1], [0, 3], [0, 4], [1, 2], [1, 4]],
    'HS20p': [[0, 1], [0, 3], [0, 4], [1, 2], [1, 4]],
    'HS20fp': [[0, 1], [0, 3], [0, 4], [1, 2], [1, 4]],
    'HS21': [[0, 1], [0, 3], [1, 2], [1, 3]],
    'HS22': [[0, 1], [0, 2], [0, 5], [2, 3], [2, 4], [2, 5]],
    'HS23': [],
    'HS24': [[0, 2], [1, 2]],
    'HS24p': [[0, 2], [1, 2]],
    'HS25': [[0, 1], [0, 2]],
    'HS25p': [[0, 1], [0, 2]],
    'HS26': [],
    'HS27': [[0, 1], [0, 3], [1, 2], [1, 3]],
    'HS27p': [[0, 1], [0, 3], [1, 2], [1, 3]],
    'HS28': [[0, 1], [0, 2], [1, 2]],
    'HS29': [[0, 1], [0, 2], [0, 3], [1, 2], [2, 3]],
    'HS30': [[0, 1], [0, 3], [0, 5], [2, 3], [3, 5], [4, 5]],
    'HS30p': [[0, 1], [0, 3], [0, 5], [2, 3], [3, 5], [4, 5]],
    'HS30fp': [[0, 1], [0, 3], [0, 5], [2, 3], [3, 5], [4, 5]],
    'HS32': [[0, 1], [0, 3], [1, 2], [1, 3]],
    'HS32p': [[0, 1], [0, 3], [1, 2], [1, 3]],
    'HS34': [[0, 1], [0, 2], [0, 3], [1, 2], [1, 4], [2, 3], [2, 5], [3, 6]],
    'HS34p': [[0, 1], [0, 2], [0, 3], [1, 2], [1, 4], [2, 3], [2, 5], [3, 6]],
    'HS35': [[0, 1], [1, 2], [1, 3]],
    'HS35p': [[0, 1], [1, 2], [1, 3]]
}


# In[7]:


fragments_lengths = {
    'HS1': 3,
    'HS2': 4,
    'HS3': 4,
    'HS3p': 4,
    'HS4': 4,
    'HS4p': 4,
    'HS6': 3,
    'HS6p': 3,
    'HS7': 4,
    'HS8': 4,
    'HS9': 3,
    'HS9p': 3,
    'HS11': 1,
    'HS12': 7,
    'HS12p': 7,
    'HS13': 5,
    'HS13p': 5,
    'HS14': 5,
    'HS14p': 5,
    'HS16': 5,
    'HS16p': 5,
    'HS17': 1,
    'HS18': 2,
    'HS19': 2,
    'HS19p': 2,
    'HS20': 5,
    'HS20p': 5,
    'HS20fp': 5,
    'HS21': 4,
    'HS22': 6,
    'HS23': 1,
    'HS24': 3,
    'HS24p': 3,
    'HS25': 3,
    'HS25p': 3,
    'HS26': 1,
    'HS27': 4,
    'HS27p': 4,
    'HS28': 4,
    'HS29': 5,
    'HS30': 6,
    'HS30p': 6,
    'HS30fp': 6,
    'HS32': 4,
    'HS32p': 4,
    'HS34': 10,
    'HS34p': 10,
    'HS35': 4,
    'HS35p': 4
}


# In[8]:


fragments_bead_types = {
    'HS1': ['TN5a', 'TP1', 'TN5a'],
    'HS2': ['TC5', 'TC5', 'TN3ar', 'TN6d'],
    'HS3': ['P1', 'Q5n', 'Q5n', 'C4h'],
    'HS3p': ['P1', 'P2', 'P2', 'C4h'],
    'HS4': ['P1', 'C4h', 'SQ5n', 'SQ5n'],
    'HS4p': ['P1', 'C4h', 'SP2', 'SP2'],
    'HS6': ['SP1', 'Q5n', 'SP1'],
    'HS6p': ['SP1', 'P2', 'SP1'],
    'HS7': ['SN6d', 'TC5', 'TN3ar', 'TN6'],
    'HS8': ['TC5', 'TC5', 'TN3ar', 'TN6'],
    'HS9': ['SP1', 'Q5n', 'TP1'],
    'HS9p': ['SP1', 'P2', 'TP1'],
    'HS11': ['P2a'],
    'HS12': ['SN6', 'TC5', 'SQ5n', 'SP1r', 'TN3ar', 'TC5e', 'TN6d'],
    'HS12p': ['SN6', 'TC5', 'SP2', 'SP1r', 'TN3ar', 'TC5e', 'TN6d'],
    'HS13': ['SN5a', 'TP1', 'TP1', 'TP1', 'SQ5n'],
    'HS13p': ['SN5a', 'TP1', 'TP1', 'TP1', 'SP2'],
    'HS14': ['SN6', 'SN6', 'SQ5n', 'SN6', 'TN4a'],
    'HS14p': ['SN6', 'SN6', 'SP2', 'SN6', 'TN4a'],
    'HS16': ['SN6', 'SC6', 'SQ5n', 'TN3ar', 'TN6'],
    'HS16p': ['SN6', 'SC6', 'SP2', 'TN3ar', 'TN6'],
    'HS17': ['C1'],
    'HS18': ['SN5a', 'TN4'],
    'HS19': ['SC2', 'SQ5n'],
    'HS19p': ['SC2', 'SP2'],
    'HS20': ['TC5e', 'TC5', 'SQ5n', 'SQ5n', 'TC5'],
    'HS20p': ['TC5e', 'TC5', 'SP2', 'SQ5n', 'TC5'],
    'HS20fp': ['TC5e', 'TC5', 'SP2', 'SP2', 'TC5'],
    'HS21': ['SN6', 'TC5', 'N4a', 'SN6'],
    'HS22': ['TC5e', 'TN2a', 'TC5e', 'TN2a', 'TN2a', 'TC5'],
    'HS23': ['N5a'],
    'HS24': ['SN5a', 'SQ5n', 'TN4'],
    'HS24p': ['SN5a', 'SP2', 'TN4'],
    'HS25': ['SP1', 'SQ5n', 'TN3ar'],
    'HS25p': ['SP1', 'SP2', 'TN3ar'],
    'HS26': ['P1'],
    'HS27': ['SN6', 'TC5', 'SQ5n', 'SN6'],
    'HS27p': ['SN6', 'TC5', 'SP2', 'SN6'],
    'HS28': ['TC5', 'TN6a', 'TC5', 'TN6a'],
    'HS29': ['SN6', 'SN6', 'TC5', 'TC5', 'TC5e'],
    'HS30': ['TC5e', 'SQ5n', 'SQ5n', 'TC5', 'SQ5n', 'TC5e'],
    'HS30p': ['TC5e', 'SQ5n', 'SQ5n', 'TC5', 'SP2', 'TC5e'],
    'HS30fp': ['TC5e', 'SP2', 'SP2', 'TC5', 'SP2', 'TC5e'],
    'HS32': ['TC5', 'TC5', 'SQ5n', 'SN6'],
    'HS32p': ['TC5', 'TC5', 'SP2', 'SN6'],
    'HS34': ['TC5', 'TC5', 'TC5', 'SN6', 'SQ5n', 'SQ5n', 'TN4', 'TC5e', 'TC5e', 'TN2a'],
    'HS34p': ['TC5', 'TC5', 'TC5', 'SN6', 'SP2', 'SP2', 'TN4', 'TC5e', 'TC5e', 'TN2a'],
    'HS35': ['SN5a', 'TN4', 'SQ5n', 'C2'],
    'HS35p': ['SN5a', 'TN4', 'SP2', 'C2']
}


# In[9]:


fragments_charges = {
    'HS1': [0.0, 0.0, 0.0],
    'HS2': [0.0, 0.0, 0.0, 0.0],
    'HS3': [0.0, -1.0, -1.0, 0.0],
    'HS3p': [0.0, 0.0, 0.0, 0.0],
    'HS4': [0.0, 0.0, -1.0, -1.0],
    'HS4p': [0.0, 0.0, 0.0, 0.0],
    'HS6': [0.0, -1.0, 0.0],
    'HS6p': [0.0, 0.0, 0.0],
    'HS7': [0.0, 0.0, 0.0, 0.0],
    'HS8': [0.0, 0.0, 0.0, 0.0],
    'HS9': [0.0, -1.0, 0.0],
    'HS9p': [0.0, 0.0, 0.0],
    'HS11': [0.0],
    'HS12': [0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
    'HS12p': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'HS13': [0.0, 0.0, 0.0, 0.0, -1.0],
    'HS13p': [0.0, 0.0, 0.0, 0.0, 0.0],
    'HS14': [0.0, 0.0, -1.0, 0.0, 0.0],
    'HS14p': [0.0, 0.0, 0.0, 0.0, 0.0],
    'HS16': [0.0, 0.0, -1.0, 0.0, 0.0],
    'HS16p': [0.0, 0.0, 0.0, 0.0, 0.0],
    'HS17': [0.0],
    'HS18': [0.0, 0.0],
    'HS19': [0.0, -1.0],
    'HS19p': [0.0, 0.0],
    'HS20': [0.0, 0.0, -1.0, -1.0, 0.0],
    'HS20p': [0.0, 0.0, 0.0, -1.0, 0.0],
    'HS20fp': [0.0, 0.0, 0.0, 0.0, 0.0],
    'HS21': [0.0, 0.0, 0.0, 0.0],
    'HS22': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'HS23': [0.0],
    'HS24': [0.0, -1.0, 0.0],
    'HS24p': [0.0, 0.0, 0.0],
    'HS25': [0.0, -1.0, 0.0],
    'HS25p': [0.0, 0.0, 0.0],
    'HS26': [0.0],
    'HS27': [0.0, 0.0, -1.0, 0.0],
    'HS27p': [0.0, 0.0, 0.0, 0.0],
    'HS28': [0.0, 0.0, 0.0, 0.0],
    'HS29': [0.0, 0.0, 0.0, 0.0, 0.0],
    'HS30': [0.0, -1.0, -1.0, 0.0, -1.0, 0.0],
    'HS30p': [0.0, -1.0, -1.0, 0.0, 0.0, 0.0],
    'HS30fp': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'HS32': [0.0, 0.0, -1.0, 0.0],
    'HS32p': [0.0, 0.0, 0.0, 0.0],
    'HS34': [0.0, 0.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
    'HS34p': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'HS35': [0.0, 0.0, -1.0, 0.0],
    'HS35p': [0.0, 0.0, 0.0, 0.0]
}


# In[10]:


fragments_vs = {
    'HS1': [],
    'HS2': [[3, 0, 1, 2, 1, -0.843, 1.043]],
    'HS3': [],
    'HS3p': [],
    'HS4': [],
    'HS4p': [],
    'HS6': [],
    'HS6p': [],
    'HS7': [[3, 0, 1, 2, 1, -0.599, 1.068]],
    'HS8': [[3, 0, 1, 2, 1, -0.756, 0.914]],
    'HS9': [],
    'HS9p': [],
    'HS11': [],
    'HS12': [[5, 1, 4, 1, 0.481], [6, 0, 4, 5, 1, 1.212, -0.983]],
    'HS12p': [[5, 1, 4, 1, 0.48], [6, 0, 4, 5, 1, 1.212, -0.981]],
    'HS13': [],
    'HS13p': [],
    'HS14': [],
    'HS14p': [],
    'HS16': [[4, 0, 1, 3, 1, -0.81, 1.013]],
    'HS16p': [[4, 0, 1, 3, 1, -0.807, 1.0]],
    'HS17': [],
    'HS18': [],
    'HS19': [],
    'HS19p': [],
    'HS20': [],
    'HS20p': [],
    'HS20fp': [],
    'HS21': [],
    'HS22': [],
    'HS23': [],
    'HS24': [],
    'HS24p': [],
    'HS25': [],
    'HS25p': [],
    'HS26': [],
    'HS27': [],
    'HS27p': [],
    'HS28': [[3, 0, 1, 2, 1, -0.663, 0.837]],
    'HS29': [[4, 3, 0, 0.12, 1, 0.308, 2, 0.242, 3, 0.33]],
    'HS30': [],
    'HS30p': [],
    'HS30fp': [],
    'HS32': [],
    'HS32p': [],
    'HS34': [[9, 0, 3, 1, 0.552], [7, 0, 1, 2, 1, -0.203, 0.426], [8, 1, 2, 3, 1, 0.126, 0.525]],
    'HS34p': [[9, 0, 3, 1, 0.552], [7, 0, 1, 2, 1, -0.21, 0.427], [8, 1, 2, 3, 1, 0.125, 0.525]],
    'HS35': [],
    'HS35p': []
}


# In[11]:


fragments_vs = {
    'HS1': {},
    'HS2': {3: {0: -0.843, 1: 1.043, 2: 0}},
    'HS3': {},
    'HS3p': {},
    'HS4': {},
    'HS4p': {},
    'HS6': {},
    'HS6p': {},
    'HS7': {3: {0: -0.599, 1: 1.068, 2: 0}},
    'HS8': {3: {0: -0.756, 1: 0.914, 2: 0}},
    'HS9': {},
    'HS9p': {},
    'HS11': {},
    'HS12': {5: {1: 0.481, 4: 0}, 6: {0: 1.212, 4: -0.983, 5: 0}},
    'HS12p': {5: {1: 0.480, 4: 0}, 6: {0: 1.212, 4: -0.981, 5: 0}},
    'HS13': {},
    'HS13p': {},
    'HS14': {},
    'HS14p': {},
    'HS16': {4: {0: -0.810, 1: 1.013, 3: 0}},
    'HS16p': {4: {0: -0.807, 1: 1.000, 3: 0}},
    'HS17': {},
    'HS18': {},
    'HS19': {},
    'HS19p': {},
    'HS20': {},
    'HS20p': {},
    'HS20fp': {},
    'HS21': {},
    'HS22': {},
    'HS23': {},
    'HS24': {},
    'HS24p': {},
    'HS25': {},
    'HS25p': {},
    'HS26': {},
    'HS27': {},
    'HS27p': {},
    'HS28': {3: {0: -0.663, 1: 0.837, 2: 0}},
    'HS29': {4: {0: 0.120, 1: 0.308, 2: 0.242, 3: 0.33}},
    'HS30': {},
    'HS30p': {},
    'HS30fp': {},
    'HS32': {},
    'HS32p': {},
    'HS34': {9: {0: 0.552, 3: 0}, 7: {0: -0.203, 1: 0.426, 2: 0}, 8: {1: 0.126, 2: 0.525, 3: 0}},
    'HS34p': {9: {0: 0.552, 3: 0}, 7: {0: -0.210, 1: 0.427, 2: 0}, 8: {1: 0.125, 2: 0.525, 3: 0}},
    'HS35': {},
    'HS35p': {}
}


# In[12]:


#DIR = '/data/ld121/ITP_FILES_OPTIMIZED'
#for FRG in names:
#    u = mda.Universe(f'{DIR}/{FRG}.itp', format = 'ITP')
#    print("'", FRG, "': ", list(u.atoms.charges), ",", sep = "")


# In[13]:


# fragments ending with an ether group
FRG_O = ['HS2', 'HS7', 'HS8', 'HS12', 'HS12p', 'HS16', 'HS16p', 'HS25', 'HS25p']
# fragments with first and last bead having the same index
FRG_same = ['HS4', 'HS4p', 'HS11' 'HS13', 'HS13p', 'HS19', 'HS19p']


# In[14]:


# translation from RDKit to VSOMM2 + atom index before branch
HS1 = np.array([[0, 1, 2, 3, 4, 5], [1, 2, 3, np.array([4, 5]), 7, 6], 4], dtype=object)
HS2 = np.array([[0, 1, 2, 3, 4, 5, 6, 7], [1, np.array([6, 7]), 8, np.array([9, 10, 11]), 12, 13, np.array([4, 5]), np.array([2, 3])], 5], dtype=object)
HS3 = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], [1, 2, 3, np.array([4, 5]), 17, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], 4], dtype=object)
HS8 = np.array([[0, 1, 2, 3, 4, 5, 6, 7], [1, np.array([2, 3]), np.array([4, 5]), 11, 12, 8, np.array([9, 10]), np.array([6, 7])], 4], dtype=object)
HS11 = np.array([[0, 1, 2, 3, 4], [1, 2, 3, 5, 4], 3], dtype=object)
HS13 = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [1, 2, 15, 3, np.array([4, 5]), 6, np.array([7, 8]), 9, 12, 13, 14, np.array([10, 11])], 2], dtype=object)
HS14 = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], [1, 2, np.array([3, 4]), 5, 6, 7, 8, 13 , np.array([14, 15]), 12, 17, 16, 9, np.array([10, 11])], 10], dtype=object)
HS16 = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [1, 2, np.array([3, 4]), 5, np.array([6, 7]), 11, 12, 13, 14, 15, 16, 8, np.array([9, 10])], 10], dtype=object)
HS17 = np.array([[0, 1, 2, 3], [1, 2, 3, 4], 3], dtype=object)
HS18 = np.array([[0, 1, 2, 3, 4], [1, 2, np.array([4, 5]), 6, 3], 3], dtype=object)
HS19 = np.array([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6], 5], dtype=object)
HS20 = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [1, 2, 3, 4, 5, np.array([12, 13]), 14, 8, 9, 10, 11, np.array([6, 7])], 6], dtype=object)
HS21 = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [1, 9, np.array([10, 11]), 15, 12, np.array([13, 14]), 4, 5, 7, 8, 6, np.array([2, 3])], 3], dtype=object)
HS22 = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [1, 5, 6, 7, np.array([11, 12]), 13, 8, 9, 10, 2, 3, 4], 5], dtype=object)
HS23 = np.array([[0, 1, 2, 3], [1, 2, 4, 3], 2], dtype=object)
HS24 = np.array([[0, 1, 2, 3, 4, 5, 6, 7], [1, 2, np.array([4, 5]), 9, 6, 7, 8, 3], 3], dtype=object)
HS26 = np.array([[0, 1, 2, 3], [1, 2, np.array([3, 4]), 5], 3], dtype=object)
HS27 = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, np.array([2, 3]), 4, 5, 6, 7, 11, np.array([12, 13]), 14, 8, np.array([9, 10])], 8], dtype=object)
HS28 = np.array([[0, 1, 2, 3, 4, 5, 6, 7], [1, np.array([6, 7]), 8, 9, np.array([4, 5]), 10, 2, 3], 5], dtype=object)
HS29 = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [1, 2, np.array([16, 17]), 18, 6, 5, 7, np.array([10, 11]), np.array([12, 13]), np.array([14, 15]), np.array([8, 9]), np.array([3, 4])], 3], dtype=object)
HS30 = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], [1, 12, 13, 14, 15, np.array([10, 11]), 6, 7, 8, 9, 16, 2, 3, 4, 5], 10], dtype=object)
HS32 = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, np.array([2, 3]), 4, 5, 6, 7, np.array([8, 9]), 10, np.array([11, 12]), 13], 9], dtype=object)
HS34 = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21], [1, 11, 12, 13, 22, np.array([23, 24]), 21, np.array([25, 26]), 27, np.array([19, 20]), 15, 16, 17, 18, 14, 10, 6, 7, 8, 9, np.array([4, 5]), np.array([2, 3])], 8], dtype=object)
HS35 = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [1, 2, np.array([4, 5]), 6, 7, 8, 9, 10, 12, 13, 11, 3], 9], dtype=object)


# In[15]:


def def_per(k, n):
    if k + n > 9:
        per = '%'
    else:
        per = ''
    return per

def merge_smiles(sequence, names, smiles, first = 'H', last = 'H'):
    indices = []
    for FRG in sequence:
        i = np.where(names == f'{FRG}')[0][0]
        indices.append(i)
    if first == 'H':
        merge = smiles[indices[0]].replace('A','')
    elif first == 'O':
        merge = smiles[indices[0]].replace('A','O')
    elif first == 'C':
        merge = smiles[indices[0]].replace('A','C')
    merge = merge.replace('X', '1').replace('Y', '2').replace('Z', '3')
    k = 3
    for i in indices[1:]:
        working_smi = merge
        per = def_per(k, 3)
        smi = smiles[i].replace('Z', f'{per}{3 + k}')
        per = def_per(k, 2)
        smi = smi.replace('Y', f'{per}{2 + k}')
        per = def_per(k, 1)
        smi = smi.replace('X', f'{per}{1 + k}')
        k += 3
        working_smi = working_smi.replace('A','').replace('E', f'{smi}')
        merge = working_smi
    if last == 'H':
        merge = merge.replace('A','').replace('(E)','')
    elif last == 'O':
        merge = merge.replace('A','').replace('(E)','(O)')
    elif last == 'C':
        merge = merge.replace('A','').replace('(E)','(C)')
    return merge


# In[16]:


# better counter for k with if statements
# shorten the code


# In[17]:


# read itp files
def read_itps(PATH, GRO):
    directory = os.fsencode(PATH)
     
    # unfortunately, residue names are not always correct assigned in the .itp files
    l = os.listdir(directory)
    for i in range(len(l)):
        l[i] = l[i].decode()
    itp_list = []
    
    u = mda.Universe(GRO) # to get correct resnames
    
    first_atoms = []
    last_atoms = []
    first_add = []
    last_add = []
    
    # sequence of fragments
    sequence = []
    
    n = 0
    for file in natsorted(l):
        filename = os.fsdecode(file)
        if filename.endswith(".itp") and filename.startswith('HS_'):
            itp_list.append(f'{file}')
            u1 = mda.Universe(f'{PATH}{file}', topology_format='ITP')
            # read out sequence of fragments, HS1 has C1 and CA
            sequence.append(u.atoms[np.add(u1.select_atoms('name C1 or name CA and not (name CA and resname HS1)').indices, n)].resnames)
            n += len(u1.atoms) # index for last atom
            if u1.atoms[0].type == 'HC':
                first_atoms.append('H')
                first_add.append(1)
            elif u1.atoms[0].type == 'CH3':
                if u1.atoms[0].name == 'C1': # this methyl group is part of the fragment by default
                    first_atoms.append('H')
                    first_add.append(0)
                else:
                    first_atoms.append('C')
                    first_add.append(1)
            elif u1.atoms[0].type == 'H': # hydrogen of hydroxy group
                first_atoms.append('O')
                first_add.append(2)
            else:
                print(f'No rule for atom type {u1.atoms[0].type}.')
            if u1.atoms[-1].type == 'HC':
                last_atoms.append('H')
                last_add.append(1)
            elif u1.atoms[-1].type == 'CH3':
                if u1.atoms[-1].name == 'C3' or u1.atoms[-1].name == 'C4' or u1.atoms[-1].name == 'C13' or u1.atoms[-1].name == 'CD2':
                    last_atoms.append('H')
                    last_add.append(0)
                else:
                    last_atoms.append('C')
                    last_add.append(1)
            elif u1.atoms[-1].type == 'H':
                if u.atoms[n - 1].resname in FRG_O:
                    last_atoms.append('H')
                    last_add.append(1)
                else:
                    last_atoms.append('O')
                    last_add.append(2)
            continue
        else:
            continue
    return first_atoms, first_add, last_atoms, last_add, sequence, itp_list


# In[18]:


def create_smiles(sequence, names, smiles, first_atoms, last_atoms):
    merged_smiles = []
    for i in range(len(sequence)):
        merged = merge_smiles(sequence[i], names, smiles, first_atoms[i], last_atoms[i])
        merged_smiles.append(merged)
    return merged_smiles


# In[19]:


# create translated list for VSOMM2
def translate_atoms(FRAGMENTS, first = 1, last = 1, first_atom = 'H', last_atom = 'H'): # adding atoms at first or last place
    RDKIT = 0
    VSOMM2 = 1
    before = 2

    MODs = []; MOD2s = []
    vsomm_list = []
    count = [0]; count2 = 0
    for ksi in range(len(FRAGMENTS)):
        FRG = FRAGMENTS[ksi]
        # add first and last atoms
        MOD = FRG[VSOMM2].copy()
        MOD2 = FRG[RDKIT].copy()
        if ksi == 0 and first == 1: # first fragment
            MOD = np.add(MOD, 1, dtype=object) # shift all indices by 1
            if first_atom == 'H':
                MOD[0] = np.array([1, MOD[0]]) # add hydrogen
            else:
                MOD = np.insert(MOD, 0, 1) # add heavy atom
                MOD2 = np.append(MOD2, len(MOD2)) # it is like shifting all and inserting 0 at the beginning
        elif ksi == len(FRAGMENTS) - 1 and last == 1: # last fragment
            if last_atom == 'H':
                MOD[FRG[before]] = np.array([MOD[FRG[before]], np.add(MOD[FRG[before]], 1)]) # add hydrogen
            else:
                if len(MOD) == FRG[before]: # atom before branch is last atom, just append
                    MOD = np.append(MOD, np.add(MOD[FRG[before]], 1))
                    MOD2 = np.append(MOD2, len(MOD2))
                else:
                    MOD = np.insert(MOD, FRG[before] + 1, np.add(MOD[FRG[before]], 1))
                    MOD2 = np.append(MOD2, len(MOD2))
        # count hidden atoms
        for j in MOD:
            if type(j) == list or type(j) == np.ndarray:
                count2 += len(j) - 1 # hidden atoms
        for i in MOD2:
            # count hidden atoms within a fragment
            vsomm_list.append(np.add(MOD[i], count[ksi]))
            if i == FRG[before] and ksi != len(FRAGMENTS) - 1: # not last fragment: 
                count_h = 0
                m = ksi
                while m >= 0:
                    count_h += len(FRAGMENTS[m][VSOMM2])
                    m -= 1
                count_h += count2
                count.append(count_h)
                MODs.append(MOD)
                MOD2s.append(MOD2)
                break
    
    for ksi in range(len(FRAGMENTS) - 1): # leave out last fragment, treated already above
        FRG = np.flip(FRAGMENTS, axis = 0)[ksi + 1]
        for i in np.flip(MOD2s, axis = 0)[ksi]:
            if i > FRG[before]:
                vsomm_list.append(np.add(np.flip(MODs, axis = 0)[ksi][i], np.flip(count)[ksi + 1]))

    length = 0
    for FRG in FRAGMENTS:
        length += len(FRG[RDKIT])
    if first_atom != 'H':
        length += 1
    if last_atom != 'H':
        length += 1
    rdkit_list = [*range(length)]
    
    return rdkit_list, vsomm_list


# In[20]:


def translate_mapping(mapping, rdkit_list, vsomm_list):
    translation = []
    for i in mapping:
        new_ind = vsomm_list[i]
        if type(new_ind) == np.ndarray or type(new_ind) == list:
            for j in new_ind:
                translation.append(j)
        else:
            translation.append(new_ind)
    return translation


# In[21]:


def get_max(l):
    m = 0
    for i in l:
        if type(i) == list or type(i) == np.ndarray:
            for j in i:
                if type(j) == list or type(j) == np.ndarray:
                    for k in j:
                        if k > m:
                            m = k
                else:
                    if j > m:
                        m = j
        else:
            if i > m:
                m = i
    return m


# In[22]:


def return_index(vsomm_list, atom_of_interest):
    # returns index of atom in list (even, if atom of interest is incorporated in list of list)
    for i, atom in enumerate(vsomm_list):
        if type(atom) == list or type(atom) == np.ndarray:
            for sub_atom in atom:
                if sub_atom == atom_of_interest:
                    index = i
                    return index
                else:
                    continue
        else:
            if atom == atom_of_interest:
                index = i
                return index
            else:
                continue
    raise ValueError('Index not found')


# In[23]:


def remove_duplicates(bead):
    return(list(dict.fromkeys(bead)))


# In[24]:


def back_translation(mapping_vsomm, vsomm_list):
    # give mapping in vsomm picture and translate to rdkit picture
    mapping_rdkit = []
    for bead in mapping_vsomm:
        translated_bead = []
        for atom in bead:
            translated_bead.append(return_index(vsomm_list, atom))
        mapping_rdkit.append(remove_duplicates(translated_bead))
    return mapping_rdkit


# In[25]:


def list_add(FRG_mapping, n):
    # add integer to elements in list of lists
    FRG_mapping_added = []
    for bead in FRG_mapping:
        bead_added = []
        for atom in bead:
            bead_added.append(atom + n)
        FRG_mapping_added.append(bead_added)
    return FRG_mapping_added


# In[26]:


def determine_no_of_atoms(FRG_mapping):
    # number of atoms within fragment (largest index)
    max = 0
    for bead in FRG_mapping:
        for atom in bead:
            if atom > max:
                max = atom
            else:
                continue
    return max


# In[27]:


def add_at_first(indices, first_add):
    # add indices at the beginning (into first bead)
    indices_add = []
    indices_add += indices[0]
    for i in reversed(range(first_add)):
        indices_add.insert(0, i + 1)
    return [indices_add] + indices[1:]
    
def add_at_last(indices, last_add, FRG):
    # add indices at the end (into last bead)
    indices_add = []
    indices_add += indices[-1]
    print(indices_add)
    no_of_atoms = determine_no_of_atoms(indices)
    for i in range(no_of_atoms, no_of_atoms + last_add):
        indices_add.append(i + 1)
    return indices[:-1] + [indices_add]

def add_at_last(indices, last_add, FRG):
    # add indices into last bead of fragment
    # index of bead, in which last atom should be added
    index = get_max(fragments_connections[FRG]) # add this to function input
    indices_add = []
    indices_add += indices[index]
    no_of_atoms = determine_no_of_atoms(indices)
    for i in range(no_of_atoms, no_of_atoms + last_add):
        indices_add.append(i + 1)
    return indices[:index] + [indices_add] + indices[index + 1:]


# In[28]:


def create_mapping_vsomm(sequence, fragments_mapping, first_add, last_add):
    # create mapping in vsomm picture
    mapping_vsomm = []
    prev_atoms = 0
    for i, FRG in enumerate(sequence):
        indices = list_add(fragments_mapping[FRG], prev_atoms + first_add)
        if i == 0:
            indices = add_at_first(indices, first_add)
        elif i == len(sequence) - 1:
            print(indices)
            indices = add_at_last(indices, last_add, FRG) # actually add at last bead of fragment
        mapping_vsomm += indices
        prev_atoms += determine_no_of_atoms(fragments_mapping[FRG])
    return mapping_vsomm


# In[29]:


def matrix_from_list(bonds, n):
    A = np.zeros((n, n), dtype = object)
    for bond in bonds:
        index1 = bond[0]
        index2 = bond[1]
        A[index1][index2] = 1
        A[index2][index1] = 1
    return A
def create_A_matrix(sequence, fragments_connections, fragments_lengths, FRG_same):
    # crate A matrix with information on connections
    bonds = []
    prev_beads = 0
    for FRG in sequence:
        indices = list_add(fragments_connections[FRG], prev_beads)
        bonds += indices
        prev_beads += fragments_lengths[FRG]
    
    prev_beads = 0
    for i in range(len(sequence) - 1):
        if sequence[i] in FRG_same:
            i_last = prev_beads
        else:
            i_last = prev_beads + determine_no_of_atoms(fragments_connections[sequence[i]]) # take highest index, not length du to VS
        prev_beads += fragments_lengths[sequence[i]]
        i_first = prev_beads
        bonds += [[i_last, i_first]]
    total_no_of_beads = prev_beads + fragments_lengths[sequence[-1]]
    return matrix_from_list(bonds, total_no_of_beads)


# In[30]:


def determine_bead_types(sequence, fragments_bead_types):
    bead_types = []
    for FRG in sequence:
        bead_types += fragments_bead_types[FRG]
    return bead_types
def determine_charges(sequence, fragments_charges):
    charges = []
    for FRG in sequence:
        charges += fragments_charges[FRG]
    return charges


# In[31]:


def determine_ring_beads(ring_atoms, beads):
    # credit to cg_param
    ring_beads = [] 
    for ring in ring_atoms:
        cgring = [] 
        for atom in ring:
            for i,bead in enumerate(beads):
                if (atom in bead) and (i not in cgring):
                    cgring.append(i)
        ring_beads.append(cgring)
    return ring_beads


# In[32]:


def get_new_virtual_sites(sequence, fragments_vs, fragments_lengths, ring_beads):
    real = []
    virtual = []
    prev_beads = 0
    for FRG in sequence:
        vs = fragments_vs[FRG].copy()
        if vs != []:
            for vs_part in vs:
                l = vs_part.copy()
                # modify indices of vs and constructing particles
                if len(vs_part) == 5: # vs 2
                    l[0] += prev_beads
                    l[1] += prev_beads 
                    l[2] += prev_beads
                elif len(vs_part) == 7: # vs 3
                    l[0] += prev_beads
                    l[1] += prev_beads
                    l[2] += prev_beads
                    l[3] += prev_beads
                elif len(vs_part) == 10: # vs n
                    l[0] += prev_beads
                    l[2] += prev_beads
                    l[4] += prev_beads
                    l[6] += prev_beads
                    l[8] += prev_beads
                virtual.append(l)
        prev_beads += fragments_lengths[FRG]
        
    # filter out vs from real
    for ring in ring_beads:
        mod = ring.copy()
        for vs in virtual:
            if vs[0] in mod:
                mod.remove(vs[0])
        print(mod)
        real.append(mod)
    return virtual, real


# In[33]:


def increase_indices(dct, increase):
    mod = {}
    for key, value in dct.items():
        if isinstance(value, dict):
            # if dictionary, than do it again
            mod[key + increase] = increase_indices(value, increase)
        else:
            mod[key + increase] = value
    return mod


# In[34]:


def get_new_virtual_sites(sequence, fragments_vs, fragmens_lengths, ring_beads):
    real = []
    virtual = {}
    prev_beads = 0
    for FRG in sequence:
        if fragments_vs[FRG] != {}:
            mod = increase_indices(fragments_vs[FRG], prev_beads)
            virtual.update(mod)
        prev_beads += fragments_lengths[FRG]
        
    # filter out vs from real
    for ring in ring_beads:
        mod = ring.copy()
        for vs in virtual.keys():
            if vs in mod:
                mod.remove(vs)
        real.append(mod)
    return virtual, real


# In[35]:


def get_standard_masses(bead_types, virtual):
    masses = []
    for bead in bead_types:
        if bead[0] == 'T':
            masses.append(36)
        elif bead[0] == 'S':
            masses.append(54)
        else:
            masses.append(72)

    for vsite,refs in virtual.items():
        vmass = masses[vsite]
        masses[vsite] = 0
        weight = len(refs.items())
        for rsite,par in refs.items():
            masses[rsite] += vmass / weight

    return masses


# In[36]:


def create_resname_list(sequence, fragments_lengths):
    resnames = []
    for FRG in sequence:
        length = fragments_lengths[FRG]
        resnames.extend([FRG]*length)
    return resnames


# In[37]:


#sequence 
# find out the rest of the fragments which you need for LHA

# mols = []
# IPythonConsole.molSize = 1600,900
#merged_smiles = []
#for i in range(len(sequence)):
#    merged = merge_smiles(sequence[i], names, smiles, first_atoms[i], last_atoms[i])
#    merged_smiles.append(merged)
#     mols.append(Chem.MolFromSmiles(merged))
# Draw.MolsToGridImage(mols, molsPerRow=3, subImgSize=(300, 300))

# cg_param
#for i in tqdm(range(len(merged_smiles))):
#    smi = merged_smiles[i] # smiles for cg_param
#    file = itp_list[i][:-4] # name of molecules, e.g. HS_1
    #!cg_param_m3_mod.py "{smi}" "{CG_PATH}{file}.gro" "{CG_PATH}{file}.itp" 0 "{CG_PATH}{file}.npy"

# replace 'Qx' with 'Q5n' in the .itp files
# but also differentiate between other ions, like 'NCC(=O)[O-]', 'CCC(=O)[O-]', 'CC(C)C(=O)[O-]', 'O=C([O-])CO', ...
# ... or not: read their publication

#for file in itp_list:
#    !sed 's/Qx /Q5n/g' -i "{CG_PATH}{file}"
#    !sed 's/MOL    2/{file[:-4]}   2/g' -i "{CG_PATH}{file}"

# what fragments do we need?
# all_frg = []
# for mol in sequence:
#     for frg in mol:
#         all_frg.append(frg)
# print(natsorted(set(all_frg)))


# In[38]:


first_atoms, first_add, last_atoms, last_add, sequence, itp_list = read_itps(PATH,GRO)
merged_smiles = create_smiles(sequence, names, smiles, first_atoms, last_atoms)

vsomm_lists = []
rdkit_lists = []
for i, mol in enumerate(sequence):
    FRAGMENTS = []
    for fragment in mol:
        FRAGMENTS.append(globals()[fragment]) # convert strings to lists
    rdkit_list, vsomm_list = translate_atoms(FRAGMENTS, first_add[i], last_add[i], first_atoms[i], last_atoms[i])
    vsomm_lists.append(vsomm_list)
    rdkit_lists.append(rdkit_list)


# In[39]:


def get_smarts_matches(mol):
    """ 
    Imported and modified from the cg_param_m3.py script
    """
    #Get matches to SMARTS strings
    smarts_strings = {
    'S([O-])(=O)(=O)O'  :    'Q2',
    '[S;!$(*OC)]([O-])(=O)(=O)'   :    'SQ4',#SQ4
    'C[N+](C)(C)C' : 'Q2',
    'CC[N+](C)(C)[O-]' : 'P6' 
    #'C(=O)O' : 'P1'#SP1
    #'CC' : 'C2',
    #'OO' : 'P5'
    #'CCC' : 'C2',
    #'CCCC': 'C2'
    }    
    ## Add function to get rid of groups with duplicate atoms 
    matched_maps = [] 
    matched_beads = [] 
    for smarts in smarts_strings:
        matches = mol.GetSubstructMatches(Chem.MolFromSmarts(smarts))
        for match in matches:
            matched_maps.append(list(match))
            matched_beads.append(smarts_strings[smarts])

    return matched_maps,matched_beads

def get_ring_atoms(mol):
    """ 
    Imported and modified from the cg_param_m3.py script
    """
    #get ring atoms and systems of joined rings 
    rings = mol.GetRingInfo().AtomRings()
    ring_systems = []
    for ring in rings:
        ring_atoms = set(ring)
        new_systems = []
        for system in ring_systems:
            shared = len(ring_atoms.intersection(system))
            if shared:
                ring_atoms = ring_atoms.union(system)
            else:
                new_systems.append(system)
        new_systems.append(ring_atoms)
        ring_systems = new_systems

    return [list(ring) for ring in ring_systems]

def get_types(beads,mol,ring_beads,matched_maps,path_matrix):
    """ 
    Imported and modified from the cg_param_m3.py script
    """
    #loops through beads and determines bead type
    #script_path = os.path.dirname(os.path.abspath(''))
    #DG_data = read_DG_data('{}/fragments-exp.dat'.format(script_path))

    DG_data = []
    bead_types = []
    charges = []
    all_smi = []
    #h_donor,h_acceptor = get_hbonding(mol,beads)
    for i,bead in enumerate(beads):
        #qbead = sum([mol.GetAtomWithIdx(int(j)).GetFormalCharge() for j in bead])
        #charges.append(qbead)
        bead_smi = get_smi(bead,mol)
        all_smi.append(bead_smi)
        #bead_types.append(param_bead(bead,bead_smi,ring_size,frag_size,any(i in ring for ring in ring_beads),qbead,i in h_donor,i in h_acceptor,DG_data,matched_maps,path_matrix))

    #tuning = 0
    #if tuning:
    #    bead_types = tune_model(beads,bead_types,all_smi)

    return bead_types,charges,all_smi,DG_data

def get_coords(mol,beads):
    """ 
    Imported and modified from the cg_param_m3.py script
    """
    #Calculates coordinates for output gro file
    mol_Hs = Chem.AddHs(mol)
    conf = mol_Hs.GetConformer(0)

    cg_coords = []
    for bead in beads:
        coord = np.array([0.0,0.0,0.0])
        total = 0.0
        for atom in bead:
            mass = mol.GetAtomWithIdx(atom).GetMass()
            #coord += conf.GetAtomPosition(atom)*mass
            coord += conf.GetAtomPosition(atom)
            total += mass
        #coord /= (total*10.0)
        coord /= (len(bead)*10.0)
        cg_coords.append(coord)

    cg_coords_a = np.array(cg_coords)

    return cg_coords_a

def get_smi(bead,mol):
    """ 
    Imported and modified from the cg_param_m3.py script
    """
    #gets fragment smiles from list of atoms

    bead_smi = Chem.rdmolfiles.MolFragmentToSmiles(mol,bead)

    #Work out aromaticity by looking for lowercase c and heteroatoms
    lc = re.compile('[cn([nH\])os]+')
    lc = string_lst = ['c','\\[nH\\]','(?<!\\[)n','o']
    lowerlist = re.findall(r"(?=("+'|'.join(string_lst)+r"))",bead_smi)

    #Construct test rings for aromatic fragments
    if lowerlist:
        frag_size = len(lowerlist)
        #For two atoms + substituents, make a 3-membered ring
        if frag_size == 2:
            subs = bead_smi.split(''.join(lowerlist))
            for i in range(len(subs)):
                if subs[i] != '':
                    subs[i] = '({})'.format(subs[i])
            try:
                bead_smi = 'c1c{}{}{}{}cc1'.format(lowerlist[0],subs[0],lowerlist[1],subs[1])
            except:
                bead_smi = Chem.rdmolfiles.MolFragmentToSmiles(mol,bead,kekuleSmiles=True)
            if not Chem.MolFromSmiles(bead_smi): #If fragment isn't kekulisable use 5-membered ring
                bead_smi = 'c1c{}{}{}{}c1'.format(lowerlist[0],subs[0],lowerlist[1],subs[1])
        #For three atoms + substituents, make a dimer
        elif len(lowerlist) == 3:
            split1 = bead_smi.split(''.join(lowerlist[:2]))
            split2 = split1[1].split(lowerlist[2])
            subs = [split1[0],split2[0],split2[1]]
            for i in range(len(subs)):
                if subs[i] != '' and subs[i][0] != '(':
                    subs[i] = '({})'.format(subs[i])
            try:
                bead_smi = 'c1c{}{}{}{}{}{}c1'.format(lowerlist[0],subs[0],lowerlist[1],subs[1],lowerlist[2],subs[2])
            except:
                bead_smi = Chem.rdmolfiles.MolFragmentToSmiles(mol,bead,kekuleSmiles=True)

            if not Chem.MolFromSmiles(bead_smi):
                bead_smi = 'c1{}{}{}{}{}{}c1'.format(lowerlist[0],subs[0],lowerlist[1],subs[1],lowerlist[2],subs[2])

    if not Chem.MolFromSmiles(bead_smi):
        bead_smi = Chem.rdmolfiles.MolFragmentToSmiles(mol,bead,kekuleSmiles=True)

    #Standardise SMILES for lookup
    bead_smi = Chem.rdmolfiles.MolToSmiles(Chem.MolFromSmiles(bead_smi))

    return bead_smi


# In[40]:


mapping = []
resnames = []
for i, smi in enumerate(merged_smiles):
    mol_name = 'MOL'
    mol = Chem.MolFromSmiles(smi)
    print(smi)
    
    matched_maps,matched_beads = get_smarts_matches(mol)
    ring_atoms = get_ring_atoms(mol)
    #A_cg,beads,ring_beads,path_matrix = mapping(mol,ring_atoms,matched_maps,3)
    A_cg = create_A_matrix(sequence[i], fragments_connections, fragments_lengths, FRG_same)
    beads = back_translation(create_mapping_vsomm(sequence[i], fragments_mapping, first_add[i], last_add[i]), vsomm_lists[i])
    ring_beads = determine_ring_beads(ring_atoms, beads)
    non_ring = [b for b in range(len(beads)) if not any(b in ring for ring in ring_beads)]
    A_atom = np.asarray(Chem.GetAdjacencyMatrix(mol))
    path_matrix = floyd_warshall(csgraph=A_atom,directed=False)
    bead_types,charges,all_smi,DG_data = get_types(beads,mol,ring_beads,matched_maps,path_matrix)
    
    charges = determine_charges(sequence[i], fragments_charges)
    bead_types = determine_bead_types(sequence[i], fragments_bead_types)
    
    nconfs = 5
    mol = Chem.AddHs(mol)
    AllChem.EmbedMultipleConfs(mol,numConfs=nconfs,randomSeed=random.randint(1,1000),useRandomCoords=True)
    AllChem.UFFOptimizeMoleculeConfs(mol)
    coords0 = get_coords(mol,beads) # coordinates of energy minimized molecules

    virtual, real = get_new_virtual_sites(sequence[i], fragments_vs, fragments_lengths, ring_beads)
    masses = get_standard_masses(bead_types, virtual)
    resname_list = create_resname_list(sequence[i], fragments_lengths)
    #write_itp(mol_name,bead_types,coords0,charges,all_smi,A_cg,ring_beads,beads,mol,nconfs,virtual,real,masses,resname_list,f'{CG_PATH}/{itp_list[i]}')
    mapping.append(beads)
    resnames.append(resname_list)
    #write_gro(mol_name,bead_types,coords0,mol,f'{CG_PATH}/test.gro')


# In[41]:


# add bonds for unwrapping
# load atomistic coordinates
#w = 0.424
#GRO = f'LHA_test/{w}/min_system.gro'
#CG_PATH = f'LHA_test/{w}'
u = mda.Universe(f'{GRO}')
# add bonds from itp files
n = 0
for i, file in enumerate(itp_list):
    u1 = mda.Universe(f'{PATH}{file}', topology_format='ITP')
    if i == 0:
        bonds = u1.bonds.indices
        n += len(u1.atoms)
    else:
        for b in u1.bonds.indices:
            bonds = np.concatenate((bonds, np.add(u1.bonds.indices, n)))
        n += len(u1.atoms)
u.add_TopologyAttr('bonds', bonds)
# unwrap
workflow = [transformations.unwrap(u.atoms)]
u.trajectory.add_transformations(*workflow)


# In[42]:


# get total number of beads
n_beads = 0
for i, file in enumerate(itp_list):
    #mapping = np.load(f'{CG_PATH}/{file[:-4]}.npy', allow_pickle=True)
    n_beads += len(mapping[i])
n = mda.Universe.empty(n_beads, n_residues = n_beads, atom_resindex = np.arange(n_beads), residue_segindex = np.zeros(n_beads))

#u = mda.Universe(f'{GRO}')
coords = []
prev_atoms = 0
resids = []; names_gro = [] #resnames = []
for i, mol in enumerate(sequence):
    #FRAGMENTS = []
    #for fragment in mol:
    #    FRAGMENTS.append(globals()[fragment]) # convert strings to lists
    #rdkit_list, vsomm_list = translate_atoms(FRAGMENTS, first_add[i], last_add[i], first_atoms[i], last_atoms[i])
    for j, bead in enumerate(mapping[i]):
        vsomm_indices = translate_mapping(bead, rdkit_lists[i], vsomm_lists[i])
        a = u.atoms[np.add(vsomm_indices, prev_atoms - 1)]
        coords.append(a.center_of_geometry())
        resids.append(i + 1); names_gro.append(f"CG{j + 1}") #resnames.append("MOL")
    prev_atoms += get_max(vsomm_lists[i])
coords = np.array(coords)
n.load_new(coords, format = mda.coordinates.memory.MemoryReader)
print(prev_atoms)


# In[43]:


# set box size to that of the atomistic frame
n.dimensions = u.dimensions
# modify names, resnames, etc.
# wrap molecules
workflow = [transformations.wrap(n.atoms)]
n.trajectory.add_transformations(*workflow)


# In[44]:


# save mapped coordinates with calcium ions
CA = u.select_atoms("resname CA2+")
for i in range(len(CA.atoms)):
    resids.append(len(sequence) + i + 1)
    resnames.append("CA")
    names_gro.append("CA")
# merge with calcium ions
merged = mda.Merge(n.select_atoms("all"), CA)
# add dimensions
merged.dimensions = u.dimensions
# add resnames, etc.
resnames_flat = [item for sublist in resnames if isinstance(sublist, list) for item in sublist] + [item for item in resnames if not isinstance(item, list)]
merged.add_TopologyAttr('resid', resids)
merged.add_TopologyAttr('resname', resnames_flat)
merged.add_TopologyAttr('name', names_gro)
merged.atoms.write(f'{CG_PATH}/mapped.gro')


# In[45]:


# create 'water.pdb' if not already present
water_pdb = """ITLE     Gromacs Runs On Most of All Computer Systems    
REMARK    THIS IS A SIMULATION BOX    
CRYST1   10.000   10.000   10.000  90.00  90.00  90.00 P 1           1    
MODEL        1    
ATOM      1  W     W     1       0.000   0.000   0.000  1.00  0.00                
TER    
ENDMDL   
"""
file = "water.pdb"
if os.path.exists(f"{CG_PATH}/{file}") == False:
    f = open(f"{CG_PATH}/{file}", "w")
    f.write(water_pdb)
    f.close()
# number of coarse-grained water molecules
N = len(u.select_atoms('name OW')) / 4
print(f"You can solvate the structure with 'gmx insert-molecules -ci water.pdb -nmol {round(N)} -f mapped.gro -radius 0.180 -try 1000 -o solvated.gro &> solvation.log'")


# In[46]:


for file in itp_list:
    with open(f"{CG_PATH}/{file}", "r") as f:
        file_data = f.read()
    file_data = file_data.replace(f"MOL    1", f"{file[:-4]}    1")
    with open(f"{CG_PATH}/{file}", "w") as f:
        f.write(file_data)


# In[ ]:




