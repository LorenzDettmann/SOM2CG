#!/usr/bin/env python3
# coding: utf-8

__author__ = "Lorenz Dettmann"
__email__ = "lorenz.dettmann@uni-rostock.de"
__version__ = "0.6.1"
__status__ = "Development"

import os
import argparse
import numpy as np
from rdkit import Chem
import MDAnalysis as mda
from MDAnalysis import transformations
import random
import math
import warnings
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore", category=Warning)

# smiles for each fragment
smiles = {
    'HS1': 'C(=O)C(O)C=O',
    'HS2': 'C1=CC(N)=C(O)C=C1',
    'HS3': 'C(CCO)CC(C=CCC([O-])=O)CC(=O)[O-]',
    'HS3p': 'C(CCO)CC(C=CCC(O)=O)CC(=O)O',
    'HS4': 'C(CO)CC(C=CC([O-])=O)CC([O-])=O',
    'HS4p': 'C(CO)CC(C=CC(O)=O)CC(O)=O',
    'HS6': 'C(CO)C(C([O-])=O)CCO',
    'HS6p': 'C(CO)C(C(O)=O)CCO',
    'HS7': 'C1=CC=C(O)C(O)=C1(N)',
    'HS8': 'C1=CC=C(O)C(O)=C1',
    'HS9': 'C(CO)C(C(=O)[O-])C(O)',
    'HS9p': 'C(CO)C(C(=O)O)C(O)',
    'HS11': 'C(=O)OC=O',
    'HS12': 'C1=C(C(N)=C(C2=C1C(=C(C(O)O2)(C([O-])=O)))(O))O',
    'HS12p': 'C1=C(C(N)=C(C2=C1C(=C(C(O)O2)(C(O)=O)))(O))O',
    'HS13': 'C(=O)CC(O)C(O)C(C([O-])=O)O',
    'HS13p': 'C(=O)CC(O)C(O)C(C(O)=O)O',
    'HS14': 'C1=C(O)C(C([O-])=O)=C(O)C(C=O)=C1(O)',
    'HS14p': 'C1=C(O)C(C(O)=O)=C(O)C(C=O)=C1(O)',
    'HS16': 'C1=C(O)C(S)=C(C([O-])=O)C(O)=C1(O)',
    'HS16p': 'C1=C(O)C(S)=C(C(O)=O)C(O)=C1(O)',
    'HS17': 'CCCC',
    'HS18': 'CC(NC)=O',
    'HS19': 'CC(C([O-])=O)C',
    'HS19p': 'CC(C(O)=O)C',
    'HS20': 'C1=C(C([O-])=O)C=CC(C([O-])=O)=C1',
    'HS20p': 'C1=C(C([O-])=O)C=CC(C(O)=O)=C1',
    'HS21': 'C1=C(O)C=C(O)C(C(OC)=O)=C1',
    'HS22': 'C1=C(OC)C=CC(OC)=C1(OC)',
    'HS23': 'CC(C)=O',
    'HS24': 'CC(NCC([O-])=O)=O',
    'HS24p': 'CC(NCC(O)=O)=O',
    'HS25': 'C(C([O-])=O)C(O)CO',
    'HS25p': 'C(C(O)=O)C(O)CO',
    'HS26': 'CC(O)C',
    'HS27': 'C1=CC(C([O-])=O)=C(O)C=C1(O)',
    'HS27p': 'C1=CC(C(O)=O)=C(O)C=C1(O)',
    'HS28': 'C1=CC(=O)C=CC1(=O)',
    'HS29': 'C1=C(C=C(C2=C1C(=C(C=C2))O))O',
    'HS30': 'C1=C(C([O-])=O)C=C(C([O-])=O)C=C1(C([O-])=O)',
    'HS30p': 'C1=C(C([O-])=O)C=C(C(O)=O)C=C1(C([O-])=O)',
    'HS32': 'C1=CC(C([O-])=O)=CC(O)=C1',
    'HS32p': 'C1=CC(C(O)=O)=CC(O)=C1',
    'HS34': 'C1=C2OC3=C(O)C(NC)=CC(C([O-])=O)=C3C2=C(C([O-])=O)C=C1',
    'HS34p': 'C1=C2OC3=C(O)C(NC)=CC(C(O)=O)=C3C2=C(C(O)=O)C=C1',
    'HS35': 'CC(NC(C([O-])=O)C(CC)C)=O',
    'HS35p': 'CC(NC(C(O)=O)C(CC)C)=O'
}

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
    'HS34': [[1, 2, 3], [4, 5, 6], [15, 19, 20], [21, 22, 23, 24], [7, 8, 9], [16, 17, 18], [25, 26, 27], [10, 11],
             [13, 14], [12]],
    'HS34p': [[1, 2, 3], [4, 5, 6], [16, 21, 22], [23, 24, 25, 26], [7, 8, 9, 10], [17, 18, 19, 20], [27, 28, 29],
              [11, 12], [14, 15], [13]],
    'HS35': [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12, 13]],
    'HS35p': [[1, 2, 3], [4, 5, 6], [7, 8, 9, 10], [11, 12, 13, 14]]
}

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

fragments_modify_first = {
    'HS22': {
        'pos': 0,
        'H': 'TC5'
    },
    'HS30': {
        'pos': 0,
        'H': 'TC5'
    },
    'HS30p': {
        'pos': 0,
        'H': 'TC5'
    },
    'HS30fp': {
        'pos': 0,
        'H': 'TC5'
    }
}

fragments_modify_last = {
    'HS2': {
        'pos': -2,
        'H': 'TN6',
        'C': 'SN2a'
    },
    'HS7': {
        'pos': -2,
        'H': 'TN6',
        'C': 'SN2a'
    },
    'HS8': {
        'pos': -2,
        'H': 'TN6',
        'C': 'SN2a'
    },
    'HS12': {
        'pos': -3,
        'H': 'TN6',
        'C': 'SN2a'
    },
    'HS12p': {
        'pos': -3,
        'H': 'TN6',
        'C': 'SN2a'
    },
    'HS16': {
        'pos': -2,
        'H': 'TN6',
        'C': 'SN2a'
    },
    'HS16p': {
        'pos': -2,
        'H': 'TN6',
        'C': 'SN2a'
    },
    'HS25': {
        'pos': -1,
        'H': 'TP1',
        'C': 'SN3r'
    },
    'HS25p': {
        'pos': -1,
        'H': 'TP1',
        'C': 'SN3r'
    },
    'HS30': {
        'pos': -1,
        'H': 'TC5'
    },
    'HS30p': {
        'pos': -1,
        'H': 'TC5'
    },
    'HS30fp': {
        'pos': -1,
        'H': 'TC5'
    }
}

# fragments ending with an ether group
FRG_O = ['HS2', 'HS7', 'HS8', 'HS12', 'HS12p', 'HS16', 'HS16p', 'HS25', 'HS25p']
# fragments with first and last bead having the same index, and having more than one bead (function for this possible)
FRG_same = ['HS4', 'HS4p', 'HS13', 'HS13p', 'HS19', 'HS19p']

# translation from RDKit to VSOMM2
fragments_vsomm_indices = {
    'HS1': np.array([1, 2, 3, np.array([4, 5]), 7, 6], dtype=object),
    'HS2': np.array([1, np.array([6, 7]), 8, np.array([9, 10, 11]), 12, 13, np.array([4, 5]), np.array([2, 3])],
                    dtype=object),
    'HS3': np.array([1, 2, 3, np.array([4, 5]), 17, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], dtype=object),
    'HS3p': np.array(
        [1, 2, 3, np.array([4, 5]), 19, 6, 7, 8, 9, 10, np.array([12, 13]), 11, 14, 15, 16, np.array([17, 18])],
        dtype=object),
    'HS4': np.array([1, 2, np.array([3, 4]), 15, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], dtype=object),
    'HS4p': np.array([1, 2, np.array([3, 4]), 17, 5, 6, 7, 8, np.array([10, 11]), 9, 12, 13, np.array([15, 16]), 14],
                     dtype=object),
    'HS6': np.array([1, 2, np.array([3, 4]), 5, 6, 7, 8, 12, 9, np.array([10, 11])], dtype=object),
    'HS6p': np.array([1, 2, np.array([3, 4]), 5, 6, np.array([8, 9]), 7, 13, 10, np.array([11, 12])], dtype=object),
    'HS7': np.array([1, np.array([2, 3]), np.array([4, 5]), 13, 14, 10, np.array([11, 12]), 6, np.array([7, 8, 9])],
                    dtype=object),
    'HS8': np.array([1, np.array([2, 3]), np.array([4, 5]), 11, 12, 8, np.array([9, 10]), np.array([6, 7])],
                    dtype=object),
    'HS9': np.array([1, 2, np.array([3, 4]), 5, 6, 7, 8, 11, np.array([9, 10])], dtype=object),
    'HS9p': np.array([1, 2, np.array([3, 4]), 5, 6, 7, np.array([8, 9]), 12, np.array([10, 11])], dtype=object),
    'HS11': np.array([1, 2, 3, 5, 4], dtype=object),
    'HS12': np.array([1, 2, 5, np.array([6, 7, 8]), 21, 20, 9, np.array([10, 11]), 12, 16,
                      np.array([17, 18]), 19, 13, 14, 15, 22, np.array([3, 4])], dtype=object),
    'HS12p': np.array([1, 2, 5, np.array([6, 7, 8]), 22, 21, 9, np.array([10, 11]), 12, 17,
                       np.array([18, 19]), 20, 13, np.array([15, 16]), 14, 23, np.array([3, 4])], dtype=object),
    'HS13': np.array([1, 2, 15, 3, np.array([4, 5]), 6, np.array([7, 8]), 9, 12, 13, 14, np.array([10, 11])],
                     dtype=object),
    'HS13p': np.array(
        [1, 2, 16, 3, np.array([4, 5]), 6, np.array([7, 8]), 9, 12, np.array([14, 15]), 13, np.array([10, 11])],
        dtype=object),
    'HS14': np.array([1, 2, np.array([3, 4]), 5, 6, 7, 8, 13, np.array([14, 15]), 12, 17, 16, 9, np.array([10, 11])],
                     dtype=object),
    'HS14p': np.array([1, 2, np.array([3, 4]), 5, 6, np.array([8, 9]), 7, 14, np.array([15, 16]), 13, 18, 17, 10,
                       np.array([11, 12])], dtype=object),
    'HS16': np.array([1, 2, np.array([3, 4]), 5, np.array([6, 7]), 11, 12, 13, 14, 15, 16, 8, np.array([9, 10])],
                     dtype=object),
    'HS16p': np.array([1, 2, np.array([3, 4]), 5, np.array([6, 7]), 11, 12, np.array([14, 15]), 13, 16, 17, 8,
                       np.array([9, 10])], dtype=object),
    'HS17': np.array([1, 2, 3, 4], dtype=object),
    'HS18': np.array([1, 2, np.array([4, 5]), 6, 3], dtype=object),
    'HS19': np.array([1, 2, 3, 4, 5, 6], dtype=object),
    'HS19p': np.array([1, 2, 3, np.array([5, 6]), 4, 7], dtype=object),
    'HS20': np.array(
        [1, 2, 3, 4, 5, np.array([12, 13]), 14, 8, 9, 10, 11, np.array([6, 7])], dtype=object),
    'HS20p': np.array([1, 2, 3, 4, 5, np.array([13, 14]), 15, 8, 9, np.array([11, 12]), 10, np.array([6, 7])],
                      dtype=object),
    'HS21': np.array([1, 9, np.array([10, 11]), 15, 12, np.array([13, 14]), 4, 5, 7, 8, 6, np.array([2, 3])],
                     dtype=object),
    'HS22': np.array([1, 5, 6, 7, np.array([11, 12]), 13, 8, 9, 10, 2, 3, 4],
                     dtype=object),
    'HS23': np.array([1, 2, 4, 3], dtype=object),
    'HS24': np.array([1, 2, np.array([4, 5]), 9, 6, 7, 8, 3], dtype=object),
    'HS24p': np.array([1, 2, np.array([4, 5]), 10, 6, np.array([8, 9]), 7, 3], dtype=object),
    'HS25': np.array([1, 2, 3, 4, 5, np.array([6, 7]), 8, 9], dtype=object),
    'HS25p': np.array([1, 2, np.array([4, 5]), 3, 6, np.array([7, 8]), 9, 10], dtype=object),
    'HS26': np.array([1, 2, np.array([3, 4]), 5], dtype=object),
    'HS27': np.array([1, np.array([2, 3]), 4, 5, 6, 7, 11, np.array([12, 13]), 14, 8, np.array([9, 10])], dtype=object),
    'HS27p': np.array(
        [1, np.array([2, 3]), 4, 5, np.array([7, 8]), 6, 12, np.array([13, 14]), 15, 9, np.array([10, 11])],
        dtype=object),
    'HS28': np.array([1, np.array([6, 7]), 8, 9, np.array([4, 5]), 10, 2, 3], dtype=object),
    'HS29': np.array([1, 2, np.array([16, 17]), 18, 6, 5, 7, np.array([10, 11]), np.array([12, 13]), np.array([14, 15]),
                      np.array([8, 9]), np.array([3, 4])], dtype=object),
    'HS30': np.array([1, 12, 13, 14, 15, np.array([10, 11]), 6, 7, 8, 9, 16, 2, 3, 4, 5], dtype=object),
    'HS30p': np.array([1, 13, 14, 15, 16, np.array([11, 12]), 6, 7, np.array([9, 10]), 8, 17, 2, 3, 4, 5],
                      dtype=object),
    'HS32': np.array([1, np.array([2, 3]), 4, 5, 6, 7, np.array([8, 9]), 10, np.array([11, 12]), 13], dtype=object),
    'HS32p': np.array([1, np.array([2, 3]), 4, 5, np.array([7, 8]), 6, np.array([9, 10]), 11, np.array([12, 13]), 14],
                      dtype=object),
    'HS34': np.array(
        [1, 11, 12, 13, 22, np.array([23, 24]), 21, np.array([25, 26]), 27, np.array([19, 20]), 15, 16, 17, 18,
         14, 10, 6, 7, 8, 9, np.array([4, 5]), np.array([2, 3])], dtype=object),
    'HS34p': np.array([1, 12, 13, 14, 24, np.array([25, 26]), 23, np.array([27, 28]), 29, np.array([21, 22]), 16, 17,
                       np.array([19, 20]), 18, 15, 11, 6, 7, np.array([9, 10]), 8, np.array([4, 5]), np.array([2, 3])],
                      dtype=object),
    'HS35': np.array([1, 2, np.array([4, 5]), 6, 7, 8, 9, 10, 12, 13, 11, 3], dtype=object),
    'HS35p': np.array([1, 2, np.array([4, 5]), 6, 7, np.array([9, 10]), 8, 11, 13, 14, 12, 3], dtype=object)
}


# read itp files
def read_itps(PATH, GRO):
    directory = os.fsencode(PATH)

    # unfortunately, residue names are not always correctly assigned in the .itp files
    l = os.listdir(directory)
    for i in range(len(l)):  # maybe to it with a numpy command
        l[i] = l[i].decode()
    itp_list = []

    if not os.path.exists(GRO):
        print(f"Error: The file '{GRO}' containing the atomistic coordinates does not exist.")
        abort_script()

    u = mda.Universe(GRO)  # to get correct resnames
    n_gro = len(u.select_atoms("not (resname SOLV or resname SOL or resname CA2+ or resname NA+)"))

    first_atoms = []
    last_atoms = []
    first_add = []
    last_add = []

    # sequence of fragments
    sequence = []

    n = 0
    sorted_list = sorted([file for file in l if file.startswith("HS_") and file.endswith(".itp")],
                         key=lambda x: int(x.split('_')[1].split('.')[0]))
    if not sorted_list:
        print(f"Error: No topology files of the form 'HS_*.itp' could be found in '{PATH}'.")
        abort_script()

    for file in sorted_list:
        itp_list.append(f'{file}')
        u1 = mda.Universe(f'{PATH}/{file}', topology_format='ITP')
        # read out sequence of fragments, HS1 has C1 and CA
        C1_CA_atoms = u.atoms[np.add(u1.select_atoms('name C1 or name CA').indices, n)]
        C1_CA_atoms -= C1_CA_atoms.select_atoms('name CA and resname HS1')
        sequence.append(C1_CA_atoms.resnames)

        n += len(u1.atoms)  # index for last atom
        if u1.atoms[0].type == 'HC':
            first_atoms.append('H')
            first_add.append(1)
        elif u1.atoms[0].type == 'CH3':
            if u1.atoms[0].name == 'C1':  # this methyl group is part of the fragment by default
                first_atoms.append('H')
                first_add.append(0)
            else:
                first_atoms.append('C')
                first_add.append(1)
        elif u1.atoms[0].type == 'CH2':
            first_atoms.append('H')
            first_add.append(0)
        elif u1.atoms[0].type == 'H':  # hydrogen of hydroxy group
            first_atoms.append('O')
            first_add.append(2)
        else:
            print(f'{file}: No rule for first atom type {u1.atoms[0].type}.')
            first_atoms.append('H')
            first_add.append(0)
        if u1.atoms[-1].type == 'HC':
            last_atoms.append('H')
            last_add.append(1)
        elif u1.atoms[-1].type == 'CH3':
            if (u1.atoms[-1].name == 'C3' or u1.atoms[-1].name == 'C4' or u1.atoms[-1].name == 'C13'
                    or u1.atoms[-1].name == 'CD2'):
                last_atoms.append('H')
                last_add.append(0)
            else:
                last_atoms.append('C')
                last_add.append(1)
        elif u1.atoms[-1].type == 'CH2':
            last_atoms.append('H')
            last_add.append(0)
        elif u1.atoms[-1].type == 'H':
            if u.atoms[n - 1].resname in FRG_O:
                last_atoms.append('H')
                last_add.append(1)
            else:
                last_atoms.append('O')
                last_add.append(2)
        else:
            print(f'{file}: No rule for last atom type {u1.atoms[0].type}.')
            last_atoms.append('H')
            last_add.append(0)

    if n != n_gro:
        print(f"Error: Number of HS atoms in '{GRO}' does not match with the number of atoms from the topology files "
              f"in '{PATH}/HS_*.itp'. Are some files missing?")
        abort_script()

    return first_atoms, first_add, last_atoms, last_add, sequence, itp_list


def create_macromolecule(sequence, first_atom='H', last_atom='H'):
    macro_mol = Chem.MolFromSmiles('')

    # combine fragments
    for FRG in sequence:
        n_before = macro_mol.GetNumHeavyAtoms()
        if n_before != 0:
            n_fragment_before = fragment_mol.GetNumHeavyAtoms()
        fragment_mol = Chem.MolFromSmiles(smiles[FRG])
        macro_mol = Chem.EditableMol(Chem.CombineMols(macro_mol, fragment_mol))
        if n_before != 0:
            macro_mol.AddBond(index_last + n_before - n_fragment_before, n_before, order=Chem.rdchem.BondType.SINGLE)
        macro_mol = macro_mol.GetMol()
        index_last = get_last(FRG)

    # add first heavy atom
    if first_atom != 'H':
        first_atom_mol = Chem.MolFromSmiles(first_atom)
        macro_mol = Chem.EditableMol(Chem.CombineMols(macro_mol, first_atom_mol))
        macro_mol.AddBond(0, macro_mol.GetMol().GetNumHeavyAtoms() - 1, order=Chem.rdchem.BondType.SINGLE)
        macro_mol = macro_mol.GetMol()
    # add last heavy atom
    if last_atom != 'H':
        last_atom_mol = Chem.MolFromSmiles(last_atom)
        macro_mol = Chem.EditableMol(Chem.CombineMols(macro_mol, last_atom_mol))
        macro_mol.AddBond(get_last(FRG) + n_before, macro_mol.GetMol().GetNumHeavyAtoms() - 1,
                          order=Chem.rdchem.BondType.SINGLE)
        macro_mol = macro_mol.GetMol()

    # remove additional hydrogens, which weren't removed by 'Chem.CombineMols'
    return Chem.RemoveHs(macro_mol)


def get_last(FRG):
    def find_max_in_array(array):
        if isinstance(array, np.ndarray):
            return np.max(array)
        else:
            return array

    # get index of last heavy atom before a brach
    index = np.argmax([find_max_in_array(element) for element in fragments_vsomm_indices[FRG]])
    return int(index)


def create_vsomm_list(sequence, first_add=1, last_add=1, first_atom='H', last_atom='H'):
    vsomm_list = np.array([], dtype=object)
    n_before = 0
    for FRG in sequence:
        FRG_vsomm_indices = np.copy(fragments_vsomm_indices[FRG])
        vsomm_list = np.concatenate((vsomm_list, np.add(FRG_vsomm_indices, n_before)))
        n_before += get_largest_index(fragments_vsomm_indices[FRG])
    # add first and last atoms
    if first_add > 0:
        if first_atom == 'H':
            vsomm_list = np.add(vsomm_list, 1, dtype=object)
            vsomm_list[0] = np.array([1, vsomm_list[0]])
        elif first_atom == 'C':
            vsomm_list = np.add(vsomm_list, 1, dtype=object)
            vsomm_list = np.append(vsomm_list, 1)
        elif first_atom == 'O':
            vsomm_list = np.add(vsomm_list, 2, dtype=object)
            vsomm_list = np.append(vsomm_list, 1)
            vsomm_list[-1] = np.array([1, 2])
    if last_add > 0:
        if last_atom == 'H':
            # get index of last atom
            index_last_atom = 0
            for i, FRG in enumerate(sequence):
                if i < len(sequence) - 1:
                    index_last_atom += len(fragments_vsomm_indices[FRG])
                else:
                    index_last_atom += get_last(FRG)
            if vsomm_list[index_last_atom] != get_largest_index(vsomm_list):
                print("Something went wrong when creating the VSOMM list.")
            vsomm_list[index_last_atom] = np.array([vsomm_list[index_last_atom], vsomm_list[index_last_atom] + 1])
        elif last_atom == 'C':
            vsomm_list = np.append(vsomm_list, get_largest_index(vsomm_list) + 1)
        elif last_atom == 'O':
            vsomm_list = np.append(vsomm_list, get_largest_index(vsomm_list) + 1)
            vsomm_list[-1] = np.array([get_largest_index(vsomm_list) + 1, get_largest_index(vsomm_list) + 2])
    return vsomm_list


def translate_mapping(mapping, vsomm_list):
    translation = []
    for i in mapping:
        new_ind = vsomm_list[i]
        if isinstance(new_ind, list) or isinstance(new_ind, np.ndarray):
            for j in new_ind:
                translation.append(j)
        else:
            translation.append(new_ind)
    return translation


def get_largest_index(input_list):
    # returns the largest index in a list (of lists)
    max_value = 0
    for element in input_list:
        if isinstance(element, list) or isinstance(element, np.ndarray):
            max_value = max(max_value, get_largest_index(element))
        else:
            max_value = max(max_value, element)

    return max_value


def return_index(vsomm_list, atom_of_interest):
    # returns index of atom in list (even, if atom of interest is incorporated in list of list)
    for i, atom in enumerate(vsomm_list):
        if isinstance(atom, list) or isinstance(atom, np.ndarray):
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


def remove_duplicates(bead):
    return list(dict.fromkeys(bead))


def back_translation(mapping_vsomm, vsomm_list):
    # give mapping in vsomm picture and translate to rdkit picture
    mapping_rdkit = []
    for bead in mapping_vsomm:
        translated_bead = []
        for atom in bead:
            translated_bead.append(return_index(vsomm_list, atom))
        mapping_rdkit.append(remove_duplicates(translated_bead))
    return mapping_rdkit


def list_add(FRG_mapping, n):
    # add integer to elements in list of lists
    FRG_mapping_added = []
    for bead in FRG_mapping:
        bead_added = []
        for atom in bead:
            bead_added.append(atom + n)
        FRG_mapping_added.append(bead_added)
    return FRG_mapping_added


def add_at_first(indices, first_add):
    # add indices at the beginning (into first bead)
    indices_add = []
    indices_add += indices[0]
    for i in reversed(range(first_add)):
        indices_add.insert(0, i + 1)
    return [indices_add] + indices[1:]


def add_at_last(indices, last_add, FRG):
    # add indices into last bead of fragment
    # index of bead, in which last atom should be added
    index = get_largest_index(fragments_connections[FRG])  # add this to function input
    indices_add = []
    indices_add += indices[index]
    no_of_atoms = get_largest_index(indices)
    for i in range(no_of_atoms, no_of_atoms + last_add):
        indices_add.append(i + 1)
    return indices[:index] + [indices_add] + indices[index + 1:]


def create_mapping_vsomm(sequence, fragments_mapping, first_add, last_add):
    # create mapping in vsomm picture
    mapping_vsomm = []
    prev_atoms = 0
    for i, FRG in enumerate(sequence):
        indices = list_add(fragments_mapping[FRG], prev_atoms + first_add)
        if i == 0:
            indices = add_at_first(indices, first_add)
        elif i == len(sequence) - 1:
            indices = add_at_last(indices, last_add, FRG)  # actually add at last bead of fragment
        mapping_vsomm += indices
        prev_atoms += get_largest_index(fragments_mapping[FRG])
    return mapping_vsomm


def matrix_from_list(bonds, n):
    A = np.zeros((n, n), dtype=object)
    for bond in bonds:
        index1 = bond[0]
        index2 = bond[1]
        A[index1][index2] = 1
        A[index2][index1] = 1
    return A


def create_A_matrix(sequence, fragments_connections, fragments_lengths, FRG_same):
    # create A matrix with information on connections
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
            i_last = prev_beads + get_largest_index(
                fragments_connections[sequence[i]])  # take the largest index not length, due to VS
        prev_beads += fragments_lengths[sequence[i]]
        i_first = prev_beads
        bonds += [[i_last, i_first]]
    total_no_of_beads = prev_beads + fragments_lengths[sequence[-1]]
    return matrix_from_list(bonds, total_no_of_beads)


def change_first_and_last_bead(resname_list, bead_types, first_atom, last_atom):
    # modify first bead in macromolecule, if fragment-atom pair is in the list
    if resname_list[0] in fragments_modify_first.keys():
        if first_atom in fragments_modify_first[resname_list[0]].keys():
            bead_types[fragments_modify_first[resname_list[0]]['pos']] = fragments_modify_first[resname_list[0]][
                first_atom]
        else:
            pass

    # same for last bead
    if resname_list[-1] in fragments_modify_last.keys():
        if last_atom in fragments_modify_last[resname_list[-1]].keys():
            bead_types[fragments_modify_last[resname_list[-1]]['pos']] = fragments_modify_last[resname_list[-1]][
                last_atom]
        else:
            pass


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


def determine_ring_beads(ring_atoms, beads):
    # credit to cg_param
    ring_beads = []
    for ring in ring_atoms:
        cgring = []
        for atom in ring:
            for i, bead in enumerate(beads):
                if (atom in bead) and (i not in cgring):
                    cgring.append(i)
        ring_beads.append(cgring)
    return ring_beads


def increase_indices(dct, increase):
    mod = {}
    for key, value in dct.items():
        if isinstance(value, dict):
            # if dictionary, than do it again
            mod[key + increase] = increase_indices(value, increase)
        else:
            mod[key + increase] = value
    return mod


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


def get_standard_masses(bead_types, virtual):
    masses = []
    for bead in bead_types:
        if bead[0] == 'T':
            masses.append(36)
        elif bead[0] == 'S':
            masses.append(54)
        else:
            masses.append(72)

    # sort virtual sites to keep order of hierarchy
    for vsite, refs in sorted(virtual.items(), key=lambda x: len(x[1]), reverse=True):
        vmass = masses[vsite]
        masses[vsite] = 0
        weight = len(refs.items())
        for rsite, par in refs.items():
            masses[rsite] += vmass / weight

    return masses


def create_resname_list(sequence, fragments_lengths):
    resnames = []
    for FRG in sequence:
        length = fragments_lengths[FRG]
        resnames.extend([FRG] * length)
    return resnames


def get_ring_atoms(mol):
    """ 
    Imported and modified from the cg_param_m3.py script
    """
    # get ring atoms and systems of joined rings
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


def get_coords(mol, beads, map_type):
    """ 
    Imported and modified from the cg_param_m3.py script
    """
    # Calculates coordinates for output gro file
    mol_Hs = Chem.AddHs(mol)
    conf = mol_Hs.GetConformer(0)

    cg_coords = []
    for bead in beads:
        cg_coords.append(bead_coords(bead, conf, mol, map_type))

    return np.array(cg_coords)


def write_itp(bead_types, coords0, charges, A_cg, ring_beads, beads, mol, n_confs, virtual, real,
              masses, resname_list, map_type, itp_name, name, i):
    """
    Imported and modified from the cg_param_m3.py script
    """
    # writes gromacs topology file
    with open(itp_name, 'w') as itp:
        itp.write('[moleculetype]\n')
        itp.write(f'{name}    1\n')
        # write atoms section
        itp.write('\n[atoms]\n')
        for b in range(len(bead_types)):
            itp.write(
                '{:5d}{:>6}{:5d}{:>6}{:>6}{:5d}{:>10.3f}{:>10.3f}\n'.format(b + 1, bead_types[b], 1 + i,
                                                                            resname_list[b],
                                                                            'CG' + str(b + 1), b + 1, charges[b],
                                                                            masses[b]))
        bonds, constraints, dihedrals = write_bonds(itp, A_cg, ring_beads, beads, real, mol, n_confs, map_type)
        write_angles(itp, bonds, constraints, beads, mol, n_confs, map_type)
        if dihedrals:
            write_dihedrals(itp, dihedrals, coords0)
        if virtual:
            write_virtual_sites(itp, virtual, beads)

    add_info(itp_name)


def write_bonds(itp, A_cg, ring_atoms, beads, real, mol, n_confs, map_type):
    """
    Imported and modified from the cg_param_m3.py script
    """
    # Writes [bonds] and [constraints] blocks in itp file
    # Construct bonded structures for ring systems, including dihedrals
    dihedrals = []
    for r, ring in enumerate(ring_atoms):
        dihedrals = ring_bonding(real[r], dihedrals)
    itp.write('\n[bonds]\n')
    bonds = [list(pair) for pair in np.argwhere(A_cg) if pair[1] > pair[0]]
    constraints = []
    k = 5000.0

    # Get average bond lengths from all conformers
    rs = np.zeros(len(bonds))
    coords = np.zeros((len(beads), 3))
    for conf in mol.GetConformers():
        for i, bead in enumerate(beads):
            coords[i] = bead_coords(bead, conf, mol, map_type)
        for b, bond in enumerate(bonds):
            rs[b] += np.linalg.norm(np.subtract(coords[bond[0]], coords[bond[1]])) / n_confs

    # Split into bonds and constraints, and write bonds
    con_rs = []
    for bond, r in zip(bonds, rs):
        share_ring = False
        for ring in ring_atoms:
            if bond[0] in ring and bond[1] in ring:
                share_ring = True
                constraints.append(bond)
                con_rs.append(r)
                break
        if not share_ring:
            itp.write('{:5d}{:3d}{:5d}{:10.3f}{:10.1f}\n'.format(bond[0] + 1, bond[1] + 1, 1, r, k))

    # Write constraints
    if len(constraints) > 0:
        itp.write('\n#ifdef min\n')
        k = 5000000.0
        for con, r in zip(constraints, con_rs):
            itp.write('{:5d}{:3d}{:5d}{:10.3f}{:10.1f}\n'.format(con[0] + 1, con[1] + 1, 1, r, k))

        itp.write('\n#else\n')
        itp.write('[constraints]\n')
        for con, r in zip(constraints, con_rs):
            itp.write('{:5d}{:3d}{:5d}{:10.3f}\n'.format(con[0] + 1, con[1] + 1, 1, r))
        itp.write('#endif\n')

    return bonds, constraints, dihedrals


def write_angles(itp, bonds, constraints, beads, mol, n_confs, map_type):
    """
    Imported and modified from the cg_param_m3.py script
    """
    # Writes [angles] block in itp file
    k = 250.0

    # Get list of angles
    angles = []
    for bi in range(len(bonds) - 1):
        for bj in range(bi + 1, len(bonds)):
            shared = np.intersect1d(bonds[bi], bonds[bj])
            if np.size(shared) == 1:
                if bonds[bi] not in constraints or bonds[bj] not in constraints:
                    x = [i for i in bonds[bi] if i != shared][0]
                    z = [i for i in bonds[bj] if i != shared][0]
                    angles.append([x, int(shared), z])

    # Calculate and write to file
    if angles:
        itp.write('\n[angles]\n')
        coords = np.zeros((len(beads), 3))
        thetas = np.zeros(len(angles))
        for conf in mol.GetConformers():
            for i, bead in enumerate(beads):
                coords[i] = bead_coords(bead, conf, mol, map_type)
            for a, angle in enumerate(angles):
                vec1 = np.subtract(coords[angle[0]], coords[angle[1]])
                vec1 = vec1 / np.linalg.norm(vec1)
                vec2 = np.subtract(coords[angle[2]], coords[angle[1]])
                vec2 = vec2 / np.linalg.norm(vec2)
                theta = np.arccos(np.dot(vec1, vec2))
                thetas[a] += theta

        thetas = thetas * 180.0 / (np.pi * n_confs)

        for a, t in zip(angles, thetas):
            itp.write('{:5d}{:3d}{:3d}{:5d}{:10.3f}{:10.1f}\n'.format(a[0] + 1, a[1] + 1, a[2] + 1, 2, t, k))


def write_dihedrals(itp, dihedrals, coords0):
    """
    Imported and modified from the cg_param_m3.py script
    """
    # Writes hinge dihedrals to itp file
    # Dihedrals chosen in ring_bonding
    itp.write('\n[dihedrals]\n')
    k = 500.0

    for dih in dihedrals:
        vec1 = np.subtract(coords0[dih[1]], coords0[dih[0]])
        vec2 = np.subtract(coords0[dih[2]], coords0[dih[1]])
        vec3 = np.subtract(coords0[dih[3]], coords0[dih[2]])
        vec1 = vec1 / np.linalg.norm(vec1)
        vec2 = vec2 / np.linalg.norm(vec2)
        vec3 = vec3 / np.linalg.norm(vec3)
        cross1 = np.cross(vec1, vec2)
        cross1 = cross1 / np.linalg.norm(cross1)
        cross2 = np.cross(vec2, vec3)
        cross2 = cross2 / np.linalg.norm(cross2)
        angle = np.arccos(np.dot(cross1, cross2)) * 180.0 / np.pi
        itp.write(
            '{:5d}{:3d}{:3d}{:3d}{:5d}{:10.3f}{:10.1f}\n'.format(dih[0] + 1, dih[1] + 1, dih[2] + 1, dih[3] + 1, 2,
                                                                 angle, k))


def write_virtual_sites(itp, virtual_sites, beads):
    """
    Imported and modified from the cg_param_m3.py script
    """
    # Write [virtual_sites] block to itp file
    # itp.write('\n[virtual_sitesn]\n')

    # vs_iter = sorted(virtual_sites.keys())
    vs_iter = dict(sorted(virtual_sites.items(), key=lambda item: (len(item[1]), item[0])))
    prev_length = None
    for vs in vs_iter:
        cs = sorted(virtual_sites[vs].items())
        if len(cs) != prev_length:
            if len(cs) == 2:
                itp.write('\n[virtual_sites2]\n')
            elif len(cs) == 3:
                itp.write('\n[virtual_sites3]\n')
            elif len(cs) == 4:
                itp.write('\n[virtual_sitesn]\n')
            prev_length = len(cs)
        if len(cs) == 4:
            itp.write(
                '{:5d}{:3d}{:5d}{:7.3f}{:5d}{:7.3f}{:5d}{:7.3f}{:5d}{:7.3f}\n'.format(vs + 1, 3, cs[0][0] + 1, cs[0][1],
                                                                                      cs[1][0] + 1, cs[1][1],
                                                                                      cs[2][0] + 1, cs[2][1],
                                                                                      cs[3][0] + 1, cs[3][1]))
        elif len(cs) == 3:
            itp.write(
                '{:5d}{:3d}{:3d}{:3d}{:5d}{:7.3f}{:7.3f}\n'.format(vs + 1, cs[0][0] + 1, cs[1][0] + 1, cs[2][0] + 1, 1,
                                                                   cs[0][1], cs[1][1]))
        elif len(cs) == 2:
            itp.write('{:5d}{:3d}{:3d}{:5d}{:7.3f}\n'.format(vs + 1, cs[0][0] + 1, cs[1][0] + 1, 1, cs[0][1]))

    itp.write('\n[exclusions]\n')

    done = []

    # Add exclusions between vs and all other beads
    for vs in vs_iter:
        excl = str(vs + 1)
        for i in range(len(beads)):
            if i != vs and i not in done:
                excl += ' ' + str(i + 1)
        done.append(vs)
        itp.write('{}\n'.format(excl))


def ring_bonding(real, dihedrals):
    """
    Imported and modified from the cg_param_m3.py script
    """
    # Construct inner frame and hinge dihedrals
    n_struts = len(real) - 3
    j = len(real) - 1
    k = 1
    struts = 0
    for s in range(int(math.ceil(n_struts / 2.0))):
        struts += 1
        i = (j + 1) % len(real)  # First one loops round to 0
        l = k + 1
        dihedrals.append([real[i], real[j], real[k], real[l]])
        k += 1
        if struts == n_struts:
            break
        struts += 1
        i = k - 1
        l = j - 1
        dihedrals.append([real[i], real[j], real[k], real[l]])
        j -= 1

    return dihedrals


def bead_coords(bead, conf, mol, map_type):
    """
    Imported and modified from the cg_param_m3.py script
    """
    # Get coordinates of a bead

    coords = np.array([0.0, 0.0, 0.0])
    total = 0.0
    for atom in bead:
        if map_type == 'com':
            mass = mol.GetAtomWithIdx(atom).GetMass()
            coords += conf.GetAtomPosition(atom) * mass
            total += mass
        else:
            coords += conf.GetAtomPosition(atom)
    if map_type == 'com':
        coords /= (total * 10.0)
    else:
        coords /= (len(bead) * 10.0)

    return coords


def abort_script():
    print('Aborted')
    exit()


def check_arguments(PATH, CG_PATH):
    # checks, if all arguments are properly set
    # additionally checks, if files are going to be overwritten
    if not os.path.exists(PATH):
        print(f"Error: The input directory '{PATH}' does not exist.")
        abort_script()

    try:
        os.makedirs(CG_PATH, exist_ok=True)
    except OSError as e:
        print(f"Error: The output directory '{CG_PATH}' could not be created. {e}")
        abort_script()

    output_prefix = 'HS_'
    output_suffix = '.itp'
    cg_coordinate_file = 'mapped.gro'
    topology_file = 'topol.top'

    output_files = [file for file in os.listdir(CG_PATH)
                    if (file.startswith(output_prefix) and file.endswith(output_suffix)) or file == cg_coordinate_file
                    or file == topology_file]

    if output_files:
        print(f"Warning: The given output directory '{CG_PATH}' does already contain topology files.")
        user_input = input("Would you like to continue and overwrite existing files? (y/n) ")
        if user_input.lower() != 'y':
            abort_script()


def positive_integer(value):
    int_value = int(value)
    if int_value <= 0:
        raise argparse.ArgumentTypeError(
            f"This number must be an integer greater than 0.")
    return int_value


def generate_structure_file(PATH, GRO, CG_PATH, itp_list, mapping, sequences, vsomm_lists, resnames, map_type):
    print(f"- Generating initial structure file from '{GRO}'.")
    # unwrap
    u = unwrapped_atomistic_structure(PATH, GRO, itp_list)

    # create mapped structure
    mapped = mapped_structure(u, itp_list, mapping, sequences, vsomm_lists, map_type)

    # add info
    add_residue_info(u, mapped, sequences, mapping, resnames)

    # save structure file
    mapped.atoms.write(f'{CG_PATH}/mapped.gro')
    add_info(f'{CG_PATH}/mapped.gro')

    # write water structure file
    write_water_file(CG_PATH, u)

    # write topology file
    write_top_file(CG_PATH, u, itp_list)


def unwrapped_atomistic_structure(PATH, GRO, itp_list):
    # unwraps the atomistic structure to correctly generate the mapped structure
    # to unwrap, the atomistic bonds have to be read and added
    # load atomistic coordinates
    u = mda.Universe(f'{GRO}')
    # add bonds from itp files
    n = 0
    for i, file in enumerate(itp_list):
        u1 = mda.Universe(f'{PATH}/{file}', topology_format='ITP')
        if i == 0:
            bonds = u1.bonds.indices
            n += len(u1.atoms)
        else:
            bonds = np.concatenate((bonds, np.add(u1.bonds.indices, n)))
            n += len(u1.atoms)
    u.add_TopologyAttr('bonds', bonds)
    # unwrap
    workflow = [transformations.unwrap(u.atoms)]
    u.trajectory.add_transformations(*workflow)

    return u


def mapped_structure(u, itp_list, mapping, sequences, vsomm_lists, map_type):
    # create the mapped structure (without ions) based on the unwrapped atomistic structure
    # wrapping at the end
    n_beads = 0  # get total number of beads
    for i, file in enumerate(itp_list):
        n_beads += len(mapping[i])
    n = mda.Universe.empty(n_beads, n_residues=n_beads, atom_resindex=np.arange(n_beads),
                           residue_segindex=np.zeros(n_beads))
    coords = []
    prev_atoms = 0
    for i, mol in enumerate(sequences):
        for j, bead in enumerate(mapping[i]):
            vsomm_indices = translate_mapping(bead, vsomm_lists[i])
            a = u.atoms[np.add(vsomm_indices, prev_atoms - 1)]
            if map_type == 'com':
                coords.append(a.center_of_mass())
            else:
                coords.append(a.center_of_geometry())

        prev_atoms += get_largest_index(vsomm_lists[i])
    n.load_new(np.array(coords), format=mda.coordinates.memory.MemoryReader)
    # set box size to that of the atomistic frame
    n.dimensions = u.dimensions
    # wrap molecules
    workflow = [transformations.wrap(n.atoms)]
    n.trajectory.add_transformations(*workflow)

    # add ions
    ions = u.select_atoms("resname CA2+ or resname NA+")
    if len(ions) > 0:
        # merge with calcium ions
        merged = mda.Merge(n.select_atoms("all"), ions)
    else:
        merged = n
    # add dimensions
    merged.dimensions = u.dimensions

    return merged


def add_residue_info(u, mapped, sequences, mapping, resnames):
    # add resids, resnames and names
    resids = []
    names_gro = []
    for i, mol in enumerate(sequences):
        for j, bead in enumerate(mapping[i]):
            resids.append(i + 1)
            names_gro.append(f"CG{j + 1}")

    ions = u.select_atoms("resname CA2+ or resname NA+")
    if len(ions) > 0:
        if ions.resnames[0] == "CA2+":
            resname = "CA"
        elif ions.resnames[0] == "NA+":
            resname = "NA"
        else:
            print(f"Error: Resname {ions.resnames[0]} of ions is unknown.")
            abort_script()
        for i in range(len(ions.atoms)):
            resids.append(len(sequences) + i + 1)
            resnames.append(resname)
            names_gro.append(resname)

    # add resnames, etc. and save trajectory
    resnames_flat = ([item for sublist in resnames if isinstance(sublist, list) for item in sublist]
                     + [item for item in resnames if not isinstance(item, list)])

    mapped.add_TopologyAttr('resid', resids)
    mapped.add_TopologyAttr('resname', resnames_flat)
    mapped.add_TopologyAttr('name', names_gro)


def write_water_file(CG_PATH, u):
    # generate gro file for solvation, if not already present
    water_gro = """Regular sized water particle
1
    1W      W      1   0.000   0.000   0.000
   1.00000   1.00000   1.00000"""
    file = "water.gro"
    if not os.path.exists(f"{CG_PATH}/{file}"):
        f = open(f"{CG_PATH}/{file}", "w")
        f.write(water_gro)

    # number of coarse-grained water molecules
    N = round(len(u.select_atoms('name OW')) / 4)

    print('- Done')
    if N > 0:
        print(
            f"You can solvate the structure with \'gmx insert-molecules -ci water.gro -nmol {round(N)}"
            + " -f mapped.gro -radius 0.180 -try 1000 -o solvated.gro &> solvation.log\'")


def write_top_file(CG_PATH, u, itp_list):
    topol_top = """#include "martini3.ff/martini_v3.0.0.itp"      
#include "martini3.ff/martini_v3.0.0_solvents_v1.itp"    
#include "martini3.ff/martini_v3.0.0_ions_v1.itp"\n
"""
    file = "topol.top"
    f = open(f"{CG_PATH}/{file}", "w")
    f.write(topol_top)
    for itp_file in itp_list:
        f.write(f'#include "{itp_file}"\n')
    f.write("""
[ system ]    
HS in water\n
[ molecules ]\n""")
    for itp_file in itp_list:
        f.write(f'{itp_file[:-4]}\t\t1\n')

    ions = u.select_atoms("resname CA2+ or resname NA+")
    N_ions = len(ions)
    N = round(len(u.select_atoms('name OW')) / 4)

    if N_ions > 0:
        if ions.resnames[0] == "CA2+":
            resname = "CA"
        elif ions.resnames[0] == "NA+":
            resname = "NA"
        else:
            print(f"Error: Resname {ions.resnames[0]} of ions is unknown.")
            abort_script()
        f.write(f'{resname}\t\t{N_ions}\n')

    if N > 0:
        f.write(f'W\t\t{N}')

    f.close()
    add_info(f"{CG_PATH}/{file}")


def parametrize(i, sequences, first_add, last_add, vsomm_lists, mapping, resnames, par, first_atoms, last_atoms,
                n_confs, map_type, CG_PATH, itp_list):
    beads = back_translation(create_mapping_vsomm(sequences[i], fragments_mapping, first_add[i], last_add[i]),
                             vsomm_lists[i])
    mapping.append(beads)
    resname_list = create_resname_list(sequences[i], fragments_lengths)
    resnames.append(resname_list)

    if par == 'yes':
        mol = create_macromolecule(sequences[i], first_atoms[i], last_atoms[i])
        ring_atoms = get_ring_atoms(mol)
        A_cg = create_A_matrix(sequences[i], fragments_connections, fragments_lengths, FRG_same)
        ring_beads = determine_ring_beads(ring_atoms, beads)
        charges = determine_charges(sequences[i], fragments_charges)
        bead_types = determine_bead_types(sequences[i], fragments_bead_types)
        change_first_and_last_bead(resname_list, bead_types, first_atoms[i], last_atoms[i])
        mol = Chem.AddHs(mol)
        Chem.AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, randomSeed=random.randint(1, 1000),
                                        useRandomCoords=True)
        Chem.AllChem.UFFOptimizeMoleculeConfs(mol)
        coords0 = get_coords(mol, beads, map_type)  # coordinates of energy minimized molecules
        virtual, real = get_new_virtual_sites(sequences[i], fragments_vs, fragments_lengths, ring_beads)
        masses = get_standard_masses(bead_types, virtual)
        write_itp(bead_types, coords0, charges, A_cg, ring_beads, beads, mol, n_confs, virtual, real, masses,
                  resname_list, map_type, f'{CG_PATH}/{itp_list[i]}', itp_list[i][:-4], i)


def replace_first_line(file, info):
    # replace first line with string (for replacing first line of .gro file with information)
    with open(file, 'r') as f:
        lines = f.readlines()

    lines[0] = info + " using MDAnalysis\n"

    with open(file, 'w') as f:
        f.writelines(lines)


def add_info(file):
    # add information on version number of this script into the file
    info = f"This file was generated with the martini-som script v{__version__}"
    with open(file, 'r') as f:
        content = f.read()

    if file.endswith('.top') or file.endswith('.itp'):
        info = "; " + info + "\n"
        with open(file, 'w') as f:
            f.write(info + content)
    elif file.endswith('.gro'):
        replace_first_line(file, info)
    else:
        print("Warning: Unknown file type for adding information.")


def main():
    parser = argparse.ArgumentParser(description='Martini SOM - A tool for converting atomistic Soil Organic Matter '
                                                 '(SOM) models from the Vienna Soil Organic Matter Modeler 2 (VSOMM2) '
                                                 'to a coarse-grained representation, compatible with the '
                                                 'Martini 3 force field.', add_help=False)

    # arguments
    parser.add_argument('-V', '--version', action='version', version=f'%(prog)s {__version__}',
                        help='Shows the version of the script')
    parser.add_argument('-h', '--help', action='help', help='Shows this help message')
    parser.add_argument('-nt', type=positive_integer, default=1, help='Number of threads')
    parser.add_argument('-input_dir', default=f'{os.path.dirname(os.path.abspath(__file__))}',
                        help='Path to the input directory with the atomistic topology files')
    parser.add_argument('-output_dir', default='INIT_cg',
                        help='Path to the output directory with the coarse-grained topology files')
    parser.add_argument('-n_confs', type=positive_integer, default=50,
                        help='Number of conformers to generate for the parametrization')
    parser.add_argument('-map', default='cog', choices=['cog', 'com'],
                        help='Apply center of geometry (cog) or center of mass (com) mapping')
    parser.add_argument('-parametrize', default='yes', choices=['yes', 'no'],
                        help='Parametrize the molecules, or only output the mapped structure file')

    args = parser.parse_args()
    # input and output locations
    PATH = args.input_dir
    GRO = f'{PATH}/min_system.gro'
    CG_PATH = args.output_dir
    map_type = args.map
    par = args.parametrize
    n_confs = args.n_confs
    num_threads = args.nt

    check_arguments(PATH, CG_PATH)

    first_atoms, first_add, last_atoms, last_add, sequences, itp_list = read_itps(PATH, GRO)

    vsomm_lists = []
    for i, sequence in enumerate(sequences):
        vsomm_list = create_vsomm_list(sequence, first_add[i], last_add[i], first_atoms[i], last_atoms[i])
        vsomm_lists.append(vsomm_list)

    mapping = []
    resnames = []
    if par == 'yes':
        print(f' - Generating output files for {len(sequences)} HS molecules.')

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(parametrize, i, sequences, first_add, last_add, vsomm_lists, mapping, resnames, par,
                                   first_atoms, last_atoms, n_confs, map_type, CG_PATH, itp_list) for i in
                   range(len(sequences))]
        # progress bar
        if par == 'yes':
            for _ in tqdm(as_completed(futures), total=len(sequences), ncols=120):
                pass

    generate_structure_file(PATH, GRO, CG_PATH, itp_list, mapping, sequences, vsomm_lists, resnames, map_type)


if __name__ == "__main__":
    main()
