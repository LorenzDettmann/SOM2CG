"""
MIT License

Copyright (c) 2024 Lorenz Friedrich Dettmann

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Parts of this code are based on work by Mark A. Miller and coworkers, used with permission.
These parts are subject to the following citations:

T. D. Potter, N. Haywood, A. Teixeira, G. Hodges, E. L. Barrett, and M. A. Miller
Partitioning into phosphatidylcholine–cholesterol membranes: liposome measurements, coarse-grained simulations,
and implications for bioaccumulation
Environmental Science: Processes & Impacts, Issue 6 (2023), https://doi.org/10.1039/D3EM00081H

T. D. Potter, E. L. Barrett, and M. A. Miller
Automated Coarse-Grained Mapping Algorithm for the Martini Force Field and Benchmarks for Membrane Water Partitioning
J. Chem. Theory Comput., 17 (2021), pp. 5777−5791, https://doi.org/10.1021/acs.jctc.1c00322

Please cite these works if you use this code in your research.
We thank Mark. A. Miller and coworkers for their contributions.
"""

__author__ = "Lorenz Dettmann"
__email__ = "lorenz.dettmann@uni-rostock.de"
__version__ = "0.11.0"
__licence__ = "MIT"

import numpy as np

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
    'HS35p': 'CC(NC(C(O)=O)C(CC)C)=O',
    'HS37': 'C(=O)CC(O)C(O)C(CC([O-])=O)O',
    'HS37p': 'C(=O)CC(O)C(O)C(CC(O)=O)O'
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
    'HS28': [[1, 6, 7], [2, 3], [8, 9], [4, 5, 10]],
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
    'HS35p': [[1, 2, 3], [4, 5, 6], [7, 8, 9, 10], [11, 12, 13, 14]],
    'HS37': [[1, 2, 16], [3, 4, 5], [6, 7, 8], [9, 10, 11, 12], [13, 14, 15]],
    'HS37p': [[1, 2, 17], [3, 4, 5], [6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
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
    'HS28': [[0, 1], [0, 2], [0, 3], [1, 3], [2, 3]],
    'HS29': [[0, 1], [0, 2], [0, 3], [1, 2], [2, 3]],
    'HS30': [[0, 1], [0, 3], [0, 5], [2, 3], [3, 5], [4, 5]],
    'HS30p': [[0, 1], [0, 3], [0, 5], [2, 3], [3, 5], [4, 5]],
    'HS30fp': [[0, 1], [0, 3], [0, 5], [2, 3], [3, 5], [4, 5]],
    'HS32': [[0, 1], [0, 3], [1, 2], [1, 3]],
    'HS32p': [[0, 1], [0, 3], [1, 2], [1, 3]],
    'HS34': [[0, 1], [0, 2], [0, 3], [1, 2], [1, 4], [2, 3], [2, 5], [3, 6]],
    'HS34p': [[0, 1], [0, 2], [0, 3], [1, 2], [1, 4], [2, 3], [2, 5], [3, 6]],
    'HS35': [[0, 1], [1, 2], [1, 3]],
    'HS35p': [[0, 1], [1, 2], [1, 3]],
    'HS37': [[0, 1], [1, 2], [2, 3], [3, 4]],
    'HS37p': [[0, 1], [1, 2], [2, 3], [3, 4]]
}

fragments_bond_fc = {
    'HS1': [11712.83, 11712.83],
    'HS2': [None, None, None],
    'HS3': [2649.18, 5538.26, 6856.66],
    'HS3p': [3000.46, 8037.8, 11722.89],
    'HS4': [5656.14, 8490.58, 5198.27],
    'HS4p': [6490.11, 6873.73, 3569.32],
    'HS6': [1543.4, 1543.4],
    'HS6p': [2116.76, 2116.76],
    'HS7': [None, None, None],
    'HS8': [None, None, None],
    'HS9': [2058.89, 2021.12],
    'HS9p': [1743.39, 1852.26],
    'HS11': [],
    'HS12': [None, None, None, 5534.56, None, None],
    'HS12p': [None, None, None, 6820.82, None, None],
    'HS13': [6205.59, 4532.58, 4532.58, 7827.56],
    'HS13p': [5731.67, 4145.78, 4145.78, 8602.67],
    'HS14': [None, None, 7262.23, None, 7706.87],
    'HS14p': [None, None, 5811.04, None, 1343.54],
    'HS16': [None, None, 8879.44, None],
    'HS16p': [None, None, 7778.23, None],
    'HS17': [],
    'HS18': [8749.05],
    'HS19': [8237.89],
    'HS19p': [10041.67],
    'HS20': [None, 7624.38, None, 7641.98, None],
    'HS20p': [None, 8199.64, None, 8378.75, None],
    'HS20fp': [None, 8671.78, None, 9111.25, None],
    'HS21': [None, None, 6623.24, None],
    'HS22': [6734.82, None, None, 9365.73, 9365.73, None],
    'HS23': [],
    'HS24': [7432.57, 6620.52],
    'HS24p': [8549.98, 10374.84],
    'HS25': [2276.35, 6639.96],
    'HS25p': [4247.43, 6334.48],
    'HS26': [],
    'HS27': [None, None, 7089.49, None],
    'HS27p': [None, None, 6865.2, None],
    'HS28': [None, None, None, None, None],
    'HS29': [None, None, None, None, None],
    'HS30': [9747.73, None, None, 7427.81, None, 9747.73],
    'HS30p': [6159.02, None, None, 9148.34, None, 6159.02],
    'HS30fp': [8831.86, None, None, 6518.41, None, 8831.86],
    'HS32': [None, None, 7734.11, None],
    'HS32p': [None, None, 6054.9, None],
    'HS34': [None, None, None, None, 7020.35, None, 7020.35, 7633.4],
    'HS34p': [None, None, None, None, 5912.08, None, 5912.08, 7308.46],
    'HS35': [7800.78, 6948.89, 5507.46],
    'HS35p':  [8248.41, 6137.88, 7201.04],
    'HS37': [7118.96, 7852.93, 7053.94, 1712.19],
    'HS37p': [6486.28, 6397.94, 6510.72, 2737.36]
}

fragments_angles_fc = {
    'HS1': {
        tuple([0, 1, 2]): 122.49,
    },
    'HS2': {},
    'HS3': {
        tuple([0, 3, 1]): 22.48,
        tuple([0, 3, 2]): 56.75,
        tuple([1, 3, 2]): 32.88,
    },
    'HS3p': {
        tuple([0, 3, 1]): 22.45,
        tuple([0, 3, 2]): 40.36,
        tuple([1, 3, 2]): 18.28,
    },
    'HS4': {
        tuple([0, 1, 2]): 53.14,
        tuple([0, 1, 3]): 34.56,
        tuple([2, 1, 3]): 35.08,
    },
    'HS4p': {
        tuple([0, 1, 2]): 70.43,
        tuple([0, 1, 3]): 29.63,
        tuple([2, 1, 3]): 28.17,
    },
    'HS6': {
        tuple([0, 1, 2]): 36.69,
    },
    'HS6p': {
        tuple([0, 1, 2]): 63.46,
    },
    'HS7': {},
    'HS8': {},
    'HS9': {
        tuple([0, 1, 2]): 48.1,
    },
    'HS9p': {
        tuple([0, 1, 2]): 55.53,
    },
    'HS11': {},
    'HS12': {
        tuple([0, 1, 2]): 78.61,
        tuple([2, 1, 3]): 285.61,
    },
    'HS12p': {
        tuple([0, 1, 2]): 75.62,
        tuple([2, 1, 3]): 149.7,
    },
    'HS13': {
        tuple([0, 1, 2]): 162.31,
        tuple([1, 2, 3]): 47.86,
        tuple([2, 3, 4]): 114.14,
    },
    'HS13p': {
        tuple([0, 1, 2]): 146.65,
        tuple([1, 2, 3]): 59.93,
        tuple([2, 3, 4]): 149.39,
    },
    'HS14': {
        tuple([0, 1, 2]): 247.77,
        tuple([2, 1, 3]): 209.28,
        tuple([0, 3, 4]): 139.66,
        tuple([1, 3, 4]): 201.22,
    },
    'HS14p': {
        tuple([0, 1, 2]): 143.92,
        tuple([2, 1, 3]): 180.35,
        tuple([0, 3, 4]): 69.23,
        tuple([1, 3, 4]): 76.56,
    },
    'HS16': {
        tuple([0, 1, 2]): 231.61,
        tuple([2, 1, 3]): 311.98,
    },
    'HS16p': {
        tuple([0, 1, 2]): 156.35,
        tuple([2, 1, 3]): 228.8,
    },
    'HS17': {},
    'HS18': {},
    'HS19': {},
    'HS19p': {},
    'HS20': {
        tuple([0, 1, 2]): 264.14,
        tuple([2, 1, 4]): 272.48,
        tuple([1, 0, 3]): 158.15,
        tuple([3, 0, 4]): 186.23,
    },
    'HS20p': {
        tuple([0, 1, 2]): 204.24,
        tuple([2, 1, 4]): 245.9,
        tuple([1, 0, 3]): 141.12,
        tuple([3, 0, 4]): 263.3,
    },
    'HS20fp': {
        tuple([0, 1, 2]): 241.03,
        tuple([2, 1, 4]): 188.94,
        tuple([1, 0, 3]): 151.79,
        tuple([3, 0, 4]): 177.33,
    },
    'HS21': {
        tuple([0, 1, 2]): 210.8,
        tuple([2, 1, 3]): 209.32,
    },
    'HS22': {
        tuple([1, 0, 2]): 65.49,
        tuple([1, 0, 5]): 87.56,
        tuple([3, 2, 0]): 268.74,
        tuple([3, 2, 5]): 326.12,
        tuple([4, 2, 0]): 303.29,
        tuple([4, 2, 5]): 256.83,
        tuple([3, 2, 4]): 203.98,
    },
    'HS23': {},
    'HS24': {
        tuple([0, 2, 1]): 20.0,
    },
    'HS24p': {
        tuple([0, 2, 1]): 14.64,
    },
    'HS25': {
        tuple([1, 0, 2]): 27.67,
    },
    'HS25p': {
        tuple([1, 0, 2]): 25.8,
    },
    'HS26': {},
    'HS27': {
        tuple([0, 1, 2]): 185.51,
        tuple([2, 1, 3]): 175.72,
    },
    'HS27p': {
        tuple([0, 1, 2]): 108.4,
        tuple([2, 1, 3]): 172.95,
    },
    'HS28': {},
    'HS29': {},
    'HS30': {
        tuple([1, 0, 3]): 196.53,
        tuple([1, 0, 5]): 260.07,
        tuple([2, 3, 0]): 166.7,
        tuple([2, 3, 5]): 243.47,
        tuple([0, 5, 4]): 204.65,
        tuple([3, 5, 4]): 228.05,
    },
    'HS30p': {
        tuple([1, 0, 3]): 269.44,
        tuple([1, 0, 5]): 224.98,
        tuple([2, 3, 0]): 301.63,
        tuple([2, 3, 5]): 281.7,
        tuple([0, 5, 4]): 327.98,
        tuple([3, 5, 4]): 191.82,
    },
    'HS30fp': {
        tuple([1, 0, 3]): 216.27,
        tuple([1, 0, 5]): 179.35,
        tuple([2, 3, 0]): 277.15,
        tuple([2, 3, 5]): 210.14,
        tuple([0, 5, 4]): 286.69,
        tuple([3, 5, 4]): 189.04,
    },
    'HS32': {
        tuple([0, 1, 2]): 214.13,
        tuple([2, 1, 3]): 222.08,
    },
    'HS32p': {
        tuple([0, 1, 2]): 171.49,
        tuple([2, 1, 3]): 103.04,
    },
    'HS34': {
        tuple([0, 1, 4]): 228.72,
        tuple([3, 2, 5]): 175.08,
        tuple([2, 1, 4]): 229.83,
        tuple([1, 2, 5]): 229.83,
        tuple([2, 3, 6]): 269.75,
        tuple([0, 3, 6]): 160.58,
        tuple([0, 2, 5]): 208.73,
    },
    'HS34p': {
        tuple([0, 1, 4]): 154.75,
        tuple([3, 2, 5]): 151.8,
        tuple([2, 1, 4]): 114.6,
        tuple([1, 2, 5]): 114.6,
        tuple([2, 3, 6]): 162.82,
        tuple([0, 3, 6]): 141.21,
        tuple([0, 2, 5]): 348.81,
    },
    'HS35': {
        tuple([0, 1, 2]): 47.92,
        tuple([0, 1, 3]): 72.06,
        tuple([2, 1, 3]): 181.38,
    },
    'HS35p': {
        tuple([0, 1, 2]): 45.65,
        tuple([0, 1, 3]): 33.4,
        tuple([2, 1, 3]): 140.84,
    },
    'HS37': {
        tuple([0, 1, 2]): 144.57,
        tuple([1, 2, 3]): 108.60,
        tuple([2, 3, 4]): 13.19,
    },
    'HS37p': {
        tuple([0, 1, 2]): 143.02,
        tuple([1, 2, 3]): 224.22,
        tuple([2, 3, 4]): 45.49,
    }
}

fragments_dihedral_angles_fc = {
    'HS1': {},
    'HS2': {},
    'HS3': {},
    'HS3p': {},
    'HS4': {},
    'HS4p': {},
    'HS6': {},
    'HS6p': {},
    'HS7': {},
    'HS8': {},
    'HS9': {},
    'HS9p': {},
    'HS11': {},
    'HS12': {
        tuple([0, 4, 3, 1]): 44.29,
    },
    'HS12p': {
        tuple([0, 4, 3, 1]): 92.34,
    },
    'HS13': {},
    'HS13p': {},
    'HS14': {},
    'HS14p': {},
    'HS16': {},
    'HS16p': {},
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
    'HS28': {
        tuple([0, 1, 3, 2]): 169.26,
    },
    'HS29': {
        tuple([0, 3, 2, 1]): 123.33,
    },
    'HS30': {},
    'HS30p': {},
    'HS30fp': {},
    'HS32': {},
    'HS32p': {},
    'HS34': {
        tuple([0, 1, 2, 3]): 108.05,
    },
    'HS34p': {
        tuple([0, 1, 2, 3]): 71.81,
    },
    'HS35': {},
    'HS35p': {},
    'HS37': {},
    'HS37p': {}
}

fragments_exclusions = {
    'HS1': [],
    'HS2':
        [[4, 1, 2, 3]],
    'HS3': [],
    'HS3p': [],
    'HS4': [],
    'HS4p': [],
    'HS6': [],
    'HS6p': [],
    'HS7':
        [[4, 1, 2, 3]],
    'HS8':
        [[4, 1, 2, 3]],
    'HS9': [],
    'HS9p': [],
    'HS11': [],
    'HS12':
        [[7, 1, 2, 3, 4, 5, 6],
         [6, 1, 2, 3, 4, 5]],
    'HS12p':
        [[7, 1, 2, 3, 4, 5, 6],
         [6, 1, 2, 3, 4, 5]],
    'HS13': [],
    'HS13p': [],
    'HS14': [],
    'HS14p': [],
    'HS16':
        [[5, 1, 2, 3, 4]],
    'HS16p':
        [[5, 1, 2, 3, 4]],
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
    'HS28':
        [[2, 3]],
    'HS29':
        [[5, 1, 2, 3, 4],
         [2, 4]],
    'HS30': [],
    'HS30p': [],
    'HS30fp': [],
    'HS32': [],
    'HS32p': [],
    'HS34':
        [[10, 1, 2, 3, 4, 5, 6, 7, 8, 9],
         [9, 1, 2, 3, 4, 5, 6, 7, 8],
         [8, 1, 2, 3, 4, 5, 6, 7]],
    'HS34p':
        [[10, 1, 2, 3, 4, 5, 6, 7, 8, 9],
         [9, 1, 2, 3, 4, 5, 6, 7, 8],
         [8, 1, 2, 3, 4, 5, 6, 7]],
    'HS35': [],
    'HS35p': [],
    'HS37': [],
    'HS37p': []
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
    'HS35p': 4,
    'HS37': 5,
    'HS37p': 5
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
    'HS28': ['TC5', 'TN6a', 'TN6a', 'TC5'],
    'HS29': ['SN6', 'SN6', 'TC5', 'TC5', 'TC5e'],
    'HS30': ['TC5e', 'SQ5n', 'SQ5n', 'TC5', 'SQ5n', 'TC5e'],
    'HS30p': ['TC5e', 'SQ5n', 'SQ5n', 'TC5', 'SP2', 'TC5e'],
    'HS30fp': ['TC5e', 'SP2', 'SP2', 'TC5', 'SP2', 'TC5e'],
    'HS32': ['TC5', 'TC5', 'SQ5n', 'SN6'],
    'HS32p': ['TC5', 'TC5', 'SP2', 'SN6'],
    'HS34': ['TC5', 'TC5', 'TC5', 'SN6', 'SQ5n', 'SQ5n', 'TN4', 'TC5e', 'TC5e', 'TN2a'],
    'HS34p': ['TC5', 'TC5', 'TC5', 'SN6', 'SP2', 'SP2', 'TN4', 'TC5e', 'TC5e', 'TN2a'],
    'HS35': ['SN5a', 'TN4', 'SQ5n', 'C2'],
    'HS35p': ['SN5a', 'TN4', 'SP2', 'C2'],
    'HS37': ['SN5a', 'TP1', 'TP1', 'SP1', 'SQ5n'],
    'HS37p': ['SN5a', 'TP1', 'TP1', 'SP1', 'SP2']
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
    'HS35p': [0.0, 0.0, 0.0, 0.0],
    'HS37': [0.0, 0.0, 0.0, 0.0, -1.0],
    'HS37p': [0.0, 0.0, 0.0, 0.0, 0.0]
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
    'HS28': {},
    'HS29': {4: {0: 0.120, 1: 0.308, 2: 0.242, 3: 0.33}},
    'HS30': {},
    'HS30p': {},
    'HS30fp': {},
    'HS32': {},
    'HS32p': {},
    'HS34': {9: {0: 0.552, 3: 0}, 7: {0: -0.203, 1: 0.426, 2: 0}, 8: {1: 0.126, 2: 0.525, 3: 0}},
    'HS34p': {9: {0: 0.552, 3: 0}, 7: {0: -0.210, 1: 0.427, 2: 0}, 8: {1: 0.125, 2: 0.525, 3: 0}},
    'HS35': {},
    'HS35p': {},
    'HS37': {},
    'HS37p': {}
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
FRG_same = ['HS4', 'HS4p', 'HS13', 'HS13p', 'HS19', 'HS19p', 'HS37', 'HS37p']

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
    'HS35p': np.array([1, 2, np.array([4, 5]), 6, 7, np.array([9, 10]), 8, 11, 13, 14, 12, 3], dtype=object),
    'HS37': np.array([1, 2, 16, 3, np.array([4, 5]), 6, np.array([7, 8]), 9, 12, 13, 14, 15, np.array([10, 11])],
                     dtype=object),
    'HS37p': np.array(
        [1, 2, 17, 3, np.array([4, 5]), 6, np.array([7, 8]), 9, 12, 13, np.array([15, 16]), 14, np.array([10, 11])],
        dtype=object)
}
