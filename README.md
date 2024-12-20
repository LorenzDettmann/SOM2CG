# Martini SOM

A tool for converting atomistic Soil Organic Matter (SOM) models from the Vienna Soil Organic Matter Modeler 2 (VSOMM2) to a coarse-grained representation, compatible with the Martini 3 force field.

## Prerequisites

The script is written in Python 3. The following packages are needed and can be installed via `pip`:
```bash
pip install MDAnalysis rdkit
```

## Execution

You can execute the script in the directory with the topology files from the VSOMM2, or modify e.g. the directory paths using the following arguments:
```bash
python3 martini_som.py [-V] [-h] [-nt 'number of threads'] 
                       [-input_dir 'input directory'] 
                       [-output_dir 'output directory'] 
                       [-n_confs 'number of conformers'] 
                       [-map '"cog" or "com" mapping'] 
                       [-parametrize '(yes/no) parametrize the molecules, or only output mapped structure file']
                       [-use_std_fc '(yes/no) use standard force constants']
                       [-with_progress_bar '(yes/no) activate a progress bar']
                       [-config 'YAML configuration file']
```

## Contributing
If you have any suggestions for improving the efficiency of the script, or for additional features, feel free to contact one of the authors of the publication.

## Acknowledgments
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

## License
This software is distributed under the MIT license.

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
