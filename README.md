# Martini SOM

A tool for converting atomistic Soil Organic Matter (SOM) models from the Vienna Soil Organic Matter Modeler 2 (VSOMM2) to a coarse-grained representation, compatible with the Martini 3 force field.

## Prerequisites

The script is written in Python 3. The following packages are needed and can be installed via `pip`:
```bash
pip install MDAnalysis rdkit
```

## Getting started

Clone the repository from Gitlab:
```bash
git clone https://gitlab.elaine.uni-rostock.de/ld207/martini-som
```
You can execute the script in the directory with the topology files from the VSOMM2, or modify e.g. the directory paths
using the following arguments:
```bash
python3 martini-som.py [-V] [-h] [-nt 'number of threads'] [-input_dir 'input directory'] [-output_dir 'output directory'] [-n_confs 'number of conformers'] [-map '"cog" or "com" mapping'] [-parametrize '(yes/no) parametrize the molecules, or only output mapped structure file']
```

## Contributing
If you have any suggestions for improving the efficiency of the script, or suggestions for any additional features, feel free to create a pull request, or simply open a new issue. Thank you!

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
