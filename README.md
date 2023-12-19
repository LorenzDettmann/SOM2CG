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
python3 martini-som.py [-V] [-h] [-input_dir 'input directory'] [-output_dir 'output directory'] [-n_confs 'number of conformers'] [-map '"cog" or "com" mapping']
```

## Roadmap
### Important / Nice-to-have
- [X] Include options for changing the standard input and output directories, as well as `nconf`
- [X] Complete information on all fragments for the conversion of the RDKit representation to the VSOMM2 representation
- [X] Generalize the script for use with sodium ions and with no ions
- [X] Remove redundant functions and dependencies on other packages
- [X] Add more information to the visual output
- [X] Add functionality for switching between COM and COG mapping
- [ ] Parallelization support
- [ ] Option to output only the coarse-grained structure file
- [ ] More accurate beads for first and last groups in molecules
### Optional
- [X] Rewrite "translate_atoms" function
- [X] Rewrite "merge_smiles" function to be able to handle arbitrary large molecules
- [ ] Rewrite and optimize functions from cg_param_m3.py

## Contributing
If you have any suggestions for improving the efficiency of the script, or suggstions for any additional features, feel free to create a pull request, or simply open a new issue. Thank you!

## License
The licence has to be clarified ...
