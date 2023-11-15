# Martini SOM

A tool for converting atomistic Soil Organic Matter (SOM) models from the Vienna Soil Organic Matter Modeler 2 (VSOMM2) to a coarse-grained representation, compatible with the Martini 3 force field.

## Prerequisites

The script is written in Python 3. The following packages are needed and can be install via `pip`:
```bash
pip install MDAnalysis rdkit scipy natsort
```

## Getting started

Clone the repository from Gitlab:
```bash
git clone https://gitlab.elaine.uni-rostock.de/ld207/martini-som
```
Execute the script in the directory with the topology files from the VSOMM2:
```bash
python3 martini-som.py
```

## Roadmap
- Include options for changing the standard input and output directories, as well as `nconf`
- Complete information on all fragments for the conversion of the RDKit representation to the VSOMM2 representation
- Add more informative visual output
- Determine more accurate beads for first and last groups in molecules
- Reduce number of additional packages to be used

## Contributing
If you have any suggestions for improving the efficiency of the script, or suggstions for any additional features, feel free to create a pull request, or simply open a new issue. Thank you!

## License
The licence has to be clarified ...
