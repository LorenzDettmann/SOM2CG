#!/usr/bin/env python3
# coding: utf-8

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
__version__ = "0.8.5"
__licence__ = "MIT"

import os
import argparse
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from functions import (positive_integer, check_arguments, read_itps, create_vsomm_list, back_translation,
                       create_mapping_vsomm, create_resname_list, parametrize, generate_structure_file)
from dictionaries import fragments_mapping, fragments_lengths

warnings.filterwarnings("ignore", category=Warning)


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
    parser.add_argument('-use_std_fc', default='no', choices=['yes', 'no'],
                        help='Use standard force constants for all bonded interactions')
    parser.add_argument('-with_progress_bar', default='yes', choices=['yes', 'no'],
                        help='Activates a progress bar, could be turned off when redirecting the output into a file')

    args = parser.parse_args()
    # input and output locations
    path = args.input_dir
    gro = f'{path}/min_system.gro'
    cg_path = args.output_dir
    map_type = args.map
    par = args.parametrize
    n_confs = args.n_confs
    num_threads = args.nt
    gen_fc = args.use_std_fc
    progress_bar = args.with_progress_bar

    check_arguments(path, cg_path)
    print(' - Reading atomistic topology files.')
    first_atoms, first_add, last_atoms, last_add, sequences, itp_list = read_itps(path, gro)

    vsomm_lists = []
    mapping = []
    resnames = []
    for i, sequence in enumerate(sequences):
        vsomm_list = create_vsomm_list(sequence, first_add[i], last_add[i], first_atoms[i], last_atoms[i])
        vsomm_lists.append(vsomm_list)
        beads = back_translation(create_mapping_vsomm(sequence, fragments_mapping, first_add[i], last_add[i]),
                                 vsomm_lists[i])
        mapping.append(beads)
        resname_list = create_resname_list(sequence, fragments_lengths)
        resnames.append(resname_list)

    if par == 'yes':
        print(f' - Generating output files for {len(sequences)} HS molecules.')

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(parametrize, i, sequences, mapping, resnames,
                                       first_atoms, last_atoms, n_confs, map_type, gen_fc, cg_path, itp_list) for i in
                       range(len(sequences))]
            # progress bar
            if progress_bar == 'yes':
                for _ in tqdm(as_completed(futures), total=len(sequences), ncols=120):
                    pass
            else:
                done_tasks = 0
                for _ in as_completed(futures):
                    done_tasks += 1
                    print(f'Progress: {done_tasks}/{len(sequences)}')

    generate_structure_file(path, gro, cg_path, itp_list, mapping, sequences, vsomm_lists, resnames, map_type, par)


if __name__ == "__main__":
    main()
