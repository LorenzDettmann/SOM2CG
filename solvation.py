"""
MIT License

Copyright (c) 2025 Lorenz Friedrich Dettmann

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
"""

__author__ = "Lorenz Dettmann"
__email__ = "lorenz.dettmann@uni-rostock.de"
__version__ = "0.1.0"
__licence__ = "MIT"

import numpy as np
import argparse
import yaml
import os
import sys

def distance_pbc(p1, p2, box):
    """
    Calculate distance between two coordinates using minimal image convention
    """
    diff = p1 - p2
    # minimal image convention
    diff -= box * np.rint(diff / box)
    return np.linalg.norm(diff)

def add_water_particles(existing_positions, n_new, max_attempts, min_distance, box):
    """
    Add 'n_new' water particles and check for distances to existing particles (min_distance)
    """
    new_positions = []
    attempts = 0

    while len(new_positions) < n_new and attempts < max_attempts:
        sys.stdout.write(f"Solvating: {len(new_positions)} / {n_new}\n")
        sys.stdout.write("\033[F")

        candidate = np.random.rand(3) * box  # random positions within the box
        valid = True

        # check distances to already existing positions
        for pos in existing_positions + new_positions:
            if distance_pbc(candidate, pos, box) < min_distance:
                valid = False
                break

        if valid:
            new_positions.append(candidate)
        attempts += 1

    if len(new_positions) < n_new:
        print(f"Warning: Only {len(new_positions)} out of {n_new} particles could be placed in the system.")
    return new_positions

def read_gro(filename):
    """
    Reads a .gro-file and returns header, atom lines and box dimensions
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
    header = lines[0].strip()
    n_atoms = int(lines[1].strip())
    atom_lines = lines[2:2+n_atoms]
    box_line = lines[2+n_atoms].strip().split()
    # orthogonal box
    box = np.array([float(x) for x in box_line[:3]])
    return header, atom_lines, box

def write_gro(filename, header, atom_lines, box, water_positions, resname="W", atomname="W"):
    """
    Outputs a new .gro file and adds entries for the water particles to existing coordinates
    """
    n_existing = len(atom_lines)
    n_new = len(water_positions)
    total_atoms = n_existing + n_new

    with open(filename, 'w') as f:
        f.write(header + "\n")
        f.write(f"{total_atoms}\n")
        atom_index = 1

        # write existing atoms
        for line in atom_lines:
            f.write(line)

        # add water particles
        res_index = 1
        for pos in water_positions:
            # format: residue number (5), residue name (5), atom name (5), atom number (5)
            # x, y, z with 8 positions each and 3 decimal places
            f.write(f"{res_index:5d}{resname:<5s}{atomname:>5s}{atom_index:5d}"
                    f"{pos[0]:8.3f}{pos[1]:8.3f}{pos[2]:8.3f}\n")
            atom_index += 1
            res_index += 1

        # write box dimensions
        f.write(f"{box[0]:10.5f}{box[1]:10.5f}{box[2]:10.5f}\n")

def main():
    parser = argparse.ArgumentParser(description="This script adds coarse-grained water particles to a system provided in the GROMACS coordinate file format (.gro).")

    #parser.add_argument('-h', '--help', action='help', help='Shows this help message')
    parser.add_argument('-f', '--file', help='GROMACS coordinate file')
    parser.add_argument('-o', '--output', help='Output coordinate file')
    parser.add_argument('-n', '--number', help='Number of water particles')
    parser.add_argument('-d', '--distance', help='Minimum distance to other particles')
    parser.add_argument('-try', '--attempts', default=10000, help='Maximum number of attempts')
    parser.add_argument('-cfg', '--config', type=str, help='YAML configuration file')

    args = parser.parse_args()

    # load yaml configuration if provided
    if args.config:
        if not os.path.exists(args.config):
            print(f"Error: The config file '{args.config}' does not exist.")
            abort_script()
        else:
            with open(args.config, 'r') as file:
                config_data = yaml.safe_load(file)
            for key, value in config_data.items():
                # overwrite default values
                if hasattr(args, key) and getattr(args, key) == parser.get_default(key):
                    setattr(args, key, value)

    input_file = args.file
    output_file = args.output
    n_particles = int(args.number)
    min_distance = float(args.distance)
    max_attempts = int(args.attempts)
    
    # read the input file
    header, atom_lines, box = read_gro(input_file)

    # extract positions of existing atoms
    existing_positions = []
    for line in atom_lines:
        try:
            # location of coordinates in a .gro file (each with 8 positions)
            x = float(line[20:28])
            y = float(line[28:36])
            z = float(line[36:44])
            existing_positions.append(np.array([x, y, z]))
        except ValueError:
            continue

    water_positions = add_water_particles(existing_positions, n_particles, max_attempts, min_distance, box)
    write_gro(output_file, header, atom_lines, box, water_positions)

if __name__ == "__main__":
    main()

