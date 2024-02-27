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
__version__ = "0.7.1"
__licence__ = "MIT"

import os
import argparse
from rdkit import Chem
from rdkit.Chem import AllChem
import MDAnalysis as mda
from MDAnalysis import transformations
import math
from dictionaries import *


# read itp files
def read_itps(path, gro):
    directory = os.fsencode(path)

    # unfortunately, residue names are not always correctly assigned in the .itp files
    file_list = [f.decode() for f in os.listdir(directory)]
    itp_list = []

    if not os.path.exists(gro):
        print(f"Error: The file '{gro}' containing the atomistic coordinates does not exist.")
        abort_script()

    u = mda.Universe(gro)  # to get correct resnames
    n_gro = len(u.select_atoms("not (resname SOLV or resname SOL or resname CA2+ or resname NA+)"))

    first_atoms = []
    last_atoms = []
    first_add = []
    last_add = []

    # sequence of fragments
    sequence = []

    n = 0
    sorted_list = sorted([file for file in file_list if file.startswith("HS_") and file.endswith(".itp")],
                         key=lambda x: int(x.split('_')[1].split('.')[0]))
    if not sorted_list:
        print(f"Error: No topology files of the form 'HS_*.itp' could be found in '{path}'.")
        abort_script()

    for file in sorted_list:
        itp_list.append(f'{file}')
        u1 = mda.Universe(f'{path}/{file}', topology_format='ITP')
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
        print(f"Error: Number of HS atoms in '{gro}' does not match with the number of atoms from the topology files "
              f"in '{path}/HS_*.itp'. Are some files missing?")
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


def get_last(frg):
    def find_max_in_array(array):
        if isinstance(array, np.ndarray):
            return np.max(array)
        else:
            return array

    # get index of last heavy atom before a brach
    index = np.argmax([find_max_in_array(element) for element in fragments_vsomm_indices[frg]])
    return int(index)


def create_vsomm_list(sequence, first_add=1, last_add=1, first_atom='H', last_atom='H'):
    vsomm_list = np.array([], dtype=object)
    n_before = 0
    for FRG in sequence:
        frg_vsomm_indices = np.copy(fragments_vsomm_indices[FRG])
        vsomm_list = np.concatenate((vsomm_list, np.add(frg_vsomm_indices, n_before)))
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


def list_add(frg_mapping, n):
    # add integer to elements in list of lists
    frg_mapping_added = []
    for bead in frg_mapping:
        bead_added = []
        for atom in bead:
            bead_added.append(atom + n)
        frg_mapping_added.append(bead_added)
    return frg_mapping_added


def add_at_first(indices, first_add):
    # add indices at the beginning (into first bead)
    indices_add = []
    indices_add += indices[0]
    for i in reversed(range(first_add)):
        indices_add.insert(0, i + 1)
    return [indices_add] + indices[1:]


def add_at_last(indices, last_add, frg):
    # add indices into last bead of fragment
    # index of bead, in which last atom should be added
    index = get_largest_index(fragments_connections[frg])  # add this to function input
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
            # if dictionary, then do it again
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
    This function is based on the work of Mark A. Miller and coworkers.
    Modifications were made for this project.
    Refer to the main license text for citation information.
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


def get_coords(mol, beads, map_type, min_energy_idx):
    """
    This function is based on the work of Mark A. Miller and coworkers.
    Modifications were made for this project.
    Refer to the main license text for citation information.
    """
    # Calculates coordinates for output gro file
    conf = mol.GetConformer(min_energy_idx)

    cg_coords = []
    for bead in beads:
        cg_coords.append(bead_coords(bead, conf, mol, map_type))

    return np.array(cg_coords)


def write_itp(bead_types, coords0, charges, A_cg, ring_beads, beads, mol, n_confs, virtual, real,
              masses, resname_list, map_type, itp_name, name, i):
    """
    This function is based on the work of Mark A. Miller and coworkers.
    Modifications were made for this project.
    Refer to the main license text for citation information.
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
    This function is based on the work of Mark A. Miller and coworkers.
    Modifications were made for this project.
    Refer to the main license text for citation information.
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
    This function is based on the work of Mark A. Miller and coworkers.
    Modifications were made for this project.
    Refer to the main license text for citation information.
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
    This function is based on the work of Mark A. Miller and coworkers.
    Modifications were made for this project.
    Refer to the main license text for citation information.
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
    This function is based on the work of Mark A. Miller and coworkers.
    Modifications were made for this project.
    Refer to the main license text for citation information.
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
    This function is based on the work of Mark A. Miller and coworkers.
    Modifications were made for this project.
    Refer to the main license text for citation information.
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
    This function is based on the work of Mark A. Miller and coworkers.
    Modifications were made for this project.
    Refer to the main license text for citation information.
    """
    # Get coordinates of a bead

    coords = np.array([0.0, 0.0, 0.0])
    total = 0.0
    for heavy_atom in bead:
        hydrogens = find_hydrogen_indices(mol, heavy_atom)
        atoms = hydrogens + [heavy_atom]
        for atom in atoms:
            if map_type == 'com':
                mass = mol.GetAtomWithIdx(atom).GetMass()
                coords += conf.GetAtomPosition(atom) * mass
                total += mass
            else:
                coords += conf.GetAtomPosition(atom)
                total += 1

    coords /= (total * 10.0)

    return coords


def find_hydrogen_indices(mol, heavy_atom_index):
    # finds the indices of hydrogen atoms connected to a given heavy atom in the molecule
    hydrogen_indices = []
    heavy_atom = mol.GetAtomWithIdx(heavy_atom_index)
    for neighbor_atom in heavy_atom.GetNeighbors():
        # hydrogen
        if neighbor_atom.GetAtomicNum() == 1:
            hydrogen_indices.append(neighbor_atom.GetIdx())
    return hydrogen_indices


def abort_script():
    print('Aborted')
    exit()


def check_arguments(path, cg_path):
    # checks, if all arguments are properly set
    # additionally checks, if files are going to be overwritten
    if not os.path.exists(path):
        print(f"Error: The input directory '{path}' does not exist.")
        abort_script()

    try:
        os.makedirs(cg_path, exist_ok=True)
    except OSError as e:
        print(f"Error: The output directory '{cg_path}' could not be created. {e}")
        abort_script()

    output_prefix = 'HS_'
    output_suffix = '.itp'
    cg_coordinate_file = 'mapped.gro'
    topology_file = 'topol.top'

    output_files = [file for file in os.listdir(cg_path)
                    if (file.startswith(output_prefix) and file.endswith(output_suffix)) or file == cg_coordinate_file
                    or file == topology_file]

    if output_files:
        print(f"Warning: The given output directory '{cg_path}' does already contain topology files.")
        user_input = input("Would you like to continue and overwrite existing files? (y/n) ")
        if user_input.lower() != 'y':
            abort_script()


def positive_integer(value):
    int_value = int(value)
    if int_value <= 0:
        raise argparse.ArgumentTypeError(
            f"This number must be an integer greater than 0.")
    return int_value


def generate_structure_file(path, gro, cg_path, itp_list, mapping, sequences, vsomm_lists, resnames, map_type, par):
    print(f" - Generating initial structure file from '{gro}'.")
    # unwrap
    u = unwrapped_atomistic_structure(path, gro, itp_list)

    # create mapped structure
    mapped = mapped_structure(u, itp_list, mapping, sequences, vsomm_lists, map_type)

    # add info
    add_residue_info(u, mapped, sequences, mapping, resnames)

    # save structure file
    mapped.atoms.write(f'{cg_path}/mapped.gro')
    add_info(f'{cg_path}/mapped.gro')

    # write water structure file
    write_water_file(cg_path, u)

    if par == 'yes':
        # write topology file
        write_top_file(cg_path, u, itp_list)


def unwrapped_atomistic_structure(path, gro, itp_list):
    # unwraps the atomistic structure to correctly generate the mapped structure
    # to unwrap, the atomistic bonds have to be read and added
    # load atomistic coordinates
    u = mda.Universe(f'{gro}')
    # add bonds from itp files
    n = 0
    for i, file in enumerate(itp_list):
        u1 = mda.Universe(f'{path}/{file}', topology_format='ITP')
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


def write_water_file(cg_path, u):
    # generate gro file for solvation, if not already present
    water_gro = """Regular sized water particle
1
    1W      W      1   0.000   0.000   0.000
   1.00000   1.00000   1.00000"""
    file = "water.gro"
    if not os.path.exists(f"{cg_path}/{file}"):
        f = open(f"{cg_path}/{file}", "w")
        f.write(water_gro)

    # number of coarse-grained water molecules
    N = round(len(u.select_atoms('name OW')) / 4)

    print(' - Done')
    if N > 0:
        print(
            f"You can solvate the structure with \'gmx insert-molecules -ci water.gro -nmol {round(N)}"
            + " -f mapped.gro -radius 0.180 -try 1000 -o solvated.gro &> solvation.log\'")


def write_top_file(cg_path, u, itp_list):
    topol_top = """#include "martini3.ff/martini_v3.0.0.itp"      
#include "martini3.ff/martini_v3.0.0_solvents_v1.itp"    
#include "martini3.ff/martini_v3.0.0_ions_v1.itp"\n
"""
    file = "topol.top"
    f = open(f"{cg_path}/{file}", "w")
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
    add_info(f"{cg_path}/{file}")


def parametrize(i, sequences, mapping, resnames, first_atoms, last_atoms,
                n_confs, map_type, cg_path, itp_list):
    mol = create_macromolecule(sequences[i], first_atoms[i], last_atoms[i])
    ring_atoms = get_ring_atoms(mol)
    A_cg = create_A_matrix(sequences[i], fragments_connections, fragments_lengths, FRG_same)
    ring_beads = determine_ring_beads(ring_atoms, mapping[i])
    charges = determine_charges(sequences[i], fragments_charges)
    bead_types = determine_bead_types(sequences[i], fragments_bead_types)
    change_first_and_last_bead(resnames[i], bead_types, first_atoms[i], last_atoms[i])
    mol = Chem.AddHs(mol)
    AllChem.EmbedMultipleConfs(mol, numConfs=n_confs)
    n_confs_generated = mol.GetNumConformers()  # actual number of conformers that were generated
    energy_list = AllChem.UFFOptimizeMoleculeConfs(mol, maxIters=1000)
    min_energy_idx = min(range(len(energy_list)), key=energy_list.__getitem__)
    coords0 = get_coords(mol, mapping[i], map_type, min_energy_idx)  # coordinates of energy minimized molecules
    virtual, real = get_new_virtual_sites(sequences[i], fragments_vs, fragments_lengths, ring_beads)
    masses = get_standard_masses(bead_types, virtual)
    write_itp(bead_types, coords0, charges, A_cg, ring_beads, mapping[i], mol, n_confs_generated, virtual, real,
              masses, resnames[i], map_type, f'{cg_path}/{itp_list[i]}', itp_list[i][:-4], i)


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
