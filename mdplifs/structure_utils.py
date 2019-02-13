import itertools
from rdkit import Chem
import numpy as np
from scipy.spatial import distance


def angle_between_vectors(v1, v2):
    return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

    return acceptor_angle, donor_angle


def normalize_vector(vector):

    norm = np.linalg.norm(vector)
    return vector/norm if not norm == 0 else vector


def is_acceptable_angle(angle, target, tolerance):

    return target - tolerance < angle < target + tolerance


def projection(plane_normal, plane_point, target_point):
    """Project coordinates of the target_point onto the plane.
    Note: Adapted from PLIPS there is likely a better numpy/
    scipy solution to this.

    Parameters
    ----------
    plane_normal: np.array
        Normal of the plane
    plane_point: np.array
        Coordinates of point on the plane
    target_point: np.array
        Coordinates of point to be projected

    Returns
    -------
    np.array
        Coordinates of point orthogonally projected on the plane
    """

    # Choose the plane normal pointing to the point to be projected
    d1 = distance.euclidean(target_point, plane_normal + plane_point)
    d2 = distance.euclidean(target_point, -1 * plane_normal + plane_point)

    if d2 < d1:
        plane_normal = -1 * plane_normal

    # Calculate the projection of target_point to the plane
    sn = -np.dot(plane_normal, target_point - plane_point)
    sd = np.dot(plane_normal, plane_normal)
    sb = sn / sd

    return target_point + sb * plane_normal


def get_ring_normal(coords):

    selected_ring_coords = [coords[x] for x in [0, 2, 4]]
    vector1 = selected_ring_coords[0] - selected_ring_coords[1]
    vector2 = selected_ring_coords[2], selected_ring_coords[0]

    return normalize_vector(np.cross(vector1, vector2))


def remove_duplicate_bonds(iterable):

    # Create a set for already seen elements
    seen = set()
    for item in iterable:
        # Lists are mutable so we need tuples for the set-operations.
        tup = (item.atom1.index, item.atom2.index)
        if tup not in seen:
            # If the tuple is not in the set append it in REVERSED order.
            seen.add(tup[::-1])
            # Include original order to remove standard duplicates
            seen.add(tup)
            yield item


def atoms_to_rdkit_mol(atoms):

    editable = Chem.EditableMol(Chem.Mol())

    idx_map = {}

    rd_idx = 0

    for atom in atoms:
        idx = atom.index

        rd_atom = Chem.Atom(atom.element.atomic_number)
        rd_atom.SetProp('original_name', atom.name)
        rd_atom.SetProp('md_index', str(atom.index))
        editable.AddAtom(rd_atom)

        idx_map[idx] = rd_idx

        rd_idx += 1

    bond_list = remove_duplicate_bonds(itertools.chain.from_iterable(
        [atom.bonds for atom in atoms]))

    for bond in bond_list:

        idx1 = bond.atom1.index
        idx2 = bond.atom2.index

        if idx1 in idx_map and idx2 in idx_map:
            rd_idx1 = idx_map[bond.atom1.index]
            rd_idx2 = idx_map[bond.atom2.index]

            # At present we have no order information here and we want topology
            # only, so this is fine but in the long run need better
            editable.AddBond(rd_idx1, rd_idx2, order=Chem.rdchem.BondType.SINGLE)

    mol = editable.GetMol()
    mol.idx_to_md_idx = {rd_idx: idx for idx, rd_idx in idx_map.items()}

    return mol

