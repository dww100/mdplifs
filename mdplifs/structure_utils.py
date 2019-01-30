import itertools
from rdkit import Chem


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


def residue_to_rdkit_mol(residue):

    editable = Chem.EditableMol(Chem.Mol())

    idx_map = {}

    rd_idx = 0

    for atom in residue.atoms:
        idx = atom.index

        rd_atom = Chem.Atom(atom.element.atomic_number)
        rd_atom.SetProp('original_name', atom.name)
        rd_atom.SetProp('md_index', atom.index)
        editable.AddAtom(rd_atom)

        idx_map[idx] = rd_idx

        rd_idx += 1

    bond_list = remove_duplicate_bonds(itertools.chain.from_iterable(
        [atom.bonds for atom in residue.atoms]))

    for bond in bond_list:

        idx1 = bond.atom1.index
        idx2 = bond.atom2.index

        if idx1 in idx_map and idx2 in idx_map:
            rd_idx1 = idx_map[bond.atom1.index]
            rd_idx2 = idx_map[bond.atom2.index]

            # At present we have no order information here and we want topology
            # only, so this is fine but in the long run need better
            editable.AddBond(rd_idx1, rd_idx2, order=Chem.rdchem.BondType.SINGLE)

    return editable.GetMol()
