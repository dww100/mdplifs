import numpy as np
import mdtraj as md


class FeatureTopology(md.Topology):

    def __init__(self, topology, prmtop_path='build/complex.prmtop',
                 ligand_selection='resname LIG', charge_tolerance=0.1):

        super().__init__()

        self.ligand_idxs = topology.select(ligand_selection)

        if len(self.ligand_idxs) == 0:
            # Replace this with a sensible exception/useful logging
            raise Exception('Nonsense input - no ligand selected')

        self.receptor_idxs = topology.select('protein')
        self.complex_idxs = np.concatenate((self.receptor_idxs, self.ligand_idxs))

        charges = []
        in_charges = False

        with open(prmtop_path) as infile:
            for line in infile:
                if in_charges:
                    if line.startswith('%FORMAT'):
                        continue
                    elif line.startswith('%FLAG'):
                        break
                    else:
                        # Note: use conversion factor to get electron charge units
                        charges += [float(x) / 18.2223 for x in line.split()]
                elif line.startswith('%FLAG CHARGE'):
                    in_charges = True

        for chain in topology.chains:

            c = self.add_chain()

            for residue in chain.residues:

                r = self.add_residue(residue.name, c, residue.resSeq, residue.segment_id)

                if r.is_protein:
                    r.rings = self.protein_ring_check(r)
                else:
                    r.rings = []

                for atom in residue.atoms:

                    self.add_atom(atom.name, atom.element, r, serial=atom.serial)

                    new_atom = self._atoms[-1]

                    new_atom.charge = charges[new_atom.index]

                    new_atom.metal_binder = self._is_metal_binder(atom)

                    if residue.is_protein:

                        (new_atom.positive,
                         new_atom.negative) = self._is_charged_protein(atom)

                    elif new_atom.index in self.ligand_idxs:
                        (new_atom.positive,
                         new_atom.negative) = self._is_charged_ligand(atom, charge_tolerance)
                        pass

                    new_atom.in_ring = False
                    new_atom.halogen_acceptor = False
                    new_atom.halogen_donor = False

                    new_atom.bonded = []
                    new_atom.bonds = []

            for bond in topology.bonds:
                a1, a2 = bond
                self.add_bond(a1, a2, type=bond.type, order=bond.order)
                new_bond = self._bonds[-1]

                new_a1 = self._atoms[a1.index]
                new_a2 = self._atoms[a2.index]

                new_a1.bonded.append(new_a2)
                new_a1.bonds.append(new_bond)

                new_a2.bonded.append(new_a1)
                new_a2.bonds.append(new_bond)

        # hydrophobicity and halogen acceptor checks
        for atom in self.atoms:
            atom.hydrophobic = self._is_hydrophobic(atom)

            if atom.index in self.receptor_idxs:
                atom.halogen_acceptor = self._is_halogen_acceptor(atom)
            elif atom.index in self.ligand_idxs:
                atom.halogen_donor = self._is_halogen_donor(atom)

        # TODO: Add ring check(s)

    @staticmethod
    def _is_hydrophobic(atom):

        hydrophobic = False

        if atom.element.symbol == 'C':

            if not [bonded.element.symbol for bonded in atom.bonded
                    if bonded.element.symbol not in ['C', 'H']]:
                hydrophobic = True

        return hydrophobic

    @staticmethod
    def _is_charged_protein(atom):

        residue = atom.residue

        positive = False
        negative = False

        if atom.element.symbol == 'N' and residue.name in ['ARG', 'HIS', 'LYS']:
            if atom.is_sidechain:
                positive = True
        elif atom.element.symbol == 'O' and residue.name in ['GLU', 'ASP']:
            if atom.is_sidechain:
                negative = True

        return positive, negative

    @staticmethod
    def _is_charged_ligand(atom, tolerance=0.1):

        positive = False
        negative = False

        if atom.charge >= tolerance:
            positive = True
        elif atom.charge <= -tolerance:
            negative = True

        return positive, negative

    @staticmethod
    def _is_halogen_acceptor(atom):

        acceptor = False

        if atom.element.symbol in ['O', 'P', 'N', 'S']:

            if len(atom.bonded) == 1 and atom.bonded[0].element.symbol in ['C', 'P', 'S']:

                acceptor = True

        return acceptor

    @staticmethod
    def _is_halogen_donor(atom):

        donor = False

        if atom.element.symbol in ['F', 'Cl', 'Br', 'I']:

            if len(atom.bonded) == 1 and atom.bonded[0].element.symbol == 'C':

                donor = True

        return donor

    @staticmethod
    def _is_metal_binder(atom):

        metal_binder = False

        residue = atom.residue

        if residue.is_protein:

            if atom.is_backbone:
                if atom.element.symbol == 'O':
                    metal_binder = True

            elif (residue.name in ['ASP', 'GLU', 'SER', 'THR', 'TYR']
                  and atom.element.symbol in ['O', 'N', 'S']):
                metal_binder = True

            elif residue.name == 'HIS' and atom.element.symbol == 'N':
                metal_binder = True

            elif residue.name == 'CYS' and atom.element.symbol == 'S':
                metal_binder = True

        return metal_binder

    @staticmethod
    def protein_ring_check(residue):

        name = residue.name
        atoms = residue.atoms

        if name in ['TRP', 'TYR', 'HIS', 'HIE', 'HID', 'HIP', 'PHE']:

            if residue.name == 'TRP':

                rings = list()
                rings.append([atom.index for atom in atoms if atom.name in ['CD2', 'CE2', 'CZ2',
                                                                            'CH2', 'CZ3', 'CE3']])
                rings.append([atom.index for atom in atoms if atom.name in ['NE1', 'CE2', 'CD2',
                                                                            'CG', 'CD1']])
                return rings

            if residue.name == 'TYR':
                return [[atom.index for atom in atoms if atom.name in ['CE2', 'CZ', 'CE1',
                                                                       'CD1', 'CG', 'CD2']]]

            if residue.name in ['HIS', 'HIE', 'HIP', 'HID']:
                return [[atom.index for atom in atoms if atom.name in ['CG', 'CD2', 'NE2',
                                                                       'CE1', 'ND1']]]

            if residue.name == 'PHE':
                return [[atom.index for atom in atoms if atom.name in ['CG', 'CD2', 'CE2',
                                                                       'CZ', 'CE1', 'CD1']]]

        else:
            return []

