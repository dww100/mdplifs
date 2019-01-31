import itertools
import numpy as np
import mdtraj as md
from .topology import FeatureTopology


class Fingerprinter:

    def __init__(self, traj, frames, ligand_selection='resname LIG',
                 receptor_selection='protein', water_cutoff=0):

        # TODO: Allow customized cut offs for each class of interaction

        selection_text = f'({receptor_selection}) or ({ligand_selection})'

        if water_cutoff > 0:
            # TODO: Allow water molecules to be added
            pass

        topology = traj.topology
        selected_atoms = topology.select(selection_text)
        self.traj = traj[frames].atomslice[selected_atoms]
        self.top = FeatureTopology(topology, ligand_selection)

        self.ligand_donor_hbonds = []
        self.receptor_donor_hbonds = []
        self.hydrophobic_interactions = []

        self.generatefingerprint()

    def generate_fingerprint(self):

        self.get_hbonds()
        self.get_hydrophobic_interactions()

        # TODO: Salt bridge
        # TODO: Pi stacking
        # TODO: Pi cation (paro/laro)
        # TODO: Halogen bonds
        # TODO: Unpaired ligand hbond donors
        # TODO: Unpaired ligand hbond acceptors
        # TODO: Water bridged interations
        # TODO: Metal complex interations

    def get_hbonds(self):

        top = self.top

        all_frames_hbonds = md.wernet_nilsson(self.traj)

        for hbonds in all_frames_hbonds:
            # TODO: Possibly need to filter multiple interactions for same atom

            self.ligand_donor_hbonds.append(hbonds[(np.isin(hbonds[:, 0], top.ligand_idxs)) &
                                                   (np.isin(hbonds[:, 2], top.receptor_idxs))])

            self.receptor_donor_hbonds.append(hbonds[(np.isin(hbonds[:, 0], top.receptor_idxs)) &
                                                     (np.isin(hbonds[:, 2], top.ligand_idxs))])

    def get_hydrophobic_interactions(self):

        hydrophobic_interactions = self.hydrophobic_interactions
        receptor_idxs = self.top.receptor_idxs
        ligand_idxs = self.top.ligand_idxs
        atom = self.top.atom

        receptor_atoms = [idx for idx in receptor_idxs if atom(idx).hydrophobic]
        ligand_atoms = [idx for idx in ligand_idxs if atom(idx).hydrophobic]

        atom_pairs = itertools.product(receptor_atoms, ligand_atoms)

        distances = md.compute_distances(self.traj, atom_pairs)

        for interactions in distances:
            hydrophobic_interactions.append(interactions[interactions <= 0.36])


class LigandFingerprinter:

    def __init__(self):

        # TODO: Get rdkit fingerprint of ligand
        # TODO: Calculate dynamic fingerprint

        pass
