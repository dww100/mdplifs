import itertools
import numpy as np
import mdtraj as md
from scipy.spatial import distance
import scipy.stats as s
from .topology import FeatureTopology
from .structure_utils import angle_between_vectors, is_acceptable_angle, projection, get_ring_normal
from .ligand import LigandFingerprinter


class Fingerprinter:
    """Generate a receptor-ligand trajectory compute the interaction fingerprint.

    TODO: Make it possible to pass only filenames for topology and trajectory

    Parameters
    ----------
    traj  : `mdtraj.Trajectory`
        Trajectory object containing topology information and coordinates over time.
    frames : slice, optional, default=slice(0,-1, 1)
        Which frames of the trajectory should be used, default to all of them.
    top_path : str_like, optional, default='build/complex.prmtop'
        Path to the topology file used to create the trajectory object.
    ligand_selection : str_like, optional, default='resname LIG'
        Selection text (using the mdtraj DSL) used to select ligand atoms.
    receptor_selection : str_like, optional, default='protein'
        Selection text (using the mdtraj DSL) used to select receptor atoms.
    water_cutoff :  int, optional, default=0
        Number of water molecules to include as part of the receptor.

    Attributes
    ----------
    traj : `mdtraj.Trajectory`
        Trajectory filtered to contain only the frames selected for analysis.
    top : FeatureTopology
        Based on traj.topology - but with additional features added to facilitate
        fingerprinting.
    ligand_donor_hbonds : list
        List of lists of the hydrogen bonds in which the ligand acts as donor
        for each frame.
    receptor_donor_hbonds : list
        List of lists of the hydrogen bonds in which the receptor acts as donor
        for each frame.
    hydrophobic_interactions : list
        List of lists of hydrophobic interactions for each frame.
    halogen_bonds : list
        List of lists of halogen bonds for each frame.
    charge_interactions_ligand_positive : list
        List of list of charge interactions in which the ligand is positive for
        each frame.
    charge_interactions_ligand_negative : list
        List of list of charge interactions in which the receptor is positive
        for each frame.
    pi_stacking_interactions : list
        List of list of pi-stacking interactions for each frame.
    pi_cation_receptor : list
        List of list of ring-cation interactions in which the receptor provides
        positive atom for each frame.
    pi_cation_ligand : list
        List of list of ring-cation interactions in which ligand provides
        positive atom for each frame.
    ligand_fingerprint : list
        Ligand fingerprints for each frame.

    """

    def __init__(self, traj, frames=slice(0, -1, 1),
                 top_path='build/complex.prmtop',
                 ligand_selection='resname LIG',
                 receptor_selection='protein',
                 water_cutoff=0):

        # TODO: Allow customized cut offs for each class of interaction

        selection_text = f'({receptor_selection}) or ({ligand_selection})'
        self.ligand_selection = ligand_selection
        self.receptor_selection = receptor_selection

        if water_cutoff > 0:
            # TODO: Allow water molecules to be added
            pass

        topology = traj.topology
        selected_atoms = topology.select(selection_text)
        self.traj = traj[frames].atom_slice(selected_atoms)

        self.top = FeatureTopology(topology, ligand_selection=ligand_selection,
                                   prmtop_path=top_path)

        self.ligand_donor_hbonds = []
        self.receptor_donor_hbonds = []
        self.hydrophobic_interactions = []
        self.halogen_bonds = []
        self.charge_interactions_ligand_positive = []
        self.charge_interactions_ligand_negative = []
        self.pi_stacking_interactions = []
        self.pi_cation_receptor = []
        self.pi_cation_ligand = []

        self.ligand_fingerprinter = None
        self.ligand_fingerprint = []

        self.generate_fingerprint()

    def generate_fingerprint(self):
        """
        Analyse the input trajectory to measure interactions and ligand shape
        changes.

        Returns
        -------

        """

        self.get_hbonds()
        self.get_hydrophobic_interactions()
        self.get_halogen_bonds()
        self.get_charge_interactions()
        self.get_pi_stacking()

        # TODO: Pi cation (receptor and ligand rings)
        # TODO: Unpaired ligand hbond donors
        # TODO: Unpaired ligand hbond acceptors
        # TODO: Water bridged interations
        # TODO: Metal complex interations
        # TODO: Strip double counting of bonds

        self.ligand_figerprinter = LigandFingerprinter(self.traj,
                                                       ligand_selection=self.ligand_selection)
        self.ligand_fingerprint = self.ligand_figerprinter.fingerprint

    def get_hbonds(self):
        """
        Measure all hydrogen bonds in each frame then log the bonds in
        `self.ligand_donor_hbonds` and `self.receptor_donor_hbonds`.

        Uses the Wernet Nilsson criteria [1] for detecting bonds as implemented in
        mdtraj. According to the mdtraj docs that means the criterion employed is:
        "r_DA < 3.3 Angstom −0.00044∗δHDA∗δHDA, where
        r_DA is the distance between donor and acceptor heavy atoms, and δHDA
        the angle made by the hydrogen atom, donor, and acceptor atoms, measured
        in degrees (zero in the case of a perfectly straight bond: D-H ... A)."

        [1] Wernet, Ph., et al. “The Structure of the First Coordination Shell
        in Liquid Water.” (2004) Science 304, 995-999.

        Returns
        -------

        """

        top = self.top

        all_frames_hbonds = md.wernet_nilsson(self.traj)

        for hbonds in all_frames_hbonds:
            # TODO: Possibly need to filter multiple interactions for same atom

            self.ligand_donor_hbonds.append(hbonds[(np.isin(hbonds[:, 0], top.ligand_idxs)) &
                                                   (np.isin(hbonds[:, 2], top.receptor_idxs))])

            self.receptor_donor_hbonds.append(hbonds[(np.isin(hbonds[:, 0], top.receptor_idxs)) &
                                                     (np.isin(hbonds[:, 2], top.ligand_idxs))])

    def get_hydrophobic_interactions(self, max_dist=0.36):
        """
        Detect interactions between atoms in ligand and receptor which are
        flagged as being hydrophobic in `self.top`. Uses a simple distance
        cut off (`max_dist`).

        Parameters
        ----------
        max_dist : float
            Maximum allowable distance for an interaction.

        Returns
        -------

        """

        hydrophobic_interactions = self.hydrophobic_interactions
        receptor_idxs = self.top.receptor_idxs
        ligand_idxs = self.top.ligand_idxs
        atom = self.top.atom

        receptor_atoms = [idx for idx in receptor_idxs if atom(idx).hydrophobic]
        ligand_atoms = [idx for idx in ligand_idxs if atom(idx).hydrophobic]

        n_ligand = len(ligand_atoms)

        atom_pairs = itertools.product(receptor_atoms, ligand_atoms)

        distances = md.compute_distances(self.traj, atom_pairs)

        for interactions in distances:
            idxs = np.where(interactions <= max_dist)[0]
            hydrophobic_interactions.append([(receptor_atoms[x // n_ligand],
                                              ligand_atoms[x % n_ligand]) for x in idxs])

    def get_atom_coords(self, idx, frame):
        """Get the coordinates of the `idx`th atom in the `frame`th frame.

        Parameters
        ----------
        idx :  int
            Atom index of interest.
        frame : int
            Trajectory frame of interest.

        Returns
        -------
        `np.array`
            X, Y and Z coordinates of the selected atom in given frame.
        """

        return self.traj.xyz[frame, idx, :]

    def get_bond_vector(self, idx, frame):
        """
        Get vector describing the bond between the `idx`th atom and the first
        bonded atom in the`frame`th frame. calculated as the difference in the
        coordinates of teh two atoms.
        Note: Not sure why this is a thing - kept here in case it becomes clear.

        Parameters
        ----------
        idx :  int
            Atom index of interest.
        frame : int
            Trajectory frame of interest.

        Returns
        -------
        `np.array`
            Describes the vector between selected atom and first onded to it.
        """

        xyz = self.traj.xyz

        bound_idx = self.top.atom(idx).bonded[0].index

        coord1 = xyz[frame, idx, :]
        coord2 = xyz[frame, bound_idx, :]

        return coord1 - coord2

    def halogen_bond_angles(self, acceptor, donor, frame):
        """
        Get the angles to be assessed in a halogen bond involving the supplied
        `donor` and `acceptor` atoms in `frame`th frame as defined by:

        acceptor     donor
          ^         ^
        Y-O ........X-C

        acceptor angle = angle_between(Y->O, O->X)
        donor angle = angle_between(X->O, X->C)

        Parameters
        ----------
        acceptor : `mdtraj.core.topology.Atom`
            Atom that will act as acceptor (O in diagram).
        donor : `mdtraj.core.topology.Atom`
            Atom that will act as acceptor (X n diagram).
        frame : int
            Frame within `self.trajectory` from which to get atomic coordinates.

        Returns
        -------
        float, float
            Acceptor and donor angles in radians.
        """

        acceptor_coords = self.get_atom_coords(acceptor.index, frame)
        acceptor_bonded_coords = self.get_atom_coords(acceptor.bonded[0].index, frame)

        donor_coords = self.get_atom_coords(donor.index, frame)
        donor_bonded_coords = self.get_atom_coords(donor.bonded[0].index, frame)

        vec1 = acceptor_coords - acceptor_bonded_coords
        vec2 = acceptor_coords - donor_coords
        vec3 = donor_coords - acceptor_coords
        vec4 = donor_coords - donor_bonded_coords

        acceptor_angle = angle_between_vectors(vec1, vec2)
        donor_angle = angle_between_vectors(vec3, vec4)

        return acceptor_angle, donor_angle

    def get_halogen_bonds(self, max_dist=0.4, acceptor_angle=np.rad2deg(120),
                          donor_angle=np.rad2deg(165),
                          tolerance=np.rad2deg(30)):
        """
        Creates a list of lists of atom index pairs representing atoms involved
        in halogen bonds for each frame as stores in `self.halogen_bonds`. To be
        bonded we need a Y-O...X-C configuration, where X is a halogen donor,
        O a halogen acceptor (detected in the topology) which meets the following
        criteria:
        angle_between(Y->O, O->X) within `tolerance` of  acceptor_angle
        angle_between(X->O, X->C) within `tolerance` of  donor_angle
        distange_between(O, X) <= max_dist

        Defaults for the targets are taken from PLIPS which gets them from:
        P. Auffinger, et al., "Halogen bonds in biological molecules", 2004, 101 (48)
        https://doi.org/10.1073/pnas.0407607101

        Parameters
        ----------
        max_dist : float, optional, default=0.4
            Maximum bonding distance, default Includes +0.05 as in PLIPS.
        acceptor_angle : float
            Optimal acceptor angle (radians).
        donor_angle : float
            Optimal donor angle (radians).
        tolerance : float
            Tolerance on angles to accept bond (radians).

        Returns
        -------

        """

        atom = self.top.atom

        halogen_bonds = self.halogen_bonds
        receptor_idxs = self.top.receptor_idxs
        ligand_idxs = self.top.ligand_idxs

        acceptor_atoms = [idx for idx in receptor_idxs if atom(idx).hydrophobic]
        donor_atoms = [idx for idx in ligand_idxs if atom(idx).hydrophobic]

        n_donors = len(donor_atoms)

        atom_pairs = itertools.product(acceptor_atoms, donor_atoms)

        distances = md.compute_distances(self.traj, atom_pairs)

        for frame, interactions in enumerate(distances):

            idxs = np.where(interactions <= max_dist)[0]
            candidate_bonds = [(acceptor_atoms[x // n_donors],
                                donor_atoms[x % n_donors]) for x in idxs]

            candidate_angles = [self.halogen_bond_angles(atom(x), atom(y), frame)
                                for x, y in candidate_bonds]

            bonds = []
            for tmp_idx, angles in enumerate(candidate_angles):

                if (is_acceptable_angle(angles[0], acceptor_angle, tolerance) and
                        is_acceptable_angle(angles[1], donor_angle, tolerance)):

                    bonds.append(candidate_bonds[tmp_idx])

            halogen_bonds.append(bonds)

    def get_charge_interactions(self, max_dist=0.55):
        """
        Records lists of charged interactions between ligand and receptor, where
        a positive and negative atom are within `max_dist` of one another, for
        each frame in the trajectory. Interactions are is stored in
        `self.charge_interactions_ligand_positive` and
        `self.charge_interactions_ligand_negative` according to the charge of
        the ligand atom involved.

        Default criteria from PLIPS originate in:
        [1] D.J.Barlow and J.M.Thornton, 'Ion-pairs in proteins', JMB, 1983,
        168 (4), https://doi.org/10.1016/S0022-2836(83)80079-5

        Parameters
        ----------
        max_dist : float, optional, default=0.55
            Maximum dist between charges to be counted as an interaction (default
            taken from [1] but + 0.15 as is PLIPS).

        Returns
        -------

        """

        atom = self.top.atom

        ligand_positive = self.charge_interactions_ligand_positive
        ligand_negative = self.charge_interactions_ligand_negative

        receptor_idxs = self.top.receptor_idxs
        ligand_idxs = self.top.ligand_idxs

        positive_ligand_atoms = [idx for idx in ligand_idxs if atom(idx).positive]
        negative_receptor_atoms = [idx for idx in receptor_idxs if atom(idx).negative]
        ligand_positive = self._interactions_distance_filter(negative_receptor_atoms,
                                                             positive_ligand_atoms,
                                                             max_dist=max_dist)

        positive_receptor_atoms = [idx for idx in receptor_idxs if atom(idx).positive]
        negative_ligand_atoms = [idx for idx in ligand_idxs if atom(idx).negative]
        ligand_negative = self._interactions_distance_filter(positive_receptor_atoms,
                                                             negative_ligand_atoms,
                                                             max_dist=max_dist)

    def _interactions_distance_filter(self, receptor_idxs, ligand_idxs, max_dist=0.35):
        """
        Filters distances between pairs of residues for those less than `max_dist`.
        All vs all check between the receptor and ligand input lists for all
        frames in `self.traj`.

        Parameters
        ----------
        receptor_idxs : list of int
            Indexes of the receptor residues of interest.
        ligand_idxs : list of int
            Indexes of the ligand residues of interest.
        max_dist : float
            Maximum dist between charges to be counted as an interaction.

        Returns
        -------
        list
            List of lists of tuples containing pairs of atoms within the cut off
            for each frame in `self.traj`.

        """

        n_ligand = len(ligand_idxs)

        atom_pairs = itertools.product(receptor_idxs, ligand_idxs)

        distances = md.compute_distances(self.traj, atom_pairs)

        results = []

        for interactions in distances:
            idxs = np.where(interactions <= max_dist)[0]
            results.append([(receptor_idxs[x // n_ligand],
                             ligand_idxs[x % n_ligand]) for x in idxs])

        return results

    def get_pi_stacking(self, dist_max=0.55,
                        angle_dev=np.deg2rad(30), max_offset=0.2):
        """
        Detect pi-stacking interactions between rings in receptor and ligand for
        all frames in `self.traj`. Results stored in `self.pistacking_interactions`

        Note: default limits from (McGaughey, 1998) as in PLIPS,
        offset = radius of benzene + 0.5 A

        Parameters
        ----------
        dist_max : float, optional, default=0.55
            Maximum distance between rings for an interactions to be counted (nm).
        angle_dev : float, optional, default=np.deg2rad(30)
            Maximum angle between ring normals for an interactions to be counted
            (radians).
        max_offset
            Maximum distance between projection of ring centre to the centre of
            the other ring for an interactions to be counted (nm).

        Returns
        -------

        """

        traj = self.traj
        traj_coords = traj.xyz
        top = self.top
        ligand_rings = top.ligand_rings
        receptor_rings = top.receptor_ring_list()

        ring_pairs = itertools.product(receptor_rings, ligand_rings)

        # TODO: implement ring planarity check

        stacking_interactions = []

        for frame in range(traj.n_frames):

            for receptor_ring, ligand_ring in ring_pairs:

                ring_list = []

                receptor_ring_coords = traj_coords[frame][receptor_ring]
                ligand_ring_coords = traj_coords[frame][ligand_ring]

                receptor_ring_centre = np.apply_along_axis(np.mean, 0, receptor_ring_coords)
                ligand_ring_centre = np.apply_along_axis(np.mean, 0, ligand_ring_coords)

                d = distance.euclidean(receptor_ring_centre,
                                       ligand_ring_centre)

                # Need to calculate this - currently place holder
                receptor_ring_normal = get_ring_normal(receptor_ring_coords)
                ligand_ring_normal = get_ring_normal(ligand_ring_coords)

                b = angle_between_vectors(receptor_ring_normal,
                                          ligand_ring_normal)

                angle = min(b, np.pi - b if not np.pi - b < 0 else b)

                # Ring centre offset calculation
                proj1 = projection(ligand_ring_normal, ligand_ring_centre, receptor_ring_centre)
                proj2 = projection(receptor_ring_normal, receptor_ring_centre, ligand_ring_centre)
                offset = min(distance.euclidean(proj1, ligand_ring_centre),
                             distance.euclidean(proj2, receptor_ring_normal))

                ptype = None
                if not 0.05 < d < dist_max:
                    continue
                if 0 < angle < angle_dev and offset < max_offset:
                    ptype = 'P'
                elif np.deg2rad(90) - angle_dev < angle < np.deg2rad(90) + angle_dev and offset < max_offset:
                    ptype = 'T'

                if ptype is not None:
                    # May want better specification but this'll do for now

                    receptor_residue = top.atom(receptor_ring[0]).residue
                    ligand_residue = top.atom(ligand_ring[0]).residue
                    ring_list.append((receptor_residue, ligand_residue, ptype))

            stacking_interactions.append(ring_list)

        self.pistacking_interactions = stacking_interactions

    def get_pi_cation_interactions(self, dist_max=0.55,
                                   dist_min=0.5,
                                   max_offset=0.2,
                                   target_rings='receptor',
                                   angle_dev=np.deg2rad(30)):
        """
        Detect pi-stacking interactions between rings in receptor and ligand for
        all frames in `self.traj`. Results stored in `self.pi_cation_ligand` or
        `self.pi_cation_receptor` depending on `target_rings`.

        Note: limits from (McGaughey, 1998) as in PLIPS,
        offset = radius of benzene + 0.5 A

        Parameters
        ----------
        dist_max : float, optional, default=0.55
            Maximum distance to count as an interaction (nm).
        dist_min : float, optional, default=0.5
            Minimum distance to count as a valid interaction (nm).
        max_offset : float, optional, default=0.2
            Maximum offset between projection of cation and ring centre to count
            as an interaction.
        target_rings : float, optional, default='receptor'
            Are we looking for interactions with 'receptor' or 'ligand' rings.
        angle_dev : float, optional, default=np.deg2rad(30)
            Angle tolerance for use in cases with


        Returns
        -------

        """

        traj = self.traj
        top = self.top
        atom = self.top.atom

        if target_rings == 'receptor':
            rings = top.receptor_rings
            component_idxs = top.receptor_idxs
            result = self.pi_cation_receptor
            # TODO: Add angle check for interactions with tertiary/quaternary
            #       amines in the ligand

        else:
            rings = top.ligand_rings
            component_idxs = top.ligand_idxs
            result = self.pi_cation_ligand

        cations = [idx for idx in component_idxs if atom(idx).positive]

        for frame in range(traj.n_frames):

            ring_list = []

            for ring in rings:

                ring_coords = traj[frame][ring]
                ring_centre = np.apply_along_axis(np.mean, 0, ring_coords)
                ring_residue = atom(ring[0]).residue

                for cation_idx in cations:

                    cat_coord = self.get_atom_coords(cation_idx, frame)
                    d = distance.euclidean(ring_centre, cat_coord)
                    cation_residue = atom(cation_idx[0]).residue

                    if dist_min < d < dist_max:

                        ring_normal = get_ring_normal(ring_coords)
                        proj = projection(ring_normal, ring_centre, cat_coord)
                        offset = distance.euclidean(proj, ring_centre)

                        if offset < max_offset:
                            ring_list.append((ring_residue, cation_residue))

            result.append(ring_list)
