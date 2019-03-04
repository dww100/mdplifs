import itertools
import numpy as np
import mdtraj as md
from scipy.spatial import distance
import scipy.stats as s
from .topology import FeatureTopology
from .structure_utils import angle_between_vectors, is_acceptable_angle, projection, get_ring_normal


class Fingerprinter:

    def __init__(self, traj, frames, ligand_selection='resname LIG',
                 receptor_selection='protein', water_cutoff=0):

        # TODO: Allow customized cut offs for each class of interaction

        selection_text = f'({receptor_selection}) or ({ligand_selection})'
        self.ligand_selection = ligand_selection
        self.receptor_selection = receptor_selection

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
        self.halogen_bonds = []
        self.charge_interactions_ligand_positive = []
        self.charge_interactions_ligand_negative = []
        self.pi_stacking_interactions = []
        self.pi_cation_receptor = []
        self.pi_cation_ligand = []

        self.ligand_fingerprint = []

        self.generatefingerprint()

    def generate_fingerprint(self):

        self.get_hbonds()
        self.get_hydrophobic_interactions()
        self.get_halogen_bonds()
        self.get_charge_interactions()
        self.get_pi_stacking()

        # TODO: Pi cation (paro/laro)
        # TODO: Unpaired ligand hbond donors
        # TODO: Unpaired ligand hbond acceptors
        # TODO: Water bridged interations
        # TODO: Metal complex interations
        # TODO: Strip double counting of bonds

        self.ligand_fingerprint = LigandFingerprinter(self.traj,
                                                      ligand_selection=self.ligand_selection)

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

        n_ligand = len(ligand_atoms)

        atom_pairs = itertools.product(receptor_atoms, ligand_atoms)

        distances = md.compute_distances(self.traj, atom_pairs)

        for interactions in distances:
            idxs = np.where(interactions <= 0.36)[0]
            hydrophobic_interactions.append([(receptor_atoms[x // n_ligand],
                                              ligand_atoms[x % n_ligand]) for x in idxs])

    def get_atom_coords(self, idx, frame):

        return self.xyz[frame, idx, :]

    def get_bond_vector(self, idx, frame):

        xyz = self.traj.xyz

        bound_idx = self.top.atom(idx).bonded[0].index

        coord1 = xyz[frame, idx, :]
        coord2 = xyz[frame, bound_idx, :]

        return coord1 - coord2

    def halogen_bond_angles(self, acceptor, donor, frame):

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

    def get_halogen_bonds(self):

        atom = self.top.atom

        # Halogen bonds in biological molecules., Auffinger
        max_dist = 0.4  # Includes +0.05 as in PLIPS
        acceptor_angle = np.rad2deg(120)
        donor_angle = np.rad2deg(165)
        tolerance = np.rad2deg(30)

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

                if (is_acceptable_angle(candidate_angles[0], acceptor_angle, tolerance) and
                   is_acceptable_angle(candidate_angles[1], donor_angle, tolerance)):

                    bonds.append(candidate_bonds[tmp_idx])

            halogen_bonds.append(bonds)

    def get_charge_interactions(self, max_dist=0.55):

        # max_dist from (Barlow and Thornton, 1983) + 0.15 as in PLIPS

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

        # Limits from (McGaughey, 1998) as in PLIPS
        # offset = radius of benzene + 0.5 A

        traj = self.traj
        top = self.top
        ligand_rings = top.ligand_rings
        receptor_rings = top.receptor_rings

        ring_pairs = itertools.product(receptor_rings, ligand_rings)

        # TODO: implement ring planarity check

        stacking_interactions = []

        for frame in range(traj.n_frames):

            for receptor_ring, ligand_ring in ring_pairs:

                ring_list = []

                receptor_ring_coords = traj[frame][receptor_ring]
                ligand_ring_coords = traj[frame][ligand_ring]

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
                                   angle_dev=np.deg2rad(30),
                                   max_offset=0.2,
                                   target_rings='receptor'):

        # Limits from (McGaughey, 1998) as in PLIPS
        # offset = radius of benzene + 0.5 A

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

                    cat_coord = traj[frame, cation_idx]
                    d = distance.euclidean(ring_centre, cat_coord)
                    cation_residue = atom(cation_idx[0]).residue

                    if dist_min < d < dist_max:

                        ring_normal = get_ring_normal(ring_coords)
                        proj = projection(ring_normal, ring_centre, cat_coord)
                        offset = distance.euclidean(proj, ring_centre)

                        if offset < max_offset:
                            ring_list.append((ring_residue, cation_residue))

            result.append(ring_list)


class LigandFingerprinter:
    """
    Create a ligand fingerprint based on the shape over an MD trajectory.
    This is based on the work of Ash and Fourches which generalized the shape
    metrics defined by Ballester et al.

    Shape is defined by the set of all atomic distances from four molecular locations:
    the molecular centroid (ctd), the closest atom to ctd (cst),
    the farthest atom to ctd (fct), and the farthest atom to fct (ftf).

    The shape is defined by the moments of the distribution of these values in each frame.
    Each moment is averaged to give a trajectory value in the fingerprint.

    Ballester, P. J. and Richards, W. G. (2007),
    Ultrafast shape recognition to search compound databases for similar molecular shapes.
    J. Comput. Chem., 28: 1711-1723. doi:10.1002/jcc.20681

    Ash, J. and Fourches, D. (2017),
    Characterizing the Chemical Space of ERK2 Kinase Inhibitors Using Descriptors Computed
    from Molecular Dynamics Trajectories
    J. Chem. Inf. Model., 57 (6): 1286-1299. DOI: 10.1021/acs.jcim.7b00048
    """

    def __init__(self, traj, ligand_selection='resname LIG',
                 n_moments=10):

        if n_moments < 2:
            raise ValueError('n_moments must be 2 or greater')

        self.n_moments = n_moments

        ligand_atoms = traj.topology.select(ligand_selection)
        self.traj = traj.atomslice[ligand_atoms]
        self.top = self.traj.topology

        self.ctds = None
        self.csts = None
        self.fcts = None
        self.ftfs = None

        self.calculate_distance_metrics()

        self.ctd_frame_moments = self._calculate_metric_moments(self.ctds,
                                                                n_moments)
        self.cst_frame_moments = self._calculate_metric_moments(self.csts,
                                                                n_moments)
        self.fct_frame_moments = self._calculate_metric_moments(self.fcts,
                                                                n_moments)
        self.ftf_frame_moments = self._calculate_metric_moments(self.ftfs,
                                                                n_moments)

        self.ctd_metrics = np.array(self.ctd_frame_moments).mean(axis=0)
        self.cst_metrics = np.array(self.cst_frame_moments).mean(axis=0)
        self.fct_metrics = np.array(self.fct_frame_moments).mean(axis=0)
        self.ftf_metrics = np.array(self.ftf_frame_moments).mean(axis=0)

        self.fingerprint = np.concatenate((self.ctd_metrics, self.cst_metrics,
                                           self.fct_metrics, self.ftf_metrics))

    def calculate_centre_points(self):

        traj = self.traj

        centres = np.zeros((traj.n_frames, 3))

        for i, x in enumerate(traj.xyz):
            centres[i, :] = x.mean(axis=0)

        return centres

    def _calculate_distance_difference(self, ref_coords):

        traj = self.traj

        metrics = np.zeros((traj.n_frames, traj.n_atoms))

        for frame, coords in enumerate(traj.xyz):
            # Compute distance between each coordinate and reference coordinate
            metrics[frame, :] = np.apply_along_axis(distance.euclidean, 1,
                                                    coords, ref_coords[frame])

        return metrics

    def calculate_distance_metrics(self):

        centres = self.calculate_centre_points()

        # Calculate distance of every atom to the centre in each frame
        ctds = self._calculate_distance_difference(centres)
        self.ctds = ctds

        # Distances of every atom to the closest atom to the centre in each frame
        closest_atom_to_centre = np.array([frame_ctds[np.argmin(frame_ctds)]
                                           for frame_ctds in ctds])
        self.csts = self._calc_dist_metric(closest_atom_to_centre)

        # Distances of every atom to the furthest atom to the centre in each frame
        furthest_atom_to_centre = np.array([frame_ctds[np.argmax(frame_ctds)]
                                            for frame_ctds in ctds])
        self.fcts = self._calc_dist_metric(furthest_atom_to_centre)

        # Distances of every atom to furthest atom from teh furhest atom for the centre
        furthest_from_furthest_atom = np.array([frame_fcts[np.argmax(frame_fcts)]
                                                for frame_fcts in furthest_atom_to_centre])

        self.ftfs = self._calc_dist_metric(furthest_from_furthest_atom)

    @staticmethod
    def _calculate_metric_moments(traj_metric, n_moments=10):

        moments = []

        for frame_values in traj_metric:
            frame_mean = np.mean(frame_values)
            moments.append(np.append(frame_mean, s.moment(frame_values, range(2, n_moments))))

        return moments

