import numpy as np
from scipy.spatial import distance
import scipy.stats as s


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

    Parameters
    ----------
    traj  : `mdtraj.Trajectory`
        Trajectory object containing topology information and coordinates over time.
    frames : slice, optional, default=slice(0,-1, 1)
        Which frames of the trajectory should be used, default to all of them.
    ligand_selection : str_like, optional, default='resname LIG'
        Selection text (using the mdtraj DSL) used to select ligand atoms.
    n_moments : int, optional, default=10
        Number of moments of each distance metric to calculate

    Attributes
    ----------
    traj : `mdtraj.Trajectory`
        Trajectory filtered to contain only the frames and atoms selected for analysis.
    top : `mdtraj.topology`
        Ligand topology corresponding to the `ligand_selection` from the input `traj`.
    ctds : np.array
        Distances of atoms to centroid of each frame
    csts : np.array
        Distances of atoms to closest atom to the centre for each frame
    fcts : np.array
        Distances of atoms to furthest atom from centroid of each frame
    ftfs : np.array
        Distances of atoms to furthest atom from furthest atom from centroid of each
        frame
    ctd_frame_moments :  np.array
        `n_moments` moments of the distribution of ctds for each frame.
    cst_frame_moments :  np.array
        `n_moments` moments of the distribution of csts for each frame.
    fct_frame_moments :  np.array
        `n_moments` moments of the distribution of fcts for each frame.
    ftf_frame_moments :  np.array
        `n_moments` moments of the distribution of ftfs for each frame.
    ctd_metrics :  np.array
        Average across all frames of each of the `n_moments` moments of the ctds.
    cst_metrics :  np.array
        Average across all frames of each of the `n_moments` moments of the csts.
    fct_metrics :  np.array
        Average across all frames of each of the `n_moments` moments of the fcts.
    ftf_metrics :  np.array
        Average across all frames of each of the `n_moments` moments of the ftfs.
    fingerprint :  np.array
        Concatenation of the averages of all metrics.
    """

    def __init__(self, traj, ligand_selection='resname LIG',
                 frames=slice(0, -1, 1), n_moments=10):

        if n_moments < 2:
            raise ValueError('n_moments must be 2 or greater')

        self.n_moments = n_moments

        ligand_atoms = traj.topology.select(ligand_selection)
        self.traj = traj[frames].atom_slice(ligand_atoms)
        self.top = self.traj.topology

        self.ctds = np.array([])
        self.csts = np.array([])
        self.fcts = np.array([])
        self.ftfs = np.array([])

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

        # TODO: Should be a function
        self.fingerprint = np.concatenate((self.ctd_metrics, self.cst_metrics,
                                           self.fct_metrics, self.ftf_metrics))

    def calculate_centre_points(self):
        """
        Compute the centre of each frame in `self.traj`.

        Returns
        -------
        np.array
            Array containing centroid for each frame of the trajectory.
        """

        traj = self.traj

        centres = np.zeros((traj.n_frames, 3))

        for i, x in enumerate(traj.xyz):
            centres[i, :] = x.mean(axis=0)

        return centres

    def _calculate_distance_difference(self, ref_coords):
        """
        Calculate distances of each atom in `self.traj` to a reference
        coordinate in the same frame (provided in `ref_coords`).

        Parameters
        ----------
        ref_coords  :  np.array
            Array containing a set of reference coordinates for each frame in
            `self.traj`.

        Returns
        -------
        np.array
            Array of distance between each coordinate in each frame of `self.traj`
            and the corresponding reference coordinate.
        """

        traj = self.traj

        metrics = np.zeros((traj.n_frames, traj.n_atoms))

        for frame, coords in enumerate(traj.xyz):

            metrics[frame, :] = np.apply_along_axis(distance.euclidean, 1,
                                                    coords, ref_coords[frame])

        return metrics

    def calculate_distance_metrics(self):
        """
        Calculate all shape mertics for each frame in `self.traj`. Metrics are set of all
        atomic distances from four molecular locations:
        the molecular centroid (ctd), the closest atom to ctd (cst),
        the farthest atom to ctd (fct), and the farthest atom to fct (ftf).
        All of which are stored in corresponding array in the object, i.e.
        `self.ctds` is an array of ctd values for each frame.

        """

        centres = self.calculate_centre_points()

        # Calculate distance of every atom to the centre in each frame
        ctds = self._calculate_distance_difference(centres)
        self.ctds = ctds

        # Distances of every atom to the closest atom to the centre in each frame
        closest_atom_to_centre = np.array([frame_ctds[np.argmin(frame_ctds)]
                                           for frame_ctds in ctds])
        self.csts = self._calculate_distance_difference(closest_atom_to_centre)

        # Distances of every atom to the furthest atom to the centre in each frame
        furthest_atom_to_centre = np.array([frame_ctds[np.argmax(frame_ctds)]
                                            for frame_ctds in ctds])
        self.fcts = self._calculate_distance_difference(furthest_atom_to_centre)

        # Distances of every atom to furthest atom from the furthest atom for the centre
        furthest_from_furthest_atom = np.array([frame_fcts[np.argmax(frame_fcts)]
                                                for frame_fcts in self.fcts])

        self.ftfs = self._calculate_distance_difference(furthest_from_furthest_atom)

    @staticmethod
    def _calculate_metric_moments(traj_metric, n_moments=10):
        """
        Calculate the first `n_moments` of the distribution of the metric in the
        `traj_metric` array for each frame.

        Parameters
        ----------
        traj_metric  :  np.array
            Containing values of metric for each atom in each frame in trajectory.
        n_moments  :  int
            Number of moments of distribution of metrics to calculate.

        Returns
        -------
            `n_moments` of the distribution of the input metric for each frame

        """

        moments = []

        for frame_values in traj_metric:
            frame_mean = np.mean(frame_values)
            moments.append(np.append(frame_mean, s.moment(frame_values,
                                                          range(2, n_moments))))

        return moments
