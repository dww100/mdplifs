from docopt import docopt
import mdtraj
from . import featurization


usage = """
mdplifs

Usage:
  mdplifs (<topology> <trajectory>) [<ligand_selection>]
  
Arguments:
  topology            System topology (currently only AMBER prmtop supported)
  trajectory          Molecular dynamics trajectory
  ligand_selection    Selection text for the ligand (default='resname LIG')
"""


def main(argv=None):

    args = docopt(usage, argv=argv, version='0.0.1')

    topology_filename = args['<topology>']
    trajectory_filename = args['<trajectory>']
    ligand_selection = args['<ligand_selection>']

    traj = mdtraj.load(trajectory_filename, top=topology_filename)

    if ligand_selection is None:
        featurization.Fingerprinter(traj, top_path=topology_filename)
    else:
        featurization.Fingerprinter(traj, top_path=topology_filename,
                                    ligand_selection=ligand_selection)
