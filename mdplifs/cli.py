import docopt
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

