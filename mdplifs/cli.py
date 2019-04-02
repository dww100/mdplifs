import sys
import logging
from docopt import docopt
import mdtraj
from . import featurization

logger = logging.getLogger(__name__)

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)

usage = """mdplifs

Usage:
  mdplifs (<topology> <trajectory>) [<ligand_selection>] [--first=<first> --last=<last> --stride=<stride>]
  mdplifs -h | --help 

Arguments:
  topology            System topology (currently only AMBER prmtop supported)
  trajectory          Molecular dynamics trajectory
  ligand_selection    Selection text for the ligand (default='resname LIG')

Options:
  -h --help           Show this screen.
  --first=<first>     First frame to use in analysis [default: 1].
  --last=<last>       Last frame to use in analysis [default: -1].
  --stride=<stride>   Step between frames to be used in analysis [default: 1].
"""


def get_frame_range(traj, first, last, stride):

    n_frames = traj.n_frames

    try:
        first = int(first)
        last = int(last)
        stride = int(stride)
    except ValueError:
        logger.critical('Frame selections (first, last & stride) must all be '
                        'integers')
        sys.exit(1)

    if last < -1:
        logger.critical('Last frame must be > -1 (selects final frame of '
                        'trajectory)')
        sys.exit(1)
    elif last > n_frames:
        logger.warning('Last frame selected after end of trajectory, '
                       'using last frame.')
        last = -1

    if first < -1 or (first > last and last != -1):
        logger.critical('First frame must be > -1 and occur before the last'
                        ' frame selected.')
        sys.exit(1)

    if stride > last - first and last != -1:
        logger.warning('Selected frame stride greater than difference '
                       'between first and last frame')

    return slice(first, last, stride)


def main(argv=None):

    args = docopt(usage, argv=argv, version='0.0.1')

    topology_filename = args['<topology>']
    trajectory_filename = args['<trajectory>']
    ligand_selection = args['<ligand_selection>']
    first = args['--first']
    last = args['--last']
    stride = args['--stride']

    traj = mdtraj.load(trajectory_filename, top=topology_filename)

    frames = get_frame_range(traj, first, last, stride)

    if ligand_selection is None:
        featurization.Fingerprinter(traj, frames=frames,
                                    top_path=topology_filename)
    else:
        featurization.Fingerprinter(traj, frames=frames,
                                    top_path=topology_filename,
                                    ligand_selection=ligand_selection)
