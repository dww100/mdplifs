import numpy as np
import scipy.stats as s


def describe_distribution(data, n_moments=8):
    """
    Calculate the `mean` and `n_moments` of the distribution of the input data.

    Parameters
    ----------
    data :  `numpy.array_like`
        List of input datapoints.
    n_moments : int
        Number of moments to calculate.

    Returns
    -------
    np.array
        Contains the mean and moments describing the input distribution.
    """

    # TODO: Check input is 1D

    average = np.mean(data)
    moments = s.moment(data, range(2, 2+n_moments))

    return np.append(average, moments)
