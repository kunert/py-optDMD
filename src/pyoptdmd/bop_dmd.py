import munkres
import scipy
import numpy as np


def match_vectors(vector1, vector2):
    """ Wrapper for MUNKRES converted from MATLAB.

    Sets up a cost function so that the indices
    returned by munkres correspond to the permutation
    which minimizes the 1-norm of the difference
    between vector1[indices] and vector2[indices ~= 0].

    The python version of munkres returns both the column and
    row indices whereas the MATLAB version returns just the
    column indices.

    Example:

    row_indices, col_indices = match_vectors(vector1,vector2)

    See also the munkres package.
    """

    m = munkres.Munkres()

    # Deliberately expand to have matlab-like vectors
    if vector1.ndim == 1:
        vector1 = vector1[:, np.newaxis]
    if vector2.ndim == 1:
        vector2 = vector2[:, np.newaxis]

    if scipy.sparse.issparse(vector1) or scipy.sparse.issparse(vector2):
        raise ValueError('Numpy Kronecker product can fail silently with sparse matrices')

    cost_matrix = np.abs(np.kron(vector1.T, np.ones((len(vector2), 1))) - np.kron(vector2,
                                                                                  np.ones(
                                                                                      (1,
                                                                                       len(vector1)))))
    # indices, cost = munkres(costmat)
    indices = m.compute(cost_matrix)

    # For the use cases currently implementing this method, it is desirable to have the
    # row and column indices separate.
    row_indices, col_indices = list(zip(*indices))

    # Return numpy arrays instead of lists
    return np.array(row_indices), np.array(col_indices)