import munkres
import scipy
import numpy as np
from pyoptdmd import optimalDMD


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


def fit(xdata, ts, modes, num_ensembles=10, ensemble_size=None, verbose=False, seed=None):
    """

    Warnings: currently only works on 2D data with a spatial dimension and a time
    dimension. Anything else will require more careful specification of dimensions.

    """

    spatial_length = xdata.shape[0]
    time_length = ts.size

    if not time_length == xdata.shape[-1]:
        raise ValueError('xdata and ts did not have the expected shape.')

    # Create the lambda vector for ensembleDMD cycle
    e_ensembleDMD = np.zeros((modes, num_ensembles)).astype(complex)
    w_ensembleDMD = np.zeros((spatial_length, modes, num_ensembles)).astype(complex)
    b_ensembleDMD = np.zeros((modes, num_ensembles)).astype(complex)

    # Substantiate the random generator.
    rng = np.random.default_rng(seed)

    if ensemble_size is None:
        ensemble_size = ts.size // 2

    # Fit the non-optimal variable projection DMD?
    # phi_DMD, lam_DMD, b_DMD, sig_DMD = DMD(xdata(:,1:end-1), xdata(:,2:end), 3);

    # Try the optdmd without bagging to get an initial guess.
    w_opt, e_opt, b_opt, _ = optimalDMD.optdmd(xdata, ts, modes, 1)

    linear_algebra_error_counter = 0
    j = 0

    while j < num_ensembles:
        # Randomly select time indices for this ensemble member.
        ind = rng.integers(low=0, high=ts.size - 1, size=ensemble_size)

        # Sort the index to be in ascending order, generating variable length time steps.
        ind = np.sort(ind)

        # Create the sub-selected data for this ensemble member using the sorted indices.
        xdata_cycle = xdata[:, ind]
        ts_ind = ts[:, ind]

        # For very high levels of noise the linear least squares svd solver
        # can fail. Catch those and pass on to the next ensemble member.
        try:
            # Solve optdmd for this ensemble member. Use optDMD modes as initial
            # conditions for BOP-DMD
            w_cycle, e_cycle, b_cycle, exit_mode = optimalDMD.optdmd(
                xdata_cycle, ts_ind, modes, 0, alpha_init=e_opt, verbose=verbose
            )

            # Match the eigenvalues to those from first optdmd call using the Munkres
            # algorithm. This step ensures that the eigenvalues have about the right
            # ordering.
            _, indices = match_vectors(e_cycle, e_opt)

            # if exit_mode.lambda_solver == 'predicted' and exit_mode.exit_state == 'maxiter':
            #     continue
            # else:
            # Assign to the outer container using the correct ordering.
            e_ensembleDMD[:, j] = e_cycle[indices].flatten()
            w_ensembleDMD[:, :, j] = w_cycle[:, indices]
            b_ensembleDMD[:, j] = b_cycle[indices].flatten()
            j += 1

        except np.linalg.LinAlgError:
            linear_algebra_error_counter += 1

        if linear_algebra_error_counter > num_ensembles:
            raise ValueError('Excessive failures in BOP-DMD solutions.')

    return e_ensembleDMD, w_ensembleDMD, b_ensembleDMD


if __name__ == "__main__":
    # Generate the synthetic data.

    # Set up modes in space.
    x0 = 0
    x1 = 1
    nx = 200

    # Space component is evenly spaced originally.
    xspace = np.linspace(x0, x1, nx)

    # Set up the spatial modes
    f1 = np.sin(xspace)[:, np.newaxis]
    f2 = np.cos(xspace)[:, np.newaxis]
    f3 = np.tanh(xspace)[:, np.newaxis]

    # Set up the time dynamics.
    t0 = 0
    t1 = 1
    nt = 100
    ts = np.linspace(t0, t1, nt)[np.newaxis, :]

    # Eigenvalues for each mode
    e1 = 1 + 0j
    e2 = -2 + 0j
    e3 = 0 + 1j
    true_eigenvalues = np.array([e1, e2, e3])

    # Generate the clean, noiseless dynamics.
    xclean = f1.dot(np.exp(e1 * ts)) + f2.dot(np.exp(e2 * ts)) + f3.dot(np.exp(e3 * ts))

    # Set the random seed
    rng = np.random.default_rng(1)

    # Number of time points
    n = len(ts)
    sigma = 1e-1

    # Create data for noise cycle (add random noise)
    xdata = xclean + sigma * rng.standard_normal(xclean.shape)

    e, w, b = fit(xdata, ts, 3, num_ensembles=10, ensemble_size=50, verbose=False,
                  seed=None)
