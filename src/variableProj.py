import numpy as np
import scipy as sci
from scipy.sparse import csc_matrix
import copy
import warnings


warnings.simplefilter("ignore", np.ComplexWarning)


# @ToDo: Clean up VisibleDeprecationWarning from creating an ndarray from ragged nested
#  sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different
#  lengths or shapes) since this is now deprecated in numpy.
def backslash(A, B):
    # I initially replaced MATLAB's backslash command with a single call to
    # np.linalg.lstsq which should be mostly equivalent most of the time. However,
    # giving the
    x = []
    for k in range(B.shape[1]):
        b = B[:, k][:, None]
        x.append(np.linalg.lstsq(A, b, rcond=None)[0])
    return np.hstack(x)


def varpro2expfun(alphaf, tf):
    alpha = copy.copy(alphaf)
    t = copy.copy(tf)
    return np.exp(np.reshape(t, (-1, 1)).dot(np.reshape(alpha, (1, -1))))


def varpro2dexpfun(alphaf, tf, i):
    alpha = copy.copy(alphaf)
    t = copy.copy(tf)
    m = t.size
    n = alpha.size
    if (i < 0) | (i >= n):
        raise ValueError('varpro2dexpfun: i outside of index range for alpha')

    # @ToDo: Fix sparse matrix cluster and SparseEfficiencyWarning.
    # I believe this segment is where a SparseEfficiencyWarning is raised. It seems
    # reasonable to follow the advice given in the 2nd answer to the stackoverflow
    # question and make this a dense matrix at first before converting to sparse when
    # passing A out of the function.
    #
    # stackoverflow.com/questions/33091397/sparse-efficiency-warning-while-changing
    # -the-column

    # Original python implementation:
    A = csc_matrix((m, n), dtype=complex)
    ttemp = np.reshape(t, (m, 1))
    A[:, i] = ttemp * np.exp(alpha[i] * ttemp)

    # @ToDo: fix inconsistent matrix sizes.
    # The dense implementation revealed that the matrix sizes were inconsistent. But,
    # since the csc_matrix sparse type appears to be (too) flexible in this regard.
    # A = np.zeros((m, n), dtype=complex)
    # ttemp = np.reshape(t, (m, 1))
    # A[:, i] = (ttemp * np.exp(alpha[i] * ttemp)).squeeze()

    return A


def varpro2_solve_special(R, D, b):
    A = np.concatenate((R, D), 0)
    b = copy.copy(b)
    m, n = R.shape
    ma, na = A.shape
    if (ma != len(b)) | (ma != (m + n)) | (na != n):
        raise ValueError('Input matrix dimensions inconsistent')
    for i in range(n):
        ind = np.array([i] + [m + k for k in range(i + 1)])
        u = A[ind, i][:, None]
        sigma = np.linalg.norm(u)
        beta = 1.0 / (sigma * (sigma + np.abs(u[0])))
        u[0] = np.sign(u[0]) * (sigma + abs(u[0]))
        A[ind, i:] += -(beta * u).dot((u.conj().T.dot(A[ind, i:])))
        b[ind] += -(beta * u).dot(u.conj().T.dot(b[ind]))
    RA = np.triu(A)[:n, :n]
    return backslash(RA, b[:n])


def checkinputrange(xname, xval, xmin, xmax):
    if xval > xmax:
        max_input_string = ('Option {:} with value {:} is greater than {:}, which is '
                            'not recommended.')
        print(max_input_string.format(
            xname, xval, xmin, xmax)
        )
    if xval < xmin:
        min_input_string = ('Option {:} with value {:} is greater than {:}, which is '
                            'not recommended')
        print(min_input_string.format(xname, xval, xmin, xmax))


class varpro_opts(object):
    def __init__(self, lambda0=1.0, maxlam=52, lamup=2.0, lamdown=2.0, ifmarq=1,
                 maxiter=30, tol=1.0e-6, eps_stall=1.0e-12, iffulljac=1):
        checkinputrange('lambda0', lambda0, 0.0, 1.0e16)
        checkinputrange('maxlam', maxlam, 0, 200)
        checkinputrange('lamup', lamup, 1.0, 1.0e16)
        checkinputrange('lamdown', lamdown, 1.0, 1.0e16)
        checkinputrange('ifmarq', ifmarq, -np.Inf, np.Inf)
        checkinputrange('maxiter', maxiter, 0, 1e12)
        checkinputrange('tol', tol, 0, 1e16)
        checkinputrange('eps_stall', eps_stall, -np.Inf, np.Inf)
        checkinputrange('iffulljac', iffulljac, -np.Inf, np.Inf)
        self.lambda0 = float(lambda0)
        self.maxlam = int(maxlam)
        self.lamup = float(lamup)
        self.lamdown = float(lamdown)
        self.ifmarq = int(ifmarq)
        self.maxiter = int(maxiter)
        self.tol = float(tol)
        self.eps_stall = float(eps_stall)
        self.iffulljac = int(iffulljac)

    def unpack(self):
        return self.lambda0, self.maxlam, self.lamup, self.lamdown, self.ifmarq, self.maxiter, self.tol, self.eps_stall, self.iffulljac


def varpro2(y, t, phi, dphi, m, iss, ia, alpha_init, opts=None, verbose=False):
    if opts is None:
        opts = varpro_opts()
    lambda0, maxlam, lamup, lamdown, ifmarq, maxiter, tol, eps_stall, iffulljac = opts.unpack()

    # initialize values
    alpha = alpha_init
    alphas = np.zeros((len(alpha), maxiter)).astype(complex)
    djacmat = np.zeros((m * iss, ia)).astype(complex)
    rhstemp = np.zeros((m * iss, 1))
    err = np.zeros((maxiter,))
    res_scale = np.linalg.norm(y, 'fro')
    scales = np.zeros((ia,))
    gamma = np.zeros((ia, ia))

    phimat = phi(alpha, t)
    [U, S, V] = np.linalg.svd(phimat, full_matrices=False)

    S = np.diag(S)
    sd = np.diag(S)
    tolrank = m * np.finfo(float).eps
    irank = np.sum(sd > (tolrank * sd[0]))
    U = U[:, :irank]
    S = S[:irank, :irank]
    V = V[:, :irank].T

    b, _, _, _ = np.linalg.lstsq(phimat, y, rcond=None)

    res = y - phimat.dot(b)
    # Note: gamma is all zeros when not using the Tikhonov regularization (not
    # implemented).
    errlast = 0.5 * (
            np.linalg.norm(res, 'fro') ** 2
            + np.linalg.norm(np.dot(gamma, alpha), 2) ** 2
    )
    imode = 0

    for itern in range(maxiter):
        # Build Jacobian matrix, looping over alpha indices.
        for j in range(ia):
            dphitemp = dphi(alpha, t, j).astype(complex)
            # The matrix sizes are inconsistent here but are saved by something going
            # on in the sparse matrix data type being used.
            djaca = (dphitemp - csc_matrix(U * csc_matrix(U.T.conj() * dphitemp))).dot(b)
            if iffulljac == 1:
                # Use the full expression for the Jacobian.
                djacb = U.dot(backslash(S, V.T.conj().dot(dphitemp.T.conj().dot(res))))
                djacmat[:, j] = -(djaca.ravel(order='F') + djacb.ravel(order='F'))
            else:
                djacmat[:, j] = -djaca.ravel()
            scales[j] = 1.0
            if ifmarq == 1:
                scales[j] = np.minimum(np.linalg.norm(djacmat[:, j]), 1.0)
                scales[j] = np.maximum(scales[j], 1.0e-6)

        # Determine lambda using a loop (lambda gives the levenberg part). Pre-compute
        # components which don't depend on step-size (lambda) and get pivots and
        # lapack-style qr for Jacobian matrix.
        # Note: Intermediate terms do not correspond to the matlab equivalents but
        # yield identical results at the end of this segment.

        rhstemp = res.ravel('F')
        _, _, _, work, _ = sci.linalg.lapack.zgeqp3(djacmat)

        # @ToDo: Figure out how to handle the ComplexWarning.
        # The following lines appear raise an erroneous ComplexWarning (i.e., warning
        # when assigning to arbitrary new variables but all variables have types that
        # match the expected output).
        djacout, jpvt, tau, _, _ = sci.linalg.lapack.zgeqp3(djacmat, work)

        rjac = np.triu(djacout)
        lwork = \
        sci.linalg.lapack.zunmqr('L', 'C', djacout, tau, res.ravel(order='F')[:, None],
                                 -1)[1]
        rhstop = \
        sci.linalg.lapack.zunmqr('L', 'C', djacout, tau, res.ravel(order='F')[:, None],
                                 lwork)[0]
        scalespvt = scales[jpvt - 1]
        rhs = np.concatenate((rhstop, np.zeros((ia, 1)).astype(complex)), 0)

        # Delta0 aligns again with the matlab code.
        delta0 = varpro2_solve_special(rjac, lambda0 * np.diag(scalespvt), rhs)
        delta0 = delta0[jpvt - 1]

        # The sign of delta0 is flipped relative to the matlab code.
        alpha0 = alpha.ravel('F') - delta0.ravel('F')

        phimat = phi(alpha0, t)
        # Replaced with the built-in backslash equivalent function.
        # b0_old = backslash(phimat, y)
        b0, _, _, _ = np.linalg.lstsq(phimat, y, rcond=None)
        res0 = y - phimat.dot(b0)
        # Return to using the reference matlab expression instead of the expression from
        # the original python conversion (below).
        # err0 = np.linalg.norm(res0, 'fro') / res_scale
        err0 = 0.5 * (
            np.linalg.norm(res0, 'fro') ** 2
            + np.linalg.norm(np.dot(gamma, alpha0), 2) ** 2
        )

        # Determine the new step size based on the ratio of expected to actual
        # improvement.
        g = np.dot(djacmat.conj().T, rhstemp)
        actual_improvement = errlast - err0
        predicted_improvement = np.real(0.5 * np.dot(delta0.conj().T, g))
        improvement_ratio = actual_improvement / predicted_improvement

        # If the residuals improved, update the values.
        if err0 < errlast:
            # Rescale lambda based on actual vs predicted improvement
            lambda0 = lambda0 * max(
                [1.0 / 3.0, 1 - (2 * improvement_ratio - 1) ** 3]
            )
            alpha = alpha0
            errlast = err0
            b = b0
            res = res0

        else:
            # If the residuals did not improve, increase lambda until something works.
            # This makes the algorithm more like gradient descent
            for j in range(maxlam):
                lambda0 = lambda0 * lamup
                delta0 = varpro2_solve_special(rjac, lambda0 * np.diag(scalespvt), rhs)
                delta0 = delta0[jpvt - 1]

                # The sign of delta0 is flipped relative to the reference matlab code.
                alpha0 = alpha.ravel('F') - delta0.ravel('F')

                # Replacing the original python conversion with a more direct conversion
                # of the matlab code. The original python conversion is below for
                # reference.
                # phimat = phi(alpha0, t)
                # b0 = backslash(phimat, y)
                # res0 = y - phimat.dot(b0)
                # Replaced with the built-in backslash equivalent function.
                phimat = phi(alpha0, t)
                b0, _, _, _ = np.linalg.lstsq(phimat, y, rcond=None)
                res0 = y - phimat.dot(b0)

                # As above, use a direct expression from the reference matlab instead
                # of the expression from the original python conversion (below).
                # err0 = np.linalg.norm(res0, 'fro') / res_scale
                err0 = 0.5 * (
                        np.linalg.norm(res0, 'fro') ** 2
                        + np.linalg.norm(np.dot(gamma, alpha0), 2) ** 2
                )

                if err0 < errlast:
                    break

            if err0 < errlast:
                # The residual improved so save and move to the next iteration.
                alpha = copy.copy(alpha0)
                errlast = copy.copy(err0)
                b = copy.copy(b0)
                res = copy.copy(res0)
            else:
                # No appropriate step length was found. Exit and return the current
                # values.
                niter = itern
                err[itern] = errlast
                imode = 4
                step_length_error_string = (
                    'Failed to find appropriate step length at iteration {:}\n'
                    ' Current residual {:}'
                )
                if verbose:
                    print(step_length_error_string.format(itern, errlast))
                warnings.resetwarnings()
                return b, alpha, niter, err, imode, alphas

        alphas[:, itern] = alpha
        err[itern] = errlast

        if verbose:
            print('step {} err {} lambda {}'.format(itern, errlast, lambda0))

        # The tolerance was met and the results are passed back out.
        if errlast < tol:
            niter = itern
            return b, alpha, niter, err, imode, alphas

        # Error handling for not converging.
        if itern > 0:
            if err[itern - 1] - err[itern] < eps_stall * err[itern - 1]:
                niter = itern
                imode = 8
                stall_error_string = (
                    'Stall detected: residual reduced by less than {:} \n times '
                    'residual at previous step.'
                    '\niteration: {:}\ncurrent residual: {:.5f}')
                if verbose:
                    print(stall_error_string.format(eps_stall, itern, errlast))
                warnings.resetwarnings()
                return b, alpha, niter, err, imode, alphas

        phimat = phi(alpha, t)
        [U, S, V] = np.linalg.svd(phimat, full_matrices=False)
        S = np.diag(S)
        sd = np.diag(S)
        tolrank = m * np.finfo(float).eps
        irank = np.sum(sd > (tolrank * sd[0]))
        U = U[:, :irank]
        S = S[:irank, :irank]
        V = V[:, :irank].T
        
    # Iterations failed to meet tolerance in `maxiter` number of steps.
    niter = maxiter
    imode = 1
    maxiter_tolerance_error_string = (
        'Failed to reach tolerance after maxiter={:} iterations \n current residual {:}'
    )
    if verbose:
        print(maxiter_tolerance_error_string.format(maxiter, errlast))
    # @ToDo: clean up output
    warnings.resetwarnings()
    return b, alpha, niter, err, imode, alphas
