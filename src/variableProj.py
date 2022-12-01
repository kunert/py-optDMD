import numpy as np
import scipy as sci
from scipy.sparse import csc_matrix
import copy


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
    A = csc_matrix((m, n), dtype=complex)
    ttemp = np.reshape(t, (m, 1))
    A[:, i] = ttemp * np.exp(alpha[i] * ttemp)
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

    # b_old = backslash(phimat, y)
    b, _, _, _ = np.linalg.lstsq(phimat, y, rcond=None)

    res = y - phimat.dot(b)
    # This expression differs from the original matlab code.
    # errlast = np.linalg.norm(res, 'fro') / res_scale
    # Gamma is a zeros matrix when not using the Tikhonov regularization so errlast
    # should just be this expression.
    errlast = 0.5 * (
            np.linalg.norm(res, 'fro') ** 2
            + np.linalg.norm(np.dot(gamma, alpha), 2) ** 2
    )
    imode = 0

    for itern in range(maxiter):
        # Build Jacobian matrix, looping over alpha indices.
        for j in range(ia):
            dphitemp = dphi(alpha, t, j).astype(complex)
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

        # loop to determine lambda (lambda gives the levenberg part)
        # pre-compute components which don't depend on step-size (lambda)
        # get pivots and lapack-style qr for Jacobian matrix

        # rhstemp[:m * iss] = res
        rhstemp = res.ravel('F')

        # This whole section is a mess. Intermediate terms do not correspond to the
        # matlab equivalents.
        _, _, _, work, _ = sci.linalg.lapack.zgeqp3(djacmat)
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

        # Original matlab code:
        # When not providing linear constraints, `ifproxfun == 0`.
        # if (ifproxfun == 1)
        #     alpha0 = proxfun(alpha + delta0);
        #     delta0 = alpha0 - alpha; % update
        #     delta0
        # else
        #     alpha0 = alpha + delta0;
        #
        # end
        # The sign of delta0 is flipped relative to the matlab code.
        alpha0 = alpha.ravel('F') - delta0.ravel('F')

        phimat = phi(alpha0, t)
        # Replaced with the built-in backslash equivalent function.
        # b0_old = backslash(phimat, y)
        b0, _, _, _ = np.linalg.lstsq(phimat, y, rcond=None)
        res0 = y - phimat.dot(b0)
        # Return to using the original matlab code expression.
        # The below commented expression is from the original python conversion.
        # err0 = np.linalg.norm(res0, 'fro') / res_scale
        err0 = 0.5 * (
            np.linalg.norm(res0, 'fro') ** 2
            + np.linalg.norm(np.dot(gamma, alpha0), 2) ** 2
        )

        # Original matlab code:
        # act_impr = errlast - err0;
        # pred_impr = real(0.5 * delta0'*(g));
        # impr_rat = act_impr / pred_impr;
        g = np.dot(djacmat.conj().T, rhstemp)
        actual_improvement = errlast - err0
        predicted_improvement = np.real(0.5 * np.dot(delta0.conj().T, g))
        improvement_ratio = actual_improvement / predicted_improvement

        # If the residuals improved, update the values.
        if err0 < errlast:
            # New version: rescale lambda based on actual vs predicted improvement

            # Original matlab code
            # lambda0 = lambda0 * max(1.0 / 3.0, 1 - (2 * impr_rat - 1) ^ 3);
            # alpha = alpha0;
            # errlast = err0;
            # b = b0;
            # res = res0;
            lambda0 = lambda0 * np.max(
                [1.0 / 3.0, 1 - (2 * improvement_ratio - 1) ** 3]
            )
            alpha = alpha0
            errlast = err0
            b = b0
            res = res0

            # # see if smaller lambda is better
            # lambda1 = lambda0 / lamdown
            # delta1 = varpro2_solve_special(rjac, lambda1 * np.diag(scalespvt), rhs)
            # delta1 = delta1[jpvt - 1]
            #
            # alpha1 = alpha.ravel() - delta1.ravel()
            # phimat = phi(alpha1, t)
            # b1 = backslash(phimat, y)
            # res1 = y - phimat.dot(b1)
            # err1 = np.linalg.norm(res1, 'fro') / res_scale
            #
            # if err1 < err0:
            #     lambda0 = copy.copy(lambda1)
            #     alpha = copy.copy(alpha1)
            #     errlast = copy.copy(err1)
            #     b = copy.copy(b1)
            #     res = copy.copy(res1)
            # else:
            #     alpha = copy.copy(alpha0)
            #     errlast = copy.copy(err0)
            #     b = copy.copy(b0)
            #     res = copy.copy(res0)
        else:
            # If the residuals did not improve, increase lambda until something works.
            # This makes the algorithm more like gradient descent
            for j in range(maxlam):
                lambda0 = lambda0 * lamup
                delta0 = varpro2_solve_special(rjac, lambda0 * np.diag(scalespvt), rhs)
                delta0 = delta0[jpvt - 1]

                # Original matlab code:
                # alpha0 = alpha + delta0;
                # if (ifproxfun == 1)
                #     alpha0 = proxfun(alpha0);
                #     delta0 = alpha0 - alpha;
                # end
                # The sign of delta0 is flipped relative to the reference matlab code.
                alpha0 = alpha.ravel() - delta0.ravel()

                phimat = phi(alpha0, t)
                b0 = backslash(phimat, y)
                res0 = y - phimat.dot(b0)

                # err0 = np.linalg.norm(res0, 'fro') / res_scale
                err0 = 0.5 * (
                        np.linalg.norm(res0, 'fro') ** 2
                        + np.linalg.norm(np.dot(gamma, alpha0), 2) ** 2
                )

                if err0 < errlast:
                    # print 'HERE' #-- triggered on both
                    break

            if err0 < errlast:
                # print 'HERE'
                alpha = copy.copy(alpha0)
                errlast = copy.copy(err0)
                b = copy.copy(b0)
                res = copy.copy(res0)
            else:
                # No appropriate step length was found.
                niter = itern
                err[itern] = errlast
                imode = 4
                step_length_error_string = (
                    'Failed to find appropriate step length at iteration {:}\n'
                    ' Current residual {:}'
                )
                print(step_length_error_string.format(itern, errlast))
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
                print(stall_error_string.format(eps_stall, itern, errlast))
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
        
    # Iterations failed to meet tolerance in maxiter number of steps.
    niter = maxiter
    imode = 1
    maxiter_tolerance_error_string = (
        'failed to reach tolerance after maxiter={:} iterations \n current residual {:}'
    )
    print(maxiter_tolerance_error_string.format(maxiter, errlast))