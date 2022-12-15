import numpy as np
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import src.variableProj as variableProj


def optdmd(X, t, r, imode, alpha_init=None, verbose=False, opts=None):
    # Pre-compute U.
    u, _, _ = np.linalg.svd(X, full_matrices=False)

    if alpha_init is None:
        # Use the projected trapezoidal rule approximation for the initial guess.
        ux1 = np.dot(u.conj().T, X[:, :-1])
        ux2 = np.dot(u.conj().T, X[:, 1:])

        t1 = t[:, :-1]
        t2 = t[:, 1:]

        dx = (ux2 - ux1) * 1.0 / (t2 - t1)
        xin = (ux1 + ux2) / 2.0

        [u1, s1, v1] = np.linalg.svd(xin, full_matrices=False)
        v1 = v1.conj().T
        u1 = u1[:, :r]
        v1 = v1[:, :r]
        s1inv = np.diag(1.0 / s1[:r])

        atilde = u1.conj().T.dot(dx.dot(v1.dot(s1inv)))
        alpha_init = np.linalg.eig(atilde)[0]

    # Fit all of the data (imode == 1).
    m = t.size
    n = u.shape[1]
    ia = r
    iss = X.shape[0]

    y = X.T
    # @ToDo: Clean up these statements.
    t = t
    phi = lambda a, t: variableProj.varpro2expfun(a, t)
    dphi = lambda a, t, i: variableProj.varpro2dexpfun(a, t, i)

    if opts is None:
        opts = variableProj.varpro_opts()

    w, e, niter, err, exit_mode, alphas = variableProj.varpro2(
        y, t, phi, dphi, m, iss, ia, alpha_init, opts, verbose=verbose)

    w = w.T

    # Normalize
    b = np.sqrt(np.sum(np.absolute(w) ** 2.0, 0)).T
    w = w.dot(np.diag(1.0 / b))

    return w, e, b, exit_mode
