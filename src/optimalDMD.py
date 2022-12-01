import numpy as np
from .variableProj import *
import sys


def optdmd(X, t, r, imode, alpha_init=None, verbose=False):
    # Pre-compute U.
    u, _, _ = np.linalg.svd(X, full_matrices=False)

    if alpha_init is None:

        # Check for varpro options and use defaults when necessary.
        # opt = varpro_opts()

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
    # These statements need cleaning up.
    t = t
    phi = lambda a, t: varpro2expfun(a, t)
    dphi = lambda a, t, i: varpro2dexpfun(a, t, i)
    # alpha_init = alpha_init
    opts = varpro_opts()

    w, e, niter, err, imode, alphas = varpro2(y, t, phi, dphi, m, iss, ia, alpha_init,
                                              opts, verbose=verbose)
    w = w.T

    # Normalize
    b = np.sqrt(np.sum(np.absolute(w) ** 2.0, 0)).T
    w = w.dot(np.diag(1.0 / b))

    return w, e, b


# if __name__ == '__main':
#     # Simple example
#     # set up modes in space
#     x0 = 0
#     x1 = 1
#     nx = 200
#
#     # space
#     xspace = np.linspace(x0, x1, nx)
#
#     # modes
#     f1 = np.sin(xspace)[:, None]
#     f2 = np.cos(xspace)[:, None]
#     f3 = np.tanh(xspace)[:, None]
#
#     # set up time dynamics
#     t0 = 0
#     t1 = 1
#     nt = 100
#
#     ts = np.linspace(t0, t1, nt)[None, :]
#
#     # eigenvalues
#     e1 = 1.0
#     e2 = -2.0
#     e3 = 1.0j
#
#     evals = np.array([e1, e2, e3])
#
#     # create clean dynamics
#     xclean = f1.dot(np.exp(e1 * ts)) + f2.dot(np.exp(e2 * ts)) + f3.dot(
#         np.exp(e3 * ts))
#
#     # add noise
#     sigma = 1e-3
#     xdata = xclean + sigma * np.random.randn(*xclean.shape)
#
#     r = 3
#     imode = 0
#     w, e, b = optdmd(xdata, ts, r, imode)