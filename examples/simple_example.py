import numpy as np
import src.optimalDMD as optimalDMD


# Simple example
# set up modes in space
x0 = 0
x1 = 1
nx = 200

# space
xspace = np.linspace(x0, x1, nx)

# modes
f1 = np.sin(xspace)[:, None]
f2 = np.cos(xspace)[:, None]
f3 = np.tanh(xspace)[:, None]

# set up time dynamics
t0 = 0
t1 = 1
nt = 100

ts = np.linspace(t0, t1, nt)[None, :]

# eigenvalues
e1 = 1.0
e2 = -2.0
e3 = 1.0j

evals = np.array([e1, e2, e3])

# create clean dynamics
xclean = f1.dot(np.exp(e1 * ts)) + f2.dot(np.exp(e2 * ts)) + f3.dot(
    np.exp(e3 * ts))

# The first implementation has no noise to enable a direct comparison
# between the reference matlab code and the python code.
sigma = 0
xdata = xclean + sigma * np.random.randn(*xclean.shape)

r = 3
imode = 0
w, e, b = optimalDMD.optdmd(xdata, ts, r, imode)

add_noise = False
if add_noise:
    # A second implementation with noise (useful when debugging). Setting an
    # identical seed and random number sequence between MATLAB and python is
    # difficult so the exact comparisons are not done.
    sigma = 1e-3
    xdata = xclean + sigma * np.random.randn(*xclean.shape)

    r = 3
    imode = 0
    w, e, b = optimalDMD.optdmd(xdata, ts, r, imode)