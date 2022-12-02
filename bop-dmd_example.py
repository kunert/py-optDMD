import numpy as np
from src import optimalDMD


# Generate the synthetic data.
# Set up modes in space.
x0 = 0
x1 = 1
nx = 200

# Space component is evenly spaced originally.
xspace = np.linspace(x0,x1,nx)

# Set up the spatial modes
f1 = np.sin(xspace)[:, np.newaxis]
f2 = np.cos(xspace)[:, np.newaxis]
f3 = np.tanh(xspace)[:, np.newaxis]

# Set up the time dynamics.
t0 = 0
t1 = 1
nt = 100
ts = np.linspace(t0,t1,nt)[np.newaxis, :]

# Eigenvalues for each mode

e1 = 1 + 0j
e2 = -2 + 0j
e3 = 0 + 1j
true_eigenvalues = np.array([e1, e2, e3])

# Generate the clean, noiseless dynamics.
# xclean = f1'*exp(e_1*ts) + f2'*exp(e_2*ts) + f3'*exp(e_3*ts);
xclean = f1.dot(np.exp(e1 * ts)) + f2.dot(np.exp(e2 * ts)) + f3.dot(np.exp(e3 * ts))

# Set the random seed
# rng(7);
rng = np.random.default_rng()

# Number of time points
n = len(ts)

## BOP-DMD

# Number you want to choose (for each ensemble member?)
p = 50
# Number of noise cycles
num_noise_cycles = 1
# number of cycles for each noise cycle
num_cycles = 100

# Create the lambda vector for DMD cycles.
lambda_vec_DMD = np.zeros((3, num_noise_cycles))

# Create the lambda vector for optdmd cycles.
lambda_vec_optDMD = np.zeros((3, num_noise_cycles))

# Create the lambda vector for optdmd cycles (how is this different?)
lambda_vec_mean_ensembleDMD = np.zeros((3, num_cycles))
niter_ensemble = np.zeros((num_cycles))
err_ensemble = np.zeros((num_cycles))
lambda_ensemble = np.zeros((num_cycles))
imode_ensemble = np.zeros((num_cycles))

for k in range(num_noise_cycles):
    # Create data for noise cycle (add random noise)
    sigma = .05
    xdata = xclean + sigma * rng.standard_normal(xclean.shape)

    # Create the lambda vector for ensembleDMD cycle
    lambda_vec_ensembleDMD = np.zeros((3, num_cycles)).astype(complex)

    # Try the regular optdmd (without bagging)
    w_opt, e_opt, b_opt, _, _, _, _ = optimalDMD.optdmd(xdata, ts, 3, 1)

    for j in range(num_cycles):
        # Try with optdmd using optDMD modes as initial conditions
        # Randomly select time indices.
        ind = rng.integers(low=0, high=ts.size - 1, size=p)

        # Sort the index to be in ascending order. This step generates variable length
        # time steps.
        ind = np.sort(ind)

        # Create the subselected dataset for this cycle using the sorted indices.
        xdata_cycle = xdata[:, ind]
        ts_ind = ts[:, ind]

        # Solve optdmd for this ensemble member.
        w_cycle, e1_cycle, b_cycle, niter_cycle, err_cycle, lambda0_cycle, imode_cycle = optimalDMD.optdmd(
            xdata_cycle, ts_ind, 3, 0, alpha_init=e_opt, verbose=True)
        lambda_vec_ensembleDMD[:, j] = e1_cycle
        niter_ensemble[j] = niter_cycle
        err_ensemble[j] = err_cycle[-1]
        lambda_ensemble[j] = lambda0_cycle
        imode_ensemble[j] = imode_cycle

    # lambda_vec_DMD[:, k] = diag(lam_DMD)
    lambda_vec_optDMD[:, k] = e_opt
