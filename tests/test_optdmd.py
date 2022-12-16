import unittest
import numpy as np
import warnings
from src import optimalDMD


class optdmdTestCase(unittest.TestCase):
    def test_exact_optDMD(self):
        # optDMD triggers several ignorable warnings. To not clutter the unittest
        # output these are turned off briefly.
        warnings.filterwarnings('ignore')

        matlab_test_values = np.array(
            [
                -1.999954504915882 - 0.000000770522912j,
                0.999990878733956 - 0.000010354786854j,
                0.000008850461444 + 1.000013991908446j
            ]
        )

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
        x_clean = f1.dot(np.exp(e1 * ts)) + f2.dot(np.exp(e2 * ts)) + f3.dot(
            np.exp(e3 * ts))

        # Explicitly specify the number of modes using r = 3.
        r = 3

        # Call the optDMD and solve.
        w, optdmd_e_no_noise, b, _ = optimalDMD.optdmd(x_clean, ts, r, 0, verbose=False)

        # Verify the python and matlab solutions are equal to a high level of precision.
        np.testing.assert_array_almost_equal(
            optdmd_e_no_noise, matlab_test_values, decimal=7, verbose=True
        )
        # Turn back on warnings.
        warnings.filterwarnings('default')


if __name__ == '__main__':
    unittest.main()
