import numpy as np

def flocal_kernel(x1, x2, q1, q2, n1, n2, nm1, nm2, sigma):
    """
    Translated subroutine from Fortran to Python with NumPy.

    Parameters:
    x1 : numpy.ndarray
        3D array of shape (nm1, max_n1, rep_size)
    x2 : numpy.ndarray
        3D array of shape (nm2, max_n2, rep_size)
    q1 : numpy.ndarray
        2D array of shape (max_n1, nm1)
    q2 : numpy.ndarray
        2D array of shape (max_n2, nm2)
    n1 : numpy.ndarray
        1D array of shape (nm1,)
    n2 : numpy.ndarray
        1D array of shape (nm2,)
    nm1 : int
        Number of molecules in set 1
    nm2 : int
        Number of molecules in set 2
    sigma : float
        Standard deviation for the Gaussian kernel

    Returns:
    kernel : numpy.ndarray
        2D array of shape (nm2, nm1)
    """
    kernel = np.zeros((nm2, nm1))
    inv_sigma2 = -1.0 / (2 * sigma**2)

    rep_size = x1.shape[2]

    for a in range(nm1):

        # Molecule 2
        for b in range(nm2):

            # Atom in Molecule 1
            for j1 in range(n1[a]):

                # Atom in Molecule 2
                for j2 in range(n2[b]):

                    if q1[j1, a] == q2[j2, b]:

                        l2 = np.sum((x1[a, j1, :] - x2[b, j2, :])**2)
                        kernel[b, a] += np.exp(l2 * inv_sigma2)

    return kernel

def get_local_kernel(X1, X2, Q1, Q2, sigma=1):
    """ Calculates the Gaussian kernel matrix K with the local decomposition where :math:`K_{ij}`:

            :math:`K_{ij} = \\sum_{I\\in i} \\sum_{J\\in j}\\exp \\big( -\\frac{\\|X_I - X_J\\|_2^2}{2\\sigma^2} \\big)`

        Where :math: X_{I}` and :math:`X_{J}` are representation vectors of the atomic environments.

        The kernel is the normal Gaussian kernel with the local decomposition of atomic environments.

        For instance atom-centered symmetry functions could be used here.
        K is calculated analytically using an OpenMP parallel Fortran routine.

        :param X1: Array of representations - shape=(N1, rep_size, max_atoms).
        :type X1: numpy array
        :param X2: Array of representations - shape=(N2, rep_size, max_atoms).
        :type X2: numpy array

        :param Q1: List of lists containing the nuclear charges for each molecule.
        :type Q1: list
        :param Q2: List of lists containing the nuclear charges for each molecule.
        :type Q2: list

        :param SIGMA: Gaussian kernel width.
        :type SIGMA: float

        :return: 2D matrix of kernel elements shape=(N1, N2),
        :rtype: numpy array
    """

    N1 = np.array([len(Q) for Q in Q1], dtype=np.int32)
    N2 = np.array([len(Q) for Q in Q2], dtype=np.int32)

    assert N1.shape[0] == X1.shape[0], "Error: List of charges does not match shape of representations"
    assert N2.shape[0] == X2.shape[0], "Error: List of charges does not match shape of representations"

    Q1_input = np.zeros((max(N1), X1.shape[0]), dtype=np.int32)
    Q2_input = np.zeros((max(N2), X2.shape[0]), dtype=np.int32)

    for i, q in enumerate(Q1):
        Q1_input[:len(q),i] = q

    for i, q in enumerate(Q2):
        Q2_input[:len(q),i] = q

    K = flocal_kernel(
            X1,
            X2,
            Q1_input,
            Q2_input,
            N1,
            N2,
            len(N1),
            len(N2),
            sigma
    )

    return K
