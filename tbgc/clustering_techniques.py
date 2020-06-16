"""Clustering techniques, both template-based and the baselines used.

The spectral clustering overloads are based on scikit-learn's, authored by:
Gael Varoquaux gael.varoquaux@normalesup.org
Brian Cheung
Wei LI <kuantkid@gmail.com>
License: BSD 3 clause

Authors:
 * Mateus Riva (mateus.riva@telecom-paris.fr)
 """

import autograd.numpy as np
import warnings
from time import time

from pymanopt.manifolds import Stiefel
from pymanopt import Problem
from pymanopt.solvers import SteepestDescent, ConjugateGradient

import networkx as nx
import sknetwork as skn

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_random_state, as_float_array, check_symmetric
from sklearn.utils.validation import check_array
from sklearn.utils.extmath import _deterministic_vector_sign_flip
from sklearn.metrics.pairwise import pairwise_kernels, rbf_kernel
from sklearn.neighbors import kneighbors_graph
#from sklearn.manifold import spectral_embedding
from sklearn.cluster.k_means_ import k_means
import sklearn.cluster as cluster

from scipy import sparse
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh, lobpcg
from scipy.sparse.csgraph import connected_components
from scipy.sparse.csgraph import laplacian as csgraph_laplacian


def compute_matching_norm(M, O, P):
    """Compute the matching norm between a model and a transformed observation."""
    # Sanity checking: conversions to ndarray
    M, O, P = np.array(M), np.array(O), np.array(P)

    OP = np.matmul(O, P)
    PTOP = np.matmul(P.T, OP)
    sub = np.subtract(M, PTOP)
    return np.linalg.norm(sub, 'fro')


def compute_gradient_norm(M, O, P):
    """Returns the gradient w.r.t. P of the norm."""
    # Sanity checking: conversions to ndarray
    M, O, P = np.array(M), np.array(O), np.array(P)

    PM = np.matmul(P, M)
    OPM = np.matmul(O, PM)
    OP = np.matmul(O, P)
    PTOP = np.matmul(P.T, OP)
    PPTOP = np.matmul(P, PTOP)
    OPPTOP = np.matmul(O, PPTOP)
    sub = np.subtract(OPPTOP, OPM)
    scaled = sub * 4
    return scaled


def find_optimal_P(M, O, manifold_P, cost=compute_matching_norm, verbosity=0):
  """Find the optimal P in a given manifold for a given model and observation.
  Takes as argument a cost function with signature (M,O,P)"""
  # redefining the cost and grad function for the pymanopt standard
  def parametrized_cost(P):
    return cost(M, O, P)

  # initializing the problem and the solver
  matching_problem = Problem(manifold=manifold_P, cost=parametrized_cost, verbosity=verbosity)
  matching_solver = SteepestDescent(maxiter=100000)
  return matching_solver.solve(matching_problem)


def normalization_P(P_prime):
    """Projects a non-negative P_prime back to non-negative right-stochastic-like space by normalising each line."""
    return (P_prime.T / np.sum(P_prime, axis=1)).T


def projection_P(P_prime):
    """Projects $P'$ back into non-negative right-stochastic-like space, line per line, using matrix operations"""
    sorted_prime = -np.sort(-P_prime, axis=1)  # Descending order sort
    cumsum_sorted = np.cumsum(sorted_prime, axis=1)  # Compute cumulative sum of lines
    rho_availability = sorted_prime > (cumsum_sorted - 1) / np.arange(1, P_prime.shape[
        1] + 1)  # Compute non-zero rho candidates
    rho = np.count_nonzero(rho_availability, axis=1)  # Compute number of non-zero values in final line (rho)
    theta = (cumsum_sorted[np.arange(len(rho)), rho - 1] - 1) / (rho)  # Compute lagrange multiplier theta
    P = (P_prime.transpose() - theta).transpose().clip(min=0)  # subtract multiplier, clip negatives

    return P


def find_optimal_P_stochastic(M, O, cost=compute_matching_norm, gradient=compute_gradient_norm, learning_rate=0.0001):
    """Find the optimal P in a given manifold for a given model and observation.
    Takes as argument cost and gradient functions with signatures (M,O,P)"""
    # Getting n and k values
    n, k = O.shape[0], M.shape[0]

    # Random initialization of P w/ normalization
    P = normalization_P(np.random.rand(n, k))

    # Stopping criteria
    max_epochs = 100000
    min_step_size = 1e-20

    # Iterative update loop
    last_cost = cost(M, O, P)
    best_cost = last_cost
    for epoch in range(max_epochs):  # trange = tqdm progress bar
        # Update P
        nabla = gradient(M, O, P)
        P_prime = P - learning_rate * nabla
        P = projection_P(P_prime)

        # Compute cost, update, check convergence
        current_cost = cost(M, O, P)
        if current_cost < best_cost:
            best_cost = current_cost
        step_size = last_cost - current_cost
        last_cost = current_cost
        if min_step_size >= step_size >= 0:
            # print("Minimal step size reached at {} epochs".format(epoch))
            break

    # if epoch == max_epochs - 1:
        # print("Max epochs reached")
    return P


def get_kmeans(P, k, verbosity=0):
  return cluster.KMeans(k, verbose=verbosity).fit(P)


def _graph_connected_component(graph, node_id):
    """Find the largest graph connected components that contains one
    given node
    Parameters
    ----------
    graph : array-like, shape: (n_samples, n_samples)
        adjacency matrix of the graph, non-zero weight means an edge
        between the nodes
    node_id : int
        The index of the query node of the graph
    Returns
    -------
    connected_components_matrix : array-like, shape: (n_samples,)
        An array of bool value indicating the indexes of the nodes
        belonging to the largest connected components of the given query
        node
    """
    n_node = graph.shape[0]
    if sparse.issparse(graph):
        # speed up row-wise access to boolean connection mask
        graph = graph.tocsr()
    connected_nodes = np.zeros(n_node, dtype=np.bool)
    nodes_to_explore = np.zeros(n_node, dtype=np.bool)
    nodes_to_explore[node_id] = True
    for _ in range(n_node):
        last_num_component = connected_nodes.sum()
        np.logical_or(connected_nodes, nodes_to_explore, out=connected_nodes)
        if last_num_component >= connected_nodes.sum():
            break
        indices = np.where(nodes_to_explore)[0]
        nodes_to_explore.fill(False)
        for i in indices:
            if sparse.issparse(graph):
                neighbors = graph[i].toarray().ravel()
            else:
                neighbors = graph[i]
            np.logical_or(nodes_to_explore, neighbors, out=nodes_to_explore)
    return connected_nodes


def _graph_is_connected(graph):
    """ Return whether the graph is connected (True) or Not (False)
    Parameters
    ----------
    graph : array-like or sparse matrix, shape: (n_samples, n_samples)
        adjacency matrix of the graph, non-zero weight means an edge
        between the nodes
    Returns
    -------
    is_connected : bool
        True means the graph is fully connected and False means not
    """
    if sparse.isspmatrix(graph):
        # sparse graph, find all the connected components
        n_connected_components, _ = connected_components(graph)
        return n_connected_components == 1
    else:
        # dense graph, find all connected components start from node 0
        return _graph_connected_component(graph, 0).sum() == graph.shape[0]


def _set_diag(laplacian, value, norm_laplacian):
    """Set the diagonal of the laplacian matrix and convert it to a
    sparse format well suited for eigenvalue decomposition
    Parameters
    ----------
    laplacian : array or sparse matrix
        The graph laplacian
    value : float
        The value of the diagonal
    norm_laplacian : bool
        Whether the value of the diagonal should be changed or not
    Returns
    -------
    laplacian : array or sparse matrix
        An array of matrix in a form that is well suited to fast
        eigenvalue decomposition, depending on the band width of the
        matrix.
    """
    n_nodes = laplacian.shape[0]
    # We need all entries in the diagonal to values
    if not sparse.isspmatrix(laplacian):
        if norm_laplacian:
            laplacian.flat[::n_nodes + 1] = value
    else:
        laplacian = laplacian.tocoo()
        if norm_laplacian:
            diag_idx = (laplacian.row == laplacian.col)
            laplacian.data[diag_idx] = value
        # If the matrix has a small number of diagonals (as in the
        # case of structured matrices coming from images), the
        # dia format might be best suited for matvec products:
        n_diags = np.unique(laplacian.row - laplacian.col).size
        if n_diags <= 7:
            # 3 or less outer diagonals on each side
            laplacian = laplacian.todia()
        else:
            # csr has the fastest matvec and is thus best suited to
            # arpack
            laplacian = laplacian.tocsr()
    return laplacian

def spectral_embedding(adjacency, n_components=8, eigen_solver=None,
                       random_state=None, eigen_tol=0.0,
                       norm_laplacian=True, drop_first=True):
    """Project the sample on the first eigenvectors of the graph Laplacian.
    The adjacency matrix is used to compute a normalized graph Laplacian
    whose spectrum (especially the eigenvectors associated to the
    smallest eigenvalues) has an interpretation in terms of minimal
    number of cuts necessary to split the graph into comparably sized
    components.
    This embedding can also 'work' even if the ``adjacency`` variable is
    not strictly the adjacency matrix of a graph but more generally
    an affinity or similarity matrix between samples (for instance the
    heat kernel of a euclidean distance matrix or a k-NN matrix).
    However care must taken to always make the affinity matrix symmetric
    so that the eigenvector decomposition works as expected.
    Note : Laplacian Eigenmaps is the actual algorithm implemented here.
    Read more in the :ref:`User Guide <spectral_embedding>`.
    Parameters
    ----------
    adjacency : array-like or sparse matrix, shape: (n_samples, n_samples)
        The adjacency matrix of the graph to embed.
    n_components : integer, optional, default 8
        The dimension of the projection subspace.
    eigen_solver : {None, 'arpack', 'lobpcg', or 'amg'}, default None
        The eigenvalue decomposition strategy to use. AMG requires pyamg
        to be installed. It can be faster on very large, sparse problems,
        but may also lead to instabilities.
    random_state : int, RandomState instance or None, optional, default: None
        A pseudo random number generator used for the initialization of the
        lobpcg eigenvectors decomposition.  If int, random_state is the seed
        used by the random number generator; If RandomState instance,
        random_state is the random number generator; If None, the random number
        generator is the RandomState instance used by `np.random`. Used when
        ``solver`` == 'amg'.
    eigen_tol : float, optional, default=0.0
        Stopping criterion for eigendecomposition of the Laplacian matrix
        when using arpack eigen_solver.
    norm_laplacian : bool, optional, default=True
        If True, then compute normalized Laplacian.
    drop_first : bool, optional, default=True
        Whether to drop the first eigenvector. For spectral embedding, this
        should be True as the first eigenvector should be constant vector for
        connected graph, but for spectral clustering, this should be kept as
        False to retain the first eigenvector.
    Returns
    -------
    embedding : array, shape=(n_samples, n_components)
        The reduced samples.
    Notes
    -----
    Spectral Embedding (Laplacian Eigenmaps) is most useful when the graph
    has one connected component. If there graph has many components, the first
    few eigenvectors will simply uncover the connected components of the graph.
    References
    ----------
    * https://en.wikipedia.org/wiki/LOBPCG
    * Toward the Optimal Preconditioned Eigensolver: Locally Optimal
      Block Preconditioned Conjugate Gradient Method
      Andrew V. Knyazev
      https://doi.org/10.1137%2FS1064827500366124
    """
    adjacency = check_symmetric(adjacency)

    try:
        from pyamg import smoothed_aggregation_solver
    except ImportError:
        if eigen_solver == "amg":
            raise ValueError("The eigen_solver was set to 'amg', but pyamg is "
                             "not available.")

    if eigen_solver is None:
        eigen_solver = 'arpack'
    elif eigen_solver not in ('arpack', 'lobpcg', 'amg'):
        raise ValueError("Unknown value for eigen_solver: '%s'."
                         "Should be 'amg', 'arpack', or 'lobpcg'"
                         % eigen_solver)

    random_state = check_random_state(random_state)

    n_nodes = adjacency.shape[0]
    # Whether to drop the first eigenvector
    if drop_first:
        n_components = n_components + 1

    if not _graph_is_connected(adjacency):
        warnings.warn("Graph is not fully connected, spectral embedding"
                      " may not work as expected.")

    laplacian, dd = csgraph_laplacian(adjacency, normed=norm_laplacian,
                                      return_diag=True)
    if (eigen_solver == 'arpack' or eigen_solver != 'lobpcg' and
       (not sparse.isspmatrix(laplacian) or n_nodes < 5 * n_components)):
        # lobpcg used with eigen_solver='amg' has bugs for low number of nodes
        # for details see the source code in scipy:
        # https://github.com/scipy/scipy/blob/v0.11.0/scipy/sparse/linalg/eigen
        # /lobpcg/lobpcg.py#L237
        # or matlab:
        # https://www.mathworks.com/matlabcentral/fileexchange/48-lobpcg-m
        laplacian = _set_diag(laplacian, 1, norm_laplacian)

        # Here we'll use shift-invert mode for fast eigenvalues
        # (see https://docs.scipy.org/doc/scipy/reference/tutorial/arpack.html
        #  for a short explanation of what this means)
        # Because the normalized Laplacian has eigenvalues between 0 and 2,
        # I - L has eigenvalues between -1 and 1.  ARPACK is most efficient
        # when finding eigenvalues of largest magnitude (keyword which='LM')
        # and when these eigenvalues are very large compared to the rest.
        # For very large, very sparse graphs, I - L can have many, many
        # eigenvalues very near 1.0.  This leads to slow convergence.  So
        # instead, we'll use ARPACK's shift-invert mode, asking for the
        # eigenvalues near 1.0.  This effectively spreads-out the spectrum
        # near 1.0 and leads to much faster convergence: potentially an
        # orders-of-magnitude speedup over simply using keyword which='LA'
        # in standard mode.
        try:
            # We are computing the opposite of the laplacian inplace so as
            # to spare a memory allocation of a possibly very large array
            laplacian *= -1
            v0 = random_state.uniform(-1, 1, laplacian.shape[0])
            lambdas, diffusion_map = eigsh(laplacian, k=n_components,
                                           sigma=1.0, which='LM',
                                           tol=eigen_tol, v0=v0)
            embedding = diffusion_map.T[n_components::-1]
            if norm_laplacian:
                embedding = embedding / dd
        except RuntimeError:
            # When submatrices are exactly singular, an LU decomposition
            # in arpack fails. We fallback to lobpcg
            eigen_solver = "lobpcg"
            # Revert the laplacian to its opposite to have lobpcg work
            laplacian *= -1

    if eigen_solver == 'amg':
        # Use AMG to get a preconditioner and speed up the eigenvalue
        # problem.
        if not sparse.issparse(laplacian):
            warnings.warn("AMG works better for sparse matrices")
        # lobpcg needs double precision floats
        laplacian = check_array(laplacian, dtype=np.float64,
                                accept_sparse=True)
        laplacian = _set_diag(laplacian, 1, norm_laplacian)
        ml = smoothed_aggregation_solver(check_array(laplacian, 'csr'))
        M = ml.aspreconditioner()
        X = random_state.rand(laplacian.shape[0], n_components + 1)
        X[:, 0] = dd.ravel()
        lambdas, diffusion_map = lobpcg(laplacian, X, M=M, tol=1.e-12,
                                        largest=False)
        embedding = diffusion_map.T
        if norm_laplacian:
            embedding = embedding / dd
        if embedding.shape[0] == 1:
            raise ValueError

    elif eigen_solver == "lobpcg":
        # lobpcg needs double precision floats
        laplacian = check_array(laplacian, dtype=np.float64,
                                accept_sparse=True)
        if n_nodes < 5 * n_components + 1:
            # see note above under arpack why lobpcg has problems with small
            # number of nodes
            # lobpcg will fallback to eigh, so we short circuit it
            if sparse.isspmatrix(laplacian):
                laplacian = laplacian.toarray()
            lambdas, diffusion_map = eigh(laplacian)
            embedding = diffusion_map.T[:n_components]
            if norm_laplacian:
                embedding = embedding / dd
        else:
            laplacian = _set_diag(laplacian, 1, norm_laplacian)
            # We increase the number of eigenvectors requested, as lobpcg
            # doesn't behave well in low dimension
            X = random_state.rand(laplacian.shape[0], n_components + 1)
            X[:, 0] = dd.ravel()
            lambdas, diffusion_map = lobpcg(laplacian, X, tol=1e-15,
                                            largest=False, maxiter=2000)
            embedding = diffusion_map.T[:n_components]
            if norm_laplacian:
                embedding = embedding / dd
            if embedding.shape[0] == 1:
                raise ValueError

    embedding = _deterministic_vector_sign_flip(embedding)
    if drop_first:
        return embedding[1:n_components].T, diffusion_map
    else:
        return embedding[:n_components].T, diffusion_map



def discretize(vectors, copy=True, max_svd_restarts=30, n_iter_max=20,
               random_state=None):
    """Search for a partition matrix (clustering) which is closest to the
    eigenvector embedding.

    Parameters
    ----------
    vectors : array-like, shape: (n_samples, n_clusters)
        The embedding space of the samples.

    copy : boolean, optional, default: True
        Whether to copy vectors, or perform in-place normalization.

    max_svd_restarts : int, optional, default: 30
        Maximum number of attempts to restart SVD if convergence fails

    n_iter_max : int, optional, default: 30
        Maximum number of iterations to attempt in rotation and partition
        matrix search if machine precision convergence is not reached

    random_state : int, RandomState instance or None (default)
        Determines random number generation for rotation matrix initialization.
        Use an int to make the randomness deterministic.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    labels : array of integers, shape: n_samples
        The labels of the clusters.

    References
    ----------

    - Multiclass spectral clustering, 2003
      Stella X. Yu, Jianbo Shi
      https://www1.icsi.berkeley.edu/~stellayu/publication/doc/2003kwayICCV.pdf

    Notes
    -----

    The eigenvector embedding is used to iteratively search for the
    closest discrete partition.  First, the eigenvector embedding is
    normalized to the space of partition matrices. An optimal discrete
    partition matrix closest to this normalized embedding multiplied by
    an initial rotation is calculated.  Fixing this discrete partition
    matrix, an optimal rotation matrix is calculated.  These two
    calculations are performed until convergence.  The discrete partition
    matrix is returned as the clustering solution.  Used in spectral
    clustering, this method tends to be faster and more robust to random
    initialization than k-means.

    """

    from scipy.sparse import csc_matrix
    from scipy.linalg import LinAlgError

    random_state = check_random_state(random_state)

    vectors = as_float_array(vectors, copy=copy)

    eps = np.finfo(float).eps
    n_samples, n_components = vectors.shape

    # Normalize the eigenvectors to an equal length of a vector of ones.
    # Reorient the eigenvectors to point in the negative direction with respect
    # to the first element.  This may have to do with constraining the
    # eigenvectors to lie in a specific quadrant to make the discretization
    # search easier.
    norm_ones = np.sqrt(n_samples)
    for i in range(vectors.shape[1]):
        vectors[:, i] = (vectors[:, i] / np.linalg.norm(vectors[:, i])) \
            * norm_ones
        if vectors[0, i] != 0:
            vectors[:, i] = -1 * vectors[:, i] * np.sign(vectors[0, i])

    # Normalize the rows of the eigenvectors.  Samples should lie on the unit
    # hypersphere centered at the origin.  This transforms the samples in the
    # embedding space to the space of partition matrices.
    vectors = vectors / np.sqrt((vectors ** 2).sum(axis=1))[:, np.newaxis]

    svd_restarts = 0
    has_converged = False

    # If there is an exception we try to randomize and rerun SVD again
    # do this max_svd_restarts times.
    while (svd_restarts < max_svd_restarts) and not has_converged:

        # Initialize first column of rotation matrix with a row of the
        # eigenvectors
        rotation = np.zeros((n_components, n_components))
        rotation[:, 0] = vectors[random_state.randint(n_samples), :].T

        # To initialize the rest of the rotation matrix, find the rows
        # of the eigenvectors that are as orthogonal to each other as
        # possible
        c = np.zeros(n_samples)
        for j in range(1, n_components):
            # Accumulate c to ensure row is as orthogonal as possible to
            # previous picks as well as current one
            c += np.abs(np.dot(vectors, rotation[:, j - 1]))
            rotation[:, j] = vectors[c.argmin(), :].T

        last_objective_value = 0.0
        n_iter = 0

        while not has_converged:
            n_iter += 1

            t_discrete = np.dot(vectors, rotation)

            labels = t_discrete.argmax(axis=1)
            vectors_discrete = csc_matrix(
                (np.ones(len(labels)), (np.arange(0, n_samples), labels)),
                shape=(n_samples, n_components))

            t_svd = vectors_discrete.T * vectors

            try:
                U, S, Vh = np.linalg.svd(t_svd)
                svd_restarts += 1
            except LinAlgError:
                print("SVD did not converge, randomizing and trying again")
                break

            ncut_value = 2.0 * (n_samples - S.sum())
            if ((abs(ncut_value - last_objective_value) < eps) or
                    (n_iter > n_iter_max)):
                has_converged = True
            else:
                # otherwise calculate rotation and continue
                last_objective_value = ncut_value
                rotation = np.dot(Vh.T, U.T)

    if not has_converged:
        raise LinAlgError('SVD did not converge')
    return labels


def spectral_clustering(affinity, n_clusters=8, n_components=None,
                        eigen_solver=None, random_state=None, n_init=10,
                        eigen_tol=0.0, assign_labels='kmeans'):
    """Apply clustering to a projection of the normalized Laplacian.

    In practice Spectral Clustering is very useful when the structure of
    the individual clusters is highly non-convex or more generally when
    a measure of the center and spread of the cluster is not a suitable
    description of the complete cluster. For instance, when clusters are
    nested circles on the 2D plane.

    If affinity is the adjacency matrix of a graph, this method can be
    used to find normalized graph cuts.

    Read more in the :ref:`User Guide <spectral_clustering>`.

    Parameters
    ----------
    affinity : array-like or sparse matrix, shape: (n_samples, n_samples)
        The affinity matrix describing the relationship of the samples to
        embed. **Must be symmetric**.

        Possible examples:
          - adjacency matrix of a graph,
          - heat kernel of the pairwise distance matrix of the samples,
          - symmetric k-nearest neighbours connectivity matrix of the samples.

    n_clusters : integer, optional
        Number of clusters to extract.

    n_components : integer, optional, default is n_clusters
        Number of eigen vectors to use for the spectral embedding

    eigen_solver : {None, 'arpack', 'lobpcg', or 'amg'}
        The eigenvalue decomposition strategy to use. AMG requires pyamg
        to be installed. It can be faster on very large, sparse problems,
        but may also lead to instabilities

    random_state : int, RandomState instance or None (default)
        A pseudo random number generator used for the initialization of the
        lobpcg eigen vectors decomposition when eigen_solver == 'amg' and by
        the K-Means initialization. Use an int to make the randomness
        deterministic.
        See :term:`Glossary <random_state>`.

    n_init : int, optional, default: 10
        Number of time the k-means algorithm will be run with different
        centroid seeds. The final results will be the best output of
        n_init consecutive runs in terms of inertia.

    eigen_tol : float, optional, default: 0.0
        Stopping criterion for eigendecomposition of the Laplacian matrix
        when using arpack eigen_solver.

    assign_labels : {'kmeans', 'discretize'}, default: 'kmeans'
        The strategy to use to assign labels in the embedding
        space.  There are two ways to assign labels after the laplacian
        embedding.  k-means can be applied and is a popular choice. But it can
        also be sensitive to initialization. Discretization is another
        approach which is less sensitive to random initialization. See
        the 'Multiclass spectral clustering' paper referenced below for
        more details on the discretization approach.

    Returns
    -------
    labels : array of integers, shape: n_samples
        The labels of the clusters.
    maps : array of floats, shape: (n_components, n_samples)
        The first k eigenvectors of the sample.

    References
    ----------

    - Normalized cuts and image segmentation, 2000
      Jianbo Shi, Jitendra Malik
      http://citeseer.ist.psu.edu/viewdoc/summary?doi=10.1.1.160.2324

    - A Tutorial on Spectral Clustering, 2007
      Ulrike von Luxburg
      http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.165.9323

    - Multiclass spectral clustering, 2003
      Stella X. Yu, Jianbo Shi
      https://www1.icsi.berkeley.edu/~stellayu/publication/doc/2003kwayICCV.pdf

    Notes
    -----
    The graph should contain only one connect component, elsewhere
    the results make little sense.

    This algorithm solves the normalized cut for k=2: it is a
    normalized spectral clustering.
    """
    if assign_labels not in ('kmeans', 'discretize'):
        raise ValueError("The 'assign_labels' parameter should be "
                         "'kmeans' or 'discretize', but '%s' was given"
                         % assign_labels)

    random_state = check_random_state(random_state)
    n_components = n_clusters if n_components is None else n_components

    # The first eigen vector is constant only for fully connected graphs
    # and should be kept for spectral clustering (drop_first = False)
    # See spectral_embedding documentation.
    maps, eigenvs = spectral_embedding(affinity, n_components=n_components,
                              eigen_solver=eigen_solver,
                              random_state=random_state,
                              eigen_tol=eigen_tol, drop_first=False)

    if assign_labels == 'kmeans':
        _, labels, _ = k_means(maps, n_clusters, random_state=random_state,
                               n_init=n_init)
    else:
        labels = discretize(maps, random_state=random_state)

    return labels, eigenvs


class SpectralClustering(BaseEstimator, ClusterMixin):
    """Apply clustering to a projection of the normalized Laplacian.

    In practice Spectral Clustering is very useful when the structure of
    the individual clusters is highly non-convex or more generally when
    a measure of the center and spread of the cluster is not a suitable
    description of the complete cluster. For instance when clusters are
    nested circles on the 2D plane.

    If affinity is the adjacency matrix of a graph, this method can be
    used to find normalized graph cuts.

    When calling ``fit``, an affinity matrix is constructed using either
    kernel function such the Gaussian (aka RBF) kernel of the euclidean
    distanced ``d(X, X)``::

            np.exp(-gamma * d(X,X) ** 2)

    or a k-nearest neighbors connectivity matrix.

    Alternatively, using ``precomputed``, a user-provided affinity
    matrix can be used.

    Read more in the :ref:`User Guide <spectral_clustering>`.

    Parameters
    ----------
    n_clusters : integer, optional
        The dimension of the projection subspace.

    eigen_solver : {None, 'arpack', 'lobpcg', or 'amg'}
        The eigenvalue decomposition strategy to use. AMG requires pyamg
        to be installed. It can be faster on very large, sparse problems,
        but may also lead to instabilities.

    random_state : int, RandomState instance or None (default)
        A pseudo random number generator used for the initialization of the
        lobpcg eigen vectors decomposition when ``eigen_solver='amg'`` and by
        the K-Means initialization. Use an int to make the randomness
        deterministic.
        See :term:`Glossary <random_state>`.

    n_init : int, optional, default: 10
        Number of time the k-means algorithm will be run with different
        centroid seeds. The final results will be the best output of
        n_init consecutive runs in terms of inertia.

    gamma : float, default=1.0
        Kernel coefficient for rbf, poly, sigmoid, laplacian and chi2 kernels.
        Ignored for ``affinity='nearest_neighbors'``.

    affinity : string, array-like or callable, default 'rbf'
        If a string, this may be one of 'nearest_neighbors', 'precomputed',
        'rbf' or one of the kernels supported by
        `sklearn.metrics.pairwise_kernels`.

        Only kernels that produce similarity scores (non-negative values that
        increase with similarity) should be used. This property is not checked
        by the clustering algorithm.

    n_neighbors : integer
        Number of neighbors to use when constructing the affinity matrix using
        the nearest neighbors method. Ignored for ``affinity='rbf'``.

    eigen_tol : float, optional, default: 0.0
        Stopping criterion for eigendecomposition of the Laplacian matrix
        when ``eigen_solver='arpack'``.

    assign_labels : {'kmeans', 'discretize'}, default: 'kmeans'
        The strategy to use to assign labels in the embedding
        space. There are two ways to assign labels after the laplacian
        embedding. k-means can be applied and is a popular choice. But it can
        also be sensitive to initialization. Discretization is another approach
        which is less sensitive to random initialization.

    degree : float, default=3
        Degree of the polynomial kernel. Ignored by other kernels.

    coef0 : float, default=1
        Zero coefficient for polynomial and sigmoid kernels.
        Ignored by other kernels.

    kernel_params : dictionary of string to any, optional
        Parameters (keyword arguments) and values for kernel passed as
        callable object. Ignored by other kernels.

    n_jobs : int or None, optional (default=None)
        The number of parallel jobs to run.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    Attributes
    ----------
    affinity_matrix_ : array-like, shape (n_samples, n_samples)
        Affinity matrix used for clustering. Available only if after calling
        ``fit``.

    labels_ :
        Labels of each point

    Examples
    --------
    >>> from sklearn.cluster import SpectralClustering
    >>> import numpy as np
    >>> X = np.array([[1, 1], [2, 1], [1, 0],
    ...               [4, 7], [3, 5], [3, 6]])
    >>> clustering = SpectralClustering(n_clusters=2,
    ...         assign_labels="discretize",
    ...         random_state=0).fit(X)
    >>> clustering.labels_
    array([1, 1, 1, 0, 0, 0])
    >>> clustering # doctest: +NORMALIZE_WHITESPACE
    SpectralClustering(affinity='rbf', assign_labels='discretize', coef0=1,
              degree=3, eigen_solver=None, eigen_tol=0.0, gamma=1.0,
              kernel_params=None, n_clusters=2, n_init=10, n_jobs=None,
              n_neighbors=10, random_state=0)

    Notes
    -----
    If you have an affinity matrix, such as a distance matrix,
    for which 0 means identical elements, and high values means
    very dissimilar elements, it can be transformed in a
    similarity matrix that is well suited for the algorithm by
    applying the Gaussian (RBF, heat) kernel::

        np.exp(- dist_matrix ** 2 / (2. * delta ** 2))

    Where ``delta`` is a free parameter representing the width of the Gaussian
    kernel.

    Another alternative is to take a symmetric version of the k
    nearest neighbors connectivity matrix of the points.

    If the pyamg package is installed, it is used: this greatly
    speeds up computation.

    References
    ----------

    - Normalized cuts and image segmentation, 2000
      Jianbo Shi, Jitendra Malik
      http://citeseer.ist.psu.edu/viewdoc/summary?doi=10.1.1.160.2324

    - A Tutorial on Spectral Clustering, 2007
      Ulrike von Luxburg
      http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.165.9323

    - Multiclass spectral clustering, 2003
      Stella X. Yu, Jianbo Shi
      https://www1.icsi.berkeley.edu/~stellayu/publication/doc/2003kwayICCV.pdf
    """

    def __init__(self, n_clusters=8, eigen_solver=None, random_state=None,
                 n_init=10, gamma=1., affinity='rbf', n_neighbors=10,
                 eigen_tol=0.0, assign_labels='kmeans', degree=3, coef0=1,
                 kernel_params=None, n_jobs=None):
        self.n_clusters = n_clusters
        self.eigen_solver = eigen_solver
        self.random_state = random_state
        self.n_init = n_init
        self.gamma = gamma
        self.affinity = affinity
        self.n_neighbors = n_neighbors
        self.eigen_tol = eigen_tol
        self.assign_labels = assign_labels
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        """Creates an affinity matrix for X using the selected affinity,
        then applies spectral clustering to this affinity matrix.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            OR, if affinity==`precomputed`, a precomputed affinity
            matrix of shape (n_samples, n_samples)

        y : Ignored

        """
        X = check_array(X, accept_sparse=['csr', 'csc', 'coo'],
                        dtype=np.float64, ensure_min_samples=2)
        if X.shape[0] == X.shape[1] and self.affinity != "precomputed":
            warnings.warn("The spectral clustering API has changed. ``fit``"
                          "now constructs an affinity matrix from data. To use"
                          " a custom affinity matrix, "
                          "set ``affinity=precomputed``.")

        if self.affinity == 'nearest_neighbors':
            connectivity = kneighbors_graph(X, n_neighbors=self.n_neighbors,
                                            include_self=True,
                                            n_jobs=self.n_jobs)
            self.affinity_matrix_ = 0.5 * (connectivity + connectivity.T)
        elif self.affinity == 'precomputed':
            self.affinity_matrix_ = X
        else:
            params = self.kernel_params
            if params is None:
                params = {}
            if not callable(self.affinity):
                params['gamma'] = self.gamma
                params['degree'] = self.degree
                params['coef0'] = self.coef0
            self.affinity_matrix_ = pairwise_kernels(X, metric=self.affinity,
                                                     filter_params=True,
                                                     **params)

        random_state = check_random_state(self.random_state)
        self.labels_, self.eigenvs_ = spectral_clustering(self.affinity_matrix_,
                                           n_clusters=self.n_clusters,
                                           eigen_solver=self.eigen_solver,
                                           random_state=random_state,
                                           n_init=self.n_init,
                                           eigen_tol=self.eigen_tol,
                                           assign_labels=self.assign_labels)
        return self

    @property
    def _pairwise(self):
        return self.affinity == "precomputed"


# PUBLIC FUNCTIONS
def cluster_spectral(A_M, A_O):
    """Performs baseline spectral clustering.
    Parameters
    ---------
    A_M : `ndarray`
        Adjacency matrix of model
    A_O : `ndarray
        Adjacency matrix of observation
    Returns
    -------
      prediction
      P
      time"""
    spectral_start_time = time()
    spectral_cluster = SpectralClustering(A_M.shape[0], affinity="precomputed").fit(A_O)
    spectral_prediction = spectral_cluster.labels_
    P_opt_spectral = spectral_cluster.eigenvs_
    spectral_time = time() - spectral_start_time

    return spectral_prediction, P_opt_spectral, spectral_time


def cluster_template(A_M, A_O, mode="adjacency", cost_function=compute_matching_norm):
    """Performs Template-based graph clustering.
    Parameters
    ---------
    A_M : `ndarray`
        Adjacency matrix of model
    A_O : `ndarray
        Adjacency matrix of observation
    mode : {"adjacency", "laplacian"}
        Mode of the template-based clustering
    Returns
    -------
      prediction
      P
      time"""
    assert mode in ["adjacency", "laplacian"], "Mode must be one of 'adjacency', 'laplacian' (got '{}')".format(mode)
    # Initializing manifold
    manifold_P = Stiefel(A_O.shape[0], A_M.shape[0])

    # Computing laplacians
    #L_M = nx.laplacian_matrix(nx.from_numpy_matrix(A_M)).toarray()
    L_M = csgraph_laplacian(A_M)
    #L_O = nx.laplacian_matrix(nx.from_numpy_matrix(A_O)).toarray()
    L_O = csgraph_laplacian(A_O)

    template_start_time = time()
    if mode=="adjacency":
        P_opt_template = find_optimal_P(A_M, A_O, manifold_P, cost=cost_function)
    elif mode=="laplacian":
        P_opt_template = find_optimal_P(L_M, L_O, manifold_P, cost=cost_function)
    template_kmeans = get_kmeans(P_opt_template, A_M.shape[0])
    template_prediction = template_kmeans.predict(P_opt_template)
    template_time = time() - template_start_time

    return template_prediction, P_opt_template, template_time


def cluster_stochastic(A_M, A_O, learning_rate=10e-5):
    """Performs stochastic template-based spectral clustering.
    Parameters
    ---------
    A_M : `ndarray`
        Adjacency matrix of model
    A_O : `ndarray
        Adjacency matrix of observation
    learning_rate : float
        Gradient descent learning rate
    Returns
    -------
      prediction
      P
      time"""
    stochastic_start_time = time()
    P_opt_stochastic = find_optimal_P_stochastic(A_M, A_O, learning_rate=learning_rate)
    stochastic_prediction = np.argmax(P_opt_stochastic, axis=1)
    stochastic_time = time() - stochastic_start_time

    return stochastic_prediction, P_opt_stochastic, stochastic_time


def cluster_stochastic_kmeans(A_M, A_O, learning_rate=10e-5):
    """Performs stochastic+kmeans template-based spectral clustering.
    Parameters
    ---------
    A_M : `ndarray`
        Adjacency matrix of model
    A_O : `ndarray
        Adjacency matrix of observation
    learning_rate : float
        Gradient descent learning rate
    Returns
    -------
      prediction
      P
      time"""
    stochastic_start_time = time()
    P_opt_stochastic = find_optimal_P_stochastic(A_M, A_O, learning_rate=learning_rate)
    stochastic_kmeans = get_kmeans(P_opt_stochastic, A_M.shape[0])
    stochastic_prediction = stochastic_kmeans.predict(P_opt_stochastic)
    stochastic_time = time() - stochastic_start_time

    return stochastic_prediction, P_opt_stochastic, stochastic_time


def cluster_modularity(A_M, A_O):
    G_O = nx.from_numpy_matrix(A_O)
    modularity_start_time = time()
    modularity_communities = nx.algorithms.community.modularity_max.greedy_modularity_communities(G_O)
    modularity_prediction = np.empty(A_O.shape[0])
    for community in range(len(modularity_communities)):
        modularity_prediction[list(map(int, modularity_communities[community]))] = community
    modularity_time = time() - modularity_start_time

    return modularity_prediction, None, modularity_time


def cluster_modularity_louvain(A_M, A_O):
    A_O = np.array(A_O)
    modularity_louv_start_time = time()
    modularity_louv_prediction = skn.clustering.Louvain(modularity="newman").fit_transform(A_O)
    modularity_louv_time = time() - modularity_louv_start_time

    return modularity_louv_prediction, None, modularity_louv_time
