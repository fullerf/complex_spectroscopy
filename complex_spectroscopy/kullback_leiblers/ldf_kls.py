import gpflow
from gpflow.base import TensorLike
import tensorflow as tf
from gpflow import covariances as cov
from ..inducing_variables import LaplacianDirichletFeatures
from gpflow import kullback_leiblers as kl
from gpflow.utilities import to_default_float

__all__ = []

@kl.prior_kl.register(LaplacianDirichletFeatures, gpflow.kernels.Kernel, TensorLike, TensorLike)
def prior_kl_ldf(inducing_variable, kernel, q_mu, q_sqrt, whiten=False):
    if whiten:
        K = None
    else:
        K = cov.Kuu(inducing_variable, kernel)
    return gauss_kl_ldf(q_mu, q_sqrt, K)


def gauss_kl_ldf(q_mu: tf.Tensor, q_sqrt: tf.Tensor, K: tf.linalg.LinearOperatorDiag):
    """
    Compute the KL divergence KL[q || p] between
          q(x) = N(m, L@L.T)
          m = Kuu @ q_mu
          L = Kuu @ q_sqrt
    and
          p(x) = N(0, K)    where K is a Diag linear operator
          p(x) = N(0, I)    if K is None
    We assume L multiple independent distributions, given by the columns of
    q_mu and the first or last dimension of q_sqrt. Returns the *sum* of the
    divergences.
    q_mu is a matrix ([M, L]), each column contains a mean.
    q_sqrt can be a 3D tensor ([L, M, M]), each matrix within is a lower
        triangular square-root matrix of the covariance of q.
    q_sqrt can be a matrix ([M, L]), each column represents the diagonal of a
        square-root matrix of the covariance of q.
    K is the covariance of p (positive-definite matrix).  In this case it must always
    be a tf.linalg.LinearOperatorDiag instance as the type hint suggests
    """
    if K is None:
        is_white = True
        is_batched_prior = False
    else:
        is_white = False
        is_batched_prior = len(K.shape) == 3
    is_diag = len(tf.shape(q_sqrt)) == 2

    M, L = tf.shape(q_mu)[0], tf.shape(q_mu)[1]

    if is_white:
        alpha = q_mu  # [M, L], implying that K is identity
    else:
        q_mu = tf.transpose(q_mu)[:, :, None] if is_batched_prior else q_mu  # [L, M, 1] or [M, L]
        alpha = K.solve(q_mu)  # [L, M, 1] or [M, L]

    if is_diag:
        # if q_sqrt is diagonal
        q_diag = tf.linalg.LinearOperatorDiag(tf.square(q_sqrt))
        # Log-determinant of the covariance of q(x); factor of 2 from fact that q_sqrt is sqrt of whole
        logdet_qcov = tf.reduce_sum(q_diag.log_abs_determinant())
    else:
        Lq = tf.linalg.band_part(q_sqrt, -1, 0)  # force lower triangle # [L, M, M]
        Lq_diag = tf.linalg.diag_part(Lq)  # [L, M]
        # Log-determinant of the covariance of q(x):
        logdet_qcov = tf.reduce_sum(tf.math.log(tf.square(Lq_diag)))

    # Mahalanobis term: μqᵀ Σp⁻¹ μq
    mahalanobis = tf.reduce_sum(q_mu * alpha)

    # Constant term: - L * M
    constant = -to_default_float(tf.size(q_mu, out_type=tf.int64))

    # Trace term: tr(Σp⁻¹ Σq)
    if is_white:
        if is_diag:
            trace = tf.reduce_sum(q_diag.trace())
        else:
            trace = tf.reduce_sum(tf.square(Lq))
    else:
        if is_diag and not is_batched_prior:
            # K is [M, M] and q_sqrt is [M, L]: fast specialisation, we skip needing to take diag_part
            trace = tf.reduce_sum(K.solve(tf.square(q_sqrt)))
        else:
            # K is [L,M,M] or [M,M] and Lq_diag is [L, M] -> [M, L]
            trace = tf.reduce_sum(K.solve(tf.square(tf.linalg.matrix_transpose(Lq_diag))))

    twoKL = mahalanobis + constant - logdet_qcov + trace

    # Log-determinant of the covariance of p(x):
    if not is_white:
        log_det_p = tf.reduce_sum(K.log_abs_determinant())
        # If K is [L, M, M], num_latent_gps is no longer implicit, no need to multiply the single kernel logdet
        scale = 1.0 if is_batched_prior else to_default_float(L)
        log_det_p *= scale
        twoKL += log_det_p

    return 0.5 * twoKL