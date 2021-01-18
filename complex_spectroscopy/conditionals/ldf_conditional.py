import gpflow
import tensorflow as tf
from gpflow.base import TensorLike
from ..inducing_variables import LaplacianDirichletFeatures
from gpflow.utilities import to_default_float
from gpflow import covariances as cov

__all__ = []

@gpflow.conditionals.conditional.register(
    TensorLike, LaplacianDirichletFeatures, gpflow.kernels.Kernel, TensorLike
)
def approx_conditional_ldf(
        Xnew,
        inducing_variable,
        kernel,
        f,
        *,
        full_cov=False,
        full_output_cov=False,
        q_sqrt=None,
        white=True,
):
    """
     - Xnew are the points of the data or minibatch, size N x D (tf.array, 2d)
     - inducing_variable is an instance of inducing_variables.InducingVariable that provides
       `Kuu` and `Kuf` methods for Laplacian Dirichlet features, this contains the limits of
       the bounding box and the frequencies
     - remainder_variable is another instance of inducing_variables.InducingVariable that specifies
       the high frequency components not selected in inducing_variable.
     - f is the value (or mean value) of the features (i.e. the weights)
     - q_sqrt (default None) is the Cholesky factor of the uncertainty about f
       (to be propagated through the conditional as per the GPflow inducing-point implementation)
     - white (defaults False) specifies whether the whitening has been applied. LDF works a lot better,
         when using vanilla gradients, if whitening has been applied, so it's the default option.

    Given the GP represented by the inducing points specified in `inducing_variable`, produce the mean
    and (co-)variance of the GP at the points Xnew.

       Xnew :: N x D
       Kuu :: M x M
       Kuf :: M x N
       f :: M x K, K = 1
       q_sqrt :: K x M x M, with K = 1
    """
    if full_output_cov:
        raise NotImplementedError

    # num_data = tf.shape(Xnew)[0]  # M
    num_func = tf.shape(f)[1]  # K
    tpose = tf.linalg.matrix_transpose  # because god damn, that's super long

    Λ = cov.Kuu(inducing_variable, kernel)  # this is now a LinearOperator
    Φ = cov.Kuf(inducing_variable, kernel, Xnew)  # still a Tensor
    Λr = cov.Kuu(inducing_variable.remainder, kernel)
    Φr = cov.Kuf(inducing_variable.remainder, kernel, Xnew)

    # compute the covariance due to the conditioning
    if full_cov:
        fvar = tpose(Φr) @ Λr.solve(Φr)
        shape = (num_func, 1, 1)
    else:
        fvar = tf.reduce_sum(Φr * Λr.solve(Φr), -2)
        shape = (num_func, 1)
    fvar = tf.expand_dims(fvar, 0) * tf.ones(
        shape, dtype=gpflow.default_float()
    )  # K x N x N or K x N

    # another backsubstitution in the unwhitened case
    if white:
        A = Λ.cholesky().solve(Φ)
    else:
        A = Λ.solve(Φ)

    # construct the conditional mean
    fmean = tf.matmul(A, f, transpose_a=True)

    if q_sqrt is not None:
        if q_sqrt.shape.ndims == 2:
            # case for q_diag = True
            LTA = Diag(tf.transpose(q_sqrt)) @ A  # K x M x N
        elif q_sqrt.shape.ndims == 3:
            LTA = tpose(q_sqrt) @ A
        else:
            raise ValueError("Bad dimension for q_sqrt: %s" % str(q_sqrt.get_shape().ndims))
        if full_cov:
            fvar = fvar + tf.matmul(LTA, LTA, transpose_a=True)  # K x N x N
        else:
            fvar = fvar + tf.reduce_sum(tf.square(LTA), 1)  # K x N
    fvar = tf.transpose(fvar)  # N x K or N x N x K

    return fmean, fvar