import gpflow
import tensorflow as tf
from gpflow.base import TensorLike
from ..inducing_variables import LaplacianDirichletFeatures
from gpflow.utilities import to_default_float
from gpflow.covariances import Kuu
from ..covariances import FreqDomainKuf as Kuf  # the dispatcher for frequency domain
from .dispatch import local_spectrum_conditional as conditional
Diag = tf.linalg.LinearOperatorDiag

__all__ = []

@conditional.register(TensorLike, LaplacianDirichletFeatures, gpflow.kernels.Kernel, TensorLike)
def local_spectrum_approx_conditional_ldf(
        Xnew,
        inducing_variable,
        kernel,
        f,
        *,
        alpha=None,
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

    Λ = Kuu(inducing_variable, kernel)  # this is now a LinearOperator
    Φ = Kuf(inducing_variable, kernel, Xnew, alpha=alpha)  # a complex Tensor
    Λr = Kuu(inducing_variable.remainder, kernel)
    Φr = Kuf(inducing_variable.remainder, kernel, Xnew, alpha=alpha)
    Φrm = Kuf(inducing_variable.remainder, kernel, -Xnew, alpha=alpha)

    # compute the covariance due to the conditioning
    if full_cov:
        Λr_inv_Φr = tf.expand_dims(tf.complex(real=1/Λr.diag_part(), imag=tf.zeros_like(Λr.diag_part())),-1) * Φr
        Λr_inv_Φrm = tf.expand_dims(tf.complex(real=1 / Λr.diag_part(), imag=tf.zeros_like(Λr.diag_part())), -1) * Φrm
        a = tf.matmul(Φr, Λr_inv_Φr, adjoint_a=True) + tf.matmul(Φr, Λr_inv_Φrm, adjoint_a=True)
        b = tf.matmul(Φr, Λr_inv_Φr, adjoint_a=True) - tf.matmul(Φr, Λr_inv_Φrm, adjoint_a=True)
        fvar_rr = tf.math.real(a)
        fvar_ii = tf.math.real(b)
        fvar_ir = tf.math.imag(a)
        fvar_ri = tf.math.imag(-b)
        fvar = tf.concat([tf.concat([fvar_rr,fvar_ir],-2), tf.concat([fvar_ri,fvar_ii],-2)], -1)  # K x 2N x 2N
        shape = (num_func, 1, 1)
    else:
        # ... x M x N -> ... x N
        Λr_inv_Φr = tf.expand_dims(tf.complex(real=1 / Λr.diag_part(), imag=tf.zeros_like(Λr.diag_part())), -1) * Φr
        Λr_inv_Φrm = tf.expand_dims(tf.complex(real=1 / Λr.diag_part(), imag=tf.zeros_like(Λr.diag_part())), -1) * Φrm
        fvar_rr = tf.reduce_sum(Φr * Λr_inv_Φr, -2) + tf.reduce_sum(Φr * Λr_inv_Φrm, -2)
        fvar_ii = tf.reduce_sum(Φr * Λr_inv_Φr, -2) - tf.reduce_sum(Φr * Λr_inv_Φrm, -2)
        fvar = tf.concat([tf.math.real(fvar_rr), tf.math.real(fvar_ii)],-1) # K x 2N x D
        shape = (num_func, 1)
    fvar = tf.expand_dims(fvar, 0) * tf.ones(
        shape, dtype=gpflow.default_float()
    )  # K x N x N or K x N
    # another backsubstitution in the unwhitened case
    if white:
        A = Λ.cholesky().solve(tf.math.real(Φ))
        B = Λ.cholesky().solve(tf.math.imag(Φ))
    else:
        A = Λ.solve(tf.math.real(Φ))
        B = Λ.solve(tf.math.imag(Φ))

    # construct the conditional mean
    fmean = tf.concat([tf.matmul(A, f, transpose_a=True),
                       tf.matmul(B, f, transpose_a=True)],-2)

    if q_sqrt is not None:
        if q_sqrt.shape.ndims == 2:
            # case for q_diag = True
            LTA1 = Diag(tf.linalg.matrix_transpose(q_sqrt)) @ A  # K x M x N
            LTA2 = Diag(tf.linalg.matrix_transpose(q_sqrt)) @ B
        elif q_sqrt.shape.ndims == 3:
            LTA1 = tf.matmul(q_sqrt, A, transpose_a=True)
            LTA2 = tf.matmul(q_sqrt, B, transpose_a=True)
        else:
            raise ValueError("Bad dimension for q_sqrt: %s" % str(q_sqrt.get_shape().ndims))
        if full_cov:
            LTA = tf.concat([LTA1, LTA2],-1)
            fvar = fvar + tf.matmul(LTA, LTA, transpose_a=True)  # K x 2N x 2N
        else:
            LTA = tf.concat([LTA1, LTA2], -1) # K x M x 2N
            fvar = fvar + tf.reduce_sum(tf.square(LTA), -2)  # K x 2N
    fvar = tf.transpose(fvar)  # 2N x K or 2N x 2N x K

    return fmean, fvar