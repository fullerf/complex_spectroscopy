from typing import Tuple

import numpy as np
import tensorflow as tf
import gpflow

from gpflow.base import Parameter
from ..inducing_variables import LaplacianDirichletFeatures
#  special conditional registry for local spectrum needed because dispatch doesn't make it easy to
# distinguish svgp from a local spectrum svgp. Separating them with different registries feels cleanest
from ..conditionals import local_spectrum_conditional as conditional
# however, the kl (and inducing space) is shared with a regular svgp, so we can use the normal kl registry
from gpflow import kullback_leiblers as kl
from ..kullback_leiblers import *  # trigger registration of LDF features case to kl registry
from gpflow.config import default_float
from gpflow.utilities import positive, triangular
from gpflow.models.model import GPModel, InputData, MeanAndVariance, RegressionData
from gpflow.models.training_mixins import ExternalDataTrainingLossMixin
from gpflow.models.util import inducingpoint_wrapper


def symmetrize_vals_no_zero(x):
    """
    x assumed to be complex

    Take x of size ...xNxD, where we assume N corresponds to a 1D axis that is sorted and
    strictly positive. Then we concatenate ...xNxD to it along dim -2 after flipping
    direction resulting in a tensor of ...x2NxD size. Real part is flipped, imag part is
    flipped and sign reversed.
    """
    xa = tf.reverse(x, axis=[-2])
    y = tf.complex(real=tf.math.real(xa), imag=-tf.math.imag(xa))
    return tf.concat([y, x], -2)


def symmetrize_vals_with_zero(x):
    """
    x assumed to be complex

    Take x of size ...xNxD, where we assume N corresponds to a 1D axis that is sorted and
    strictly positive. Then we concatenate ...xNxD to it along dim -2 after flipping
    direction resulting in a tensor of ...x(2N-1)xD size. Real part is flipped, imag part is
    flipped and sign reversed.

    In this case, we assume the first index on the -2 dim is 0 and skip it for the concatenated
    portion.
    """
    if len(tf.shape(x)) > 2:
        xa = tf.reverse(x[..., 1:, :], axis=[-2])
    else:
        xa = tf.reverse(x[1:, :], axis=[-2])
    y = tf.complex(real=tf.math.real(xa), imag=-tf.math.imag(xa))
    return tf.concat([y, x], -2)


def symmetrize_axis_no_zero(x):
    """
    x assumed to be real

    Take x of size ...xNxD, where we assume N corresponds to a 1D axis that is sorted and
    strictly positive. Then we concatenate ...xNxD to it along dim -2 after flipping
    direction and sign inverting resulting in a tensor of ...x2NxD size.
    """
    xa = tf.reverse(x, axis=[-2])
    return tf.concat([-xa, x], -2)


def symmetrize_axis_with_zero(x):
    """
    x assumed to be real

    Take x of size ...xNxD, where we assume N corresponds to a 1D axis that is sorted and
    strictly positive. Then we concatenate ...xNxD to it along dim -2 after flipping
    direction and sign inverting resulting in a tensor of ...x(2N-1)xD size

    In this case, we assume the first index on the -2 dim is 0 and skip it for the concatenated
    portion.
    """
    if len(tf.shape(x)) > 2:
        xa = tf.reverse(x[..., 1:, :], axis=[-2])
    else:
        xa = tf.reverse(x[1:, :], axis=[-2])
    return tf.concat([-xa, x], -2)


class Local_Spectrum_GPMM_SVGP1D(GPModel, ExternalDataTrainingLossMixin):
    """

    This is the 1 dimensional Sparse Variational GP (SVGP) Gaussian Process Mixture of Measurements (GPMM) model
    The key references are:
    ::
      @inproceedings{hensman2014scalable,
        title={Scalable Variational Gaussian Process Classification},
        author={Hensman, James and Matthews, Alexander G. de G. and Ghahramani, Zoubin},
        booktitle={Proceedings of AISTATS},
        year={2015}
      }

      @journal{tobar2018,
        title={Bayesian Non-parametric Spectral Estimation},
        author={Tobar, Filipe},
        booktitle={NeurlIPS},
        year={2018}
      }

      @journal{tobar2017,
        title={Recovering Latent Signals From a Mixture of Measurements Using a Gaussian Process Prior},
        author={Tobar, Filipe},
        booktitle={IEEE Signal Processing Letters},
        year={2017}
      }

    """

    def __init__(
        self,
        kernel,
        freq_axis,
        inducing_variable,
        noise_scale,
        *,
        alpha=None,
        mean_function=None,
        num_latent_gps: int = 1,
        q_diag: bool = False,
        q_mu=None,
        q_sqrt=None,
        whiten: bool = True,
        num_data=None,
    ):
        """
        - kernel, inducing_variables, mean_function are appropriate
          GPflow objects
        - noise_scale is the estimated std_dev of observational noise (for a gaussian likelihood)
        - num_latent_gps is the number of latent processes to use, defaults to 1
        - q_diag is a boolean. If True, the covariance is approximated by a
          diagonal matrix.
        - whiten is a boolean. If True, we use the whitened representation of
          the inducing points.
        - num_data is the total number of observations, defaults to X.shape[0]
          (relevant when feeding in external minibatches)
        """
        # init the super class, accept args
        likelihood = gpflow.likelihoods.Gaussian(variance=noise_scale ** 2)
        super().__init__(kernel, likelihood, mean_function, num_latent_gps)
        self.num_data = num_data
        self.q_diag = q_diag
        self.whiten = whiten
        # we require inducing variable of type LaplacianDirichletFeature
        self.inducing_variable = inducing_variable
        assert type(self.inducing_variable) is LaplacianDirichletFeatures
        self.one_sided_axis = tf.sort(tf.convert_to_tensor(freq_axis, dtype=gpflow.default_float()))
        # require axis to be positive frequencies only
        assert tf.math.reduce_all(self.one_sided_axis>=0)
        zero_axis_flag = self.one_sided_axis[0] <= 1e-6 # treat 0 to 1e-6 as zero
        if zero_axis_flag:
            self._axis_symmetrizer = symmetrize_axis_with_zero
            self._val_symmetrizer = symmetrize_vals_with_zero
        else:
            self._axis_symmetrizer = symmetrize_axis_no_zero
            self._val_symmetrizer = symmetrize_vals_no_zero
        self.axis = self._axis_symmetrizer(self.one_sided_axis)
        if alpha is None:
            self.alpha = tf.reshape(tf.constant(0.1, gpflow.default_float()),(-1))
        else:
            self.alpha = alpha
        # init variational parameters
        num_inducing = self.inducing_variable.num_inducing
        self._init_variational_parameters(num_inducing, q_mu, q_sqrt, q_diag)

    def _init_variational_parameters(self, num_inducing, q_mu, q_sqrt, q_diag):
        """
        Constructs the mean and cholesky of the covariance of the variational Gaussian posterior.
        If a user passes values for `q_mu` and `q_sqrt` the routine checks if they have consistent
        and correct shapes. If a user does not specify any values for `q_mu` and `q_sqrt`, the routine
        initializes them, their shape depends on `num_inducing` and `q_diag`.
        Note: most often the comments refer to the number of observations (=output dimensions) with P,
        number of latent GPs with L, and number of inducing points M. Typically P equals L,
        but when certain multioutput kernels are used, this can change.
        Parameters
        ----------
        :param num_inducing: int
            Number of inducing variables, typically refered to as M.
        :param q_mu: np.array or None
            Mean of the variational Gaussian posterior. If None the function will initialise
            the mean with zeros. If not None, the shape of `q_mu` is checked.
        :param q_sqrt: np.array or None
            Cholesky of the covariance of the variational Gaussian posterior.
            If None the function will initialise `q_sqrt` with identity matrix.
            If not None, the shape of `q_sqrt` is checked, depending on `q_diag`.
        :param q_diag: bool
            Used to check if `q_mu` and `q_sqrt` have the correct shape or to
            construct them with the correct shape. If `q_diag` is true,
            `q_sqrt` is two dimensional and only holds the square root of the
            covariance diagonal elements. If False, `q_sqrt` is three dimensional.
        """
        q_mu = np.zeros((num_inducing, self.num_latent_gps)) if q_mu is None else q_mu
        self.q_mu = Parameter(q_mu, dtype=default_float())  # [M, P]

        if q_sqrt is None:
            if self.q_diag:
                ones = np.ones((num_inducing, self.num_latent_gps), dtype=default_float())
                self.q_sqrt = Parameter(ones, transform=positive())  # [M, P]
            else:
                q_sqrt = [
                    np.eye(num_inducing, dtype=default_float()) for _ in range(self.num_latent_gps)
                ]
                q_sqrt = np.array(q_sqrt)
                self.q_sqrt = Parameter(q_sqrt, transform=triangular())  # [P, M, M]
        else:
            if q_diag:
                assert q_sqrt.ndim == 2
                self.num_latent_gps = q_sqrt.shape[1]
                self.q_sqrt = Parameter(q_sqrt, transform=positive())  # [M, L|P]
            else:
                assert q_sqrt.ndim == 3
                self.num_latent_gps = q_sqrt.shape[0]
                num_inducing = q_sqrt.shape[1]
                self.q_sqrt = Parameter(q_sqrt, transform=triangular())  # [L|P, M, M]

    def prior_kl(self) -> tf.Tensor:
        return kl.prior_kl(
            self.inducing_variable, self.kernel, self.q_mu, self.q_sqrt, whiten=self.whiten
        )

    def maximum_log_likelihood_objective(self, data: RegressionData) -> tf.Tensor:
        return self.elbo(data)

    def elbo(self, data: RegressionData) -> tf.Tensor:
        """
        This gives a variational bound (the evidence lower bound or ELBO) on
        the log marginal likelihood of the model.
        """
        X, Y = data
        kl = self.prior_kl()
        z_mean_real, z_mean_imag, z_var_real, z_var_imag = self.predict_f_reduced3(X)
        var_exp = (self.likelihood.variational_expectations(z_mean_real,
                                                            z_var_real, tf.math.real(Y)) +
                   self.likelihood.variational_expectations(z_mean_imag,
                                                            z_var_imag, tf.math.imag(Y)))
        if self.num_data is not None:
            num_data = tf.cast(self.num_data, kl.dtype)
            minibatch_size = tf.cast(tf.shape(X)[0], kl.dtype)
            scale = num_data / minibatch_size
        else:
            scale = tf.cast(1.0, kl.dtype)
        return tf.reduce_sum(var_exp) * scale - kl

    def predict_f_reduced(self, X):
        f_mean, f_var = self.predict_f(self.axis, full_cov=True, full_output_cov=False)  # ...NxK, NxNxK
        f_mean_r, f_mean_i = tf.split(f_mean, 2, axis=-2)
        f_var_r, f_var_i = tf.split(f_var, 2, axis=-3)  # N/2xNxK
        f_var_rr, f_var_ri = tf.split(f_var_r, 2, axis=-2)  # N/2xN/2xK
        f_var_ir, f_var_ii = tf.split(f_var_i, 2, axis=-2)  # N/2xN/2xK
        z_var_real = tf.reduce_sum(
            tf.tensordot(tf.math.real(X), f_var_rr, [-1, -3]) * tf.expand_dims(tf.math.real(X), -1), -2) - \
                     tf.reduce_sum(
                         tf.tensordot(tf.math.imag(X), f_var_ii, [-1, -3]) * tf.expand_dims(tf.math.imag(X), -1), -2)
        z_var_imag = -tf.reduce_sum(
            tf.tensordot(tf.math.real(X), f_var_ri, [-1, -3]) * tf.expand_dims(tf.math.imag(X), -1), -2) + \
                     tf.reduce_sum(
                         tf.tensordot(tf.math.imag(X), f_var_ir, [-1, -3]) * tf.expand_dims(tf.math.real(X), -1), -2)
        z_mean_real = tf.math.real(X) @ f_mean_r - tf.math.imag(X) @ f_mean_i
        z_mean_imag = tf.math.real(X) @ f_mean_i + tf.math.imag(X) @ f_mean_r
        return z_mean_real, z_mean_imag, z_var_real, z_var_imag

    def predict_f_reduced2(self, X):
        f_mean, f_var = self.predict_f(self.axis, full_cov=True, full_output_cov=False)  # ...NxK, NxNxK
        Xe = tf.concat([tf.math.real(X), tf.math.imag(X)],-1)
        XeDagger = tf.transpose(tf.concat([tf.math.real(X), -tf.math.imag(X)],-1))
        z_var = tf.transpose(tf.linalg.diag_part(Xe @ (tf.transpose(f_var, perm=[2,0,1]) @ XeDagger)))
        z_var_real, z_var_imag = tf.split(z_var,2,axis=-2)
        f_mean_r, f_mean_i = tf.split(f_mean, 2, axis=-2)
        z_mean_real = tf.math.real(X) @ f_mean_r - tf.math.imag(X) @ f_mean_i
        z_mean_imag = tf.math.real(X) @ f_mean_i + tf.math.imag(X) @ f_mean_r
        return z_mean_real, z_mean_imag, z_var_real, z_var_imag

    def predict_f_reduced3(self, X):
        """ convert X to conjugate extended form, as well as covariance"""
        f_mean, f_var = self.predict_f(self.axis, full_cov=True, full_output_cov=False)  # ...NxK, NxNxK
        Xe = tf.concat([X, tf.math.conj(X)],-1)
        f_mean_r, f_mean_i = tf.split(f_mean, 2, axis=-2)
        f_mean_c = tf.complex(real=f_mean_r, imag=f_mean_i)
        m = tf.concat([f_mean_c, tf.math.conj(f_mean_c)],-2)
        f_var = tf.transpose(f_var, perm=[2,0,1])
        f_var_r, f_var_i = tf.split(f_var, 2, axis=-2)
        f_var_rr, f_var_ri = tf.split(f_var_r, 2, axis=-1)  # N/2xN/2xK
        f_var_ir, f_var_ii = tf.split(f_var_i, 2, axis=-1)  # N/2xN/2xK
        G = tf.complex(real=f_var_rr + f_var_ii, imag=f_var_ir - f_var_ri)
        C = tf.complex(real=f_var_rr - f_var_ii, imag=f_var_ir + f_var_ri)
        Sigma_a = tf.concat([G,C],-1)
        Sigma_b = tf.concat([tf.math.conj(C), tf.math.conj(G)],-1)
        Sigma = tf.concat([Sigma_a, Sigma_b],-2)
        z_var = tf.transpose(tf.linalg.diag_part(tf.matmul(tf.matmul(Xe,Sigma),Xe,adjoint_b=True)))
        z_var_real, z_var_imag = (tf.math.real(z_var), tf.math.imag(z_var))
        mu = Xe @ m
        return tf.math.real(mu), tf.math.imag(mu), z_var_real, z_var_imag


    def predict_f(self, Xnew: InputData, full_cov=False, full_output_cov=False) -> MeanAndVariance:
        q_mu = self.q_mu
        q_sqrt = self.q_sqrt
        mu, var = conditional(
            Xnew,
            self.inducing_variable,
            self.kernel,
            q_mu,
            alpha=self.alpha,
            q_sqrt=q_sqrt,
            full_cov=full_cov,
            white=self.whiten,
            full_output_cov=full_output_cov,
        )
        # tf.debugging.assert_positive(var)  # We really should make the tests pass with this here
        return mu + self.mean_function(tf.concat([Xnew, Xnew],-2)), var