# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import tensorflow as tf
import gpflow

__all__ = ['SeparableMatern12', 'SeparableMatern32', 'SeparableMatern52']


# +
class SeparableMatern12(gpflow.kernels.Kernel):
    def __init__(self, variance=1.0, lengthscales=1.0, **kwargs):
        super().__init__()
        self.variance = gpflow.Parameter(variance, transform=gpflow.utilities.positive())
        self.lengthscales = gpflow.Parameter(lengthscales, transform=gpflow.utilities.positive())
        self._validate_ard_active_dims(self.lengthscales)
    
    @property
    def ard(self) -> bool:
        """
        Whether ARD behaviour is active.
        """
        return self.lengthscales.shape.ndims > 0
    
    def scale(self, X):
        X_scaled = X / self.lengthscales if X is not None else X
        return X_scaled
    
    def _separable_squared_distance(self, X, X2):
        """
        Returns ||X - X2ᵀ||² for each dimension D (last dim of X & X2) separately,
        while also respecting batch dims of the input (anything before -2)
        """
        if X2 is None:
            X2 = X
        d = tf.expand_dims(X,-2) - tf.expand_dims(X2,-3) # b,N,1,D x b,1,N,D = b,N,N,D
        return tf.square(d)

    def K(self, X, X2=None):
        r2 = self._separable_squared_distance(self.scale(X), self.scale(X2))
        r = tf.sqrt(tf.maximum(r2, 1e-36))
        k = tf.exp(-r)
        return self.variance * tf.math.reduce_prod(k,-1)

    def K_diag(self, X):
        return tf.fill(tf.shape(X)[:-1], tf.squeeze(self.variance))

class SeparableMatern32(gpflow.kernels.Kernel):
    def __init__(self, variance=1.0, lengthscales=1.0, **kwargs):
        super().__init__()
        self.variance = gpflow.Parameter(variance, transform=gpflow.utilities.positive())
        self.lengthscales = gpflow.Parameter(lengthscales, transform=gpflow.utilities.positive())
        self._validate_ard_active_dims(self.lengthscales)
    
    @property
    def ard(self) -> bool:
        """
        Whether ARD behaviour is active.
        """
        return self.lengthscales.shape.ndims > 0
    
    def scale(self, X):
        X_scaled = X / self.lengthscales if X is not None else X
        return X_scaled
    
    def _separable_squared_distance(self, X, X2):
        """
        Returns ||X - X2ᵀ||² for each dimension D (last dim of X & X2) separately,
        while also respecting batch dims of the input (anything before -2)
        """
        if X2 is None:
            X2 = X
        d = tf.expand_dims(X,-2) - tf.expand_dims(X2,-3) # b,N,1,D x b,1,N,D = b,N,N,D
        return tf.square(d)

    def K(self, X, X2=None):
        r2 = self._separable_squared_distance(self.scale(X), self.scale(X2))
        r = tf.sqrt(tf.maximum(r2, 1e-36))
        sqrt3 = np.sqrt(3.0)
        k = (1.0 + sqrt3 * r) * tf.exp(-sqrt3 * r)
        return self.variance * tf.math.reduce_prod(k,-1)

    def K_diag(self, X):
        return tf.fill(tf.shape(X)[:-1], tf.squeeze(self.variance))

class SeparableMatern52(gpflow.kernels.Kernel):
    def __init__(self, variance=1.0, lengthscales=1.0, **kwargs):
        super().__init__()
        self.variance = gpflow.Parameter(variance, transform=gpflow.utilities.positive())
        self.lengthscales = gpflow.Parameter(lengthscales, transform=gpflow.utilities.positive())
        self._validate_ard_active_dims(self.lengthscales)
    
    @property
    def ard(self) -> bool:
        """
        Whether ARD behaviour is active.
        """
        return self.lengthscales.shape.ndims > 0
    
    def scale(self, X):
        X_scaled = X / self.lengthscales if X is not None else X
        return X_scaled
    
    def _separable_squared_distance(self, X, X2):
        """
        Returns ||X - X2ᵀ||² for each dimension D (last dim of X & X2) separately,
        while also respecting batch dims of the input (anything before -2)
        """
        if X2 is None:
            X2 = X
        d = tf.expand_dims(X,-2) - tf.expand_dims(X2,-3) # b,N,1,D x b,1,N,D = b,N,N,D
        return tf.square(d)

    def K(self, X, X2=None):
        r2 = self._separable_squared_distance(self.scale(X), self.scale(X2))
        r = tf.sqrt(tf.maximum(r2, 1e-36))
        sqrt5 = np.sqrt(5.0)
        k = (1.0 + sqrt5 * r + 5.0 / 3.0 * tf.square(r)) * tf.exp(-sqrt5 * r)
        return self.variance * tf.math.reduce_prod(k,-1)

    def K_diag(self, X):
        return tf.fill(tf.shape(X)[:-1], tf.squeeze(self.variance))
