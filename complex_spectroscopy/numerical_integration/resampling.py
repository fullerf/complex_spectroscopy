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
import tensorflow_probability as tfp

__all__ = ['re_sample_1D_functions', 're_sample_1D_complex_functions']


# +
def re_sample_1D_functions(new_axis, sampled_funcs, 
                           old_axes_mins,
                           old_axes_maxs,
                          ):
        """
        This function will take sampled 1D functions and re-sample them, extending
        as necessary with zeros. We have some strong requirements for these samples,
        however. See below.
        
        args:
        new_axis: a tensor of size JxM of type dtype = gpflow.default_float (assumed)
        sampled_funcs: a tensor of size JxBxN of type dtype = gpflow.default_float (assumed)
        old_axes_mins: a tensor of length J of type dtype = gpflow.default_float (assumed)
        old_axes_maxs: a tensor of length J of type dtype = gpflow.default_float (assumed)
        
        Further:
        sampled_funcs: tensors should correspond to B batches of 1D functions samples of 
            of length N, stored in columns. The samples must be take on an even spacing
            from old_axes_mins[j] to old_axes_maxs[j].
            
        returns: a tensor of size JxBxM
        
        """
        sampled_funcs = sampled_funcs  # now of size JxBxN
        xmins = old_axes_mins[:,None]  # Jx1
        xmaxs = old_axes_maxs[:,None]  # Jx1
        new_axis_b = new_axis[:,None,:]  # Jx1xM
        return tfp.math.batch_interp_regular_1d_grid(new_axis_b, xmins, xmaxs, sampled_funcs,
                                                  fill_value_above=0.,
                                                  fill_value_below=0.)
    
def re_sample_1D_complex_functions(new_axis, sampled_funcs, 
                           old_axes_mins,
                           old_axes_maxs,
                          ):
        """
        This function will take sampled 1D complex-valued functions and re-sample them,
        extending as necessary with zeros. We have some strong requirements for these
        samples, however. See below. Furthermore, we assume the functions accept only
        real arguments.
        
        args:
        new_axis: a tensor of size JxM of type dtype = gpflow.default_float (assumed)
        sampled_funcs: a tensor of size JxBxN of type dtype corresponding to gpflow.default_float (assumed)
        old_axes_mins: a tensor of length J of type dtype = gpflow.default_float (assumed)
        old_axes_maxs: a tensor of length J of type dtype = gpflow.default_float (assumed)
        
        Further:
        sampled_funcs: tensors should correspond to B batches of 1D functions samples of 
            of length N, stored in columns. The samples must be take on an even spacing
            from old_axes_mins[j] to old_axes_maxs[j].
            
        returns: a tensor of size JxBxM
        
        """
        real_samples = tf.math.real(sampled_funcs)
        imag_samples = tf.math.imag(sampled_funcs)
        real_resampled = re_sample_1D_functions(new_axis, real_samples, old_axes_mins,
                                                old_axes_maxs)
        imag_resampled = re_sample_1D_functions(new_axis, imag_samples, old_axes_mins,
                                                old_axes_maxs)
        return tf.complex(real=real_resampled, imag=imag_resampled)
