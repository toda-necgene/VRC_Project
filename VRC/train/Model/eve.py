from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.training import optimizer
from tensorflow.python.training import training_ops

import tensorflow as tf

class EveOptimizer(optimizer.Optimizer):
    """INPUT:
        params: collection of Tensorflow variables to optimize with respect to the cost

        cost: objective function that will be optimized with respect to the parameters indicated

        lr: initial learning rate of the optimization algorithm
            0.001 has been shown to be a good starting value in practice

        k: lowerbound threshold value used for numerical stability during computation
            0.1 has been shown to be a good value in practice.

        K: upperbound threshold value used for numerical stability during computation
            10 has been shown to be a good value in practice

        B1: Adam's smoothing factor for computation of the first moment of the gradient
            0.9 has been shown to be a good value in practice

        B2: Adam's smoothing factor for computation of the second moment of the gradient
            0.999 has been shown to be a good value in practice

        B3: Eve's smoothing factor for tracking the relative change of the objective function
            0.999 has been shown to be a good value in practice

    OUTPUT:
        A single Tensorflow op for parameter updates

    """

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999,beta3=0.999,small_k=0.1,large_k=10.0, epsilon=1e-8,
               use_locking=False, name="Eve"):



        super(EveOptimizer, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2
        self._beta3 = beta3
        self._epsilon = epsilon
        self._small_k = small_k
        self._large_k = large_k
        # Tensor versions of the constructor arguments, created in _prepare().
        self._lr_t = None
        self._beta1_t = None
        self._beta2_t = None
        self._epsilon_t = None

        # Variables to accumulate the powers of the beta parameters.
        # Created in _create_slots when we know the variables to optimize.
        self._beta1_power = None
        self._beta2_power = None

        # Created in SparseApply if needed.
        self._updated_lr = None

    def _get_beta_accumulators(self):
        return self._beta1_power, self._beta2_power

    def _non_slot_variables(self):
        return self._get_beta_accumulators()

    def _create_slots(self, var_list):
    # Create the beta1 and beta2 accumulators on the same device as the first
    # variable. Sort the var_list to make sure this device is consistent across
    # workers (these need to go on the same PS, otherwise some updates are
    # silently ignored).
        first_var = min(var_list, key=lambda x: x.name)

        create_new = self._beta1_power is None
        if not create_new and context.in_graph_mode():
            create_new = (self._beta1_power.graph is not first_var.graph)

        if create_new:
            with ops.colocate_with(first_var):
                self._beta1_power = variable_scope.variable(self._beta1,
                                                        name="beta1_power",
                                                        trainable=False)
                self._beta2_power = variable_scope.variable(self._beta2,
                                                        name="beta2_power",
                                                        trainable=False)
        # Create slots for the first and second moments.
        for v in var_list:
            self._zeros_slot(v, "m", self._name)
            self._zeros_slot(v, "v", self._name)
            self._get_or_make_slot_with_initializer(v,tf.constant_initializer(1.),v.shape,tf.float32, "eve_d", self._name)
            self._get_or_make_slot_with_initializer(v,tf.constant_initializer(1.),v.shape,tf.float32, "eve_fhat", self._name)

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
        self._beta1_t = ops.convert_to_tensor(self._beta1, name="beta1")
        self._beta2_t = ops.convert_to_tensor(self._beta2, name="beta2")
        self._epsilon_t = ops.convert_to_tensor(self._epsilon, name="epsilon")

    def _apply_dense(self, grad, var):
        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")
        return training_ops.apply_adam(
            var, m, v,
            math_ops.cast(self._beta1_power, var.dtype.base_dtype),
            math_ops.cast(self._beta2_power, var.dtype.base_dtype),
            math_ops.cast(self._lr_t, var.dtype.base_dtype),
            math_ops.cast(self._beta1_t, var.dtype.base_dtype),
            math_ops.cast(self._beta2_t, var.dtype.base_dtype),
            math_ops.cast(self._epsilon_t, var.dtype.base_dtype),
            grad, use_locking=self._use_locking).op

    def _resource_apply_dense(self, grad, var):
        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")
        return training_ops.resource_apply_adam(
            var.handle, m.handle, v.handle,
            math_ops.cast(self._beta1_power, grad.dtype.base_dtype),
            math_ops.cast(self._beta2_power, grad.dtype.base_dtype),
            math_ops.cast(self._lr_t, grad.dtype.base_dtype),
            math_ops.cast(self._beta1_t, grad.dtype.base_dtype),
            math_ops.cast(self._beta2_t, grad.dtype.base_dtype),
            math_ops.cast(self._epsilon_t, grad.dtype.base_dtype),
            grad, use_locking=self._use_locking)

    def _apply_sparse_shared(self, grad, var, indices, scatter_add):
        ''' Eve optimizer '''
        beta1_power = math_ops.cast(self._beta1_power, var.dtype.base_dtype)
        beta2_power = math_ops.cast(self._beta2_power, var.dtype.base_dtype)
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
        epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)
        lr = (lr_t * math_ops.sqrt(1 - beta2_power) / (1 - beta1_power))
        # m_t = beta1 * m + (1 - beta1) * g_t
        m = self.get_slot(var, "m")
        m_scaled_g_values = grad * (1 - beta1_t)
        m_t = state_ops.assign(m, m * beta1_t,
                               use_locking=self._use_locking)
        with ops.control_dependencies([m_t]):
            m_t = scatter_add(m, indices, m_scaled_g_values)
        # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
        v = self.get_slot(var, "v")
        v_scaled_g_values = (grad * grad) * (1 - beta2_t)
        v_t = state_ops.assign(v, v * beta2_t, use_locking=self._use_locking)
        with ops.control_dependencies([v_t]):
            v_t = scatter_add(v, indices, v_scaled_g_values)
        v_sqrt = math_ops.sqrt(v_t)

        d = self.get_slot(1., 'eve_d')
        f_hat = self.get_slot(1., 'eve_fhat')

        def f3(): return tf.constant(self._small_k+1, dtype = tf.float32), tf.constant(self._large_k+1, dtype = tf.float32)
        def f4(): return tf.constant( 1./(self._large_k + 1) ), tf.constant(1./(self._small_k + 1))

        delta1_t, delta2_t = tf.cond( tf.greater_equal(grad, f_hat), f3, f4 )
        c_t = tf.minimum(tf.maximum(delta1_t, tf.div(grad, f_hat) ), delta2_t)

        r_t = tf.div( tf.abs( f_hat*(c_t - 1) ), tf.minimum(c_t*f_hat, f_hat) )
        d_scaled_r_values=(1. - self._beta3)*r_t
        d_t = state_ops.assign(d,self._beta3*d ,use_locking=self._use_locking)
        f_hat = state_ops.assign(f_hat, grad, use_locking=self._use_locking)
        with ops.control_dependencies([d_t]):
            d_t = scatter_add(d, indices, d_scaled_r_values)
        var_update = state_ops.assign_sub(var,
                                          lr * m_t / (d_t*v_sqrt + epsilon_t),
                                          use_locking=self._use_locking)
        return tf.group(*[var_update, m_t, v_t, d_t, f_hat])


    def _apply_sparse(self, grad, var):
        return self._apply_sparse_shared(
            grad.values, var, grad.indices,
            lambda x, i, v: state_ops.scatter_add( x, i, v, use_locking=self._use_locking))

    def _resource_scatter_add(self, x, i, v):
        with ops.control_dependencies(
            [resource_variable_ops.resource_scatter_add(
                x.handle, i, v)]):
            return x.value()

    def _resource_apply_sparse(self, grad, var, indices):
        return self._apply_sparse_shared(grad, var, indices, self._resource_scatter_add)

    def _finish(self, update_ops, name_scope):
        # Update the power accumulators.
        with ops.control_dependencies(update_ops):
            with ops.colocate_with(self._beta1_power):
                update_beta1 = self._beta1_power.assign(
                    self._beta1_power * self._beta1_t,
                    use_locking=self._use_locking)
                update_beta2 = self._beta2_power.assign(
                    self._beta2_power * self._beta2_t,
                    use_locking=self._use_locking)
                return control_flow_ops.group(*update_ops + [update_beta1, update_beta2],name=name_scope)

# ECE 411, Computational Graphs for Machine Learning
# Professor Chris Curro

# Midterm Project: A Tensorflow implementation of Eve that improves upon Adam SGD optimization
# By Frank Longueira

# Acknowledgements: The paper describing this improvement is named
# "Improving Stochastic Gradient Descent with Feedback" by Jayanth Koushik & Hiroaki Hayashi
# of Carnegie Mellon University. The implementation found in the code below is based on the
# following code implementing Adam: https://github.com/openai/weightnorm/blob/master/tensorflow/nn.py




def adam_updates(params, cost_or_grads, lr=0.001, B1=0.9, B2=0.999):
    ''' Adam optimizer '''
    updates = []
    if type(cost_or_grads) is not list:
        grads = tf.gradients(cost_or_grads, params)
    else:
        grads = cost_or_grads
    t = tf.Variable(1., 'adam_t')
    for p, g in zip(params, grads):
        v = tf.Variable(tf.zeros(p.get_shape()), p.name + '_adam_v')
        if B1>0:
            m = tf.Variable(tf.zeros(p.get_shape()), p.name + '_adam_m')
            m_t = B1*m + (1. - B1)*g
            m_hat = m_t / (1. - tf.pow(B1,t))
            updates.append(m.assign(m_t))
        else:
            m_hat = g
        v_t = B2*v + (1. - B2)*tf.square(g)
        v_hat = v_t / (1. - tf.pow(B2,t))
        g_t = m_hat / tf.sqrt(v_hat + 1e-8)
        p_t = p - lr * g_t
        updates.append(v.assign(v_t))
        updates.append(p.assign(p_t))
    updates.append(t.assign_add(1))
    return tf.group(*updates)