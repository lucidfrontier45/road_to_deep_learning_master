__author__ = 'shiqiaodu'

from unittest import TestCase
from nose.tools import ok_, eq_
import numpy as np
from road_to_deep_learning.naive import mlp
from road_to_deep_learning.utils import functions

model = mlp.MLPRegression(3, 2, 2, batch_size=100, tol=1e-7, lr=0.001,
                          n_iter=100000, report=1000, method="L-BFGS-B",
                          activation=functions.Identity(),
                          lambda_input=0.0, lambda_hidden=0.0)
w_input = np.array([[1, 1, 1], [1, -1, 0]])
w_hidden = np.array([[1, 1], [1, -1]])


class MLPCase(TestCase):
    def setUp(self):
        self.f = functions.Sigmoid()
        model.w_input = w_input
        model.w_hidden = w_hidden

    def tearDown(self):
        pass

    def testSigmoid(self):
        x = 0
        eq_(0.5, self.f(x))
        eq_(0.25, self.f.grad(x))

    def testForward(self):
        x = np.array([[3, 2]])
        a, z, y_hat = model.forward(x)

        a_true = np.array([[6, 1]])
        ok_(np.allclose(a, a_true))

        y_true = np.array([[7, 5]])
        ok_(np.allclose(y_hat, y_true))

    def testBackward(self):
        x = np.array([[3, 2, 1]])
        y = np.array([[6, 2]])
        w = mlp._pack_w(model.w_input, model.w_hidden)
        e, w_grad = model._error_grad_func(w, x, y)
        g_input, g_hidden = mlp._unpack_w(w_grad, x, y)
        eq_(e, 5.0)
        g_hidden_true = np.array([[6, 1], [18, 3]]).T
        ok_(np.allclose(g_hidden, g_hidden_true))
        g_input_true = np.array([[12, 8, 4], [-6, -4, -2]])
        ok_(np.allclose(g_input, g_input_true))

    def testDelta(self):
        x = np.array([[3, 2]])
        y = np.array([[6, 2]])
        a, z, y_hat = model.forward(x)
        delta_hidden, delta_input = mlp._compute_delta(model.w_hidden, y, model.activation.grad, a, y_hat)
        ok_(np.allclose(delta_hidden, np.array([[1, 3]])))
        ok_(np.allclose(delta_input, np.array([[4, -2]])))
