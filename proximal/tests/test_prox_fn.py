from proximal.tests.base_test import BaseTest
from proximal.prox_fns import (norm1, sum_squares, sum_entries, nonneg, weighted_norm1,
                               weighted_nonneg, diff_fn, sum_of_sep_diff_fn, schatten)
from proximal.lin_ops import Variable
from proximal.halide.halide import Halide, halide_installed
from proximal.utils.utils import im2nparray, tic, toc
import cvxpy as cvx
import numpy as np

import os
from PIL import Image


class TestProxFn(BaseTest):

    def test_sum_squares(self):
        """Test sum squares prox fn.
        """
        # No modifiers.
        tmp = Variable(10)
        fn = sum_squares(tmp)
        rho = 1
        v = np.arange(10) * 1.0
        x = fn.prox(rho, v.copy())
        self.assertItemsAlmostEqual(x, v * rho / (2 + rho))

        rho = 2
        x = fn.prox(rho, v.copy())
        self.assertItemsAlmostEqual(x, v * rho / (2 + rho))

        # With modifiers.
        mod_fn = sum_squares(tmp, alpha=2, beta=-1,
                             c=np.ones(10) * 1.0, b=np.ones(10) * 1.0, gamma=1)

        rho = 2
        v = np.arange(10) * 1.0
        x = mod_fn.prox(rho, v.copy())

        # vhat = mod_fn.beta*(v - mod_fn.c/rho)*rho/(rho+2*mod_fn.gamma) - mod_fn.b
        # rho_hat = rho/(mod_fn.alpha*np.sqrt(np.abs(mod_fn.beta)))
        # xhat = fn.prox(rho_hat, vhat)
        x_var = cvx.Variable(10)
        cost = 2 * cvx.sum_squares(-x_var - np.ones(10)) + \
            np.ones(10).T * x_var + cvx.sum_squares(x_var) + \
            (rho / 2) * cvx.sum_squares(x_var - v)
        prob = cvx.Problem(cvx.Minimize(cost))
        prob.solve()

        self.assertItemsAlmostEqual(x, x_var.value, places=3)

    def test_sum_of_sep_diff_fn(self):
        func = lambda x, idx: np.sum(x**2, axis=-1)
        grad = lambda x, idx: 2*x
        def hessian(x, idx):
            n = x.shape[1]
            res = np.zeros((x.shape[0], n, n))
            for i in range(n):
                res[:,i,i] = 2
            return res

        dim = (5,2)
        tmp = Variable(dim)
        fn = sum_of_sep_diff_fn(tmp, func, grad, hessian)
        rho = 1
        v = np.reshape(np.arange(dim[0]*dim[1]), dim) * 1.0
        x = fn.prox(rho, v.copy())
        self.assertItemsAlmostEqual(x, v * rho / (2 + rho))

        rho = 2
        x = fn.prox(rho, v.copy())
        self.assertItemsAlmostEqual(x, v * rho / (2 + rho))

        # With modifiers.
        mod_fn = sum_of_sep_diff_fn(tmp, func, grad, hessian,
                                    alpha=2, beta=-1,
                                    c=np.ones(dim) * 1.0,
                                    b=np.ones(dim) * 1.0, gamma=1)

        rho = 2
        v = np.reshape(np.arange(dim[0]*dim[1]), dim) * 1.0
        x = mod_fn.prox(rho, v.copy())

        # vhat = mod_fn.beta*(v - mod_fn.c/rho)*rho/(rho+2*mod_fn.gamma) - mod_fn.b
        # rho_hat = rho/(mod_fn.alpha*np.sqrt(np.abs(mod_fn.beta)))
        # xhat = fn.prox(rho_hat, vhat)
        flatdim = dim[0]*dim[1]
        x_var = cvx.Variable(flatdim)
        cost = 2 * cvx.sum_squares(-x_var - np.ones(flatdim)) + \
            np.ones(flatdim).T * x_var + cvx.sum_squares(x_var) + \
            (rho / 2) * cvx.sum_squares(x_var - v.flatten())
        prob = cvx.Problem(cvx.Minimize(cost))
        prob.solve()

        self.assertItemsAlmostEqual(x.flatten(), x_var.value, places=3)

    def test_norm1(self):
        """Test L1 norm prox fn.
        """
        # No modifiers.
        tmp = Variable(10)
        fn = norm1(tmp)
        rho = 1
        v = np.arange(10) * 1.0 - 5.0
        x = fn.prox(rho, v.copy())
        self.assertItemsAlmostEqual(x, np.sign(v) * np.maximum(np.abs(v) - 1.0 / rho, 0))

        rho = 2
        x = fn.prox(rho, v.copy())
        self.assertItemsAlmostEqual(x, np.sign(v) * np.maximum(np.abs(v) - 1.0 / rho, 0))

        # With modifiers.
        mod_fn = norm1(tmp, alpha=0.1, beta=5,
                       c=np.ones(10) * 1.0, b=np.ones(10) * -1.0, gamma=4)

        rho = 2
        v = np.arange(10) * 1.0
        x = mod_fn.prox(rho, v.copy())

        # vhat = mod_fn.beta*(v - mod_fn.c/rho)*rho/(rho+2*mod_fn.gamma) - mod_fn.b
        # rho_hat = rho/(mod_fn.alpha*mod_fn.beta**2)
        # xhat = fn.prox(rho_hat, vhat)
        x_var = cvx.Variable(10)
        cost = 0.1 * cvx.norm1(5 * x_var + np.ones(10)) + np.ones(10).T * x_var + \
            4 * cvx.sum_squares(x_var) + (rho / 2) * cvx.sum_squares(x_var - v)
        prob = cvx.Problem(cvx.Minimize(cost))
        prob.solve()

        self.assertItemsAlmostEqual(x, x_var.value, places=3)

        # With weights.
        tmp = Variable(10)
        v = np.arange(10) * 1.0 - 5.0
        fn = weighted_norm1(tmp, -v + 1)
        rho = 2
        x = fn.prox(rho, v.copy())
        self.assertItemsAlmostEqual(x, np.sign(v) *
                                    np.maximum(np.abs(v) - np.abs(-v + 1) / rho, 0))

    def test_nonneg(self):
        """Test I(x >= 0) prox fn.
        """
        # No modifiers.
        tmp = Variable(10)
        fn = nonneg(tmp)
        rho = 1
        v = np.arange(10) * 1.0 - 5.0
        x = fn.prox(rho, v.copy())
        self.assertItemsAlmostEqual(x, np.maximum(v, 0))

        rho = 2
        x = fn.prox(rho, v.copy())
        self.assertItemsAlmostEqual(x, np.maximum(v, 0))

        # With modifiers.
        mod_fn = nonneg(tmp, alpha=0.1, beta=5,
                        c=np.ones(10) * 1.0, b=np.ones(10) * -1.0, gamma=4)

        rho = 2
        v = np.arange(10) * 1.0
        x = mod_fn.prox(rho, v.copy())

        vhat = mod_fn.beta * (v - mod_fn.c / rho) * rho / (rho + 2 * mod_fn.gamma) - mod_fn.b
        rho_hat = rho / (mod_fn.alpha * np.sqrt(np.abs(mod_fn.beta)))
        xhat = fn.prox(rho_hat, vhat)

        self.assertItemsAlmostEqual(x, (xhat + mod_fn.b) / mod_fn.beta)

        # With weights.
        tmp = Variable(10)
        v = np.arange(10) * 1.0 - 5.0
        fn = weighted_nonneg(tmp, -v - 4)
        rho = 2
        new_v = v.copy()
        idxs = (-v - 4 != 0)
        new_v[idxs] = np.maximum((-v - 4)[idxs] * v[idxs], 0.) / (-v - 4)[idxs]
        x = fn.prox(rho, v.copy())
        self.assertItemsAlmostEqual(x, new_v)

    def test_norm1_halide(self):
        """Halide Norm 1 test
        """
        if halide_installed():
            # Load image
            testimg_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                            'data', 'angela.jpg')
            img = Image.open(testimg_filename)  # opens the file using Pillow - it's not an array yet
            np_img = np.asfortranarray(im2nparray(img))

            # Convert to gray
            np_img = np.mean(np_img, axis=2)

            # Test problem
            v = np_img
            theta = 0.5

            # Output
            output = np.zeros_like(np_img)
            Halide('prox_L1.cpp').prox_L1(v, theta, output)  # Call

            # Reference
            output_ref = np.maximum(0.0, v - theta) - np.maximum(0.0, -v - theta)

            self.assertItemsAlmostEqual(output, output_ref)

    def test_isonorm1_halide(self):
        """Halide Norm 1 test
        """
        if halide_installed():
            # Load image
            testimg_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                            'data', 'angela.jpg')
            img = Image.open(testimg_filename)
            np_img = np.asfortranarray(im2nparray(img))

            # Convert to gray
            np_img = np.mean(np_img, axis=2)

            # Test problem
            theta = 0.5

            f = np_img
            if len(np_img.shape) == 2:
                f = f[..., np.newaxis]

            ss = f.shape
            fx = f[:, np.r_[1:ss[1], ss[1] - 1], :] - f
            fy = f[np.r_[1:ss[0], ss[0] - 1], :, :] - f
            v = np.asfortranarray(np.stack((fx, fy), axis=-1))

            # Output
            output = np.zeros_like(v)
            Halide('prox_IsoL1.cpp').prox_IsoL1(v, theta, output)  # Call

            # Reference
            normv = np.sqrt(np.multiply(v[:, :, :, 0], v[:, :, :, 0]) +
                            np.multiply(v[:, :, :, 1], v[:, :, :, 1]))
            normv = np.stack((normv, normv), axis=-1)
            with np.errstate(divide='ignore'):
                output_ref = np.maximum(0.0, 1.0 - theta / normv) * v

            self.assertItemsAlmostEqual(output, output_ref)

    def test_poisson_halide(self):
        """Halide Poisson norm test
        """
        if halide_installed():
            # Load image
            testimg_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                            'data', 'angela.jpg')
            img = Image.open(testimg_filename)
            np_img = np.asfortranarray(im2nparray(img))

            # Convert to gray
            np_img = np.mean(np_img, axis=2)

            # Test problem
            v = np_img
            theta = 0.5

            mask = np.asfortranarray(np.random.randn(*list(np_img.shape)).astype(np.float32))
            mask = np.maximum(mask, 0.)
            b = np_img * np_img

            # Output
            output = np.zeros_like(v)

            tic()
            Halide('prox_Poisson.cpp').prox_Poisson(v, mask, b, theta, output)  # Call
            print('Running took: {0:.1f}ms'.format(toc()))

            # Reference
            output_ref = 0.5 * (v - theta + np.sqrt((v - theta) * (v - theta) + 4 * theta * b))
            output_ref[mask <= 0.5] = v[mask <= 0.5]

            self.assertItemsAlmostEqual(output, output_ref)

    def test_diff_fn(self):
        """Test generic differentiable function operator.
        """
        # Least squares.
        tmp = Variable(10)
        fn = diff_fn(tmp, lambda x: np.square(x).sum(), lambda x: 2 * x)
        rho = 1
        v = np.arange(10) * 1.0 - 5.0
        x = fn.prox(rho, v.copy())
        self.assertItemsAlmostEqual(x, v / (2 + rho))

        # -log
        n = 5
        tmp = Variable(n)
        fn = diff_fn(tmp, lambda x: -np.log(x).sum(), lambda x: -1.0 / x)
        rho = 2
        v = np.arange(n) * 2.0 + 1
        x = fn.prox(rho, v.copy())
        val = (v + np.sqrt(v**2 + 4 / rho)) / 2
        self.assertItemsAlmostEqual(x, val)

    def test_sum_entries(self):
        """Sum of entries of lin op.
        """
        tmp = Variable(10)
        v = np.arange(10) * 1.0 - 5.0
        fn = sum_entries(tmp)
        x = fn.prox(1.0, v.copy())
        self.assertItemsAlmostEqual(x, v - 1)

    def test_schatten(self):

        for p in [1,2,np.inf]:

            if p == 1:
                cvx_prox_inner = cvx.normNuc
            elif p == 2:
                cvx_prox_inner = lambda x: cvx.pnorm(x.flatten(), p=2)
            elif np.isinf(p):
                cvx_prox_inner = cvx.sigma_max

            # 2 norm
            for rho in [1, 0.1, 10]:
                # a single matrix
                tmp = Variable((2,2))
                v = np.reshape(np.arange(tmp.size), tmp.shape).astype(np.float) - tmp.size/2
                fn = schatten(tmp, p)
                res = fn.prox(rho, v.copy())

                ctmp = cvx.Variable((2,2))
                prob = (1/rho)*cvx_prox_inner(ctmp) + 0.5*cvx.sum_squares(ctmp - v)
                cvx.Problem(cvx.Minimize(prob)).solve()

                resval1 = (1/rho)*fn.eval(res) + 0.5*np.sum((res - v)**2)
                resval2 = (1/rho)*fn.eval(ctmp.value) + 0.5*np.sum((ctmp.value - v)**2)

                print("p", p, "rho %5.1f"%rho, "merrit(proximal) %.3e"%resval1, "merrit(cvx) %.3e"%resval2, "cvxprob.value %.3e"% prob.value)
                self.assertItemsAlmostEqual(ctmp.value, res)

                # a vector of matrices
                tmp = Variable((5, 3, 2))
                v = np.reshape(np.arange(tmp.size), tmp.shape).astype(np.float) - tmp.size/2
                fn = schatten(tmp, p)
                res = fn.prox(rho, v.copy())

                ctmp = [cvx.Variable(tmp.shape[1:]) for i in range(tmp.shape[0])]
                prob = (1/rho)*(cvx_prox_inner(ctmp[0]) +
                                cvx_prox_inner(ctmp[1]) +
                                cvx_prox_inner(ctmp[2]) +
                                cvx_prox_inner(ctmp[3]) +
                                cvx_prox_inner(ctmp[4])) + 0.5*(cvx.sum_squares(ctmp[0] - v[0,...]) +
                                                                cvx.sum_squares(ctmp[1] - v[1,...]) +
                                                                cvx.sum_squares(ctmp[2] - v[2,...]) +
                                                                cvx.sum_squares(ctmp[3] - v[3,...]) +
                                                                cvx.sum_squares(ctmp[4] - v[4,...]) )
                cvx.Problem(cvx.Minimize(prob)).solve()

                ctmpval = np.array([ctmp[i].value for i in range(v.shape[0])])
                resval1 = (1/rho)*fn.eval(res) + 0.5*np.sum((res - v)**2)
                resval2 = (1/rho)*fn.eval(ctmpval) + 0.5*np.sum((ctmpval - v)**2)

                print("p", p, "rho %5.1f"%rho, "merrit(proximal) %.3e"%resval1, "merrit(cvx) %.3e"%resval2, "cvxprob.value %.3e"% prob.value)
                self.assertItemsAlmostEqual(ctmpval, res, places=2)


    def test_overloading(self):
        """Test operator overloading.
        """
        x = Variable(1)
        fn = sum_squares(x, b=1)
        val = fn.prox(2.0, 0)
        self.assertItemsAlmostEqual([val], [0.5])

        fn = 2 * sum_squares(x, b=1)
        val = fn.prox(4.0, 0)
        self.assertItemsAlmostEqual([val], [0.5])

        fn = sum_squares(x, b=1) * 2
        val = fn.prox(4.0, 0)
        self.assertItemsAlmostEqual([val], [0.5])

        fn = sum_squares(x, b=1) / 2
        val = fn.prox(1.0, 0)
        self.assertItemsAlmostEqual([val], [0.5])

        fn1 = sum_squares(x, b=1)
        fn2 = norm1(x)
        arr = fn1 + fn2
        self.assertEqual(type(arr), list)
        self.assertEqual(len(arr), 2)
        arr = arr + fn2
        self.assertEqual(type(arr), list)
        self.assertEqual(len(arr), 3)

if __name__ == "__main__":
    TestProxFn().test_schatten()

