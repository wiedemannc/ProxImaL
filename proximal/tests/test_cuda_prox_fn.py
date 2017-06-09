from proximal.tests.base_test import BaseTest

from proximal.prox_fns.sum_squares import sum_squares
from proximal.prox_fns.group_norm1 import group_norm1
from proximal.lin_ops.variable import Variable
from proximal.utils.cuda_codegen import PyCudaAdapter

from numpy import random
import numpy as np

class TestCudaProxFn(BaseTest):

    def test_sum_squares(self):
        rho = 2.5
        for alpha in [1.0, 3.0]:
            for beta in [1.0, 5.0]:
                for b in [0.0, np.ones((10,6))]:
                    for c in [0.0, np.ones((10,6))]:
                        for gamma in [0.0, 2.5]:
                            for d in [0.0, 4.0]:

                                x = Variable((10,6))
                                f = sum_squares(x)
                                v = np.reshape(np.arange(10*6), (10,6)).astype(np.float32)

                                xhat1 = f.prox(rho, v.copy(), alpha=alpha, beta=beta, b=b, c=c, gamma=gamma, d=d)
                                adapter = PyCudaAdapter()
                                xhat2 = f.prox_cuda(adapter, rho, v.copy(),  alpha=alpha, beta=beta, b=b, c=c, gamma=gamma, d=d).get()

                                if not np.all(np.abs(xhat1 - xhat2) < 1e-4):
                                    print(f.cuda_code)
                                    print("failed: %f" % np.amax(np.abs(xhat1-xhat2)))
                                self.assertTrue(np.all(np.abs(xhat1 - xhat2) < 1e-4))

        rho = np.random.randn(10,6).astype(np.float32)
        for alpha in [1.0, 3.0]:
            for beta in [1.0, 5.0]:
                for b in [0.0, np.ones((10,6))]:
                    for c in [0.0, np.ones((10,6))]:
                        for gamma in [0.0, 2.5]:
                            for d in [0.0, 4.0]:

                                x = Variable((10,6))
                                f = sum_squares(x)

                                xhat1 = f.prox(rho, v.copy(), alpha=alpha, beta=beta, b=b, c=c, gamma=gamma, d=d)
                                adapter = PyCudaAdapter()

                                xhat2 = f.prox_cuda(adapter, rho, v.copy(),  alpha=alpha, beta=beta, b=b, c=c, gamma=gamma, d=d).get()

                                if not np.all(np.abs(xhat1 - xhat2) < 1e-4):
                                    print(f.cuda_code)
                                    print("failed: %f" % np.amax(np.abs(xhat1-xhat2)))
                                self.assertTrue(np.all(np.abs(xhat1 - xhat2) < 1e-4))




    def test_group_norm1(self):
        random.seed(0)

        x = Variable((10,10,2,3))
        f = group_norm1(x, [2,3])

        v = np.reshape(np.arange(10*10*2*3), (10,10,2,3)).astype(np.float32)
        xhat1 = f.prox(1, v.copy())
        adapter = PyCudaAdapter()
        xhat2 = f.prox_cuda(adapter, 1, v.copy()).get()

        if not np.all(np.abs(xhat1 - xhat2) < 1e-4):
            print(f.cuda_code)
            print("failed: %f" % np.amax(np.abs(xhat1-xhat2)))
        self.assertTrue(np.all(np.abs(xhat1 - xhat2) < 1e-4))

        eps = 1e-5
        maxeps = 0
        for i in range(50):
            x = Variable((10,10,2,3))
            f = group_norm1(x, [2,3])

            v = random.rand(10,10,2,3).astype(np.float32)
            if i < 25:
                rho = np.abs(random.rand(1))
            else:
                rho = np.abs(random.rand(*x.shape))
            xhat1 = f.prox(rho, v.copy())
            xhat2 = f.prox_cuda(adapter, rho, v.copy()).get()

            err = np.amax(np.abs(xhat1 - xhat2))
            if not err < eps:
                print(f.cuda_code)
                print("failed: %f" % np.amax(np.abs(xhat1-xhat2)))
            self.assertTrue(err < eps)
            maxeps = max(err,maxeps)

        for i in range(50):
            x = Variable((10,10,2,3))
            f = group_norm1(x, [2,3])
            v = random.rand(10,10,2,3).astype(np.float32)
            rho = np.abs(random.rand(1))
            alpha = np.abs(random.rand(1))
            beta = np.abs(random.rand(1))
            gamma = np.abs(random.rand(1))
            c = np.abs(random.rand(*f.c.shape))
            b = np.abs(random.rand(*f.b.shape))

            xhat1 = f.prox(rho, v.copy(), alpha=alpha, beta=beta, gamma=gamma, c=c, b=b)
            xhat2 = f.prox_cuda(adapter, rho, v.copy()).get()

            err = np.amax(np.abs(xhat1 - xhat2))
            if not err < eps:
                print(f.cuda_code)
                print("failed: %f" % np.amax(np.abs(xhat1-xhat2)))
            self.assertTrue(err < eps)
            maxeps = max(err,maxeps)

        print("Max proxfn error: %.2e" % maxeps)

if __name__ == "__main__":
    import logging
    logging.getLogger().setLevel(logging.DEBUG)
    t = TestCudaProxFn()
    t.test_sum_squares()
    import unittest
    unittest.main()
