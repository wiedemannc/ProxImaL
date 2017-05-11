from proximal.tests.base_test import BaseTest

from proximal import conv_nofft, grad, mul_elemwise, scale, subsample, uneven_subsample, sum, transpose, CompGraph, Variable, reshape, hstack
from proximal.lin_ops.pxwise_matrixmult import pxwise_matrixmult
from proximal.lin_ops.vstack import vstack
import numpy as np
import numpy.random

class TestSparseMat(BaseTest):

    def _generic_test(self, opfactory, inshapes):

        T = [np.zeros(s) for s in inshapes]

        op = opfactory(*T)

        sA = op.sparse_matrix()
        tA = np.zeros(sA.shape)

        for i in range(tA.shape[1]):
            off = 0
            ok = 0
            for t in T:
                m = int(np.prod(t.shape))
                if i >= off and i < off + m:
                    t[np.unravel_index(i-off, t.shape)] = 1
                    tmp = np.zeros(op.shape)
                    op.forward(T, [tmp])
                    tA[:,i] = tmp.flat
                    t[np.unravel_index(i-off, t.shape)] = 0
                    ok = 1
                    break
                off += m
            assert(ok)

        self.assertTrue(np.allclose(sA.toarray(), tA))

        R = [numpy.random.rand(*s) for s in inshapes]
        r1 = np.zeros(op.shape)
        op.forward(R, [r1])
        r2 = np.reshape(sA.dot(np.concatenate([t.flat for t in R])), op.shape)

        self.assertTrue(np.allclose(r1,r2))

    def test_sum(self):
        self._generic_test(lambda *args: sum(args), [(5,6), (5,6), (5,6)])

    def test_transpose(self):
        self._generic_test(lambda x: transpose(x, (1,2,0)), [(3,4,5)])

    def test_uneven_subsample(self):
        self._generic_test(lambda x: uneven_subsample(x, ([[0,1,4], [0,3,4]], [[0,1,4], [3,4,5]])), [(5,6)])

    def test_subsample(self):
        self._generic_test(lambda x: subsample(x, [2,3]), [(5,6)])

    def test_scale(self):
        self._generic_test(lambda x: scale(4.0, x), [(5,6)])

    def test_pxwisematrixmult(self):
        A = numpy.random.rand(5,6, 2,3)
        self._generic_test(lambda x: pxwise_matrixmult(A, x), [(5,6,3)])

    def test_mulelemwise(self):
        self._generic_test(lambda x: mul_elemwise(numpy.random.rand(*x.shape)*100, x), [(5,6)])

    def test_grad(self):
        self._generic_test(grad, [(5,6)])

    def test_conv_nofft(self):
        k = np.reshape(np.arange(1, 16), (3,5))
        self._generic_test(lambda x: conv_nofft(k, x), [(5,6)])

    def test_reshape(self):
        self._generic_test(lambda x: reshape(x, (5,3,2)), [(5,6)])

    def test_vstack(self):
        self._generic_test(lambda *args: vstack(args), [(5,6), (2,5)])

    def test_hstack(self):
        self._generic_test(lambda *args: hstack(args), [(5,6), (30,)])

    def test_comp_graph(self):
        x = Variable((5,6))
        y = Variable((2,5))
        A = numpy.random.rand(2,5,2,2)
        gy = grad(y)
        cg = CompGraph(vstack([x*10, gy + pxwise_matrixmult(A, gy)]))
        sm = cg.sparse_matrix()

        x = numpy.random.rand(cg.input_size)
        y1 = np.zeros(cg.output_size)
        cg.forward(x, y1)
        y2 = sm.dot(x)

        self.assertTrue(np.allclose(y1,y2))

        #print(sm)

if __name__ ==  "__main__":
    t = TestSparseMat()
    t.test_reshape()
    t.test_hstack()
    import unittest
    unittest.main()

