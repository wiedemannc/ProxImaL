from .prox_fn import ProxFn
import numpy as np

def project_to_b1(x, axis=None):
    if axis is not None and axis < 0:
        axis = len(x.shape) + axis
    x_shape = x.shape
    nx = np.linalg.norm(x, 1, axis=axis, keepdims=True)
    cond = nx <= 1
    sgn = np.sign(x)
    x = np.abs(x)
    x_sorted = np.sort(x, axis=axis)
    if axis is None:
        assert len(x_sorted.shape) == 1
        n = x_sorted.shape[0]
        x_sorted = np.reshape(x_sorted, (1,n,1))
    else:
        n = x_sorted.shape[axis]
        k = int(np.prod(x_sorted.shape[axis+1:]))
        x_sorted = np.reshape(x_sorted, (-1,n,k))

    x = np.reshape(x, x_sorted.shape)
    cond = np.reshape(cond, x_sorted.shape[:1] + x_sorted.shape[2:])
    sgn = np.reshape(sgn, x_sorted.shape)

    lmb = -np.ones(x_sorted.shape[:1] + x_sorted.shape[2:])
    for i in range(n):
        x_subset = x_sorted[:,i:,:]
        # sum(x_subset, axis=1) - n*lmb = 1
        # lmb = (sum(x_subset, axis=1)-1)/n
        lmb_t = (np.sum(x_subset, axis=1) - 1)/(n-i)
        v = np.sum(np.maximum(x_sorted - lmb_t[:,np.newaxis,:], 0), axis=1)
        valid = np.logical_and(v >= 0.999999, v <= 1.0000001)
        lmb[valid] = lmb_t[valid]
    assert np.all(lmb[np.logical_not(cond)] >= 0)
    res = sgn*np.choose(cond[:,np.newaxis,:], (np.maximum(x - lmb[:,np.newaxis,:],0) , x))
    res = np.reshape(res, x_shape)
    assert np.all( np.sum(np.abs(res), axis=axis) <= 1.000001)
    return res


class schatten(ProxFn):
    def __init__(self, lin_op, p, **kwargs):
        """
        """
        assert len(lin_op.shape) >= 2
        self.p = p
        if self.p == 1:
            # projection to l_inf ball
            self.q = np.inf
        elif np.isinf(self.p):
            self.q = 1
        else:
            self.q = self.p/(self.p-1)
        super(schatten, self).__init__(lin_op, **kwargs)

    def _prox(self, rho, v, *args, **kwargs):
        U,s,V = np.linalg.svd(v)

        tau = 1/rho
        if self.p == 1:
            # soft thresholding
            s = np.sign(s)*np.maximum(0, np.abs(s) - tau)
        elif self.p == 2:
            ns = np.linalg.norm(s, axis=-1, keepdims=True)
            s = np.choose(ns >= tau, (0, (ns-tau)/ns*s))
        elif np.isinf(self.p):
            s = s - tau*project_to_b1(s/tau, axis=-1)
        else:
            raise RuntimeError("Not implemented: %s-norm" % self.p)

        S = np.zeros(s.shape[:-1] + (U.shape[-1],V.shape[-1]))
        for i in range(s.shape[-1]):
            S[...,i,i] = s[...,i]
        #S = np.diag(s)
        res = np.matmul(U, np.matmul(S, V))
        return res

    def _eval(self, v):
        s = np.linalg.svd(v, compute_uv=False)
        if self.p == 1:
            return np.sum(s)
        elif self.p == 2:
            res_S = np.sum(np.linalg.norm(s, axis=-1))
            if 0:
                # check that this is equivalent to frobenius norm
                res_F = np.sum(np.linalg.norm(np.reshape(v, v.shape[:-2] + (v.shape[-2]*v.shape[-1],)), axis=-1))
                assert np.allclose(res_F, res_S)
            return res_S
        elif np.isinf(self.p):
            return np.sum( np.max(s, axis=-1) )
        else:
            raise RuntimeError("Not implemented")

    def get_data(self):
        return [self.p]


