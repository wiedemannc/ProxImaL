from .prox_fn import ProxFn
import numpy as np
import scipy.optimize as opt

def newton_solve_seperable(x0, f, grad_f, hessian_f, newtonIts=5, maxLineSearchIts=10, lineSearchAlpha=0.1, lineSearchBeta=0.5, assertSanity=True):
    x = x0
    f0 = None
    for nit in range(newtonIts):
        ls_non_conv = np.ones(x.shape[0], np.bool)
        # calculate newton steps dx (n x k)
        gfx = grad_f(x, ls_non_conv)
        def elemdot(A,b):
            # elementwise dot product of A:
            # R[:,0] = A[:,0,0]*b[:,0] + A[:,0,1]*b[:,1] + ...
            # R[:,1] = A[:,1,0]*b[:,0] + A[:,1,1]*b[:,1] + ...
            # ...
            R = np.zeros(A.shape[:-1])
            for i in range(A.shape[1]):
                R[:,i] = np.sum(A[:,i,:]*b, axis=-1)
            return R
        hinv = np.linalg.inv( hessian_f(x, ls_non_conv) )
        dx = -elemdot(hinv, gfx)
        fx = f(x, ls_non_conv)
        if assertSanity and f0 is None: f0 = fx
        # calculate t by backtracking line search
        t = np.ones(x.shape[0])
        for lit in range(maxLineSearchIts):
            xtest = x[ls_non_conv] + t[ls_non_conv,np.newaxis]*dx[ls_non_conv,:]
            fxtest = f(xtest, ls_non_conv)
            ls_non_conv[ls_non_conv] = (fxtest > fx[ls_non_conv] +
                                                    lineSearchAlpha*t[ls_non_conv]*
                                                        np.sum(gfx[ls_non_conv,:]*dx[ls_non_conv], axis=-1))
            if np.sum(ls_non_conv) == 0:
                break
            t[ls_non_conv] *= lineSearchBeta
        x = x + t[:,np.newaxis]*dx
    if assertSanity: 
        idx = np.ones(x.shape[0], np.bool)
        assert np.all(f(x, idx) <= f0)
        grad_f(x, idx)
        hessian_f(x, idx)
    return x



class sum_of_sep_diff_fn(ProxFn):
    def __init__(self, lin_op, func, grad, hessian,
                 newtonParams=dict(newtonIts=5, maxLineSearchIts=10, lineSearchAlpha=0.1, lineSearchBeta=0.5, assertSanity=True),
                 **kwargs):
        """
        represents the sum of seperable, 2 times differentiable, function. Solution is found by a fixed number
        of Newton iterations.

        func   : func(x) applies the seperable function on all k dimensional elements in x [n x k] -> [n]
        grad   : grad(x) returns n k dimensional gradients -> [n x k]
        hessian: func_hessian(x) returns n hessians, each [k x k] -> [n x k x k]
        """
        assert len(lin_op.shape) == 2
        self.func = func
        self.grad = grad
        self.hessian = hessian
        self.newtonParams = newtonParams
        self.sep_prox_func = lambda v, rho, x, idx: self.func(x, idx) + np.sum( (rho*0.5)*(x - v)**2, axis=-1 )
        self.sep_prox_grad = lambda v, rho, x, idx: self.grad(x, idx) + rho*(x - v)
        self.last_res = None
        def rho2diag(rho, shape):
            res = np.zeros(shape[:1] + (shape[-1]**2,))
            idx_eye = np.ravel_multi_index((range(shape[-1]),)*2, (shape[-1],shape[-1]) )
            res[:,idx_eye] = rho
            return np.reshape(res, (shape + (shape[-1],)))
        self.sep_prox_hessian = lambda v, rho, x, idx: (self.hessian(x, idx) + rho2diag(rho, x.shape))
                                                   #np.repeat((np.eye(x.shape[-1])*rho)[np.newaxis,...],
                                                   #          x.shape[0], axis=0))
        super(sum_of_sep_diff_fn, self).__init__(lin_op, **kwargs)

    def _prox(self, rho, v, *args, **kwargs):
        f = lambda x, idx: self.sep_prox_func(v[idx], rho if np.isscalar(rho) else rho[idx,:], x, idx)
        grad_f = lambda x, idx: self.sep_prox_grad(v[idx], rho if np.isscalar(rho) else rho[idx,:], x, idx)
        hessian_f = lambda x, idx: self.sep_prox_hessian(v[idx], rho if np.isscalar(rho) else rho[idx,:], x, idx)
        if self.last_res is None:
            self.last_res = v
        res = newton_solve_seperable(self.last_res, f, grad_f, hessian_f, **self.newtonParams)
        self.last_res = res
        return res

    def _eval(self, v):
        return np.sum(self.func(v, np.ones(v.shape[0], np.bool)))

    def get_data(self):
        return [self.func, self.grad, self.hessian, self.newtonParams]
