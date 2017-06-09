from __future__ import division
import abc
import numpy as np
from proximal.utils import Impl
from proximal.utils.cuda_codegen import (sub2ind, ind2subCode, indent, gpuarray,
     compile_cuda_kernel, cuda_function)

class ProxFn(object):
    """Represents alpha*f(beta*x - b) + <c,x> + gamma*<x,x> + d
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, lin_op, alpha=1.0, beta=1.0, b=0.0, c=0.0,
                 gamma=0.0, d=0.0, implem=None):
        # Error checking.
        for elem, name in zip([b, c], ["b", "c"]):
            if not (np.isscalar(elem) or elem.shape == lin_op.shape):
                raise Exception("Invalid dimensions of %s." % name)
        for elem, name in zip([alpha, gamma, d], ["alpha", "gamma"]):
            if not np.isscalar(elem) or elem < 0:
                raise Exception("%s must be a nonnegative scalar." % name)
        for elem, name in zip([beta, d], ["beta", "d"]):
            if not np.isscalar(elem):
                raise Exception("%s must be a scalar." % name)

        self.implem_key = implem
        self.implementation = Impl['numpy']
        if implem is not None:
            self.set_implementation(implem)

        self.lin_op = lin_op
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.b = b
        self.c = c
        if np.isscalar(b):
            self.b = b * np.ones(self.lin_op.shape)
        if np.isscalar(c):
            self.c = c * np.ones(self.lin_op.shape)
        self.gamma = float(gamma)
        self.d = float(d)
        self.init_tmps()
        self.kernel_cuda_prox = None
        self._rho_hat = None
        self._xhat = None
        self._bgpu = None
        self._cgpu = None
        super(ProxFn, self).__init__()

    def set_implementation(self, im):
        if im in Impl.values():
            self.implementation = im
        elif im in Impl.keys():
            self.implementation = Impl[im]
        else:
            raise Exception("Invalid implementation.")

        return self.implementation

    def implementation(self, im):
        return self.implementation

    def variables(self):
        """Return a list of the variables in the problem.
        """
        return self.lin_op.variables()

    def init_tmps(self):
        """Initialize temporary variables for _prox method.
        """
        pass

    @abc.abstractmethod
    def _prox(self, rho, v, *args, **kwargs):
        """The prox function for a specific atom.
        """
        return NotImplemented

    def prox(self, rho, v, *args, **kwargs):
        """Wrapper on the prox function to handle alpha, etc.
           It is here the iteration for debug purposese etc.
        """
        rho_hat = (rho + 2 * self.gamma) / (self.alpha * self.beta**2)
        # vhat = (rho*v - c)*beta/(rho + 2*gamma) - b
        # Modify v in-place. This is important for the Python to be performant.
        v *= rho
        v -= self.c
        v *= self.beta / (rho + 2 * self.gamma)
        v -= self.b
        xhat = self._prox(rho_hat, v, *args, **kwargs)
        # x = (xhat + b)/beta
        # Modify result in-place.
        xhat += self.b
        xhat /= self.beta
        return xhat

    def cuda_additional_buffers(self):
        res = []
        if not np.all(self.c == 0):
            res.append( ("c", self.c) )
        if not np.all(self.b == 0):
            res.append( ("b", self.b) )
        return res

    def gen_cuda_code(self):
        shape = self.lin_op.shape
        dimv = int(np.prod(shape))
        self.cuda_args = []
        argnames = []
        for aname,aval in self.cuda_additional_buffers():
            argnames.append(aname)
            aval = gpuarray.to_gpu(aval.astype(np.float32))
            self.cuda_args.append( (aname, aval) )
        argstring = "".join( ", const float *%s" % arg for arg in argnames)
        ccode = "- c[vidx]" if "c" in argnames else ""
        bcode1 = " - b[vidx]" if "b" in argnames else ""
        bcode2 = "xhatc += b[vidx];\n" if "b" in argnames else ""
        if self.gamma == 0:
            plus_2gamma = ""
        else:
            plus_2gamma = " + %.8ef" % (self.gamma * 2)
        alpha_beta_s = self.alpha * self.beta**2;
        if alpha_beta_s == 1:
            div_alpha_beta_s = ""
        else:
            div_alpha_beta_s = " * %.8ef" % (1./alpha_beta_s)
        beta = self.beta
        if beta == 1:
            xhatc_div_beta = ""
            mul_beta = ""
        else:
            xhatc_div_beta = "xhatc *= %.8ef;" % (1./beta)
            mul_beta = " * %.8ef" % beta

        idxvars = ["subidx%d" % d for d in range(len(shape))]
        ind2subcode = ind2subCode("vidx", shape, idxvars)

        v_divisor = "float vDiv = 1.f / (rho%(plus_2gamma)s);\n" % locals()

        gen_v = lambda idx: "(((v[%(idx)s] * rho)%(ccode)s)%(mul_beta)s)*vDiv%(bcode1)s" % dict(
                    idx=sub2ind(idx, shape) if len(idx) > 1 else idx[0],
                    ccode=ccode % locals(),
                    mul_beta=mul_beta,
                    plus_2gamma=plus_2gamma,
                    bcode1=bcode1 % locals(),
                )


        cucode = self._prox_cuda("rho_hat", gen_v, idxvars, "vidx", "xhatc")
        cucode = indent(cucode, 8)
        ind2subcode = indent(ind2subcode, 8)

        code = """
__global__ void prox_off_and_fac(const float *v, float *xhat, float rho, const float *offset, float factor %(argstring)s)
{
    float rho_hat = (rho%(plus_2gamma)s)%(div_alpha_beta_s)s;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for( int vidx = index; vidx < %(dimv)d; vidx += stride )
    {
        %(v_divisor)s
        %(ind2subcode)s
        %(cucode)s
        %(bcode2)s
        %(xhatc_div_beta)s

        xhat[vidx] = offset[vidx] + factor * xhatc;
    }
}

__global__ void prox(const float *v, float *xhat, float rho %(argstring)s)
{
    float rho_hat = (rho%(plus_2gamma)s)%(div_alpha_beta_s)s;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for( int vidx = index; vidx < %(dimv)d; vidx += stride )
    {
        %(v_divisor)s
        %(ind2subcode)s
        %(cucode)s
        %(bcode2)s
        %(xhatc_div_beta)s

        xhat[vidx] = xhatc;
    }
}
""" % locals()

        gen_vppd = lambda idx: "(((v[%(idx)s] * rho[vidx])%(ccode)s)%(mul_beta)s)*vDiv%(bcode1)s" % dict(
                    idx=sub2ind(idx, shape) if len(idx) > 1 else idx[0],
                    ccode=ccode % locals(),
                    mul_beta=mul_beta,
                    plus_2gamma=plus_2gamma,
                    bcode1=bcode1 % locals(),
                )

        cucode_ppd = self._prox_cuda("rho_hat", gen_vppd, idxvars, "vidx", "xhatc")
        cucode_ppd = indent(cucode_ppd, 8)

        code += """
__global__ void prox_ppd_off_and_fac(const float *v, float *xhat, const float *rho, const float *offset, const float *factor %(argstring)s)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for( int vidx = index; vidx < %(dimv)d; vidx += stride )
    {
        float rho_hat = (rho[vidx] %(plus_2gamma)s)%(div_alpha_beta_s)s;
        float vDiv = 1.f / (rho[vidx] %(plus_2gamma)s%(bcode1)s);

        %(ind2subcode)s
        %(cucode_ppd)s
        %(bcode2)s
        %(xhatc_div_beta)s

        xhat[vidx] = offset[vidx] + factor[vidx] * xhatc;
    }
}

__global__ void prox_ppd(const float *v, float *xhat, const float *rho %(argstring)s)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for( int vidx = index; vidx < %(dimv)d; vidx += stride )
    {
        float rho_hat = (rho[vidx] %(plus_2gamma)s)%(div_alpha_beta_s)s;
        float vDiv = 1.f / (rho[vidx] %(plus_2gamma)s%(bcode1)s);

        %(ind2subcode)s
        %(cucode_ppd)s
        %(bcode2)s
        %(xhatc_div_beta)s

        xhat[vidx] = xhatc;
    }
}
""" % locals()

        #print(code)
        self.cuda_source = code
        mod = compile_cuda_kernel(self.cuda_source)
        const_vals = tuple(x[1] for x in self.cuda_args)
        self.kernel_cuda_prox = cuda_function(mod, "prox", int(np.prod(shape)), const_vals)
        self.kernel_cuda_prox_off_and_fac = cuda_function(mod, "prox_off_and_fac", int(np.prod(shape)), const_vals)
        self.kernel_ppd_prox = cuda_function(mod, "prox_ppd", int(np.prod(shape)), const_vals)
        self.kernel_ppd_cuda_prox_off_and_fac = cuda_function(mod, "prox_ppd_off_and_fac", int(np.prod(shape)), const_vals)

    def _cuda_kernel_available(self):
        return hasattr(self, "_prox_cuda")

    def prox_cuda(self, adapter, rho, v, *args, **kwargs):
        if hasattr(rho, "shape") and len(rho.shape) > 0 and rho.shape[0] > 1:
            ppd = True
        else:
            ppd = False
        if self._cuda_kernel_available():
            if self.kernel_cuda_prox is None:
                self.gen_cuda_code()
            if not type(v) == gpuarray.GPUArray:
                v = gpuarray.to_gpu(v.astype(np.float32))
            if self._xhat is None:
                self._xhat = gpuarray.zeros(v.shape, dtype=np.float32)
            xhat = self._xhat
            if not ppd:
                kernel_cuda_prox_off_and_fac = self.kernel_cuda_prox_off_and_fac
                kernel_cuda_prox = self.kernel_cuda_prox
                rho = np.float32(rho)
                factor = np.float32(kwargs.get("factor", 1))
            else:
                kernel_cuda_prox_off_and_fac = self.kernel_ppd_cuda_prox_off_and_fac
                kernel_cuda_prox = self.kernel_ppd_prox
                factor = kwargs.get("factor", None)
            if "offset" in kwargs:
                assert(kwargs["offset"].flags.c_contiguous and kwargs["offset"].dtype == np.float32)
                kernel_cuda_prox_off_and_fac(v, xhat, rho, kwargs["offset"], factor)
            else:
                kernel_cuda_prox(v, xhat, rho)
            return xhat
        else:
            if self._xhat is None:
                self._xhat = gpuarray.zeros(v.shape, dtype=np.float32)
            if self._cgpu is None:
                self._cgpu = gpuarray.to_gpu(self.c.astype(np.float32))
            if self._bgpu is None:
                self._bgpu = gpuarray.to_gpu(self.b.astype(np.float32))
            c = self._cgpu
            b = self._bgpu
            cuda_fun = lambda rho, v, *args, **kw: gpuarray.to_gpu(self._prox(rho.get() if ppd else rho, v.get(), *args, **kw).astype(np.float32()))

            gamma = adapter.scalar(self.gamma)
            beta = adapter.scalar(self.beta)
            alpha = adapter.scalar(self.alpha)

            if type(rho) is adapter.arraytype():
                if self._rho_hat is None:
                    self._rho_hat = gpuarray.zeros(rho.shape, dtype=np.float32)
                rho_hat = self._rho_hat
                adapter.elem_wise_wrapper("rho_hat = (rho + 2.f * gamma) / (alpha * beta * beta)", (("rho_hat",rho_hat),("rho",rho),("gamma",gamma),("alpha",alpha),("beta",beta)))
            else:
                rho_hat = (rho + 2 * self.gamma) / (self.alpha * self.beta**2)

            # vhat = (rho*v - c)*beta/(rho + 2*gamma) - b
            # Modify v in-place. This is important for the Python to be performant.
            xhat = self._xhat
            adapter.elem_wise_wrapper("v = (v*rho - c)*beta / (rho + 2.f*gamma) - b", (("v",v),("rho",rho), ("c",c), ("beta",beta), ("gamma",gamma), ("b",b)))
            #v *= rho
            #v -= c
            #v *= self.beta / (rho + 2 * self.gamma)
            #v -= b
            #print(np.amax(np.abs((test - v).get())), test.get().flatten()[65], v.get().flatten()[65], rho.shape, v.shape)
            xhat = cuda_fun(rho_hat, v, *args, **kwargs)

            if "offset" in kwargs:
                offset = kwargs["offset"]
                factor = kwargs.get("factor", adapter.scalar(1))
                adapter.elem_wise_wrapper("xhat = ((xhat + b)/beta)*factor + offset", (("xhat",xhat),("b",b),("beta",beta),("factor",factor),("offset",offset)))
            else:
                adapter.elem_wise_wrapper("xhat = ((xhat + b)/beta)", (("xhat",xhat),("b",b),("beta",beta)))
            # x = (xhat + b)/beta
            # Modify result in-place.
            #xhat += b
            #xhat /= self.beta
            #if "offset" in kwargs:
            #    xhat = kwargs["offset"] + kwargs.get("factor",1) * xhat
            return xhat

    @abc.abstractmethod
    def _eval(self, v):
        """Evaluate the function on v (ignoring parameters).
        """
        return NotImplemented

    def eval(self, v):
        """Evaluate the function on v.
        """
        #print(self, "norm", v.flatten())
        return self.alpha * self._eval(self.beta * v - self.b) + \
            np.sum(self.c * v) + self.gamma * np.square(v).sum() + self.d

    def eval_cuda(self, v):
        #print(self, "cuda", v.get().flatten())
        alpha = np.float32(self.alpha)
        beta = np.float32(self.beta)
        if self.b is 0.0:
            b = np.float32(self.b)
        else:
            # todo: do this only once
            b = gpuarray.to_gpu(self.b)
        if self.c is 0.0:
            c = np.float32(self.c)
        else:
            c = gpuarray.to_gpu(self.c)
        gamma = np.float32(self.gamma)
        d = np.float32(self.d)
        return float((alpha * self._eval( ((beta * v) - b).get() ) + gpuarray.sum(c * v) + gamma * gpuarray.sum(v*v) + d).get())

    @property
    def value(self):
        return self.eval(self.lin_op.value)

    def __str__(self):
        """Default to string is name of class.
        """
        return self.__class__.__name__

    def __add__(self, other):
        """ProxFn + ProxFn(s).
        """
        if isinstance(other, ProxFn):
            return [self, other]
        elif type(other) == list:
            return [self] + other
        else:
            return NotImplemented

    def __radd__(self, other):
        """Called for list + ProxFn.
        """
        if type(other) == list:
            return other + [self]
        else:
            return NotImplemented

    def __mul__(self, other):
        """ProxFn * Number.
        """
        # Can only multiply by scalar constants.
        if np.isscalar(other) and other > 0:
            return self.copy(alpha=self.alpha * other)
        else:
            raise TypeError("Can only multiply by a positive scalar.")

    def __rmul__(self, other):
        """Called for Number * ProxFn.
        """
        return self * other

    def __div__(self, other):
        """Called for ProxFn/Number.
        """
        return (1. / other) * self

    def __truediv__(self, other):
        """ProxFn / integer.
        """
        return self.__div__(other)

    def copy(self, lin_op=None, **kwargs):
        """Returns a shallow copy of the object.

        Used to reconstruct an object tree.

        Parameters
        ----------
        args : list, optional
            The arguments to reconstruct the object. If args=None, use the
            current args of the object.

        Returns
        -------
        Expression
        """
        if lin_op is None:
            lin_op = self.lin_op
        data = self.get_data()
        curr_args = {'alpha': self.alpha,
                     'beta': self.beta,
                     'gamma': self.gamma,
                     'c': self.c,
                     'b': self.b,
                     'd': self.d,
                     'implem': self.implem_key}
        for key in curr_args.keys():
            if key not in kwargs:
                kwargs[key] = curr_args[key]
        return type(self)(lin_op, *data, **kwargs)

    def get_data(self):
        """Returns info needed to reconstruct the object besides the args.

        Returns
        -------
        list
        """
        return []
