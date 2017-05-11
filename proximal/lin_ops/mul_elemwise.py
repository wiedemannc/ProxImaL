from .lin_op import LinOp
from ..utils.cuda_codegen import sub2ind
import numpy as np
from proximal.utils.utils import Impl
from proximal.halide.halide import Halide
import scipy.sparse

class mul_elemwise(LinOp):
    """Elementwise multiplication weight*X with a fixed constant.
    """

    def __init__(self, weight, arg, implem=None):
        assert arg.shape == weight.shape
        #assert np.all(weight > 0.0)
        self.weight = weight
        shape = arg.shape

        # Halide temp
        if len(shape) in [2, 3]:
            self.weight = np.asfortranarray(self.weight.astype(np.float32))
            self.tmpout = np.zeros(arg.shape, dtype=np.float32, order='F')

        super(mul_elemwise, self).__init__([arg], shape, implem)

    def forward(self, inputs, outputs):
        """The forward operator.

        Reads from inputs and writes to outputs.
        """
        if self.implementation == Impl['halide'] and (len(self.shape) in [2, 3]):

            # Halide implementation
            tmpin = np.asfortranarray(inputs[0].astype(np.float32))
            Halide('A_mask.cpp').A_mask(tmpin, self.weight, self.tmpout)  # Call
            np.copyto(outputs[0], self.tmpout)

        else:
            # Numpy
            np.multiply(inputs[0], self.weight, outputs[0])

    def adjoint(self, inputs, outputs):
        """The adjoint operator.

        Reads from inputs and writes to outputs.
        """
        self.forward(inputs, outputs)

    def sparse_matrix(self):
        n = self.weight.flatten().shape[0]
        return scipy.sparse.diags([self.weight.flatten()], [0], shape=(n,n), dtype=np.float32)

    def cuda_additional_buffers(self):
        return [("mul_elemwise_%d" % self.linop_id, np.ascontiguousarray(self.weight.astype(np.float32)))]

    def cuda_kernel_base(self, cg, num_tmp_vars, abs_idx, parent, fn):
        arg = self.cuda_additional_buffers()[0][0]
        linidx = sub2ind(abs_idx, self.shape)

        res = "res_%d" % num_tmp_vars
        num_tmp_vars += 1

        code = """/* mul_elemwise */
float %(res)s = %(arg)s[%(linidx)s];
""" % locals()
        num_tmp_vars += 1
        n = cg.input_nodes(self)[0]
        icode, var, num_tmp_vars = fn(cg, num_tmp_vars, abs_idx, self)
        code += """
%(icode)s
%(res)s *= %(var)s;
""" % locals()

        return code, res, num_tmp_vars

    def forward_cuda_kernel(self, cg, num_tmp_vars, abs_idx, parent):
        return self.cuda_kernel_base(cg, num_tmp_vars, abs_idx, parent, cg.input_nodes(self)[0].forward_cuda_kernel)

    def adjoint_cuda_kernel(self, cg, num_tmp_vars, abs_idx, parent):
        return self.cuda_kernel_base(cg, num_tmp_vars, abs_idx, parent, cg.output_nodes(self)[0].adjoint_cuda_kernel)

    def is_diag(self, freq=False):
        """Is the lin op diagonal (in the frequency domain)?
        """
        return not freq and self.input_nodes[0].is_diag(freq)

    def get_diag(self, freq=False):
        """Returns the diagonal representation (A^TA)^(1/2).

        Parameters
        ----------
        freq : bool
            Is the diagonal representation in the frequency domain?
        Returns
        -------
        dict of variable to ndarray
            The diagonal operator acting on each variable.
        """
        assert not freq
        var_diags = self.input_nodes[0].get_diag(freq)
        self_diag = np.reshape(self.weight, self.size)
        for var in var_diags.keys():
            var_diags[var] = var_diags[var] * self_diag
        return var_diags

    def norm_bound(self, input_mags):
        """Gives an upper bound on the magnitudes of the outputs given inputs.

        Parameters
        ----------
        input_mags : list
            List of magnitudes of inputs.

        Returns
        -------
        float
            Magnitude of outputs.
        """
        return np.max(np.abs(self.weight)) * input_mags[0]
