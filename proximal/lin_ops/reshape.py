from .lin_op import LinOp
import numpy as np
import scipy.sparse
from ..utils.cuda_codegen import sub2ind, ind2subCode

class reshape(LinOp):
    """A variable.
    """

    def __init__(self, arg, shape):
        assert arg.size == np.prod(shape)
        self.inshape = arg.shape
        super(reshape, self).__init__([arg], shape)

    def forward(self, inputs, outputs):
        """The forward operator.

        Reads from inputs and writes to outputs.
        """
        shaped_input = np.reshape(inputs[0], self.shape)
        np.copyto(outputs[0], shaped_input)

    def adjoint(self, inputs, outputs):
        """The adjoint operator.

        Reads from inputs and writes to outputs.
        """
        shaped_input = np.reshape(inputs[0], self.input_nodes[0].shape)
        np.copyto(outputs[0], shaped_input)

    def sparse_matrix(self):
        return scipy.sparse.eye(int(np.prod(self.shape)), dtype=np.float32)

    def cuda_kernel_base(self, cg, num_tmp_vars, absidx, parent, in_shape, out_shape, fn):
        linidx = sub2ind(absidx, out_shape)
        linidxvar = "idx_%d" % num_tmp_vars
        num_tmp_vars += 1
        newidxvars = ["idx_%d" % i for i in range(num_tmp_vars, num_tmp_vars + len(in_shape))]
        num_tmp_vars += len(in_shape)
        idxdefcode = ind2subCode(linidx, in_shape, newidxvars)

        icode, var, num_tmp_vars = fn(cg, num_tmp_vars, newidxvars, self)
        code = """/* reshape */
int %(linidxvar)s = %(linidx)s;
(void)%(linidxvar)s;
%(idxdefcode)s
%(icode)s
""" % locals()
        return code, var, num_tmp_vars

    def forward_cuda_kernel(self, cg, num_tmp_vars, absidx, parent):
        return self.cuda_kernel_base(cg, num_tmp_vars, absidx, parent, self.inshape, self.shape, cg.input_nodes(self)[0].forward_cuda_kernel)

    def adjoint_cuda_kernel(self, cg, num_tmp_vars, absidx, parent):
        return self.cuda_kernel_base(cg, num_tmp_vars, absidx, parent, self.shape, self.inshape, cg.output_nodes(self)[0].adjoint_cuda_kernel)

    def is_diag(self, freq=False):
        """Is the lin op diagonal (in the frequency domain)?
        """
        return self.input_nodes[0].is_diag(freq)

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
        return self.input_nodes[0].get_diag(freq)

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
        return input_mags[0]
