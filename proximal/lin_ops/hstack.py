from .lin_op import LinOp
from ..utils.cuda_codegen import ind2subCode, sub2ind, indent, NodeReverseInOut
import numpy as np
import scipy.sparse

class hstack(LinOp):
    """Horizontally concatenates vector inputs.
    """

    def __init__(self, input_nodes, implem=None):
        height = input_nodes[0].size
        width = len(input_nodes)
        super(hstack, self).__init__(input_nodes, (height, width))

    def forward(self, inputs, outputs):
        """The forward operator.

        Reads from inputs and writes to outputs.
        """
        for idx, input_data in enumerate(inputs):
            outputs[0][:, idx] = input_data.flatten()

    def adjoint(self, inputs, outputs):
        """The adjoint operator.

        Reads from inputs and writes to outputs.
        """
        for idx, output_data in enumerate(outputs):
            data = inputs[0][:, idx]
            output_data[:] = np.reshape(data, output_data.shape)

    def sparse_matrix(self):
        height = self.shape[0]
        width = self.shape[1]
        V = np.zeros((height,width), dtype=np.float32)
        I = np.zeros((height,width), dtype=np.float32)
        J = np.zeros((height,width), dtype=np.float32)
        for j in range(width):
            V[:,j] = 1
            I[:,j] = np.arange(height, dtype=np.int32)*width + j
            J[:,j] = np.arange(height, dtype=np.int32) + j*height

        return scipy.sparse.coo_matrix((V.flat, (I.flat,J.flat)), shape=(height*width, height*width), dtype=np.float32)


    def forward_cuda_kernel(self, cg, num_tmp_vars, abs_idx, parent):
        #print("vstack:forward:cuda")
        # multiple reshaped output in, linear index out
        res = "var_%(num_tmp_vars)d" % locals()
        num_tmp_vars += 1
        nidx = abs_idx[1]
        code = """/*hstack*/
float %(res)s = 0;
int hstack_test_idx_%(num_tmp_vars)d = %(nidx)s;
""" % locals()
        nidx = "hstack_test_idx_%(num_tmp_vars)d" % locals()
        num_tmp_vars += 1

        shape = self.shape
        input_nodes = cg.input_nodes(self)
        assert (len(input_nodes) == shape[1])

        for cidx, n in enumerate(input_nodes):
            if cidx < shape[1]-1:
                terminate = " else "
            else:
                terminate = ""
            idxvars = ["hstack_idx_%d" % i for i in range(num_tmp_vars, num_tmp_vars + len(n.shape))]
            num_tmp_vars += len(n.shape)
            idxdefs = ind2subCode(abs_idx[0], n.shape, idxvars)
            icode, var, num_tmp_vars = n.forward_cuda_kernel(cg, num_tmp_vars, idxvars, self)
            icode = indent(icode, 4)

            code += """\
if( %(nidx)s == %(cidx)d )
{
    %(idxdefs)s
    %(icode)s
    %(res)s = %(var)s;
}%(terminate)s""" % locals()
        return code, res, num_tmp_vars

    def adjoint_cuda_kernel(self, cg, num_tmp_vars, abs_idx, parent):
        #print("vstack:adjoint:cuda")
        input_nodes = cg.input_nodes(self)
        found = False
        #idx = input_nodes.index(parent)
        for idx,n in enumerate(input_nodes):
            while isinstance(n, NodeReverseInOut):
                n = n.n
            if n is parent:
                found = True
                break
        assert(found)

        linidx = sub2ind(abs_idx, n.shape)
        sidx0 = "hstack_idx_%d" % num_tmp_vars
        sidx1 = "hstack_idx_%d" % (num_tmp_vars + 1)
        num_tmp_vars += 1
        icode, var, num_tmp_vars = cg.output_nodes(self)[0].adjoint_cuda_kernel(cg, num_tmp_vars, [sidx0,sidx1], self)

        code = """/* hstack */
int %(sidx0)s = %(linidx)s;
int %(sidx1)s = %(idx)d;

%(icode)s
""" % locals()

        return code, var, num_tmp_vars

    def is_gram_diag(self, freq=False):
        """Is the lin op's Gram matrix diagonal (in the frequency domain)?
        """
        return all([arg.is_gram_diag(freq) for arg in self.input_nodes])

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
        var_diags = {var: np.zeros(var.size) for var in self.variables()}
        for arg in self.input_nodes:
            arg_diags = arg.get_diag(freq)
            for var, diag in arg_diags.items():
                var_diags[var] = var_diags[var] + diag * np.conj(diag)
        # Get (A^TA)^{1/2}
        for var in self.variables():
            var_diags[var] = np.sqrt(var_diags[var])
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
        return np.linalg.norm(input_mags, 2)
