import re
import numpy as np
import logging

try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    from pycuda.compiler import SourceModule, DEFAULT_NVCC_FLAGS
    from pycuda import gpuarray
    from pycuda.elementwise import ElementwiseKernel

    cuda_available = True

except ImportError:
    cuda_available = False
    class gpuarray:
        pass
except Exception as e:
    import traceback
    print("Warning: caught exception; continuing without pycuda.")
    traceback.print_exc()
    cuda_available = False
    class gpuarray:
        pass

def compile_cuda_kernel(cuda_kernel_code):
    """
    compiles a cuda kernel and return compiled module
    """
    try:
        cuda_code = cuda_kernel_code if 1 else replace_local_floats_with_double(cuda_kernel_code)
        logging.debug("Compiling cuda code:\n" + cuda_code)
        mod = SourceModule(cuda_code, options=DEFAULT_NVCC_FLAGS + ['--use_fast_math'])
    except cuda.CompileError as e:
        logging.error(cuda_code)
        logging.error("CUDA compilation error:")
        logging.error(e.stderr)
        raise e
    return mod

class CudaFunc:
    def __init__(self, func, block, grid, additional_arguments, function_name):
        self.func = func
        self.block = block
        self.grid = grid
        self.additional_arguments = additional_arguments
        self.function_name = function_name

    def __call__(self, *args):
        #for a in args + self.additional_arguments:
        #    if not type(a) is pycuda.gpuarray.GPUArray:
        #        print(self.func, "using non gpu array input")
        t = self.func(*(args+self.additional_arguments), grid=self.grid, block=self.block, time_kernel=True)
        logging.debug("Cuda function %s execution time: %.2f ms", self.function_name, t*1000)
        return t

def cuda_function(mod, function_name, datadim, additional_arguments = ()):
    """
    Returns a callable for function <function_name> from the compile cuda kernel <mod>
    setting up block and grid sizes for a 1D data block with datadim elements.

    The callable's signature matches the cuda kernel signature (additional_arguments are appended).
    """
    cuda_func = mod.get_function(function_name)
    block = (min(datadim, cuda_func.MAX_THREADS_PER_BLOCK), 1, 1)
    grid = (datadim//block[0],1,1)
    #print("cuda function: %s, block=%d, grid=%d, MAX=%d" % (function_name, block[0], grid[0], cuda_func.MAX_THREADS_PER_BLOCK))
    return CudaFunc(cuda_func, block, grid, additional_arguments, function_name)
    #result = lambda *args: logger(cuda_func(*(args+additional_arguments), grid=grid, block=block, time_kernel=True))
    #return result

class NumpyAdapter:
    """
    To keep algorithms free of cuda clutter, we use these adapter classes to map
    calls to either numpy or gpuarray.
    """
    def __init__(self, floattype = np.float64):
        self.floattype = floattype
        self.elem_wise_expressions = {}

    def zeros(self, *args, **kw):
        kw['dtype'] = self.floattype
        return np.zeros(*args, **kw)

    def to_np(self, matrix):
        return matrix

    def from_np(self, matrix):
        return matrix.astype(self.floattype)

    def scalar(self, s):
        return self.floattype(s)

    def reshape(self, *args):
        return np.reshape(*args)

    def flatten(self, matrix):
        return matrix.flatten()

    def copyto(self, tgt, src):
        np.copyto(tgt, src)

    def sum(self, matrix):
        return np.sum(matrix)

    def abs(self, matrix):
        return np.abs(matrix)

    def elem_wise_wrapper(self, expression, variables):
        """
        expression must be a string containing a statement like "a = b*c+(-c/s)"
        """
        vtypes = tuple([type(vval) for (vname, vval) in variables])
        if not (expression, vtypes) in self.elem_wise_expressions:
            sexp = expression.split("=")
            assert len(sexp) == 2
            new_exp = "np.copyto(" + sexp[0] + ", " + sexp[1] + ")"
            self.elem_wise_expressions[(expression, vtypes)] = new_exp
        exec(self.elem_wise_expressions[(expression, vtypes)], globals(), dict(variables))

    def arraytype(self):
        return np.ndarray

    def implem(self):
        return 'numpy'

class PyCudaAdapter:
    """
    To keep algorithms free of cuda clutter, we use these adapter classes to map
    calls to either numpy or gpuarray.
    """
    def __init__(self, floattype = np.float32):
        self.floattype = floattype
        self.elem_wise_expressions = {}

    def zeros(self, *args, **kw):
        kw['dtype'] = self.floattype
        return gpuarray.zeros(*args, **kw)

    def to_np(self, matrix):
        return matrix.get()

    def from_np(self, matrix):
        return gpuarray.to_gpu(matrix.astype(self.floattype))

    def scalar(self, s):
        return self.floattype(s)

    def reshape(self, *args):
        return gpuarray.reshape(*args)

    def flatten(self, matrix):
        return gpuarray.reshape(matrix, int(np.prod(matrix.shape)))

    def copyto(self, tgt, src):
        tgt[:] = src

    def sum(self, matrix):
        return gpuarray.sum(matrix)

    def abs(self, matrix):
        return matrix.__abs__()

    def elem_wise_wrapper(self, expression, variables):
        """
        expression must be a string containing a statement like "a = b*c+(-c/s)"
        """
        vtypes = tuple([type(vval) is self.floattype for (vname, vval) in variables])
        if not (expression, vtypes) in self.elem_wise_expressions:
            for _,v in variables:
                if hasattr(v, "dtype"):
                    assert v.dtype == self.floattype
            cexp = expression
            args = []
            for vname, vval in variables:
                if type(vval) is self.floattype:
                    args.append("float " + vname)
                else:
                    args.append("float *" + vname)
                    cexp = re.sub(r"\b%s\b" % vname, vname + "[i]", cexp)
            args = ", ".join(args)
            #print(expression, vtypes, args, cexp)
            self.elem_wise_expressions[(expression, vtypes)] = ElementwiseKernel(
                    args,
                    cexp,
                    "elem_wise_wrapper"
                    )
        self.elem_wise_expressions[(expression, vtypes)](*(vval for (_, vval) in variables))

    def arraytype(self):
        return gpuarray.GPUArray

    def implem(self):
        return 'pycuda'

def float_constant(v):
    """
    return a floating point constant for usage in cuda kernels (a string).
    """
    # TODO maybe use hexadecimal floating point constants here
    return "%.10ef" % v

def indent(code, level):
    """
    indent code to the given level
    """
    return code.replace("\n", "\n" + (" "*level))

def sub2ind(idx, shape):
    """
    Calculates a linear index from multuple sub indices (like the matlab function).
    Intended to be used in cuda kernel generators.
    """
    assert(len(idx) == len(shape))
    if all([not isinstance(i, str) for i in idx]):
        res = 0
        for d,i in enumerate(idx):
            res += i*np.prod(shape[(d+1):])
        return res
    else:

        res = "(" + ("+".join(["(%s)*%d" % (i, np.prod(shape[(d+1):])) for d,i in enumerate(idx)])) + ")"
    return res

def ind2sub(ind, shape):
    """
    Calculates a sub indices from a linear index (like the matlab function).
    Intended to be used in cuda kernel generators.
    """
    if not isinstance(ind, str):
        res = []
        for d in range(len(shape)):
            res.append( (ind % int(np.prod(shape[d:])) // int(np.prod(shape[d+1:]))) )
    else:
        res = []
        for d in range(len(shape)):
            exp1 = ""
            exp2 = ""
            if d > 0:
                exp1 = " %% %d" % int(np.prod(shape[d:]))
            if d < len(shape)-1:
                exp2 = "/ %d" % int(np.prod(shape[(d+1):]))
            res.append( "((%(ind)s)%(exp1)s)%(exp2)s" % locals() )
    return res

def _magic_division_numbers(d):
    with np.errstate(over='ignore'):
        # see https://github.com/milakov/int_fastdiv/blob/master/int_fastdiv.h
        # (M,s,n_add_sign)
        assert(d > 0)

        if d == 1: return (0, -1,  1)
        if d == -1: return (0, -1, -1)
        two31 = np.uint32(0x80000000)
        ad = np.uint32(1 if d == 0 else abs(d))
        t = np.uint32(two31 + (np.uint32(d) >> 31))
        anc = np.uint32(t - 1 - t % ad)
        p = np.int32(31)
        q1 = np.uint32(two31 // anc)
        r1 = np.uint32(two31 - q1*anc)
        q2 = np.uint32(two31 // ad)
        r2 = np.uint32(two31 - q2 * ad)
        while 1:
            p += np.int32(1)
            q1 = np.uint32(q1*2)
            r1 = np.uint32(r1*2)
            if (r1 >= anc):
                q1 = np.uint32(q1+1)
                r1 = np.uint32(r1-anc)
            q2 = np.uint32(q2*2)
            r2 = np.uint32(r2*2)
            if (r2 >= ad):
                q2 = np.uint32(q2+1)
                r2 = np.uint32(r2-ad)
            delta = np.int32(ad - r2)
            if not (q1 < delta or (q1 == delta and r1 == 0)):
                break
        M = np.int32(q2 + 1)
        if (d < 0):
            M = np.int32(-M)
        s = np.int32(p - 32)

        if ((d > 0) and (M < 0)):
            n_add_sign = np.int32(1)
        elif ((d < 0) and (M > 0)):
            n_add_sign = np.int32(-1)
        else:
            n_add_sign = np.int32(0)
        return (M,s,n_add_sign)

def ind2subCode(ind, shape, target_var_names):
    """
    Same as ind2sub, but it will be more efficient by reusing results.
    Pass the variable names to be defined and initialized in target_var_names.
    """
    code = ""
    av = target_var_names[-1]
    code += "int %(av)s = 0;\n" % locals()
    for d in range(len(shape)):
        v = target_var_names[d]
        divisor = int(np.prod(shape[(d+1):]))
        if d != len(shape) - 1:
            code += "int "
        if 1 or divisor in [1 << k for k in range(1,31)]:
            code += "%(v)s = ( (%(ind)s) + %(av)s ) / %(divisor)d;\n" % locals()
        elif divisor == 1:
            code += "%(v)s = ( (%(ind)s) + %(av)s );\n" % locals()
        elif divisor > 1:
            M,s,n_add_sign = _magic_division_numbers(divisor)

            code += """\
%(v)s = (%(ind)s) + %(av)s; /* optimized integer division by %(divisor)d */
asm("mul.hi.s32 %%0, %%1, %%2;" : "=r"(%(v)s) :  "r"(%(M)d), "r"(%(v)s));
%(v)s += ((%(ind)s) + %(av)s) * %(n_add_sign)d;
""" % locals()
            if s > 0:
                if s < 32:
                    code += "%(v)s >>= %(s)d;" % locals()
                else:
                    code += "%(v)s = (%(v)s >= 0) ? 0 : -1;\n" % locals()
                code += "%(v)s += (((unsigned int)%(v)s) >> 31);\n" % locals()

        if d != len(shape) - 1:
            code += "%(av)s -= %(v)s*%(divisor)d;\n" % locals()

        code += "(void)%(v)s;\n" % locals()
    return code

class NodeReverseInOut(object):
    """
    When generating cuda kernels, the graphs are traversed implicitely.
    Sometimes it is necessary to "reverse" the node such that the forward
    operation gets the adjoint and vice versa. This class simulates a single
    reversed node.
    """
    def __init__(self, n, parent):
        self.n = n
        self.parent = parent

    def adjoint_cuda_kernel(self, *args, **kw):
        args = (a if not a is self.parent else self.parent.o for a in args)
        return self.n.forward_cuda_kernel(*args, **kw)

    def forward_cuda_kernel(self, *args,**kw):
        args = (a if not a is self.parent else self.parent.o for a in args)
        return self.n.adjoint_cuda_kernel(*args, **kw)

    def cuda_kernel_available(self):
        return self.n.cuda_kernel_available()

    @property
    def size(self):
        return self.n.size

class ReverseInOut(object):
    """
    When generating cuda kernels, the graphs are traversed implicitely.
    Sometimes it is necessary to "reverse" the node such that the forward
    operation gets the adjoint and vice versa. This is the helper class for
    the operation.
    """
    def __init__(self, o, reverseNodes = True):
        self.o = o
        self.reverseNodes = reverseNodes
        self.in_nodes = {}
        self.out_nodes = {}

    def input_nodes(self, n):
        if self.reverseNodes:
            if not n in self.in_nodes:
                self.in_nodes[n] = list([NodeReverseInOut(x, self) for x in self.o.output_nodes(n)])
            return self.in_nodes[n]
        else:
            return self.o.output_nodes(n)

    def output_nodes(self, n):
        if self.reverseNodes:
            if not n in self.out_nodes:
                self.out_nodes[n] = list([NodeReverseInOut(x, self) for x in self.o.input_nodes(n)])
            return self.out_nodes[n]
        else:
            return self.o.input_nodes(n)

def replace_local_floats_with_double(src):
    """
    This function replaces all internal float variables and constants with doubles,
    but keeps the pointer types.
    """
    Rd = re.compile(r"\bfloat\b(?! *[*])")
    Rc = re.compile(r"\b(-?(0(\.\d*)?|([1-9]\d*\.?\d*)|(\.\d+))([Ee][+-]?\d+)?)f\b")

    return Rc.subn(r"\1", Rd.subn("double",src)[0])[0]

class ProxyNode:
    """
    This class is used as a dummy node when the computation graph is split up
    into multiple parts.
    """
    def __init__(self, n, argname, shape):
        self.n = n
        self.argname = argname
        self.shape = shape

    def adjoint_cuda_kernel(self, cg, num_tmp_vars, idx, parent):
        var = "var_%d" % num_tmp_vars
        num_tmp_vars += 1
        argname = self.argname
        linidx = sub2ind(idx, self.shape)
        code = "float %(var)s = %(argname)s[%(linidx)s];\n" % locals()
        return code, var, num_tmp_vars

    def forward_cuda_kernel(self, *args):
        return self.adjoint_cuda_kernel(*args)

    @property
    def size(self):
        return self.n.size

class CudaSubGraph:
    """
    This class splits a linear operation graph into multiple computable subgraphs.
    The constructor is intended to be called with the end node of a comp graph.
    The comp graph is split in pieces with single kernels, and the pieces
    are recursively stored in the "dependent_subgraphs" member.
    """
    instance_cnt = 0

    def __init__(self, get_input_nodes, get_output_nodes, endnode):
        assert endnode.cuda_kernel_available()
        self.subgraph_id = CudaSubGraph.instance_cnt
        CudaSubGraph.instance_cnt += 1
        self.end = endnode
        self._input_nodes = {}
        self._output_nodes = {}
        self.kernel_nodes = []
        self.nokernel_nodes = [] # where the graph must be splitted
        self.nokernel_results = {}
        self.nokernel_inputs = {}
        self.nokernel_proxynodes = {}
        active_nodes = [self.end]
        visited_nodes = {}
        nokernel_innodes = []
        from ..lin_ops.sum import copy
        from ..lin_ops.vstack import vstack
        while len(active_nodes) > 0:
            n = active_nodes.pop(0)
            if n in visited_nodes:
                continue
            visited_nodes[n] = True
            try:
                if not n in self._output_nodes:
                    self._output_nodes[n] = get_output_nodes(n)
            except KeyError:
                pass
            if n.cuda_kernel_available():
                self.kernel_nodes.append(n)
                try:
                    innodes = get_input_nodes(n)
                    # avoid situations where the same node instance is referenced
                    # multiple times in innodes. This situation is solved by
                    # inserting copy nodes
                    new_innodes = []
                    replacements = {}
                    for innidx, inn in enumerate(innodes):
                        if not isinstance(n, vstack) and inn in innodes[innidx+1:]:
                            # insert a copy node between inn and n
                            nn = copy(n.shape)
                            if not inn in replacements: replacements[inn] = []
                            replacements[inn].append(nn)
                            new_innodes.append(nn)
                            self._output_nodes[nn] = [n]
                            self._input_nodes[nn] = [inn]
                            active_nodes.append(inn)
                        else:
                            new_innodes.append(inn)
                            active_nodes.append(inn)
                    for inn in replacements:
                        output_nodes = get_output_nodes(inn)
                        new_output_nodes = []
                        idx = 0
                        for onn in output_nodes:
                            if onn is n and idx < len(replacements[inn]):
                                new_output_nodes.append(replacements[inn][idx])
                                idx += 1
                            else:
                                new_output_nodes.append(onn)
                        self._output_nodes[inn] = new_output_nodes
                    self._input_nodes[n] = new_innodes
                except KeyError:
                    pass
            else:
                logging.info("%s: no cuda kernel available", str(n))
                cn = [n]
                while 1:
                    innodes = get_input_nodes(cn[0])
                    assert (len(innodes) == 1)
                    assert (len(get_output_nodes(cn[0])) == 1)
                    if innodes[0].cuda_kernel_available():
                        nokernel_innodes += innodes
                        break
                    cn = [innodes[0]] + cn
                self.nokernel_nodes.append(cn)
        self.dependent_subgraphs = []
        for n in nokernel_innodes:
            dsg = CudaSubGraph(get_input_nodes, get_output_nodes, n)
            self.dependent_subgraphs.append(dsg)

        self.orig_input_nodes = get_input_nodes
        self.orig_output_nodes = get_output_nodes

    def __str__(self):
        res = "CudaSubGraph(\n"
        for n in self.kernel_nodes:
            res += "  %s <- " % repr(n)
            res += ", ".join(["%s" % repr(x) for x in self._input_nodes.get(n, [])])
            res += "\n"
        res += ")"
        return res

    def visualize(self, dot = None):
        import graphviz
        from IPython.display import display
        root = False
        if not dot:
            root = True
            dot = graphviz.Digraph()
        for csg in self.dependent_subgraphs:
            csg.visualize(dot)
        nodes = {}
        visited = {}
        active = [self.end]
        while len(active) > 0:
            n = active.pop(0)
            if not n in nodes:
                nodes[n] = 'N%d' % len(nodes)
                dot.node(nodes[n], str(type(n)))
                try:
                    innodes = self.input_nodes(n)
                    for inn in innodes:
                        active.append(inn)
                except KeyError:
                    pass
        active = [self.end]
        while len(active) > 0:
            n = active.pop(0)
            if not n in visited:
                visited[n] = True
                try:
                    innodes = self.input_nodes(n)
                    for inn in self.input_nodes(n):
                        active.append(inn)
                        dot.edge(nodes[inn], nodes[n])
                except KeyError:
                    pass
        if root:
            display(dot)

    def input_nodes(self, n):
        """
        returns the input nodes of node n using proxy nodes wherever necessary
        """
        innodes = self._input_nodes[n]
        res = []
        for inn in innodes:
            noKernelNode = False
            for cn in self.nokernel_nodes:
                if inn in cn:
                    noKernelNode = True
            if noKernelNode:
                res.append(self.nokernel_proxynodes[inn])
            else:
                res.append(inn)
        #print("innodes(%s) -> %s" % (repr(n), res))
        return res

    def output_nodes(self, n):
        """
        returns the output nodes of node n (this is called seldomly)
        """
        return self._output_nodes[n]

    def gen_code(self, fcn, parent = None, shape = None):
        """
        generates the cuda kernel code. fcn should either be "forward_cuda_kernel" or "adjoint_cuda_kernel"
        """
        self.cuda_args = []
        self.fcn = fcn
        # do we need additional arguments for the kernel except x (=input) and y (=output)
        for n in self.kernel_nodes:
            try:
                buffers = n.cuda_additional_buffers()
            except AttributeError:
                buffers = []
            for aname, aval in buffers:
                if aname in [ca[0] for ca in self.cuda_args]:
                    continue
                if type(aval) == np.ndarray and aval.dtype == np.int32:
                    aval = gpuarray.to_gpu(aval)
                    self.cuda_args.append( (aname, aval, "int") )
                else:
                    aval = gpuarray.to_gpu(aval.astype(np.float32))
                    self.cuda_args.append( (aname, aval, "float") )

        for cn in self.nokernel_nodes:
            for n in cn:
                o = self.orig_output_nodes(n)
                assert(len(o) == 1)
                o = o[0]
                rshape = n.shape if fcn == "forward_cuda_kernel" else o.shape
                o = gpuarray.zeros(rshape, dtype=np.float32)
                self.nokernel_results[n] = o
            n = cn[-1]
            self.cuda_args.append( ("linop_proxy_output_%d" % n.linop_id, o, "float") )
            self.nokernel_proxynodes[n] = ProxyNode(n, self.cuda_args[-1][0], rshape)

        add_args = "".join((", %s *%s" % (x[2], x[0]) for x in self.cuda_args))

        cg = self if fcn == "forward_cuda_kernel" else ReverseInOut(self, reverseNodes=False)
        self.shape = shape if not shape is None else self.end.shape
        # generate the cuda kernel for this subgraph
        cucode, var, num_tmp_vars = getattr(self.end, fcn)(cg, 0, ind2sub("yidx", self.shape), parent)
        cucode = indent(cucode, 8)
        dimy = int(np.prod(self.shape))
        subgraph_id = self.subgraph_id
        code  = """\
__global__ void %(fcn)s_%(subgraph_id)d(const float *x, float *y%(add_args)s)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for( int yidx = index; yidx < %(dimy)d; yidx += stride )
    {
        %(cucode)s
        y[yidx] = %(var)s;
    }
}

""" % locals()

        # generate the cuda kernels for the dependent subgraphs
        for i,dsg in enumerate(self.dependent_subgraphs):
            cn = self.nokernel_nodes[i]
            if fcn == "forward_cuda_kernel":
                parent = self.orig_input_nodes(cn[0])
                assert(len(parent) == 1)
                parent = parent[0]
            else:
                parent = cn[0]
            dsg.gen_code(fcn, cn[0], parent.shape)
            for ni, n in enumerate(cn):
                self.nokernel_inputs[n] = gpuarray.zeros(dsg.shape, dtype=np.float32) if ni == 0 else self.nokernel_results[cn[ni-1]]

        self._cuda_code = code
        self.cuda_mod = compile_cuda_kernel(code)
        arg_vals = tuple(x[1] for x in self.cuda_args)
        self.cuda_kernel_func = cuda_function(self.cuda_mod, "%(fcn)s_%(subgraph_id)d" % locals(), dimy, arg_vals)

    def apply(self, x, y):
        """
        apply the compiled cuda kernels and all dependent subgraphs
        """
        t = 0.0
        for i,dsg in enumerate(self.dependent_subgraphs):
            cn = self.nokernel_nodes[i]
            n = cn[0]
            t += dsg.apply(x, self.nokernel_inputs[n])
        for i,cn in enumerate(self.nokernel_nodes):
            if self.fcn == "forward_cuda_kernel":
                for n in cn:
                    n.forward_cuda([self.nokernel_inputs[n]], [self.nokernel_results[n]])
            else:
                for n in cn:
                    n.adjoint_cuda([self.nokernel_inputs[n]], [self.nokernel_results[n]])
        t += self.cuda_kernel_func(x, y)
        self.output = gpuarray.reshape(y, self.shape)
        return t

    @property
    def cuda_code(self):
        """
        return the cuda code for this and the dependent kernels
        """
        return self._cuda_code + "\n".join([x.cuda_code for x in self.dependent_subgraphs])

if __name__ == "__main__":
    def div(n, M, s, n_add_sign):
        v = np.int32(np.uint32(  (np.int64(n)*np.int64(M)) >> 32) )
        v += np.int32(n) * np.int32(n_add_sign)
        if s >= 0:
            if s < 32:
                v >>= s
            else:
                v = 0 if v >= 0 else -1
            v += np.int32(np.uint32(v) >> 31)
        return v

    for d in range(3,100):
        M,s,n_add_sign = _magic_division_numbers(d)
        for n in range(2,10000):
            r = div(n, M, s, n_add_sign)
            print(n, "/", d, "=", r)
            assert( r == n // d )

