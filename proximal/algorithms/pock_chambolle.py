from __future__ import print_function
import tempfile
import os
from proximal.lin_ops import (CompGraph, est_CompGraph_norm, Variable,
                              vstack, lin_op)
from proximal.utils.timings_log import TimingsLog, TimingsEntry
from proximal.utils.utils import graph_visualize
from proximal.utils.cuda_codegen import NumpyAdapter
from .invert import get_least_squares_inverse, max_diag_set
import numpy as np

class PCUniformConvexGorF:
    """This class can be used to implement algorithm 2 of the Pock Chambolle
    paper for 1/N^2 convergence rate if either Psi or Omega is uniformly convex.

    From the paper: in case G is uniform convex, there exists gamma such that
    for any x in dom subgrad G
        G(x') >= G(x) + <p, x'-x> + gamma/2*||x-x'||^2 for all p in subgrad G(x), x' in X
    """
    def __init__(self, gamma, tau0, L = None, try_fast_norm = False):
        self._lastIt = 0
        self._gamma = gamma
        self._tau = tau0
        self._theta = 1.0 / np.sqrt(1 + gamma*tau0)
        self._L = L
        self._try_fast_norm = try_fast_norm

    def L(self, K):
        if self._L is None:
            self._L = est_CompGraph_norm(K, self._try_fast_norm)
        return self._L

    def _recalculate(self, it, K):
        self._lastIt = it
        self._tau = self._theta*self._tau
        self._theta = 1.0 / np.sqrt(1 + self._gamma*self._tau)

    def tau(self, it, K, adapter):
        if self._lastIt != it:
            self._recalculate(it,K)
        return adapter.scalar(self._tau)

    def sigma(self, it, K, adapter):
        if self._lastIt != it:
            self._recalculate(it,K)
        return adapter.scalar(1.0/((self.L(K)**2)*self._tau))

    def theta(self, it, K, adapter):
        if self._lastIt != it:
            self._recalculate(it,K)
        return adapter.scalar(self._theta)

    def tau_last(self):
        return self._tau

class PCUniformConvexGorF_PPD:
    def __init__(self, gamma, tau0, ppd_alpha = 1.):
        self._lastIt = 0
        self._gamma = gamma
        self._tau = tau0
        self._theta = 1.0 / np.sqrt(1 + gamma*tau0)
        self._vtau = None
        self._vsigma = None
        self._ppd_alpha = ppd_alpha
        self._wrotenote = False

    def ppd(self, K, adapter):
        if self._vtau is None:
            sm = K.sparse_matrix()
            # abs(K)
            # alpha = 1:
            sm = sm.maximum(-sm) # assume sm is real and take absolute value of individual entries
            self._vsigma = adapter.zeros(K.output_size)
            self._vtau = adapter.zeros(K.input_size)
            if self._ppd_alpha == 1:
                self._vsigma[:] = adapter.scalar(1.)/np.maximum(sm.dot(np.ones(sm.shape[1], dtype=np.float32)),adapter.scalar(1e-5))
                self._vtau[:] = adapter.scalar(1.)/np.maximum(sm.T.dot(np.ones(sm.shape[0], dtype=np.float32)),adapter.scalar(1e-5))
            else:
                self._vsigma[:] = adapter.scalar(1.)/np.maximum(sm.power(self._ppd_alpha).dot(np.ones(sm.shape[1], dtype=np.float32)),adapter.scalar(1e-5))
                self._vtau[:] = adapter.scalar(1.)/np.maximum(sm.power(2-self._ppd_alpha).T.dot(np.ones(sm.shape[0], dtype=np.float32)),adapter.scalar(1e-5))
        return self._vsigma, self._vtau

    def _recalculate(self, it, K):
        self._lastIt = it
        if self._tau > 1e-3:
            self._tau = self._theta*self._tau
            self._theta = 1.0 / np.sqrt(1 + self._gamma*self._tau)
        else:
            if not self._wrotenote:
                print("it=%d: Stopping with acceleration." % it)
            self._wrotenote = True

    def tau(self, it, K, adapter):
        if self._lastIt != it:
            self._recalculate(it, K)
        vsigma, vtau = self.ppd(K, adapter)
        return adapter.scalar(self._tau)*vtau

    def sigma(self, it, K, adapter):
        if self._lastIt != it:
            self._recalculate(it, K)
        vsigma, vtau = self.ppd(K, adapter)
        return adapter.scalar(1./self._tau)*vsigma

    def theta(self, it, K, adapter):
        if self._lastIt != it:
            self._recalculate(it, K)
        return adapter.scalar(self._theta)

    def tau_last(self):
        return self._tau

def partition(prox_fns, try_diagonalize=True):
    """Divide the proxable functions into sets Psi and Omega.
    """
    # Omega must be a single function.
    # Merge quadratic functions into the x update.
    # Automatically try to split the problem.
    quad_fns = max_diag_set(prox_fns)
    split_fn = []
    omega_fns = []
    if len(quad_fns) == 0:
        for fn in prox_fns:
            if type(fn.lin_op) == Variable:
                split_fn = [fn]
                break
        omega_fns = split_fn
    else:
        # Proximal solve for:
        # G(x) + 1/(2*tau) * ||x - v||^2_2, with G containing all quadratics
        quad_ops = []
        const_terms = []
        for fn in quad_fns:
            fn = fn.absorb_params()
            quad_ops.append(fn.beta * fn.lin_op)
            const_terms.append(fn.b.flatten())

        b = np.hstack(const_terms)
        # Get optimize inverse (tries spatial and frequency diagonalization)
        x_update = get_least_squares_inverse(quad_ops, b, try_diagonalize)
        omega_fns = [x_update]

    psi_fns = [func for func in prox_fns if func not in split_fn + quad_fns]
    return psi_fns, omega_fns

dsp_cnt = 0
def display_matrix(M):
    import pickle
    global dsp_cnt
    f = "/tmp/matrix%d.pickle" % dsp_cnt
    pickle.dump(M, open(f, "wb"))
    dsp_cnt = (dsp_cnt + 1) % 30
    # display the corners of M and a point in the middle
    print(f)
    idx = [0] * len(M.shape)
    while 1:
        print(("%3d "*len(idx)) % tuple(idx), "-> %+02.03e" % M.item(*idx))
        ok = False
        i = -1
        while not ok:
            if idx[i] == 0 and M.shape[i] > 1:
                ok = True
                idx[i] = M.shape[i]-1
            else:
                idx[i] = 0
            i = i-1
            if -i > len(idx):
                break
        if not ok:
            break
    idx = [s//2 for s in M.shape]
    print(("%3d "*len(idx)) % tuple(idx), "-> %+02.03e" % M.item(*idx))

def solve(psi_fns, omega_fns, tau=None, sigma=None, theta=None,
          max_iters=1000, eps_abs=1e-3, eps_rel=1e-3, x0=None,
          lin_solver="cg", lin_solver_options=None, conv_check=100,
          try_diagonalize=True, try_fast_norm=False, scaled=True,
          metric=None, convlog=None, verbose=0, callback=None, adapter = NumpyAdapter(), ppd=False, cuda_code_file=None):
    # psi_fns corresponds with F(Kx) from the paper
    # omega_fns corresponds with G(x) from the paper

    # Can only have one omega function.
    assert len(omega_fns) <= 1
    prox_fns = psi_fns + omega_fns
    stacked_ops = vstack([fn.lin_op for fn in psi_fns])
    K = CompGraph(stacked_ops)

    #graph_visualize(prox_fns)

    if adapter.implem() == 'numpy':
        K_forward = K.forward
        K_adjoint = K.adjoint
        prox_off_and_fac = lambda offset, factor, fn, *args, **kw: offset + factor*fn.prox(*args, **kw)
        prox = lambda fn, *args, **kw: fn.prox(*args, **kw)

    elif adapter.implem() == 'pycuda':
        K_forward = K.forward_cuda
        K_adjoint = K.adjoint_cuda
        prox_off_and_fac = lambda offset, factor, fn, *args, **kw: fn.prox_cuda(adapter, *args, **kw, offset=offset, factor=factor)
        prox = lambda fn, *args, **kw: fn.prox_cuda(adapter, *args, **kw)

        if not cuda_code_file is None:
            K.gen_cuda_code()
            open(cuda_code_file, "w").write("/* CUDA FORWARD IMPLEMENTAIOTN */\n" + K.cuda_forward_subgraphs.cuda_code + "\n/*CUDA ADJOINT IMPLEMENTAIOTN */" + K.cuda_adjoint_subgraphs.cuda_code)

    else:
        raise RuntimeError("Implementation %s unknown" % adapter.implem())

    z_eq_a_plus_b_times_c = lambda z,a,b,c: adapter.elem_wise_wrapper("z = a+b*c", (("a",a),("b",b),("c",c),("z",z)))
    z_eq_m_a = lambda z,a: adapter.elem_wise_wrapper("z = -a", (("a",a), ("z",z)))
    z_eq_a_div_b = lambda z,a,b: adapter.elem_wise_wrapper("z = a/b", (("a",a),("b",b),("z",z)))
    z_eq_z_minus_a_times_b = lambda z,a,b: adapter.elem_wise_wrapper("z = z - a*b", (("a",a),("b",b),("z",z)))
    z_eq_z_plus_a_times_b_minus_c = lambda z,a,b,c: adapter.elem_wise_wrapper("z = z + a*(b-c)", (("a",a),("b",b),("c",c),("z",z)))

    # Select optimal parameters if wanted
    if tau is None or sigma is None or theta is None:
        if type(ppd) is tuple:
            ppd_alpha = ppd[1]
            ppd = ppd[0]
        else:
            ppd_alpha = 1
        if not ppd:
            tau, sigma, theta = est_params_pc(K, tau, sigma, verbose, scaled, try_fast_norm)
        else:
            sm = K.sparse_matrix()
            # abs(K)
            # alpha = 1:
            sm = sm.maximum(-sm) # assume sm is real and take absolute value of individual entries
            sigma = adapter.zeros(K.output_size)
            tau = adapter.zeros(K.input_size)
            if ppd_alpha == 1:
                sigma[:] = adapter.scalar(1.)/np.maximum(sm.dot(np.ones(sm.shape[1], dtype=np.float32)),adapter.scalar(1e-5))
                tau[:] = adapter.scalar(1.)/np.maximum(sm.T.dot(np.ones(sm.shape[0], dtype=np.float32)),adapter.scalar(1e-5))
            else:
                sigma[:] = adapter.scalar(1.)/np.maximum(sm.power(ppd_alpha).dot(np.ones(sm.shape[1], dtype=np.float32)),adapter.scalar(1e-5))
                tau[:] = adapter.scalar(1.)/np.maximum(sm.power(2-ppd_alpha).T.dot(np.ones(sm.shape[0], dtype=np.float32)),adapter.scalar(1e-5))
            theta = 1

            #import matplotlib.pyplot as plt
            #plt.plot(tau); plt.show()
            #plt.plot(sigma); plt.show()
            #xxx
    elif callable(tau) or callable(sigma) or callable(theta):
        pass
        #if scaled:
        #    L = 1
        #else:
        #    L = est_CompGraph_norm(K, try_fast_norm)

    if len(omega_fns) >= 1:
        assert len(omega_fns) == 1
        if len(omega_fns[0].lin_op.variables()) == 1:
            v = omega_fns[0].lin_op.variables()[0]
            offset = K.var_info[v.uuid]
            omega_slc = slice(offset, offset + omega_fns[0].lin_op.size, None)
        else:
            omega_slc = slice(0, K.output_size, None)

    # Initialize
    x = adapter.zeros(K.input_size)
    y = adapter.zeros(K.output_size)
    xbar = adapter.zeros(K.input_size)
    u = adapter.zeros(K.output_size)
    z = adapter.zeros(K.output_size)

    if x0 is not None:
        x[:] = adapter.reshape(adapter.from_np(x0), K.input_size)
    else:
        x[:] = adapter.from_np(K.x0())

    K_forward(x, y)
    xbar[:] = x

    # Buffers.
    Kxbar = adapter.zeros(K.output_size)
    Kx = adapter.zeros(K.output_size)
    KTy = adapter.zeros(K.input_size)
    KTu = adapter.zeros(K.input_size)
    s = adapter.zeros(K.input_size)

    prev_x = x.copy()
    prev_Kx = Kx.copy()
    prev_z = z.copy()
    prev_u = u.copy()

    # Log for prox ops.
    prox_log = TimingsLog(prox_fns)
    prox_log_tot = TimingsLog(prox_fns)
    # Time iterations.
    iter_timing = TimingsLog(["pc_iteration_tot",
                              "copyprev",
                              "calcz",
                              "calcx",
                              "omega_fn",
                              "xbar",
                              "conv_check"])

    # Convergence log for initial iterate
    if convlog is not None or conv_check > 0 or callback is not None:
        K.update_vars(adapter.to_np(x))
        objval = []
        for f in prox_fns:
            evp = f.value
            #print(str(f), '->', f.value)
            objval += [evp]
        if verbose > 0:
            print("obj_val = %02.03e [%s]" % (sum(objval), " ".join(["%02.03e" % v for v in objval])))
        if convlog is not None:
            convlog.record_objective(sum(objval))
            convlog.record_timing(0.0)
        #if callback is not None:
        #    callback(adapter.to_np(x), 0)

    for i in range(max_iters):
        iter_timing["pc_iteration_tot"].tic()
        if convlog is not None:
            convlog.tic()

        if callable(sigma):
            csigma = sigma(i, K, adapter)
        else:
            csigma = sigma
        if callable(tau):
            ctau = tau(i, K, adapter)
        else:
            ctau = tau
        if callable(theta):
            ctheta = theta(i, K, adapter)
        else:
            ctheta = theta

        if not type(csigma) is adapter.arraytype(): csigma = adapter.scalar(csigma)
        if not type(ctau)   is adapter.arraytype(): ctau   = adapter.scalar(ctau)
        if not type(ctheta) is adapter.arraytype(): ctheta = adapter.scalar(ctheta)

        # Keep track of previous iterates
        iter_timing["copyprev"].tic()
        adapter.copyto(prev_x, x)
        adapter.copyto(prev_z, z)
        adapter.copyto(prev_u, u)
        adapter.copyto(prev_Kx, Kx)
        iter_timing["copyprev"].toc()

        # Compute z
        iter_timing["calcz"].tic()
        K_forward(xbar, Kxbar)
        #z = y + csigma * Kxbar
        z_eq_a_plus_b_times_c(z,y,csigma,Kxbar)
        iter_timing["calcz"].toc()

        # Update y.
        if i == 0:
            z_div_csigma = adapter.zeros(z.shape)
        if type(csigma) is adapter.arraytype():
            if i == 0:
                m_csigma = adapter.zeros(csigma.shape)
            z_eq_m_a(m_csigma, csigma)
        else:
            m_csigma = -csigma

        z_eq_a_div_b(z_div_csigma, z, csigma)

        offset = 0
        for fn in psi_fns:
            prox_log_tot[fn].tic()
            slc = slice(offset, offset + fn.lin_op.size, None)
            z_slc = adapter.reshape(z[slc], fn.lin_op.shape)
            if ppd:
                csigma_slc = adapter.reshape(csigma[slc], fn.lin_op.shape)
                m_csigma_slc = adapter.reshape(m_csigma[slc], fn.lin_op.shape)
            else:
                csigma_slc = csigma
                m_csigma_slc = -csigma
            z_div_csigma_slc = adapter.reshape(z_div_csigma[slc], fn.lin_op.shape)

            # Moreau identity: apply and time prox.
            prox_log[fn].tic()
            y[slc] = adapter.flatten( prox_off_and_fac(z_slc, m_csigma_slc, fn, csigma_slc, z_div_csigma_slc, i) )
            prox_log[fn].toc()
            offset += fn.lin_op.size
            prox_log_tot[fn].toc()

        iter_timing["calcx"].tic()
        if offset < y.shape[0]:
            y[offset:] = 0
        # Update x
        K_adjoint(y, KTy)
        #x -= ctau * KTy
        z_eq_z_minus_a_times_b(x, ctau, KTy)
        iter_timing["calcx"].toc()

        iter_timing["omega_fn"].tic()
        if len(omega_fns) > 0:
            fn = omega_fns[0]
            prox_log_tot[fn].tic()
            xtmp = adapter.reshape(x[omega_slc], fn.lin_op.shape)
            prox_log[fn].tic()
            if ppd:
                one_div_ctau = adapter.reshape(adapter.scalar(1.0) / ctau[omega_slc], fn.lin_op.shape)
            else:
                one_div_ctau = adapter.scalar(1.0) / ctau
            x[omega_slc] = adapter.flatten( prox(fn, one_div_ctau, xtmp, x_init=prev_x,
                                     lin_solver=lin_solver, options=lin_solver_options) )
            prox_log[fn].toc()
            prox_log_tot[fn].toc()
        iter_timing["omega_fn"].toc()

        iter_timing["xbar"].tic()
        # Update xbar
        adapter.copyto(xbar, x)
        z_eq_z_plus_a_times_b_minus_c(xbar, ctheta, x, prev_x)
        #xbar += ctheta * (x - prev_x)
        iter_timing["xbar"].toc()

        # Convergence log
        if convlog is not None:
            convlog.toc()
            if adapter.implem() == 'pycuda':
                K_forward(x, Kx)
                iobjval2 = []
                offset = 0
                for fn in psi_fns:
                    slc = slice(offset, offset + fn.lin_op.size, None)
                    Kx_slc = adapter.reshape(Kx[slc], fn.lin_op.shape)
                    iobjval2.append(fn.eval_cuda(Kx_slc))
                    offset += fn.lin_op.size
                for fn in omega_fns:
                    xtmp = adapter.reshape(x, fn.lin_op.shape)
                    iobjval2.append(fn.eval_cuda(xtmp))
                objval = sum(iobjval2)
            else:
                K.update_vars(adapter.to_np(x))
                iobjval = list([fn.value for fn in prox_fns])
                objval = sum(iobjval)
            convlog.record_objective(objval)

        """ Old convergence check
        #Very basic convergence check.ŝ
        r_x = np.linalg.norm(x - prev_x)
        r_xbar = np.linalg.norm(xbar - prev_xbar)
        r_ybar = np.linalg.norm(y - prev_y)
        error = r_x + r_xbar + r_ybar
        """

        # Residual based convergence check
        if i % conv_check in [0, conv_check-1]:
            iter_timing["conv_check"].tic()
            K_forward(x, Kx)
            u = adapter.scalar(1.0) / csigma * y + ctheta * (Kx - prev_Kx)
            z = prev_u + prev_Kx - adapter.scalar(1.0) / csigma * y
            iter_timing["conv_check"].toc()

        # Iteration order is different than
        # lin-admm (--> start checking at iteration 1)
        if i > 0 and i % conv_check == 0:

            # Check convergence
            r = prev_Kx - z
            K_adjoint(csigma * (z - prev_z), s)
            eps_pri = np.sqrt(K.output_size) * eps_abs + eps_rel * \
                max([np.linalg.norm(adapter.to_np(prev_Kx)), np.linalg.norm(adapter.to_np(z))])

            K_adjoint(u, KTu)
            if hasattr(csigma, "shape"):
                eps_dual = np.sqrt(K.input_size) * eps_abs + eps_rel * np.linalg.norm(adapter.to_np(KTu/ctau))
            else:
                eps_dual = np.sqrt(K.input_size) * eps_abs + eps_rel * np.linalg.norm(adapter.to_np(KTu)) / csigma

            if not callback is None or verbose == 2:
                K.update_vars(adapter.to_np(x))
            if not callback is None:
                callback(adapter.to_np(x), i)

            # Progress
            if verbose > 0:
                # Evaluate objective only if required (expensive !)
                objstr = ''
                if verbose == 2:
                    ov = list([fn.value for fn in prox_fns])
                    objval = sum(ov)
                    objstr = ", obj_val = %02.03e [%s] " % (objval, ", ".join("%02.03e" % x for x in ov))

                """ Old convergence check
                #Evaluate metric potentially
                metstr = '' if metric is None else ", {}".format( metric.message(x.copy()) )
                print "iter [%04d]:" \
                      "||x - x_prev||_2 = %02.02e " \
                      "||xbar - xbar_prev||_2 = %02.02e " \
                      "||y - y_prev||_2 = %02.02e " \
                      "SUM = %02.02e (eps=%02.03e)%s%s" \
                        % (i, r_x, r_xbar, r_ybar, error, eps, objstr, metstr)
                """

                # Evaluate metric potentially
                metstr = '' if metric is None else ", {}".format(metric.message(v))
                print(
                    "iter %d: ||r||_2 = %.3f, eps_pri = %.3f, ||s||_2 = %.3f, eps_dual = %.3f%s%s"
                    % (i, np.linalg.norm(adapter.to_np(r)), eps_pri, np.linalg.norm(adapter.to_np(s)), eps_dual, objstr, metstr)
                )

            iter_timing["pc_iteration_tot"].toc()
            if np.linalg.norm(adapter.to_np(r)) <= eps_pri and np.linalg.norm(adapter.to_np(s)) <= eps_dual:
                break

        else:
            iter_timing["pc_iteration_tot"].toc()

        """ Old convergence check
        if error <= eps:
            break
        """

    # Print out timings info.
    if verbose > 0:
        print(iter_timing)
        print("prox funcs total:")
        print(prox_log_tot)
        print("prox funcs inner:")
        print(prox_log)
        print("K forward ops:")
        print(K.forward_log)
        print("K adjoint ops:")
        print(K.adjoint_log)

    # Assign values to variables.
    K.update_vars(adapter.to_np(x))
    if not callback is None:
        callback(adapter.to_np(x))
    # Return optimal value.
    return sum([fn.value for fn in prox_fns])


def est_params_pc(K, tau=None, sigma=None, verbose=True, scaled=False, try_fast_norm=False):

    # Select theta constant and sigma larger 0
    theta = 1.0
    sigma = 1.0 if sigma is None else sigma

    # Estimate Lipschitz constant and comput tau
    if scaled:
        L = 1
    else:
        L = est_CompGraph_norm(K, try_fast_norm)
    tau = 1.0 / (sigma * L**2)

    if verbose:
        print("Estimated params [sigma = %3.3f | tau = %3.3f | theta = %3.3f | L_est = %3.4f]"
              % (sigma, tau, theta, L))

    return tau, sigma, theta
