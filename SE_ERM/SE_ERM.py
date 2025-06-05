import numpy as np
import numpy.random as rnd
import numpy.linalg as linalg
import time
import argparse
from scipy.linalg import sqrtm
from scipy.optimize import minimize
from scipy.special import logsumexp
from numba import njit
from mpi4py import MPI

# Functions for the State Evolution loop functions

def Softplus(x):
    return logsumexp([0, x])

def DSoftplus(x):
    return np.exp(-logsumexp([0, -x]))

def DDSoftplus(x):
    return np.exp(-x -2*logsumexp([0, -x]))

def Loss(y, z):
    return ((y - z[0])*(y - z[0])/(2*Softplus(z[1])) + np.log(Softplus(z[1])))/2

def DDzLoss(y, z):
    Sz = Softplus(z[1])
    Dsz = DSoftplus(z[1])
    return np.array([[1/Sz, (y - z[0])*Dsz/(Sz*Sz)],[(y - z[0])*Dsz/(Sz*Sz), (Sz*DDSoftplus(z[1])*(Sz - (y - z[0])*(y - z[0])) + Dsz*Dsz*(2*(y - z[0])*(y - z[0]) - Sz))/(2*Sz*Sz*Sz)]])

def Prox(mu, Omega, f):
    ToOptimize = lambda x: np.einsum("i,ij,j->",(x - mu),linalg.inv(Omega), (x - mu))/2 + f(x)
    Prox = minimize(ToOptimize, x0=[0, 0])
    return Prox.x

def T(mhat, qhat, chihat, W, xi):
    sqrtqhat = sqrtm(qhat).real
    return linalg.inv(chihat) @ (mhat @ W + sqrtqhat @ xi)

def fw(R, SigmaInv, Lambda):
    return linalg.inv(SigmaInv + Lambda*np.eye(2)) @ SigmaInv @ R

def fc(SigmaInv, Lambda):
    return linalg.inv(SigmaInv + Lambda*np.eye(2))

def phi(z, A):
    return z[0] + A*np.sqrt(Softplus(z[1]))

def Dz_phi(z, A):
    return np.array([1, np.divide(A*DSoftplus(z[1]), 2*np.sqrt(Softplus(z[1])))])

def gout(zStar, w, V):
    return linalg.inv(V) @ (zStar - w)

def Domega_gout(zStar, y, V):
    InvV = linalg.inv(V)
    return InvV @ (linalg.inv(InvV + DDzLoss(y, zStar)) @ InvV - np.eye(2))

def Dy_gout(zStar, y, V):
    InvV = linalg.inv(V)
    Sz = Softplus(zStar[1])
    Dsz = DSoftplus(zStar[1])
    return InvV @ (linalg.inv(InvV + DDzLoss(y, zStar)) @ np.array([1/Sz, (Dsz*(y - zStar[0]))/(Sz*Sz)]))

# State Evolution Loop functions

def RealFuncs(mhat, qhat, chihat, W, xi, Lambda):
    fwval = fw(T(mhat, qhat, chihat, W, xi), chihat, Lambda)
    m = np.einsum("i,j->ij", fwval, W)
    q = np.einsum("i,j->ij", fwval, fwval)
    sigma = fc(chihat, Lambda)
    return m, q, sigma

def HatFuncs(sigma, z, w, A):
    y = phi(z, A)
    Optimizedz = Prox(w, sigma, lambda z: Loss(y, z))
    qhat = np.einsum("i,j->ij", gout(Optimizedz, w, sigma), gout(Optimizedz, w, sigma))
    mhat = np.einsum("i,j->ij", Dy_gout(Optimizedz, y, sigma), Dz_phi(z, A))
    chihat = Domega_gout(Optimizedz, y, sigma)
    return qhat, mhat, chihat

# State Evolution sampling functions

def TrueRandSampleReal():
    W = rnd.normal(0, 1, 2)
    xi = rnd.normal(0, 1, 2)
    return W, xi

def TrueRandSampleHat(L):
    zw = L @ rnd.normal(0, 1, 4)
    A = rnd.normal(0, 1)
    return np.array([zw[0], zw[1]]), np.array([zw[2], zw[3]]), A

# State Evolution Expectation functions

def TrueRandExpectReal(mhat, qhat, chihat, Lambda, Nsample):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    local_N = Nsample // size
    m, q, sigma = np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2))
    for _ in range(local_N):
        W, xi = TrueRandSampleReal()
        newm, newq, newsigma = RealFuncs(mhat, qhat, chihat, W, xi, Lambda)
        m += newm
        q += newq
        sigma += newsigma
    m_global = np.zeros_like(m)
    q_global = np.zeros_like(q)
    sigma_global = np.zeros_like(sigma)
    comm.Allreduce(m, m_global, op=MPI.SUM)
    comm.Allreduce(q, q_global, op=MPI.SUM)
    comm.Allreduce(sigma, sigma_global, op=MPI.SUM)
    return m_global/Nsample, q_global/Nsample, sigma_global/Nsample

def TrueRandExpectHat(q, m, sigma, alpha, Nsample):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    local_N = Nsample // size
    qhat, mhat, chihat = np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2))
    L = linalg.cholesky(np.vstack([np.hstack([np.eye(2), m]), np.hstack([m, q])]) + 1e-4*np.eye(4))
    for _ in range(local_N):
        z, w, A = TrueRandSampleHat(L)
        newqhat, newmhat, newchihat = HatFuncs(sigma, z, w, A)
        qhat += newqhat
        mhat += newmhat
        chihat -= newchihat
    qhat_global = np.zeros_like(qhat)
    mhat_global = np.zeros_like(mhat)
    chihat_global = np.zeros_like(chihat)
    comm.Allreduce(qhat, qhat_global, op=MPI.SUM)
    comm.Allreduce(mhat, mhat_global, op=MPI.SUM)
    comm.Allreduce(chihat, chihat_global, op=MPI.SUM)
    return alpha*qhat_global/Nsample, alpha*mhat_global/Nsample, alpha*chihat_global/Nsample

# State Evolution runner

def TrueRandSE_ERM(alpha, Lambda, q0, m0, sigma0, Damping, Nsample, MaxIter, EpsConvergence, Verbose, VerboseRate, DebugVerbose):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    q, m, sigma = q0, m0, sigma0
    qhat, mhat, chihat = TrueRandExpectHat(q, m, sigma, alpha, Nsample)
    NIter = 0
    Conv = 1
    while((Conv > EpsConvergence) and (NIter < MaxIter)):
        IterStart = time.time()
        newqhat, newmhat, newchihat = TrueRandExpectHat(q, m, sigma, alpha, Nsample)
        qhat, mhat, chihat = Damping*qhat + (1-Damping)*newqhat, Damping*mhat + (1-Damping)*newmhat, Damping*chihat + (1-Damping)*newchihat
        newm, newq, newsigma = TrueRandExpectReal(mhat, qhat, chihat, Lambda, Nsample)
        Conv = (np.abs(q[0,0] - newq[0,0]) + np.abs(q[1,1] - newq[1,1]))/(np.abs(newq[0,0]) + np.abs(newq[1,1]))
        m, q, sigma = Damping*m + (1-Damping)*newm, Damping*q + (1-Damping)*newq, Damping*sigma + (1-Damping)*newsigma
        IterEnd = time.time()
        IterTime = IterEnd - IterStart
        if(Verbose and NIter%VerboseRate == 0 and rank == 0):
            print("Iteration %s" % NIter, flush=True)
            print("Current convergence criterion %s" % Conv, flush=True)
        if(DebugVerbose and rank == 0):
            print("qhat", qhat, flush=True)
            print("mhat", mhat, flush=True)
            print("chihat", chihat, flush=True)
            print("m", m, flush=True)
            print("q", q, flush=True)
            print("sigma", sigma, flush=True)
            print("Eigenvalues of Q", linalg.eigvals(np.vstack([np.hstack([np.eye(2), m]), np.hstack([m, q])])), flush=True)
            print("Iteration time : ", IterTime, flush=True)
        NIter += 1
    return q, m, sigma

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    parser = argparse.ArgumentParser(description="Run State Evolution ERM with MPI")
    parser.add_argument("--alpha", type=float, default=10, help="Value of alpha")
    parser.add_argument("--Lambda", type=float, default=1, help="Value of Lambda")
    parser.add_argument("--Damping", type=float, default=0, help="Damping coefficient")
    parser.add_argument("--Nsample", type=int, default=10000, help="Total number of Monte Carlo samples")
    parser.add_argument("--MaxIter", type=int, default=1e4, help="Maximum number of SE iterations")
    parser.add_argument("--EpsConvergence", type=float, default=1e-6, help="Convergence threshold")
    parser.add_argument("--Verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--VerboseRate", type=int, default=1, help="Frequence of outputs, if verbose is enabled")
    parser.add_argument("--Debug", action="store_true", help="Enable debug verbosity")

    args = parser.parse_args()

    # Only rank 0 prints to avoid clutter
    if args.Verbose and rank == 0:
        print("Running State Evolution with MPI", flush=True)
        print(f"Parameters: alpha={args.alpha}, Lambda={args.Lambda}, Nsample={args.Nsample}", flush=True)

    q0 = np.array([[1, 0], [0, 0.81]])#0.5*np.eye(2)
    m0 = np.array([[0.95, 0], [0, 0.75]])#0.2*np.eye(2)
    sigma0 = 0.5*np.eye(2)

    # Run the State Evolution algorithm
    q, m, sigma = TrueRandSE_ERM(args.alpha, args.Lambda, q0, m0, sigma0, Damping = args.Damping, Nsample = args.Nsample, MaxIter = args.MaxIter,
                                EpsConvergence = args.EpsConvergence, Verbose = args.Verbose, VerboseRate = args.VerboseRate,
                                DebugVerbose = args.Debug)

    if rank == 0:
        print("Final Results:")
        print("q =\n", q)
        print("m =\n", m)
        print("sigma =\n", sigma)
    
    np.savetxt("q.txt", q, fmt="%.6f")
    np.savetxt("m.txt", m, fmt="%.6f")
    np.savetxt("sigma.txt", sigma, fmt="%.6f")

if __name__ == "__main__":
    main()
