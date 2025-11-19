import numpy as np
import numpy.random as rnd
import numpy.linalg as linalg
import time
import argparse
import h5py
from scipy.linalg import sqrtm
from scipy.optimize import minimize
from scipy.special import softplus
from mpi4py import MPI

# Basic math functions for the State Evolution loop functions

def Loss(y, z):
    return (y - z[0])*(y - z[0])/2

def Prox(mu, Omega, f, StartPoint):
    ToOptimize = lambda x: np.einsum("i,ij,j->",(x - mu), linalg.inv(Omega), (x - mu))/2 + f(x)
    Proximal = minimize(ToOptimize, x0=StartPoint)
    return Proximal.x

def T(mhat, qhat, chihat, W, xi):
    sqrtqhat = sqrtm(qhat).real
    return linalg.inv(chihat) @ (mhat @ W + sqrtqhat @ xi)

def fw(R, SigmaInv, Lambda):
    return linalg.inv(SigmaInv + np.diag(Lambda)) @ SigmaInv @ R

def fc(SigmaInv, Lambda):
    return linalg.inv(SigmaInv + np.diag(Lambda))

def phi(z, A):
    return z[0] + A*np.sqrt(softplus(z[1]))

def gout(zStar, w, V):
    return linalg.inv(V) @ (zStar - w)

# State Evolution Loop functions

def RealFuncs(mhat, qhat, chihat, W, xi, Lambda):
    fwval = fw(T(mhat, qhat, chihat, W, xi), chihat, Lambda)
    m = np.einsum("i,j->ij", fwval, W)
    q = np.einsum("i,j->ij", fwval, fwval)
    sigma = fc(chihat, Lambda)
    return m, q, sigma

def HatFuncsSteins(sigma, z, w, A, SigmaInv):
    gOut = gout(Prox(w, sigma, lambda zToOptimize: Loss(phi(z, A), zToOptimize), z), w, sigma)
    SteinsExtraTerm = np.einsum("ij,j->i", SigmaInv, np.hstack([z, w]))
    qhat = np.einsum("i,j->ij", gOut , gOut)
    mhat = np.einsum("i,j->ij", gOut, SteinsExtraTerm[0:2])
    minuschihat = np.einsum("i,j->ij", gOut, SteinsExtraTerm[2:4])
    return qhat, mhat, minuschihat

# State Evolution sampling functions for the expectation

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
    size = comm.Get_size()
    local_N = Nsample // size
    m, q, sigma = np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2))
    m2, q2 = np.zeros((2, 2)), np.zeros((2, 2))
    for _ in range(local_N):
        W, xi = TrueRandSampleReal()
        newm, newq, newsigma = RealFuncs(mhat, qhat, chihat, W, xi, Lambda)
        m += newm
        m2 += np.square(newm)
        q += newq
        q2 += np.square(newq)
        sigma += newsigma
    m_global = comm.allreduce(m, op=MPI.SUM)
    m2_global = comm.allreduce(m2, op=MPI.SUM)
    q_global = comm.allreduce(q, op=MPI.SUM)
    q2_global = comm.allreduce(q2, op=MPI.SUM)
    sigma_global = comm.allreduce(sigma, op=MPI.SUM)
    return m_global/Nsample, (m2_global - np.square(m_global)/Nsample)/(Nsample - 1), q_global/Nsample, (q2_global - np.square(q_global)/Nsample)/(Nsample - 1), sigma_global/Nsample

def TrueRandExpectHat(q, m, sigma, alpha, Nsample):
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    local_N = Nsample // size
    qhat, mhat, chihat = np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2))
    Sigma = np.vstack([np.hstack([np.eye(2), m]), np.hstack([m, q])]) + 1e-4*np.eye(4)
    L = linalg.cholesky(Sigma)
    for _ in range(local_N):
        z, w, A = TrueRandSampleHat(L)
        newqhat, newmhat, newminuschihat = HatFuncsSteins(sigma, z, w, A, linalg.inv(Sigma))
        qhat += newqhat
        mhat += newmhat
        chihat -= newminuschihat
    qhat_global = comm.allreduce(qhat, op=MPI.SUM)
    mhat_global = comm.allreduce(mhat, op=MPI.SUM)
    chihat_global = comm.allreduce(chihat, op=MPI.SUM)
    return alpha*qhat_global/Nsample, alpha*mhat_global/Nsample, alpha*chihat_global/Nsample

# State Evolution runner

def TrueRandSE_ERM(alpha, Lambda, q0, m0, sigma0, Damping, Nsample, MaxIter, EpsConvergence, Verbose, VerboseRate, DebugVerbose, FileName):
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
        newm, varm, newq, varq, newsigma = TrueRandExpectReal(mhat, qhat, chihat, Lambda, Nsample)
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
        if rank == 0:
            with h5py.File(FileName, "a") as f:
                QERM = f["q_ERM"]
                MERM = f["m_ERM"]
                SERM = f["sigma_ERM"]
                vQERM = f["varq_ERM"]
                vMERM = f["varm_ERM"]
                n = QERM.shape[2]
                QERM.resize(n + 1, axis=2)
                MERM.resize(n + 1, axis=2)
                SERM.resize(n + 1, axis=2)
                vQERM.resize(n + 1, axis=2)
                vMERM.resize(n + 1, axis=2)
                QERM[:, :, n] = q
                MERM[:, :, n] = m
                SERM[:, :, n] = sigma
                vQERM[:, :, n] = varq
                vMERM[:, :, n] = varm

        NIter += 1
    return q, varq, m, varm

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    parser = argparse.ArgumentParser(description="Run State Evolution ERM with MPI")
    parser.add_argument("--alpha", type=float, default=10, help="Value of alpha")
    parser.add_argument("--Lambda0", type=float, default=1, help="Value of Lambda0")
    parser.add_argument("--Lambda1", type=float, default=1, help="Value of Lambda1")
    parser.add_argument("--Damping", type=float, default=0, help="Damping coefficient")
    parser.add_argument("--Nsample", type=int, default=10000, help="Total number of Monte Carlo samples")
    parser.add_argument("--MaxIter", type=int, default=1e2, help="Maximum number of SE iterations")
    parser.add_argument("--EpsConvergence", type=float, default=1e-6, help="Convergence threshold")
    parser.add_argument("--Verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--VerboseRate", type=int, default=1, help="Frequence of outputs, if verbose is enabled")
    parser.add_argument("--Debug", action="store_true", help="Enable debug verbosity")

    args = parser.parse_args()

    # Only rank 0 prints to avoid clutter
    if args.Verbose and rank == 0:
        print("Running State Evolution with MPI", flush=True)
        print(f"Parameters: alpha={args.alpha}, Lambda={args.Lambda}, Nsample={args.Nsample}", flush=True)

    q0 = np.array([[0.64, 0.01], [0.01, 0.23]])#0.5*np.eye(2)
    m0 = np.array([[0.64, 0.01], [0.01, 0.16]])#0.2*np.eye(2)
    sigma0 = 0.5*np.eye(2)

    filename = f"SE_ERM_Ridge_{args.alpha:.5f}_Lambda0_{args.Lambda0:.5f}_Lambda1_{args.Lambda1:.5f}_Samples_{args.Nsample:.5f}.mat"
    if rank == 0:
        with h5py.File(filename, "w") as f:
            f.create_dataset("q_ERM", shape=(2, 2, 0), maxshape=(2, 2, None), dtype='float64')
            f.create_dataset("m_ERM", shape=(2, 2, 0), maxshape=(2, 2, None), dtype='float64')
            f.create_dataset("sigma_ERM", shape=(2, 2, 0), maxshape=(2, 2, None), dtype='float64')
            f.create_dataset("varq_ERM", shape=(2, 2, 0), maxshape=(2, 2, None), dtype='float64')
            f.create_dataset("varm_ERM", shape=(2, 2, 0), maxshape=(2, 2, None), dtype='float64')

    # Run the State Evolution algorithm
    StartTime = time.time()
    q, varq, m, varm = TrueRandSE_ERM(args.alpha, np.array([args.Lambda0, args.Lambda1]), q0, m0, sigma0, Damping = args.Damping, Nsample = args.Nsample, MaxIter = args.MaxIter,
                                EpsConvergence = args.EpsConvergence, Verbose = args.Verbose, VerboseRate = args.VerboseRate,
                                DebugVerbose = args.Debug, FileName = filename)

    if rank == 0:
        print("Final Results:")
        print("q =\n", q)
        print("m =\n", m)
        print("Elapsed time : ", time.time() - StartTime)

if __name__ == "__main__":
    main()
