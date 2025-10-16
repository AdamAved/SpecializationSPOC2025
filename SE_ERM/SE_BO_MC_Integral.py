import numpy as np
import numpy.random as rnd
import numpy.linalg as linalg
import time
import argparse
import h5py
from scipy.integrate import nquad
from scipy.linalg import sqrtm
from scipy.stats import norm, multivariate_normal
from scipy.special import softplus
from mpi4py import MPI

# Basic math functions for the State Evolution loop functions

def Pout(z, y):
    return (norm(loc = z[0], scale = np.sqrt(softplus(z[1])))).pdf(y)

def Z_out(y, w, V, IntSamples):
    Vhalf = linalg.cholesky(V)
    Zout = 0
    for _ in range(IntSamples):
        z = w + Vhalf @ rnd.normal(0, 1, 2)
        Zout += Pout(z,y)
    return Zout/IntSamples

def z_avg_out(y, w, V, IntSamples):
    Vhalf = linalg.cholesky(V)
    Z_avg_Unnorm = np.zeros(2)
    for _ in range(IntSamples):
        z = w + Vhalf @ rnd.normal(0, 1, 2)
        Z_avg_Unnorm += z*Pout(z,y)
    Z_avg_Unnorm /= IntSamples
    return Z_avg_Unnorm/Z_out(y, w, V, IntSamples)

def T(qhat, W, xi):
    return W + sqrtm(linalg.inv(qhat)).real @ xi

def fw(R, SigmaInv):
    return linalg.inv(SigmaInv + np.eye(2)) @ SigmaInv @ R

def phi(z, A):
    return z[0] + A*np.sqrt(softplus(z[1]))

def gout(z_avg, w, V):
    return linalg.inv(V) @ (z_avg - w)

# State Evolution Loop functions

def RealFuncs(qhat, W, xi):
    fwval = fw(T(qhat, W, xi), qhat)
    q = np.einsum("i,j->ij", fwval, fwval)
    return q

def HatFuncs(sigma, z, w, A, IntSamples):
    gOut = gout(z_avg_out(phi(z, A), w, sigma, IntSamples), w, sigma)
    qhat = np.einsum("i,j->ij", gOut , gOut)
    return qhat

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

def TrueRandExpectReal(qhat, Nsample):
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    local_N = Nsample // size
    q = np.zeros((2, 2))
    q2 = np.zeros((2, 2))
    for _ in range(local_N):
        W, xi = TrueRandSampleReal()
        newq = RealFuncs(qhat, W, xi)
        q += newq
        q2 += np.square(newq)
    q_global = comm.allreduce(q, op=MPI.SUM)
    q2_global = comm.allreduce(q2, op=MPI.SUM)
    return q_global/Nsample, (q2_global - np.square(q_global)/Nsample)/(Nsample - 1)

def TrueRandExpectHat(q, alpha, Nsample, IntSamples):
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    local_N = Nsample // size
    sigma = np.eye(2) - q
    qhat = np.zeros((2, 2))
    Sigma = np.vstack([np.hstack([np.eye(2), q]), np.hstack([q, q])]) + 1e-4*np.eye(4)
    L = linalg.cholesky(Sigma)
    for _ in range(local_N):
        z, w, A = TrueRandSampleHat(L)
        newqhat = HatFuncs(sigma, z, w, A, IntSamples)
        qhat += newqhat
    qhat_global = comm.allreduce(qhat, op=MPI.SUM)
    return alpha*qhat_global/Nsample

# State Evolution runner

def TrueRandSE_BO(alpha, q0, Damping, Nsample, IntSamples, MaxIter, EpsConvergence, Verbose, VerboseRate, DebugVerbose, FileNameQ):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    q = q0
    qhat= TrueRandExpectHat(q, alpha, Nsample, IntSamples)
    NIter = 0
    Conv = 1
    while((Conv > EpsConvergence) and (NIter < MaxIter)):
        IterStart = time.time()
        newqhat = TrueRandExpectHat(q, alpha, Nsample, IntSamples)
        qhat = Damping*qhat + (1-Damping)*newqhat
        newq, varq = TrueRandExpectReal(qhat, Nsample)
        Conv = (np.abs(q[0,0] - newq[0,0]) + np.abs(q[1,1] - newq[1,1]))/(np.abs(newq[0,0]) + np.abs(newq[1,1]))
        q = Damping*q + (1-Damping)*newq
        IterEnd = time.time()
        IterTime = IterEnd - IterStart
        if(Verbose and NIter%VerboseRate == 0 and rank == 0):
            print("Iteration %s" % NIter, flush=True)
            print("Current convergence criterion %s" % Conv, flush=True)
        if(DebugVerbose and rank == 0):
            print("qhat", qhat, flush=True)
            print("q", q, flush=True)
            print("Eigenvalues of Q", linalg.eigvals(np.vstack([np.hstack([np.eye(2), q]), np.hstack([q, q])])), flush=True)
            print("Iteration time : ", IterTime, flush=True)
        if rank == 0:
            with h5py.File(FileNameQ, "a") as f:
                dset = f[FileNameQ]
                n = dset.shape[0]
                dset.resize(n + 1, axis=0)
                dset[n, :, :] = q
        NIter += 1
    return q, varq

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    parser = argparse.ArgumentParser(description="Run State Evolution ERM with MPI")
    parser.add_argument("--alpha", type=float, default=10, help="Value of alpha")
    parser.add_argument("--Damping", type=float, default=0, help="Damping coefficient")
    parser.add_argument("--Nsample", type=int, default=10000, help="Total number of Monte Carlo samples")
    parser.add_argument("--IntSamples", type=int, default=20, help="Number of MC samples for integral")
    parser.add_argument("--MaxIter", type=int, default=1e4, help="Maximum number of SE iterations")
    parser.add_argument("--EpsConvergence", type=float, default=1e-6, help="Convergence threshold")
    parser.add_argument("--Verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--VerboseRate", type=int, default=1, help="Frequence of outputs, if verbose is enabled")
    parser.add_argument("--Debug", action="store_true", help="Enable debug verbosity")

    args = parser.parse_args()

    # Only rank 0 prints to avoid clutter
    if args.Verbose and rank == 0:
        print("Running State Evolution with MPI", flush=True)
        print(f"Parameters: alpha={args.alpha}, Nsample={args.Nsample}", flush=True)

    q0 = np.array([[0.64, 0.01], [0.01, 0.23]])#0.5*np.eye(2)
    m0 = np.array([[0.64, 0.01], [0.01, 0.16]])#0.2*np.eye(2)
    sigma0 = 0.5*np.eye(2)

    filenameQ = f"q_alpha_{args.alpha:.5f}_Samples_{args.Nsample:.5f}_SE_BO.mat"
    if rank == 0:
        with h5py.File(filenameQ, "w") as f:
            dset = f.create_dataset("q_BO", shape=(0, 2, 2), maxshape=(None, 2, 2), dtype='float64')

    # Run the State Evolution algorithm
    StartTime = time.time()
    q, varq = TrueRandSE_BO(args.alpha, q0, Damping = args.Damping, Nsample = args.Nsample, IntSamples = np.abs(args.IntSamples), MaxIter = args.MaxIter,
                                EpsConvergence = args.EpsConvergence, Verbose = args.Verbose, VerboseRate = args.VerboseRate,
                                DebugVerbose = args.Debug, FileNameQ = filenameQ)

    if rank == 0:
        print("Final Results:")
        print("q =\n", q)
        print("m =\n", m)
        print("Elapsed time : ", time.time() - StartTime)
    
    np.savetxt(f"q_alpha_{args.alpha}_Samples_{args.Nsample}_SE_BO.txt", q, fmt="%.6f")
    np.savetxt(f"varq_alpha_{args.alpha}_Samples_{args.Nsample}_SE_BO.txt",  varq, fmt="%.6f")

if __name__ == "__main__":
    main()
