import numpy as np
import numpy.random as rnd
import time
import argparse
import h5py
from scipy.special import softplus, expit

# Gradient and Loss functions

def GradERM(X, y, w, v, sigmafunc, Dsigmafunc, LambdaRegularization):
    zw = np.matmul(X, w)
    zv = np.matmul(X, v)
    GradwTotal = np.matmul(np.transpose(X), - np.divide(y - zw,sigmafunc(zv))) + LambdaRegularization[0]*w
    GradvTotal = (np.matmul(np.transpose(X), np.multiply((np.divide(1,(2*sigmafunc(zv))) - np.divide(np.power((y - zw),2),(2*np.power(sigmafunc(zv),2)))), Dsigmafunc(zv))) 
                  + LambdaRegularization[1]*v)
    return(np.stack((GradwTotal, GradvTotal), axis = 1))

def LossERM(X, y, w, v, sigmafunc, LambdaRegularization):
    zw = np.matmul(X, w)
    zv = np.matmul(X, v)
    return( np.sum(np.divide(np.power((y - zw),2),(2*sigmafunc(zv))) + np.log(sigmafunc(zv))/2, 0) + (LambdaRegularization[0]*np.dot(w,w) + LambdaRegularization[1]*np.dot(v,v))/2 )

# GD step

def GDStepERM(X, y, wv, sigmafunc, Dsigmafunc, LambdaRegularization, LearningRate):
    wv -= LearningRate*GradERM(X, y, wv[:,0], wv[:,1], sigmafunc, Dsigmafunc, LambdaRegularization)
    return(LossERM(X, y, wv[:,0], wv[:,1], sigmafunc, LambdaRegularization))

# Full GD function

def GDERM(X, y, wv, wvTrue, sigmafunc, Dsigmafunc, Dim, LambdaRegularization, LearningRate, MaxIter, EpsConvergence, Verbose, VerboseRate, FileName):
    Conv = 1
    NIter = 0
    Losses = [LossERM(X, y, wv[:,0], wv[:,1], sigmafunc, LambdaRegularization)]
    if(Verbose):
        print("Iteration %s" % NIter)
        print("Current loss %s" % Losses[NIter])
    while((NIter < MaxIter) and (Conv > EpsConvergence)):
        Losses.append(GDStepERM(X, y, wv, sigmafunc, Dsigmafunc, LambdaRegularization, LearningRate))
        NIter = NIter + 1
        Conv = np.abs(Losses[NIter] - Losses[NIter-1])/np.abs(Losses[NIter])
        q = np.einsum("ji,jk->ik", wv, wv)/Dim
        m = np.einsum("ji,jk->ik", wv, wvTrue)/Dim
        with h5py.File(FileName, "a") as f:
                QERM = f["q_ERM"]
                MERM = f["m_ERM"]
                lERM = f["Loss_ERM"]
                n = QERM.shape[2]
                QERM.resize(n + 1, axis=2)
                MERM.resize(n + 1, axis=2)
                lERM.resize(n + 1, axis=1)
                QERM[:, :, n] = q
                MERM[:, :, n] = m
                lERM[0, n] = Losses[NIter]
    return(np.array(Losses))

# Different type of sigma functions

def sigmaSquare(zv):
    return(np.power(zv,2))

def DsigmaSquare(zv):
    return(2*zv)

def sigmaSoftplus(zv):
    return(softplus(zv))

def DsigmaSoftplus(zv):
    return(expit(zv))

# Main

def main():

    parser = argparse.ArgumentParser(description="Run State Evolution ERM with MPI")
    parser.add_argument("--Nrep", type=int, default=1, help="Repetition number")
    parser.add_argument("--dim", type=int, default=1000, help="Value of dimension")
    parser.add_argument("--alpha", type=float, default=10, help="Value of alpha")
    parser.add_argument("--Lambda0", type=float, default=1, help="Value of Lambda0")
    parser.add_argument("--Lambda1", type=float, default=1, help="Value of Lambda1")
    parser.add_argument("--LearningRate", type=float, default=0.002, help="Value of the learning rate")
    parser.add_argument("--MaxIter", type=int, default=1e6, help="Maximum number of GD iterations")
    parser.add_argument("--EpsConvergence", type=float, default=1e-6, help="Convergence threshold")
    parser.add_argument("--Verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--VerboseRate", type=int, default=100, help="Frequence of outputs, if verbose is enabled")

    args = parser.parse_args()

    if args.Verbose:
        print("Running GD", flush=True)
        print(f"Parameters: alpha={args.alpha}, Lambda0={args.Lambda0}, Lambda1={args.Lambda1}, Nrep={args.Nrep}", flush=True)

    filename = f"GD_ERM_{args.alpha:.5f}_Lambda0_{args.Lambda0:.5f}_Lambda1_{args.Lambda1:.5f}_Dim_{args.dim:.5f}_Rep_{args.Nrep:.5f}.mat"
    with h5py.File(filename, "w") as f:
        f.create_dataset("q_ERM", shape=(2, 2, 0), maxshape=(2, 2, None), dtype='float64')
        f.create_dataset("m_ERM", shape=(2, 2, 0), maxshape=(2, 2, None), dtype='float64')
        f.create_dataset("Loss_ERM", shape=(1, 0), maxshape=(1, None), dtype='float64')

    # Simulation Preparation
    M = int(args.alpha*args.dim)
    X = rnd.normal(0, 1/np.sqrt(args.dim), size = (M, args.dim))
    wvTrue = rnd.normal(0, 1, size = (args.dim, 2))
    wvLearned = rnd.normal(0, 1, size = (args.dim, 2))
    yNoise = np.einsum("ij,j->i", X, wvTrue[:,0]) + rnd.normal(0, np.sqrt(sigmaSoftplus(np.matmul(X, wvTrue[:,1]))))
    

    # Run the State Evolution algorithm
    StartTime = time.time()
    _ = GDERM(X, yNoise, wvLearned, wvTrue, sigmaSoftplus, DsigmaSoftplus, Dim=args.dim, LearningRate=args.LearningRate,
        LambdaRegularization=[args.Lambda0, args.Lambda1], MaxIter = args.MaxIter, Verbose = args.Verbose,
        VerboseRate = args.VerboseRate, EpsConvergence=args.EpsConvergence, FileName=filename)

    print("Elapsed time : ", time.time() - StartTime)

if __name__ == "__main__":
    main()
