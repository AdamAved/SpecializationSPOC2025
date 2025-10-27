foreach ($alpha in 1.0, 3.0, 5.0, 7.0, 10.0) {
    foreach ($lambda in 1, 2, 5) {
        Write-Output "Running with alpha=$alpha, Lambda=$lambda"
        mpiexec -n 20 python SE_ERM.py `
            --alpha $alpha `
            --Lambda $lambda `
            --Nsample 100000 `
            --Damping 0.6 `
            --EpsConvergence 1e-3
    }
}
