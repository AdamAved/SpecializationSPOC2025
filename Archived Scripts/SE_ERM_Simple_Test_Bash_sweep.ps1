foreach ($alpha in 7.0) {
    foreach ($lambda in 1) {
        Write-Output "Running with alpha=$alpha, Lambda=$lambda"
        mpiexec -n 20 python SE_ERM_SimpleTest.py `
            --alpha $alpha `
            --Lambda $lambda `
            --Nsample 100000 `
            --Damping 0.6 `
            --EpsConvergence 1e-3
    }
}
