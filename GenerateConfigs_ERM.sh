#!/bin/bash
# Generates all combinations of A, B, and C parameter values into params.txt

# Other parameters
module load gcc
module load python
source myenv/bin/activate

# Generate log-spaced values using Python
lambda_0=1 # ($(python3 -c "import numpy as np; print(' '.join(map(str, np.logspace(-1, 1, 11))))"))
lambda_1=1 # ($(python3 -c "import numpy as np; print(' '.join(map(str, np.logspace(-1, 1, 11))))"))
alpha=($(python3 -c "import numpy as np; print(' '.join(map(str, np.logspace(-1, 2, 61))))"))

# Output file
outfile="Params_Alpha_Scan.txt"
> "$outfile"  # clear or create file

# Generate combinations
for a in "${alpha[@]}"; do
    for l0 in "${lambda_0[@]}"; do
        for l1 in "${lambda_1[@]}"; do
            echo "$a $l0 $l1" >> "$outfile"
        done
    done
done

echo "✅ Generated $outfile with $(wc -l < $outfile) parameter combinations."
