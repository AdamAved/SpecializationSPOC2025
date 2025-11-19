#!/bin/bash
# Generates all combinations of A, B, and C parameter values into params.txt

# Other parameters
module load gcc
module load python
source myenv/bin/activate

# Generate log-spaced values using Python
# lambda=1 # ($(python3 -c "import numpy as np; print(' '.join(map(str, np.logspace(-1, 1, 11))))"))
lambda=($(python3 -c "import numpy as np; print(' '.join(map(str, np.logspace(-1, 1, 41))))"))
alpha=5 # ($(python3 -c "import numpy as np; print(' '.join(map(str, np.logspace(-1, 1.62, 10))))"))

# Output file
outfile="Params_Lambda_Scan_Basic.txt"
> "$outfile"  # clear or create file

# Generate combinations
for a in "${alpha[@]}"; do
    for l in "${lambda[@]}"; do
        echo "$a $l $l" >> "$outfile"
    done
done

echo "✅ Generated $outfile with $(wc -l < $outfile) parameter combinations."
