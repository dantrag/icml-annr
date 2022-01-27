#!/bin/bash

set -e

mkdir -p "data"
mkdir -p "logs"
true > data/ball6.txt

# Train ANNR
python scripts/annr_train.py configs/ball6d.py

# Train DEFER
python scripts/defer_train.py configs/ball6d.py

# Get main data folder
folder=`python -c "import sys; sys.stdout = open('/dev/null', 'w'); sys.stderr = open('/dev/null'); sys.path += [\"scripts\", \"configs\"]; import ball6d as cfg; sys.stdout = sys.__stdout__; print(f'{cfg.root_data_folder}/{cfg.name}')"`

# Temporary shell file for DEFER parallelization
tmp="temp.sh"

# Selected radii
rs=(0.50 0.60 0.70 0.80 0.90 0.925 0.95 0.975 1.00 1.025 1.05 1.075 1.10 1.20 1.30 1.4 1.5)

# Generate test data
true > ${tmp}
for i in "${!rs[@]}"; do
  r=${rs[$i]}
  echo "echo Started $i && python scripts/testing_ball.py \"${folder}\" --data_tag none --test_tag uniform -r $r --mae_out data/ball6.txt --n_test_multiplier 100 --exec \"cfg.test_seed=$(( $i+7148 ))\" > \"logs/log_${i}.txt\" 2>/dev/null && echo Done $i" >> $tmp
done
parallel --lb :::: ${tmp}

# Test ANNR and nANNR
true > ${tmp}
for i in "${!rs[@]}"; do
  r=${rs[$i]}
  for k in "annr" "uniform"; do
    name="${i}_${k}"
    echo "Started ${name}" && python scripts/testing_ball.py "${folder}" --data_tag $k --test_tag uniform -r $r --mae_out data/ball6.txt --n_test_multiplier 100 --exec "cfg.test_seed=$(( $i+7148 ))" && echo "Done ${name}"
  done
done
bash -e ${tmp}

# Test DEFER (in parallel)
true > ${tmp}
for i in "${!rs[@]}"; do
  r=${rs[$i]}
  for k in "defer"; do
    name="${i}_${k}"
    echo "echo Started ${name} && python scripts/testing_ball.py \"${folder}\" --data_tag $k --test_tag uniform -r $r --mae_out data/ball6.txt --n_test_multiplier 100 --exec \"cfg.test_seed=$(( $i+7148 ))\" > \"logs/log_${name}.txt\" 2>/dev/null && echo Done ${name}" >> $tmp
  done
done
parallel --lb :::: ${tmp}

# Plot curves
python scripts/plot_mae_over_radius.py "data/ball6.txt"
python scripts/plot_hist_over_radius.py "${folder}"
