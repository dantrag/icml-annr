#!/bin/bash

set -e

# Either "gw_clipped" or "latent"
data_name="${1}"

# 150.0 for "gw_clipped" and 0.05 for "latent"
lambda="${2}"

# "uniform" for "gw_clipped" and "grid" for "latent"
test_tag="${3}"

angle="1.553343"

cfg_folder="configs/${data_name}"
mae_file="data/${data_name}.txt"

mkdir -p "${cfg_folder}"
mkdir -p data
true > "${mae_file}"

# Temporary shell file for DEFER parallelization
tmp="temp.sh"

# Train ANNR & DEFER, 10 runs
for i in {239..248}
do
  cfg_path="${cfg_folder}/${data_name}_${i}.py"
	 _RS=$i _LAMBDA=${lambda} _ANGLE=${angle} envsubst < "configs/${data_name}_template.py" > "${cfg_path}"
	name=${i}
	echo Started ANNR "${name}" && python scripts/annr_train.py "${cfg_path}" && echo Done ANNR "${name}"
	echo Started DEFER "${name}" && python scripts/defer_train.py "${cfg_path}" && echo Done DEFER "${name}"
done

# Folder of the first run
folder239=`python -c "import sys; sys.stdout = open('/dev/null', 'w'); sys.stderr = open('/dev/null'); sys.path += [\"scripts\", \"${cfg_folder}\"]; import ${data_name}_239 as cfg; sys.stdout = sys.__stdout__; print(f'{cfg.root_data_folder}/{cfg.name}')"`

# Generate test data
python scripts/testing.py "${folder239}" --data_tag none --test_tag ${test_tag} --n_test_multiplier 100

true > "${tmp}"
for i in {239..248}
do
  folder=`python -c "import sys; sys.stdout = open('/dev/null', 'w'); sys.stderr = open('/dev/null'); sys.path += [\"scripts\", \"${cfg_folder}\"]; import ${data_name}_${i} as cfg; sys.stdout = sys.__stdout__; print(f'{cfg.root_data_folder}/{cfg.name}')"`
	name=${i}
	if [ ${i} -ne 239 ] ; then
    cp "${folder239}/${test_tag}_test_data.npy" "${folder}/"
    cp "${folder239}/true_${test_tag}_test_values.npy" "${folder}/"
  fi
	echo Started ANNR "${name}" && python scripts/testing.py "${folder}" --data_tag annr --test_tag "${test_tag}" --n_test_multiplier 100 --mae_out "${mae_file}" && echo Done ANNR "${name}"
	echo Started nANNR "${name}" && python scripts/testing.py "${folder}" --data_tag uniform --test_tag "${test_tag}" --n_test_multiplier 100 --mae_out "${mae_file}" && echo Done nANNR "${name}"
	echo "echo Started DEFER \"${name}\" && python scripts/testing.py \"${folder}\" --data_tag defer --test_tag \"${test_tag}\" --n_test_multiplier 100 --mae_out \"${mae_file}\" && echo Done DEFER \"${name}\"" >> ${tmp}
done
parallel --lb :::: "${tmp}"

# Print mean and std MAE
python scripts/print_mae.py ${mae_file}
