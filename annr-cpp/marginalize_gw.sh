#!/bin/bash

set -e

cfg_path="${1}"
folder=`python -c "import sys; sys.stdout = open('/dev/null', 'w'); sys.stderr = open('/dev/null', 'w'); sys.path += [\"scripts\"]; import utils; cfg = utils.load_config(\"${cfg_path}\"); sys.stdout = sys.__stdout__; print(f'{cfg.root_data_folder}/{cfg.name}')"`
echo "Folder: ${folder}"

g=32
m=50000

tmp="temp.sh"

true > ${tmp}
for slice in {0..14}; do
  python scripts/marginalize.py "${folder}" --data_tag annr -g "$g" -m "$m" --slice "${slice}" --nosave
  echo "python scripts/marginalize.py \"${folder}\" --data_tag true -g \"$g\" -m \"$m\" --slice \"${slice}\"  --nosave"  >> ${tmp}
  echo "python scripts/marginalize.py \"${folder}\" --data_tag defer -g \"$g\" -m \"$m\" --slice \"${slice}\"  --nosave" >> ${tmp}
done
parallel --lb :::: ${tmp}

python scripts/marginalize_plot_all.py "${folder}" -g "$g" -m "$m"
