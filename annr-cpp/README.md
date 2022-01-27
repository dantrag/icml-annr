This is an implementation of the ANNR in C++ with provided Python interface. We also provide the scripts which reproduce results from the paper.

## Compilation
The current version has only been tested on Linux, and might not work on other platforms.

The code relies on `Eigen 3.4+`, `boost` (and particularly `boost-python`), `OpenMP` and `ZLib` libraries, as well as on availability of `cmake`, `make` and requires C++17 standard.

To compile the code, run:
```shell
mkdir build && cd build
cmake ..
make
```

The `annr` library will appear in the `build/annr/` folder.

## Experiments
The scripts require `parallel` for the execution of some commands.

The following commands will reproduce the results from the paper:
- `bash ball6d.sh` <p>
   Reproduces the figures that corresponds to the investigation of the curse of dimensionality by generates diagrams `ball6_mae.png` and `ball6_queries.png`

- `bash train_several.sh gw 150.0 uniform` <p>
  Runs 10 training procedures for the 'gravitational wave' dataset and reports MAE for ANNR, DEFER and nANNR over those runs.
  
- `bash train_several.sh latent 0.005 grid` <p>
  Runs 10 training procedures for the 'VAE latent space' dataset and reports MAE for ANNR, DEFER and nANNR over those runs.
  
- `bash marginalize_gw.sh configs/gw/gw_239.py` <p>
  Produces marginals for the first run of 'gravitational wave' in `pics` folder, reproducing the corresponding figures from the table.
  
## Python scripts
These are the main scripts for method training and numeric comparison.

- `annr_train.py` Training of the ANNR.
- `defer_train.py` Training of the DEFER.
- `testing.py` Computation of MAE for a trained method.

The methods require a configuration file, see for example `configs/example_config.py`

## Other
`checkpoints` contains an already trained VAE.