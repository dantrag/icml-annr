# annr-python

This repository contains the basic Python implementation of the ANNR algorithm (folder `pyannr`). To reduce dependencies, the code for a baseline method [DEFER](https://github.com/bodin-e/defer) is also included here (folder `pydefer`, renamed to avoid conflicts with a `python-defer` module).

In addition, several simple examples are included, some of which also appear in the manuscript. The code is designed to reproduce the figures presented in the manuscript.

### Dependencies

The following packages are required to run this code (joint dependencies for `pyannr` and `pydefer`). Package versions are the ones that have been tested, other versions might work as well.
```
matplotlib==3.4.2
numpy==1.17.4
scipy==1.4.1
tqdm==4.62.2
xxhash==2.0.2
```
These dependencies can be installed by running the following command:
`pip install -r requirements.txt`

## Quick start

Run a function approximation experiment, where function is a standard Gaussian:

```
python main.py
```

Four png files - two with approximated diagrams, one with ground truth plot and one with sqores - should appear in the same folder.

### Other examples

For other examples, go to the corresponding folder and run it as follows:

```
cd examples/l1
python l1.py
```

### Your own functions

To create your own function, create a class derived from class `Function` (see `function.py`) and implement method `__call__(self, x)`, returning the value of a function, given input point `x`.  An example can be found in `examples/narrow_domain/narrow_domain.py`.
