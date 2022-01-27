import os
import time
import numpy as np

def load_config(config_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location('cfg', config_path)
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    print(f'Using configuration at {config_path}')
    return cfg

class timed:
    def __init__(self, name):
        self.name = name
        self.start = None

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        finish = time.time()
        print(f'{self.name}: {finish - self.start} sec.')

class its_own_random:
    def __init__(self, seed):
        self.seed = seed
        self.outer_state = None

    def __enter__(self):
        self.outer_state = np.random.get_state()
        np.random.seed(self.seed)

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert self.outer_state is not None
        np.random.set_state(self.outer_state)

def try_load(filename_npy, compute):
    if os.path.isfile(filename_npy):
        print(f'Loading {filename_npy}')
        return np.load(filename_npy)
    print(f'Computing {filename_npy}')
    result = compute()
    np.save(filename_npy, result)
    return result

def load_random(filename_pkl, seed):
    import pickle
    if os.path.isfile(filename_pkl):
        print(f'Loaded random state')
        state = pickle.load(open(filename_pkl, 'rb'))
        np.random.set_state(state)
    else:
        print(f'Generating new random state')
        np.random.seed(seed)
        state = np.random.get_state()
        pickle.dump(state, open(filename_pkl, 'wb'))

def f2sim(name, fn, dim, lower=None, upper=None, dtype=np.float128):
    from density_functions.core import Simulator
    from density_functions.misc import unit_domain
    from defer.bounded_space import BoundedSpace
    if lower is None:
        domain = unit_domain(dim)
    else:
        if isinstance(lower, (int, float)):
            lower = np.ones(dim, dtype=dtype) * lower
        if isinstance(upper, (int, float)):
            upper = np.ones(dim, dtype=dtype) * upper
        domain = BoundedSpace(lower, upper)
        # print(f'{domain.dtype}')
        # exit()
    return Simulator(name=name, domain=domain, fn=fn)


def as_float128(sim):
    from density_functions.core import Simulator
    from defer.bounded_space import BoundedSpace
    old_dtype = sim.domain.lower_limit_vector.dtype
    return Simulator(name=sim.name,
                     domain=BoundedSpace(sim.domain.lower_limit_vector.astype(np.float128),
                                         sim.domain.upper_limit_vector.astype(np.float128)),
                     fn=lambda theta: np.float128(sim._simulate(old_dtype.type(theta))))


def normalize_function(sim):
    from density_functions.core import Simulator
    from density_functions.misc import unit_domain
    old_name = sim.name
    old_lower = sim.domain.lower_limit_vector
    old_upper = sim.domain.upper_limit_vector
    old_callbacks = sim._callbacks_by_event  # note: currently forgetting those
    old_fn = sim._simulate

    new_name = f'{old_name}_normalized'
    new_domain = unit_domain(old_lower.shape[0], dtype=old_lower.dtype)

    def new_fn(new_theta):
        old_theta = (new_theta * (old_upper - old_lower) + old_lower).astype(old_lower.dtype)
        old_theta = np.maximum(old_lower, np.minimum(old_upper, old_theta))
        return old_fn(old_theta)

    return Simulator(name=new_name, domain=new_domain, fn=new_fn), sim.domain

def unlog_function(sim, cnst=0):
    from density_functions.core import Simulator
    from density_functions.misc import unit_domain
    old_name = sim.name
    old_domain = sim.domain
    old_callbacks = sim._callbacks_by_event  # note: currently forgetting those
    old_fn = sim._simulate

    new_name = f'{old_name}_nonlog'

    assert old_domain.lower_limit_vector.dtype == np.float128

    def new_fn(theta):
        return np.exp(np.array(old_fn(theta), dtype=np.float128) + cnst)

    return Simulator(name=new_name, domain=old_domain, fn=new_fn)


def generate_grid(linspace, dim):
    if not isinstance(linspace, list):
        linspace = [linspace] * dim
    grid = np.meshgrid(*linspace)
    grid = [np.reshape(a, (-1, 1)) for a in grid]
    grid = np.concatenate(grid, axis=1)
    assert grid.shape[1] == dim
    return grid


def uni_sample(domain, n_samples):
    import annr
    if isinstance(domain, annr.BoundingBox):
        lower = domain.lower
        upper = domain.upper
        dim = lower.size
        return np.random.uniform(
            lower.astype(np.float64), upper.astype(np.float64), (n_samples, dim)).astype(lower.dtype)
    elif isinstance(domain, annr.BoundingSphere):
        center = domain.center
        radius = domain.radius
        dim = center.size
        directions = np.random.normal(0, 1, (n_samples, dim))
        directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
        radii = radius * (np.random.random((n_samples, 1)) ** (1. / dim))

        data = directions * radii
        return data.astype(center.dtype)
    else:
        raise Exception(f'Unknown domain type: {type(domain)}')


def gen_sphere(n, d, r, dtype=np.float128):  # (d-1)-dim sphere
    directions = np.random.normal(0, 1, (n, d))
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    radii = r

    data = directions * radii
    return data.astype(dtype)


def fstr(template):
    return eval(f"f'{template}'")


def load_interpolate(cfg, data_tag, iteration=0):
    import annr
    from tqdm import tqdm

    def make_compute_values(function, _data):
        def func():
            # _values = Parallel(n_jobs=-1, verbose=5)(delayed(function)(x) for x in _data)
            # return np.array(_values)
            _values = []
            for x in tqdm(_data):
                _values.append(function(x))
            return np.array(_values)
        return func

    save_path = f'{cfg.root_data_folder}/{cfg.name}'
    if data_tag in ['annr', 'uniform']:
        if data_tag == 'annr':
            data = np.load(f'{save_path}/{data_tag}_data.npy')
            if data.shape[0] < cfg.n_queries:
                raise Exception(f'Not enough points! {data.shape[0]} out of {cfg.n_queries}')
        elif data_tag == 'uniform':
            with its_own_random(cfg.uniform_data_seed if 'uniform_data_seed' in dir(cfg) else cfg.random_seed + 542):
                data = try_load(f'{save_path}/{data_tag}_data.npy',
                                lambda: uni_sample(cfg.domain, cfg.n_queries))
    
        values = try_load(f'{save_path}/{data_tag}_values.npy',
                          make_compute_values(cfg.function, data))
    
        if iteration > 0:
            data = data[:iteration, :]
            values = values[:iteration]
    
        graph = annr.VoronoiGraph(cfg.strategy)
        graph.initialize(data, annr.Unbounded())
    
        interpolator = annr.ActiveInterpolator(
            cfg.random_seed + 614, graph, values, cfg.criterion)
    
        interpolate = interpolator.interpolate
    
    elif data_tag == 'defer' or data_tag == 'defernnr':
        from defer.helpers import Variables, DensityFunctionApproximation, construct
        from defer.variables import Variable
        x = Variable(
            lower=cfg.function.domain.lower_limit_vector,
            upper=cfg.function.domain.upper_limit_vector,
            name="x"
        )
        variables = Variables([x])
        approx: DensityFunctionApproximation = construct(
            fn=cfg.function,
            is_log_fn=False,
            variables=variables,
            num_fn_calls=1,
            callback=lambda i, density:
            print("#Evals: %s. Log Z: %.2f" %
                  (density.num_partitions, np.log(density.z))),
            callback_freq_fn_calls=1000,
            is_vectorized_fn=False
        )
        if args.iteration <= 0:
            print(f'Loading {save_path}/defer.pkl')
            approx.load(f'{save_path}/defer.pkl')
        else:
            print(f'Loading {save_path}/extra/{args.iteration}_defer.pkl')
            approx.load(f'{save_path}/extra/{args.iteration}_defer.pkl')
        if data_tag == 'defer':
            interpolate = lambda points: np.array([approx(np.float64(x)) for x in tqdm(points)])
        elif data_tag == 'defernnr':
            data = np.array([.5 * (p.domain.lower_limit_vector + p.domain.upper_limit_vector) for p in approx.all_partitions()]).astype(np.float128)
            values = np.array([p.f for p in approx.all_partitions()]).astype(np.float128)
    
            graph = annr.VoronoiGraph(cfg.strategy)
            graph.initialize(data, annr.Unbounded())
    
            interpolator = annr.ActiveInterpolator(
                cfg.random_seed + 614, graph, values, cfg.criterion)
            interpolate = interpolator.interpolate
        else:
            raise Exception()
    
    elif data_tag == 'true':
        interpolate = lambda points: np.array([cfg.function(x) for x in tqdm(points)], dtype=np.float128)
    
    else:
        raise Exception(f'Unknown data tag: {data_tag}')
    return interpolate
