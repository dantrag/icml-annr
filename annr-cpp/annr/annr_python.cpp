#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

#include "algo/ActiveInterpolator.h"

namespace py = boost::python;
namespace numpy = boost::python::numpy;

void copy_numpy_data_float(const std::string &dtype, const numpy::ndarray &array, size_t size, ftype *out) {
    if (dtype == "float32") {
        assert(sizeof(float) == 4);
        float *array_data = reinterpret_cast<float *>(array.get_data());
        std::transform(array_data, array_data + size, out, [](float a) -> ftype {return static_cast<ftype>(a);});
    } else if (dtype == "float64") {
        assert(sizeof(double) == 8);
        double *array_data = reinterpret_cast<double *>(array.get_data());
        std::transform(array_data, array_data + size, out, [](double a) -> ftype {return static_cast<ftype>(a);});
    } else if (dtype == "float128") {
        assert(sizeof(long double) == 16);
        long double *array_data = reinterpret_cast<long double *>(array.get_data());
        std::transform(array_data, array_data + size, out, [](long double a) -> ftype {return static_cast<ftype>(a);});
    } else {
        throw std::runtime_error("Unknown dtype: " + dtype);
    }
}

dmatrix numpy_to_dmatrix(const numpy::ndarray &array) {
    assert(array.get_nd() == 2);
    auto shape = array.get_shape();
    dmatrix res(shape[1], shape[0]);
    std::string dtype(py::extract<char const *>(py::str(array.get_dtype())));
    copy_numpy_data_float(dtype, array, shape[0] * shape[1], res.data());
    return res;
}

dvector numpy_to_dvector(const numpy::ndarray &_array) {
    const numpy::ndarray &array = _array.squeeze();
    assert(array.get_nd() == 1);
    auto shape = array.get_shape();
    dvector res(shape[0]);
    std::string dtype(py::extract<char const *>(py::str(array.get_dtype())));
    copy_numpy_data_float(dtype, array, shape[0], res.data());
    return res;
}

vec<ftype> numpy_to_vector(const numpy::ndarray &_array) {
    const numpy::ndarray &array = _array.squeeze();
    assert(array.get_nd() == 1);
    auto shape = array.get_shape();
    vec<ftype> res(shape[0]);
    std::string dtype(py::extract<char const *>(py::str(array.get_dtype())));
    copy_numpy_data_float(dtype, array, shape[0], res.data());
    return res;
}

numpy::ndarray vector_to_numpy(const vec<ftype> &data) {
    // At the moment, always saves to np.float128
    Py_intptr_t shape[1] = { static_cast<Py_intptr_t>(data.size()) };
    numpy::ndarray result = numpy::zeros(1, shape, numpy::dtype(py::object("float128")));
    std::transform(data.begin(), data.end(), reinterpret_cast<long double*>(result.get_data()),
                   [](ftype a) {return static_cast<long double>(a);});
    return result;
}

numpy::ndarray veci_to_numpy(const vec<int> &data) {
    // At the moment, always saves to np.float128
    Py_intptr_t shape[1] = { static_cast<Py_intptr_t>(data.size()) };
    numpy::ndarray result = numpy::zeros(1, shape, numpy::dtype::get_builtin<int>());
    std::copy(data.begin(), data.end(), reinterpret_cast<int*>(result.get_data()));
    return result;
}

numpy::ndarray dvector_to_numpy(const dvector &data) {
    return vector_to_numpy(vec<ftype>(data.begin(), data.end()));
}

numpy::ndarray dmatrix_to_numpy(const dmatrix &data) {
    // At the moment, always saves to np.float128
    Py_intptr_t shape[2] = { static_cast<Py_intptr_t>(data.cols()), static_cast<Py_intptr_t>(data.rows()) };
    numpy::ndarray result = numpy::zeros(2, shape, numpy::dtype(py::object("float128")));
    std::transform(data.data(), data.data() + data.size(), reinterpret_cast<long double*>(result.get_data()),
                   [](ftype a) {return static_cast<long double>(a);});
    return result;
}

class WrapperFuncs {
public:
    static ptr<BoundingBox> BoundingBox_constructor(const numpy::ndarray &min, const numpy::ndarray &max) {
        return std::make_shared<BoundingBox>(numpy_to_dvector(min), numpy_to_dvector(max));
    }

    static void BoundingBox_constructor_wrapper(py::object &self, const numpy::ndarray &min, const numpy::ndarray &max) {
        auto constructor = py::make_constructor(&BoundingBox_constructor);
        constructor(self, min, max);
    }

    static numpy::ndarray BoundingBox_lower_getter(BoundingBox *self) {
        return dvector_to_numpy(self->get_mn());
    }

    static numpy::ndarray BoundingBox_upper_getter(BoundingBox *self) {
        return dvector_to_numpy(self->get_mx());
    }

    static ptr<VoronoiGraph> VoronoiGraph_constructor(RayStrategyType strategy) {
        return std::make_shared<VoronoiGraph>(strategy, DataType::EUCLIDEAN);
    }

    static void VoronoiGraph_constructor_wrapper(py::object &self, RayStrategyType strategy) {
        auto constructor = py::make_constructor(&VoronoiGraph_constructor);
        constructor(self, strategy);
    }

    static void VoronoiGraph_initialize(VoronoiGraph *self, const numpy::ndarray &points, const ptr<Bounds> &bounds) {
        self->initialize(numpy_to_dmatrix(points), bounds);
    }

    static numpy::ndarray VoronoiGraph_get_data(VoronoiGraph *self) {
        return dmatrix_to_numpy(self->get_data());
    }

    static ptr<ActiveInterpolator> ActiveInterpolator_constructor(
            int seed, const ptr<VoronoiGraph> &graph, const numpy::ndarray &f_values, const ptr<Criterion> &criterion) {
        return std::make_shared<ActiveInterpolator>(seed, graph, numpy_to_vector(f_values), criterion);
    }

    static void ActiveInterpolator_constructor_wrapper(
            py::object &self, int seed, const ptr<VoronoiGraph> &graph, const numpy::ndarray &f_values,
            const ptr<Criterion> &criterion) {
        auto constructor = py::make_constructor(&ActiveInterpolator_constructor);
        constructor(self, seed, graph, f_values, criterion);
    }

//    static VoronoiGraph::Polytope ActiveInterpolator_search_simplex(
    static py::tuple ActiveInterpolator_search_simplex(
            ActiveInterpolator *self, const numpy::ndarray &inits, int n_steps, int tape_size) {
//        VoronoiGraph::Polytope p = self->search_simplex(numpy_to_dmatrix(inits), n_steps, tape_size, nullptr);
//        return py::make_tuple(p);
        dmatrix barycenters;
        VoronoiGraph::Polytope p = self->search_simplex(numpy_to_dmatrix(inits), n_steps, tape_size, &barycenters);
        return py::make_tuple(p, dmatrix_to_numpy(barycenters));
    }

    static void ActiveInterpolator_insert_point(ActiveInterpolator *self, const numpy::ndarray &point, ftype value) {
        self->insert_point(numpy_to_dvector(point), value);
    }

    static numpy::ndarray ActiveInterpolator_interpolate(ActiveInterpolator *self, const numpy::ndarray &queries) {
        return vector_to_numpy(self->interpolate(numpy_to_dmatrix(queries)));
    }

    static numpy::ndarray ActiveInterpolator_get_values(ActiveInterpolator *self) {
        return vector_to_numpy(self->get_values());
    }

    static numpy::ndarray Polytope_dual_getter(VoronoiGraph::Polytope *self) {
        return veci_to_numpy(self->dual);
    }

    static numpy::ndarray Polytope_ref_getter(VoronoiGraph::Polytope *self) {
        return dvector_to_numpy(self->ref);
    }

    static bool Bounds_contains(Bounds *self, const numpy::ndarray &ref) {
        return self->contains(numpy_to_dvector(ref));
    }

    static ftype Bounds_max_length(Bounds *self, const numpy::ndarray &ref, const numpy::ndarray &u) {
        return self->max_length(numpy_to_dvector(ref), numpy_to_dvector(u));
    }

    static numpy::ndarray BoundingSphere_radius_getter(BoundingSphere *self) {
        return dvector_to_numpy(self->get_center());
    }

};


BOOST_PYTHON_MODULE(annr) {
    numpy::initialize();

    py::enum_<RayStrategyType>("RayStrategyType")
            .value("BRUTE_FORCE", BRUTE_FORCE)
            .value("BIN_SEARCH", BIN_SEARCH)
//            .value("BRUTE_FORCE_GPU", BRUTE_FORCE_GPU)
            ;

    py::class_<Bounds, boost::noncopyable>("Bounds", py::no_init)
            .def("contains", &WrapperFuncs::Bounds_contains)
            .def("max_length", &WrapperFuncs::Bounds_max_length);
    py::class_<Unbounded, py::bases<Bounds>>("Unbounded", py::init<>());
    py::class_<BoundingBox, py::bases<Bounds>>("BoundingBox", py::init<int, ftype>())
            .def("__init__", &WrapperFuncs::BoundingBox_constructor_wrapper)
            .add_property("lower", &WrapperFuncs::BoundingBox_lower_getter)
            .add_property("upper", &WrapperFuncs::BoundingBox_upper_getter);
    py::class_<BoundingSphere, py::bases<Bounds>>("BoundingSphere", py::init<int, ftype>())
            .add_property("center", &WrapperFuncs::BoundingSphere_radius_getter)
            .add_property("radius", &BoundingSphere::get_radius);

    py::class_<Criterion, boost::noncopyable>("Criterion", py::no_init);
    py::class_<CircumradiusCriterion, py::bases<Criterion>>("CircumradiusCriterion", py::init<>());
    py::class_<VarianceCriterion, py::bases<Criterion>>("VarianceCriterion", py::init<>());
    py::class_<CayleyMengerCriterion, py::bases<Criterion>>("CayleyMengerCriterion", py::init<ftype>());
    py::class_<ClippedCayleyMengerCriterion, py::bases<Criterion>>("ClippedCayleyMengerCriterion", py::init<ftype, ftype>());
    py::class_<SlopeCriterion, py::bases<Criterion>>("SlopeCriterion", py::init<>());

    py::class_<VoronoiGraph::Polytope>("Polytope", py::no_init)
            .add_property("dual", &WrapperFuncs::Polytope_dual_getter)
            .add_property("ref", &WrapperFuncs::Polytope_ref_getter)
            ;

    py::class_<VoronoiGraph>("VoronoiGraph", py::no_init)
            .def("__init__", &WrapperFuncs::VoronoiGraph_constructor_wrapper)
            .def("initialize", &WrapperFuncs::VoronoiGraph_initialize,
                 py::with_custodian_and_ward_postcall<1, 2, py::with_custodian_and_ward_postcall<1, 3>>())
            .def("get_data", &WrapperFuncs::VoronoiGraph_get_data)
            ;

    py::class_<ActiveInterpolator>("ActiveInterpolator", py::no_init)
            .def("__init__", &WrapperFuncs::ActiveInterpolator_constructor_wrapper,
                 py::return_internal_reference<3, py::return_internal_reference<4>>())
            .def("search_simplex", &WrapperFuncs::ActiveInterpolator_search_simplex)
            .def("insert_point", &WrapperFuncs::ActiveInterpolator_insert_point)
            .def("interpolate", &WrapperFuncs::ActiveInterpolator_interpolate)
            .def("get_values", &WrapperFuncs::ActiveInterpolator_get_values)
            ;
}
