
#include "../algo/ActiveInterpolator.h"

ftype f(const dvector &x) {
    return 1000 * math::exp(-(sqr(x(0, 0)) + sqr(x(1, 0))) / 10);
}

int main() {
    auto npy = cnpy::npy_load("data/test1kk.npy");
    auto data = npy2matrix(npy);
    int dim = data.rows();
    auto bounds = std::make_shared<BoundingBox>(dim, 1);
    auto criterion = std::make_shared<CayleyMengerCriterion>();

//    auto graph = std::make_shared<VoronoiGraph>(BRUTE_FORCE);
    auto graph = std::make_shared<VoronoiGraph>(BIN_SEARCH);
    graph->initialize(data, bounds);

    vec<ftype> values(data.cols());
    for (int i = 0; i < data.cols(); i++) {
        values[i] = f(data.col(i));
    }

    auto inits = npy2matrix(cnpy::npy_load("data/test1kk" + std::to_string(0) + ".npy"));

    auto interpolator = ActiveInterpolator(240, graph, values, criterion);
    for (int i = 0; i < 100; i++) {
        dmatrix barycenters;
        auto vertex = interpolator.search_simplex(inits, 10, 10, &barycenters);
        inits = barycenters;

        ftype value = f(vertex.ref);

        std::cout << i << ": " << vertex.dual << " " << vertex.ref << std::endl;

        interpolator.insert_point(vertex.ref, value);
    }


    return 0;
}
