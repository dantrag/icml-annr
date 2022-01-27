#include <queue>
#include <chrono>
using namespace std::chrono;

#include "ActiveInterpolator.h"


ftype ConstCriterion::eval(const VoronoiGraph &graph, const VoronoiGraph::Polytope &p,
                           const vec<ftype> &f_values) const {
    return 0;
}

ActiveInterpolator::ActiveInterpolator(int seed, const ptr<VoronoiGraph> &graph, const vec<ftype> &f_values,
                                       const ptr<Criterion> &criterion):
        re(seed), graph(graph), f_values(f_values), criterion(criterion) {
}

VoronoiGraph::Polytope ActiveInterpolator::search_simplex(const dmatrix &inits, int n_steps, int tape_size,
                                                          dmatrix *barycenters_ptr) {
    ensure(graph->get_data_size() == f_values.size(), "Number of points not coinciding"
                                                      " with the number of function values");
    int dim = graph->get_data_dim();

    #pragma omp parallel
    re.fix_random_engines();

    using Polytope = VoronoiGraph::Polytope;
    int n_inits = inits.cols();

    auto time0 = high_resolution_clock::now();

    vec<VoronoiGraph::Polytope> vertices = graph->retrieve_vertices(inits, re);
    {
        vec<Polytope> new_vertices;
        for (const Polytope &p: vertices) {
            if (!p.is_none()) {
                new_vertices.push_back(p);
            }
        }
        vertices = new_vertices;
        n_inits = vertices.size();
//        if (n_inits < inits.cols()) {
//            std::cout << "Number of starting vertices: " << n_inits << " out of " << inits.cols() << std::endl;
//        }
    }

    ensure(!vertices.empty(), "Provided inits resulted in an empty list of initial vertices");

    auto time1 = high_resolution_clock::now();


    auto cmp = [](const HeapElement &a, const HeapElement &b) {return a.second > b.second;};
    set_t<IndexSet> visited;
    std::priority_queue<HeapElement, vec<HeapElement>, decltype(cmp)> heap(cmp);  // heap is used to pop smallest elements

    for (int i_step = 0; i_step <= n_steps; i_step++) {
        // 1 evaluate
        vec<ftype> value_vec(n_inits);
        #pragma omp parallel for
        for (int i = 0; i < n_inits; i++) {
            value_vec[i] = criterion->eval(*graph, vertices[i], f_values);
        }
        for (int i = 0; i < n_inits; i++) {
            if (visited.find(vertices[i].dual) == visited.end()) {
                visited.insert(vertices[i].dual);
                heap.push(std::make_pair(
                        std::make_shared<Polytope>(vertices[i]),
                        value_vec[i]));
                if (heap.size() > tape_size) {
                    heap.pop();
                }
            }
        }
        if (i_step == n_steps) break;

        // 2 find next
        vec<int> indices(n_inits);
        for (int i = 0; i < n_inits; i++) {
            indices[i] = re.current().rand_int(dim + 1);
        }
        vec<Polytope> next = graph->get_neighbors(vertices, indices, re);
        #pragma omp parallel for
        for (int i = 0; i < n_inits; i++) {
            if (!next[i].is_none()) {
                vertices[i] = next[i];
            }
        }

    }

    auto time2 = high_resolution_clock::now();

    ensure(!heap.empty(), "The final list of visited vertices is empty!");

    vec<Polytope> tape;
    while (!heap.empty()) {
        const HeapElement &he = heap.top();
        tape.push_back(*he.first);
        heap.pop();
    }

    std::reverse(tape.begin(), tape.end());

    if (barycenters_ptr) {
        (*barycenters_ptr).resize(dim, tape.size());
        #pragma omp parallel for
        for (int i = 0; i < tape.size(); i++) {
            IndexSet &dual = tape[i].dual;
            dvector bc = dvector::Zero(dim, 1);
            for (int idx : dual) {
                bc += graph->get_data().col(idx);
            }
            bc /= dual.size();
            barycenters_ptr->col(i) = bc;
        }
    }

    auto time3 = high_resolution_clock::now();

//    std::cout << "IN: "
//            << duration<double>(time1 - time0).count() << " "
//            << duration<double>(time2 - time1).count() << " "
//            << duration<double>(time3 - time2).count() << std::endl;

    return tape[0];
}

void ActiveInterpolator::insert_point(const dvector &point, ftype value) {
    dmatrix points(graph->get_data_dim(), 1);
    points.col(0) = point;
    graph->insert(points);
    f_values.push_back(value);
}

vec<ftype> ActiveInterpolator::interpolate(const dmatrix &queries) const {
    int n_queries = queries.cols();
    vec<ftype> result(n_queries);

    if (n_queries == 1) {
        result[0] = f_values[graph->get_containing_voronoi_cell(queries.col(0))];
    } else {
        my_tqdm bar(n_queries);
        #pragma omp parallel for
        for (int i = 0; i < n_queries; i++) {
            bar.atomic_iteration();
            result[i] = f_values[graph->get_containing_voronoi_cell(queries.col(i))];
        }
        bar.bar().finish();
    }

    return result;
}

const vec<ftype> &ActiveInterpolator::get_values() const {
    return f_values;
}

ftype CircumradiusCriterion::eval(const VoronoiGraph &graph, const VoronoiGraph::Polytope &p,
                                  const vec<ftype> &f_values) const {
    return (p.ref - graph.get_data().col(p.dual[0])).norm();
}

ftype
VarianceCriterion::eval(const VoronoiGraph &graph, const VoronoiGraph::Polytope &p, const vec<ftype> &f_values) const {
    int n = p.dual.size();
    vec<ftype> local_values(n);
    for (int i = 0; i < n; i++) {
        local_values[i] = f_values[p.dual[i]];
    }
    ftype mean = 0;
    for (ftype v: local_values) {
        mean += v;
    }
    mean /= n;
    ftype variance = 0;
    for (ftype v: local_values) {
        variance += sqr(v - mean);
    }
    return variance / n;
}

ftype cm_volume(const dynmatrix &simplex_points) {
    int dim = int(simplex_points.cols()) - 1;
    dynmatrix B_hat = dynmatrix::Zero(dim + 2, dim + 2);
    for (int i = 0; i <= dim; i++) {
        B_hat(i + 1, 0) = 1;
        B_hat(0, i + 1) = 1;
        for (int j = i + 1; j <= dim; j++) {
            B_hat(i + 1, j + 1) = B_hat(j + 1, i + 1) =
                    (simplex_points.col(i) - simplex_points.col(j)).squaredNorm();
        }
    }
    return math::sqrt(math::abs(B_hat.determinant()));
}

CayleyMengerCriterion::CayleyMengerCriterion(ftype lambda) : Criterion(), lambda(lambda) {
}

ftype CayleyMengerCriterion::eval(const VoronoiGraph &graph, const VoronoiGraph::Polytope &p,
                                  const vec<ftype> &f_values) const {
    ensure(graph.is_vertex(p), "simplex dimensionality should be ambient_dim");
    int dim = graph.get_data_dim();
    dynmatrix simplex_points(dim + 1, dim + 1);
    for (int i = 0; i <= dim; i++) {
        int idx = p.dual[i];
        simplex_points.block(0, i, dim, 1) = graph.get_data().col(idx);
        simplex_points(dim, i) = lambda * f_values[idx];
    }

    return cm_volume(simplex_points);
}

ClippedCayleyMengerCriterion::ClippedCayleyMengerCriterion(ftype lambda, ftype clipping_angle):
        Criterion(), lambda(lambda), lipschitz(1 / math::cos(clipping_angle)) { }

ftype ClippedCayleyMengerCriterion::eval(const VoronoiGraph &graph, const VoronoiGraph::Polytope &p,
                                         const vec<ftype> &f_values) const {
    ensure(graph.is_vertex(p), "simplex dimensionality should be ambient_dim");
    int dim = graph.get_data_dim();
    dynmatrix simplex_points = dynmatrix::Zero(dim + 1, dim + 1);
    for (int i = 0; i <= dim; i++) {
        int idx = p.dual[i];
        simplex_points.block(0, i, dim, 1) = graph.get_data().col(idx);
    }
    ftype bottom = cm_volume(simplex_points);

    for (int i = 0; i <= dim; i++) {
        int idx = p.dual[i];
        simplex_points(dim, i) = lambda * f_values[idx];
    }
    ftype top = cm_volume(simplex_points);

    return std::min(top, bottom * lipschitz);
}

ftype
SlopeCriterion::eval(const VoronoiGraph &graph, const VoronoiGraph::Polytope &p, const vec<ftype> &f_values) const {
    ensure(graph.is_vertex(p), "simplex dimensionality should be ambient_dim");
    int dim = graph.get_data_dim();
    dynmatrix simplex_points = dynmatrix::Zero(dim + 1, dim + 1);
    for (int i = 0; i <= dim; i++) {
        int idx = p.dual[i];
        simplex_points.block(0, i, dim, 1) = graph.get_data().col(idx);
    }
    ftype bottom = cm_volume(simplex_points);

    for (int i = 0; i <= dim; i++) {
        int idx = p.dual[i];
        simplex_points(dim, i) = f_values[idx];
    }
    ftype top = cm_volume(simplex_points);

    return top / bottom;
}
