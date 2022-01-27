#pragma once

#include "VoronoiGraph.h"

class Criterion {
public:
    virtual ftype eval(const VoronoiGraph &graph, const VoronoiGraph::Polytope &p,
                       const vec<ftype> &f_values) const = 0;
};

class ConstCriterion : public Criterion {
public:
    ftype eval(const VoronoiGraph &graph, const VoronoiGraph::Polytope &p,
               const vec<ftype> &f_values) const override;
};

class CircumradiusCriterion : public Criterion {
public:
    ftype eval(const VoronoiGraph &graph, const VoronoiGraph::Polytope &p,
               const vec<ftype> &f_values) const override;
};

class VarianceCriterion : public Criterion {
public:
    ftype eval(const VoronoiGraph &graph, const VoronoiGraph::Polytope &p, const vec<ftype> &f_values) const override;
};

class CayleyMengerCriterion : public Criterion {
public:
    CayleyMengerCriterion(ftype lambda = 1);
    ftype eval(const VoronoiGraph &graph, const VoronoiGraph::Polytope &p, const vec<ftype> &f_values) const override;
private:
    ftype lambda;
};

class ClippedCayleyMengerCriterion : public Criterion {
public:
    ClippedCayleyMengerCriterion(ftype lambda = 1, ftype clipping_angle = .5 * PI_ftype);
    ftype eval(const VoronoiGraph &graph, const VoronoiGraph::Polytope &p, const vec<ftype> &f_values) const override;
private:
    ftype lambda;
    ftype lipschitz;
};

class SlopeCriterion : public Criterion {
public:
    ftype eval(const VoronoiGraph &graph, const VoronoiGraph::Polytope &p, const vec<ftype> &f_values) const override;
};

class ActiveInterpolator {
public:
    using HeapElement = std::pair<ptr<VoronoiGraph::Polytope>, ftype>;

    ActiveInterpolator(int seed, const ptr<VoronoiGraph> &graph, const vec<ftype> &f_values,
                       const ptr<Criterion> &criterion);

    VoronoiGraph::Polytope search_simplex(const dmatrix &inits, int n_steps, int tape_size,
                                          dmatrix *barycenters_ptr = nullptr);

    void insert_point(const dvector &point, ftype value);

    vec<ftype> interpolate(const dmatrix &queries) const;

    const vec<ftype> &get_values() const;

private:
    RandomEngineMultithread re;

    ptr<VoronoiGraph> graph;
    vec<ftype> f_values;
    ptr<Criterion> criterion;

};
