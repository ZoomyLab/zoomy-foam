#pragma once
#include "List.H"
#include "vector.H"
#include "scalar.H"
#include "word.H"

namespace Model
{
constexpr int n_dof_q    = 2;
constexpr int n_dof_qaux = 2;
constexpr int n_parameters = 2;
constexpr int dimension  = 1;
const Foam::List<Foam::word> map_boundary_tag_to_function_index{ "inflow", "outflow", "wall" };
const Foam::List<Foam::word> parameter_names{ "g", "n_m" };
inline Foam::List<Foam::scalar> default_parameters() { return { 9.81, 0.033 }; }

inline Foam::List<Foam::List<Foam::scalar>> flux_x(
    const Foam::List<Foam::scalar>& Q,
    const Foam::List<Foam::scalar>& Qaux,
    const Foam::List<Foam::scalar>& p)
{
    auto res = Foam::List<Foam::List<Foam::scalar>>(2, Foam::List<Foam::scalar>(1, 0.0));
    res[0][0] = Q[1];
    res[1][0] = 0.5*p[0]*Foam::pow(Q[0], 2) + Foam::pow(Q[1], 2)/Q[0];
    return res;
}


inline Foam::List<Foam::List<Foam::scalar>> nonconservative_matrix_x(
    const Foam::List<Foam::scalar>& Q,
    const Foam::List<Foam::scalar>& Qaux,
    const Foam::List<Foam::scalar>& p)
{
    auto res = Foam::List<Foam::List<Foam::scalar>>(2, Foam::List<Foam::scalar>(2, 0.0));
    res[0][0] = 0;
    res[0][1] = 0;
    res[1][0] = 0;
    res[1][1] = 0;
    return res;
}


inline Foam::List<Foam::List<Foam::scalar>> quasilinear_matrix_x(
    const Foam::List<Foam::scalar>& Q,
    const Foam::List<Foam::scalar>& Qaux,
    const Foam::List<Foam::scalar>& p)
{
    auto res = Foam::List<Foam::List<Foam::scalar>>(2, Foam::List<Foam::scalar>(2, 0.0));
    res[0][0] = 0;
    res[0][1] = 1;
    res[1][0] = 1.0*p[0]*Q[0] - Foam::pow(Q[1], 2)/Foam::pow(Q[0], 2);
    res[1][1] = 2*Q[1]/Q[0];
    return res;
}


inline Foam::List<Foam::List<Foam::scalar>> eigenvalues(
    const Foam::List<Foam::scalar>& Q,
    const Foam::List<Foam::scalar>& Qaux,
    const Foam::List<Foam::scalar>& p,
    const Foam::vector& n)
{
    Foam::scalar t0 = Foam::pow(p[0], 1.0/2.0)*Foam::pow(Foam::pow(Q[0], 5), 1.0/2.0);
    Foam::scalar t1 = n.x()/Foam::pow(Q[0], 2);
    auto res = Foam::List<Foam::List<Foam::scalar>>(2, Foam::List<Foam::scalar>(1, 0.0));
    res[0][0] = t1*(Q[0]*Q[1] - t0);
    res[1][0] = t1*(Q[0]*Q[1] + t0);
    return res;
}


inline Foam::List<Foam::List<Foam::scalar>> source(
    const Foam::List<Foam::scalar>& Q,
    const Foam::List<Foam::scalar>& Qaux,
    const Foam::List<Foam::scalar>& p)
{
    auto res = Foam::List<Foam::List<Foam::scalar>>(2, Foam::List<Foam::scalar>(1, 0.0));
    res[0][0] = 0;
    res[1][0] = -Qaux[1]*p[0]*Q[0] - p[0]*Foam::pow(Q[0], -2.3333333333333335)*Q[1]*Foam::pow(p[1], 2)*Foam::mag(Q[1]);
    return res;
}


inline Foam::List<Foam::scalar> boundary_conditions(
    const int bc_idx,
    const Foam::scalar& time,
    const Foam::vector& X,
    const Foam::scalar& dX,
    const Foam::List<Foam::scalar>& Q,
    const Foam::List<Foam::scalar>& Qaux,
    const Foam::List<Foam::scalar>& p,
    const Foam::vector& n)
{
    if (bc_idx == 0 || bc_idx == 1 || bc_idx == 2) {
        auto res = Foam::List<Foam::scalar>(2, 0.0);
        res[0] = Q[0];
        res[1] = Q[1];
        return res;
    }
    auto default_res = Foam::List<Foam::scalar>(2, 0.0);
    return default_res;
}

} // namespace Model