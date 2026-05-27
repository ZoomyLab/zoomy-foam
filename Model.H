#pragma once
#include "List.H"
#include "vector.H"
#include "scalar.H"
#include "word.H"

namespace Model
{
constexpr int n_dof_q    = 3;
constexpr int n_dof_qaux = 0;
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
    auto res = Foam::List<Foam::List<Foam::scalar>>(3, Foam::List<Foam::scalar>(1, 0.0));
    res[0][0] = 0;
    res[1][0] = Q[2];
    res[2][0] = 0.5*p[0]*Foam::pow(Q[1], 2) + Foam::pow(Q[2], 2)/Q[1];
    return res;
}


inline Foam::List<Foam::List<Foam::scalar>> nonconservative_matrix_x(
    const Foam::List<Foam::scalar>& Q,
    const Foam::List<Foam::scalar>& Qaux,
    const Foam::List<Foam::scalar>& p)
{
    auto res = Foam::List<Foam::List<Foam::scalar>>(3, Foam::List<Foam::scalar>(3, 0.0));
    res[0][0] = 0;
    res[0][1] = 0;
    res[0][2] = 0;
    res[1][0] = 0;
    res[1][1] = 0;
    res[1][2] = 0;
    res[2][0] = p[0]*Q[1];
    res[2][1] = 0;
    res[2][2] = 0;
    return res;
}


inline Foam::List<Foam::List<Foam::scalar>> quasilinear_matrix_x(
    const Foam::List<Foam::scalar>& Q,
    const Foam::List<Foam::scalar>& Qaux,
    const Foam::List<Foam::scalar>& p)
{
    Foam::scalar t0 = p[0]*Q[1];
    auto res = Foam::List<Foam::List<Foam::scalar>>(3, Foam::List<Foam::scalar>(3, 0.0));
    res[0][0] = 0;
    res[0][1] = 0;
    res[0][2] = 0;
    res[1][0] = 0;
    res[1][1] = 0;
    res[1][2] = 1;
    res[2][0] = t0;
    res[2][1] = 1.0*t0 - Foam::pow(Q[2], 2)/Foam::pow(Q[1], 2);
    res[2][2] = 2*Q[2]/Q[1];
    return res;
}


inline Foam::List<Foam::List<Foam::scalar>> eigenvalues(
    const Foam::List<Foam::scalar>& Q,
    const Foam::List<Foam::scalar>& Qaux,
    const Foam::List<Foam::scalar>& p,
    const Foam::vector& n)
{
    Foam::scalar t0 = Q[2]*n.x()/Q[1];
    Foam::scalar t1 = n.x()*Foam::pow(p[0]*Foam::pow(Q[1], 5), 1.0/2.0)/Foam::pow(Q[1], 2);
    auto res = Foam::List<Foam::List<Foam::scalar>>(3, Foam::List<Foam::scalar>(1, 0.0));
    res[0][0] = 0.0;
    res[1][0] = t0 - t1;
    res[2][0] = t0 + t1;
    return res;
}


inline Foam::List<Foam::List<Foam::scalar>> source(
    const Foam::List<Foam::scalar>& Q,
    const Foam::List<Foam::scalar>& Qaux,
    const Foam::List<Foam::scalar>& p)
{
    auto res = Foam::List<Foam::List<Foam::scalar>>(3, Foam::List<Foam::scalar>(1, 0.0));
    res[0][0] = 0;
    res[1][0] = 0;
    res[2][0] = -p[0]*Foam::pow(Q[1], -2.3333333333333335)*Q[2]*Foam::pow(p[1], 2)*Foam::mag(Q[2]);
    return res;
}

} // namespace Model