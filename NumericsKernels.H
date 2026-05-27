#pragma once
#include "List.H"
#include "vector.H"
#include "scalar.H"
#include "Model.H"

namespace Numerics
{
constexpr int n_dof_q = 2;

inline Foam::List<Foam::scalar> numerical_flux(
    const Foam::List<Foam::scalar>& Q_minus,
    const Foam::List<Foam::scalar>& Q_plus,
    const Foam::List<Foam::scalar>& Qaux_minus,
    const Foam::List<Foam::scalar>& Qaux_plus,
    const Foam::List<Foam::scalar>& p,
    const Foam::vector& n)
{
    Foam::scalar t0 = 0.5*n.x();
    Foam::scalar t1 = 0.5*Foam::max(numerics::max_wavespeed(Q_minus, Qaux_minus, p, n), numerics::max_wavespeed(Q_plus, Qaux_plus, p, n));
    Foam::scalar t2 = 0.5*p[0];
    auto res = Foam::List<Foam::scalar>(2, 0.0);
    res[0] = Q_minus[1]*t0 + Q_plus[1]*t0 - t1*(-Q_minus[0] + Q_plus[0]);
    res[1] = t0*(Foam::pow(Q_minus[0], 2)*t2 + Foam::pow(Q_minus[1], 2)/Q_minus[0]) + t0*(Foam::pow(Q_plus[0], 2)*t2 + Foam::pow(Q_plus[1], 2)/Q_plus[0]) - t1*(-Q_minus[1] + Q_plus[1]);
    return res;
}


inline Foam::List<Foam::List<Foam::scalar>> numerical_fluctuations(
    const Foam::List<Foam::scalar>& Q_minus,
    const Foam::List<Foam::scalar>& Q_plus,
    const Foam::List<Foam::scalar>& Qaux_minus,
    const Foam::List<Foam::scalar>& Qaux_plus,
    const Foam::List<Foam::scalar>& p,
    const Foam::vector& n)
{
    auto res = Foam::List<Foam::List<Foam::scalar>>(2, Foam::List<Foam::scalar>(2, 0.0));
    res[0][0] = 0;
    res[0][1] = 0;
    res[1][0] = 0;
    res[1][1] = 0;
    return res;
}


inline Foam::List<Foam::scalar> local_max_abs_eigenvalue(
    const Foam::List<Foam::scalar>& Q,
    const Foam::List<Foam::scalar>& Qaux,
    const Foam::List<Foam::scalar>& p,
    const Foam::vector& n)
{
    auto res = Foam::List<Foam::scalar>(1, 0.0);
    res[0] = numerics::max_wavespeed(Q, Qaux, p, n);
    return res;
}

} // namespace Numerics