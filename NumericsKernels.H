#pragma once
#include "List.H"
#include "vector.H"
#include "scalar.H"
#include "Model.H"

namespace Numerics
{
constexpr int n_dof_q = 3;

inline Foam::List<Foam::scalar> numerical_flux(
    const Foam::List<Foam::scalar>& Q_minus,
    const Foam::List<Foam::scalar>& Q_plus,
    const Foam::List<Foam::scalar>& Qaux_minus,
    const Foam::List<Foam::scalar>& Qaux_plus,
    const Foam::List<Foam::scalar>& p,
    const Foam::vector& n)
{
    Foam::scalar t0 = -Foam::max(Q_minus[0], Q_plus[0]);
    Foam::scalar t1 = Foam::max(0.0, Q_minus[0] + Q_minus[1] + t0);
    Foam::scalar t2 = Foam::max(1.0e-14, Q_minus[1]);
    Foam::scalar t3 = 0.5*n.x();
    Foam::scalar t4 = Foam::max(0.0, Q_plus[0] + Q_plus[1] + t0);
    Foam::scalar t5 = Foam::max(1.0e-14, Q_plus[1]);
    Foam::scalar t6 = 0.5*p[0];
    auto res = Foam::List<Foam::scalar>(3, 0.0);
    res[0] = 0;
    res[1] = Q_minus[2]*t1*t3/t2 + Q_plus[2]*t3*t4/t5;
    res[2] = t3*(Foam::pow(Q_minus[2], 2)*t1/Foam::pow(t2, 2) + Foam::pow(t1, 2)*t6) + t3*(Foam::pow(Q_plus[2], 2)*t4/Foam::pow(t5, 2) + Foam::pow(t4, 2)*t6);
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
    Foam::scalar t0 = Foam::max(Q_minus[0], Q_plus[0]);
    Foam::scalar t1 = -t0;
    Foam::scalar t2 = Foam::max(0.0, Q_minus[0] + Q_minus[1] + t1);
    Foam::scalar t3 = Foam::max(0.0, Q_plus[0] + Q_plus[1] + t1);
    Foam::scalar t4 = Q_minus[2]*t2/Foam::max(1.0e-14, Q_minus[1]);
    Foam::scalar t5 = (1.0 / Foam::pow(Foam::max(1.0e-14, Q_plus[1]), 1));
    Foam::scalar t6 = 0.5*Foam::max(numerics::max_wavespeed(t0, t2, t4, p[0], n.x()), numerics::max_wavespeed(t0, t3, Q_plus[2]*t3*t5, p[0], n.x()));
    Foam::scalar t7 = t6*(-t2 + t3);
    Foam::scalar t8 = 0.5*p[0];
    Foam::scalar t9 = t6*(Q_plus[2]*t3*t5 - t4);
    auto res = Foam::List<Foam::List<Foam::scalar>>(2, Foam::List<Foam::scalar>(3, 0.0));
    res[0][0] = 0;
    res[0][1] = t7;
    res[0][2] = n.x()*(-Foam::pow(Q_plus[1], 2)*t8 + 0.5*p[0]*Foam::pow(t3, 2)) + t9;
    res[1][0] = 0;
    res[1][1] = -t7;
    res[1][2] = n.x()*(Foam::pow(Q_minus[1], 2)*t8 - Foam::pow(t2, 2)*t8) - t9;
    return res;
}


inline Foam::List<Foam::scalar> local_max_abs_eigenvalue(
    const Foam::List<Foam::scalar>& Q,
    const Foam::List<Foam::scalar>& Qaux,
    const Foam::List<Foam::scalar>& p,
    const Foam::vector& n)
{
    auto res = Foam::List<Foam::scalar>(1, 0.0);
    res[0] = numerics::max_wavespeed(Q[0], Q[1], Q[2], p[0], n.x());
    return res;
}

} // namespace Numerics