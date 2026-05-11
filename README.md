# Variance Swaption Pricing — Bergomi Two-Factor Model

**NYU MFE — Advanced Equity Derivatives, Homework 3 (Final Project)**
*Rajvardhan Pawar (rsp9234), May 2026*

---

## Live Dashboard

**[View Dashboard](https://pawarraj8888.github.io/variance-swaption/variance_swaption_dashboard.html)**

---

## Overview

Prices an at-the-money variance swaption under the Bergomi two-factor stochastic volatility model using two methods:

- **Method B** — Black-Scholes with lognormal forward variance approximation (Bergomi eq. 7.41)
- **Method A** — Monte Carlo simulation of the full Bergomi two-factor dynamics

---

## Problem Setup

| Parameter | Value |
|---|---|
| Forward Variance F = xi0 | 0.04 = (20%)^2 |
| Strike K (ATM) | 0.04 |
| Swaption expiry T1 | 0.5 years |
| VS maturity T2 | 1.0 year |
| Discount rate r | 1% continuous |

**Swaption payoff at T1:**

```
max( hat_sigma^2_{T1,T2}(T1) - K , 0 )
```

where `hat_sigma^2_{T1,T2}(T1) = 1/(T2-T1) * integral_{T1}^{T2} xi^u_{T1} du`

---

## Model: Bergomi Two-Factor (Table 7.1, Set I)

| nu | theta | k1 | k2 | rho12 |
|---|---|---|---|---|
| 150% | 0.312 | 2.63 | 0.42 | -70% |

Forward variance dynamics under risk-neutral measure:

```
d xi^T_t / xi^T_t = nu * [ theta * exp(-k1*(T-t)) dW1_t
                          + (1-theta) * exp(-k2*(T-t)) dW2_t ]
corr(dW1, dW2) = rho12
```

---

## Results

| Method | Price (variance units) | Implied Vol |
|---|---|---|
| B: Black-Scholes (Bergomi 7.41) | 0.000671 | 5.97% |
| A: Monte Carlo (Bergomi 2F) | 0.008222 | 74.07% |
| Difference | +0.007551 | +68.1 vol pts |
| Relative difference | +1126% | |

### Key Finding

The two methods differ by a factor of 12. The Bergomi eq. 7.41 approximation `sigma_V = 2*hat_nu` is a first-order linearisation valid only for small nu. At nu = 150% it fails completely because the forward VS variance is an arithmetic average of lognormal forward variances (a basket of lognormals) — not lognormal itself. Jensen's inequality and the basket's non-lognormality push the true price far above the linearised estimate.

The Monte Carlo implied vol of 74.07% is the correct number.

---

## Method B: Black-Scholes

Uses Bergomi eq. 7.41 to approximate the vol of the forward VS variance:

```
sigma_V = 2 * hat_nu_{T1,T2}

hat_nu^2 = (1/T1) * integral_0^{T1} nu^2_{T1,T2}(t) dt

nu^2_{T1,T2}(t) = (nu*xi0/(T2-T1))^2 *
  [ theta^2*I1^2 + (1-theta)^2*I2^2 + 2*theta*(1-theta)*rho12*I1*I2 ]

Ii(t) = (exp(-ki*(T1-t)) - exp(-ki*(T2-t))) / ki
```

Then prices with Black-76: `Price = exp(-r*T1) * [F*N(d1) - K*N(d2)]`

Results: integral = 0.000446, hat_nu = 2.99%, sigma_V = 5.97%, **Price = 0.000671**

---

## Method A: Monte Carlo

Simulates `xi^u_{T1}` for each maturity u on a grid over [T1, T2]:

```
xi^u_{T1} = xi0 * exp( nu*[theta*J1(u) + (1-theta)*J2(u)] - 0.5*nu^2*QV(u) )
```

where `Ji(u)` are Ito stochastic integrals accumulated at each time step.

Forward VS variance on each path via trapezoidal integration, then discounted expected payoff.

- 200,000 paths, 100 time steps, 61 u-grid points
- Correlated increments via Cholesky decomposition
- **Price = 0.008222**, 95% CI: [0.008148, 0.008296]
- Martingale check: E[fwd_var] = 0.039941 ~ 0.04

---

## Files

```
variance-swaption/
├── var_swaption.py                    # Python pricing code
├── variance_swaption_dashboard.html   # Standalone HTML dashboard
├── hw3_writeup.pdf                    # Full writeup with derivations
└── README.md
```

---

## How to Run

```bash
pip install numpy scipy
python var_swaption.py
```

Output:

```
METHOD B: Black-Scholes
  sigma_V = 5.9726%
  Price   = 0.000671

METHOD A: Monte Carlo
  Price   = 0.008222  +/- 0.000038
  Impl vol = 74.07%
```

---

## References

1. Bergomi, L. (2016). *Stochastic Volatility Modeling*. CRC Press, Chapter 7, pp. 229-232.
2. Black, F. (1976). The pricing of commodity contracts. *Journal of Financial Economics*, 3(1-2), 167-179.
3. Gatheral, J. (2006). *The Volatility Surface: A Practitioner's Guide*. Wiley Finance.
