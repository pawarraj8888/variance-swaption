"""
Variance Swaption Pricing
Advanced Equity Derivatives - Homework 3 (Final Project)
Raj Pawar, NYU MFE

Setup:
  xi0 = 0.04 = (20%)^2  flat initial forward variance
  K   = 0.04            ATM strike
  T1  = 0.5y            swaption expiry
  T2  = 1.0y            VS maturity
  r   = 1%              discount rate
  Bergomi 2F params: Table 7.1 Set I
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.integrate import quad

xi0=0.04; K=0.04; T1=0.5; T2=1.0; r=0.01
nu=1.50; theta=0.312; k1=2.63; k2=0.42; rho12=-0.70

print("="*65)
print("Variance Swaption Pricing -- Bergomi Two-Factor Model")
print("="*65)
print(f"xi0={xi0}  K={K}  T1={T1}  T2={T2}  r={r}")
print(f"nu={nu}  theta={theta}  k1={k1}  k2={k2}  rho12={rho12}")
print()

def black_call(F,K,sigma,T):
    if sigma<=0: return max(F-K,0)*np.exp(-r*T)
    d1=(np.log(F/K)+0.5*sigma**2*T)/(sigma*np.sqrt(T))
    d2=d1-sigma*np.sqrt(T)
    return np.exp(-r*T)*(F*norm.cdf(d1)-K*norm.cdf(d2))

# ── METHOD B: Black-Scholes ───────────────────────────────────
# Bergomi eq 7.41: sigma_V = 2*hat_nu
# hat_nu^2 = (1/T1)*integral_0^T1 nu^2_{T1T2}(t) dt
# nu^2_{T1T2}(t) = (nu*xi0/(T2-T1))^2 * [theta^2*I1^2 + (1-theta)^2*I2^2
#                                         + 2*theta*(1-theta)*rho12*I1*I2]
# Ii = (exp(-ki*(T1-t)) - exp(-ki*(T2-t))) / ki

def nu_sq(t):
    I1=(np.exp(-k1*(T1-t))-np.exp(-k1*(T2-t)))/k1
    I2=(np.exp(-k2*(T1-t))-np.exp(-k2*(T2-t)))/k2
    return (nu*xi0/(T2-T1))**2*(theta**2*I1**2+(1-theta)**2*I2**2
                                +2*theta*(1-theta)*rho12*I1*I2)

int_nu_sq,_=quad(nu_sq,0,T1)
hat_nu=np.sqrt(int_nu_sq/T1)
sigma_V=2*hat_nu
price_bs=black_call(xi0,K,sigma_V,T1)

print("-"*65)
print("METHOD B: Black-Scholes (Bergomi eq 7.41 approximation)")
print("-"*65)
print(f"  Integral nu^2 dt  = {int_nu_sq:.8f}")
print(f"  hat_nu            = {hat_nu*100:.4f}%")
print(f"  sigma_V = 2*hat_nu= {sigma_V*100:.4f}%")
print(f"  BS Price          = {price_bs:.8f} (variance units)")
print(f"  BS Price          = {price_bs/xi0*100:.4f}% of fwd variance")
print()

# ── METHOD A: Monte Carlo ─────────────────────────────────────
# Simulate xi^u_{T1} = xi0 * exp(nu*[theta*J1(u)+(1-theta)*J2(u)] - 0.5*nu^2*QV(u))
# Ji(u) = integral_0^T1 exp(-ki*(u-t)) dWi_t
# fwd_var = 1/(T2-T1) * integral_T1^T2 xi^u_{T1} du
# price = exp(-r*T1) * E[max(fwd_var - K, 0)]

print("-"*65)
print("METHOD A: Monte Carlo (Bergomi 2F)")
print("-"*65)
print("  Running... (200k paths)")

np.random.seed(42)
n_paths=200000; n_steps=100; n_u=60
dt=T1/n_steps
u_grid=np.linspace(T1,T2,n_u+1)

L=np.array([[1.0,0.0],[rho12,np.sqrt(1-rho12**2)]])

J1=np.zeros((n_paths,n_u+1))
J2=np.zeros((n_paths,n_u+1))
QV=np.zeros((n_paths,n_u+1))

for step in range(n_steps):
    t=step*dt
    Z=np.random.standard_normal((n_paths,2))
    dW=(Z@L.T)*np.sqrt(dt)
    dW1=dW[:,0]; dW2=dW[:,1]
    mask=u_grid>t+1e-12
    e1=np.where(mask,np.exp(-k1*(u_grid-t)),0.0)
    e2=np.where(mask,np.exp(-k2*(u_grid-t)),0.0)
    J1+=np.outer(dW1,e1)
    J2+=np.outer(dW2,e2)
    QV+=((theta*e1)**2+((1-theta)*e2)**2+2*theta*(1-theta)*rho12*e1*e2)*dt

log_xi=nu*(theta*J1+(1-theta)*J2)-0.5*nu**2*QV
xi_T1=xi0*np.exp(log_xi)
fwd_var=np.trapezoid(xi_T1,u_grid,axis=1)/(T2-T1)
payoff=np.maximum(fwd_var-K,0.0)
price_mc=np.exp(-r*T1)*np.mean(payoff)
se=np.exp(-r*T1)*np.std(payoff)/np.sqrt(n_paths)

try:
    iv_mc=brentq(lambda s: black_call(xi0,K,s,T1)-price_mc,1e-6,50.0)
except:
    iv_mc=float('nan')

print(f"  E[fwd_var]        = {np.mean(fwd_var):.6f} (check: should be ~{xi0})")
print(f"  MC Price          = {price_mc:.8f} (variance units)")
print(f"  95% CI            = [{price_mc-1.96*se:.8f}, {price_mc+1.96*se:.8f}]")
print(f"  Std error         = {se:.8f}")
print(f"  Implied vol       = {iv_mc*100:.4f}%")
print(f"  MC Price          = {price_mc/xi0*100:.4f}% of fwd variance")
print()

print("="*65)
print("SUMMARY")
print("="*65)
print(f"  ATM Variance Swaption: K={K}, T1={T1}y, T2={T2}y")
print()
print(f"  {'Method':<38} {'Price':>10} {'Impl Vol':>10}")
print(f"  {'-'*38} {'-'*10} {'-'*10}")
print(f"  {'B: Black (Bergomi 7.41 approx)':<38} {price_bs:>10.6f} {sigma_V*100:>9.4f}%")
print(f"  {'A: Monte Carlo (Bergomi 2F)':<38} {price_mc:>10.6f} {iv_mc*100:>9.4f}%")
print()
print(f"  Difference MC-BS : {price_mc-price_bs:+.6f}")
print(f"  Relative diff    : {(price_mc-price_bs)/price_bs*100:+.2f}%")
print()
print("Interpretation:")
print(f"  The BS approximation uses sigma_V=2*hat_nu={sigma_V*100:.1f}%,")
print(f"  a linearisation valid for small nu. At nu={nu*100:.0f}% this severely")
print(f"  underestimates the true price because the forward VS variance is a")
print(f"  basket of lognormals — Jensen's inequality and the non-lognormality")
print(f"  of the basket push the MC price ({price_mc:.6f}) well above the")
print(f"  BS price ({price_bs:.6f}).")
print(f"  The MC implied vol ({iv_mc*100:.1f}%) captures the full vol-of-vol")
print(f"  effect that the 7.41 approximation misses at high nu.")
