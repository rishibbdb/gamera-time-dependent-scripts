import os
import sys
sys.path.append('/lustre/hawcz01/scratch/userspace/rbabu/J1809-Pass5.1/gamera/gam/GAMERA/lib') 
sys.path.append('rad_field_richard_tuffs')
import gappa as gp
import numpy as np
import astropy.constants as c
import astropy.units as u
import emcee
from multiprocessing import Pool
from scipy.io import readsav
from astropy.table import Table
from clean_model import *
import matplotlib.pyplot as plt

fu = gp.Utils()
bins = 200
eV_to_TeV = 1e-12
TeV_to_erg = 1.60218

known_properties = {
    #PSR J1813-1749 pulsar properties
    'distance': 7e3, #pc
    'longi': 32.638, #deg 
    'lati' : 0.527,  #deg in GCS
    'e_dot' : 9.8e46, #erg/sec
    'char_age' : 43000, # yrs                                                                                                                                  
    'P' :  38.522e-3,  #sec                                                                                                                              
    'P_dot' :  1.4224e-14, #/sec/sec  
    'br_ind' : 3.0, #pulsar braking index  (Assumed)
    'ebreak' : None, #TeV
    'alpha0' : None
}


P0 = 23.7e-03
theta = 0.05
alpha = 1.82
ecut = 3.17
b_now = 2.5e-06
model_parameters_fit_pwn = {
    #fit parameters, lowerbound, upperbound
    'b_now' : [b_now,1e-6,20e-6], #Gauss
    'P0' : [P0,18e-3,40e-3], #sec
    'theta' : [theta,0.01,1.0], #converstion frac fron pulsar lum to electrons
    'ecut' : [ecut,1,3.2], #log10(E/TeV)
    'alpha' : [alpha,1.1,3.0],
    'density' : [1,1,100]#/cm3 #low particle density region
}

define_model_parameters_scales(model_parameters_fit_pwn)
model_parameters_pwn = model_parameters_fit_pwn

for key in model_parameters_pwn.keys():
    print(key,model_parameters_pwn[key])

from data import DataDict


hessData = DataDict["HESS"]
fermiData = DataDict["Fermi"]
chandraData = DataDict["Chandra"]
fpmaData = DataDict["NuSTAR"]
km2aData = DataDict["KM2A"]
tibetData = DataDict["Tibet"]
hawcData = DataDict["HAWC"]

relic_eref = np.concatenate((hawcData['energy'],hessData[hessData['is_ul']==False]['energy'], fermiData[fermiData['is_ul']==False]['energy'], fpmaData[fpmaData['is_ul']==False]['energy'], km2aData['energy'], tibetData['energy'], chandraData['energy']))
relic_sed = np.concatenate((hawcData['flux'],hessData[hessData['is_ul']==False]['flux'], fermiData[fermiData['is_ul']==False]['flux'], fpmaData[fpmaData['is_ul']==False]['flux'], km2aData['flux'], tibetData['flux'], chandraData['flux']))
relic_sed_err_lo = np.concatenate((hawcData['flux_error_lo'],hessData[hessData['is_ul']==False]['flux_error_lo'], fermiData[fermiData['is_ul']==False]['flux_error_lo'], fpmaData[fpmaData['is_ul']==False]['flux_error_lo'], km2aData['flux_error_lo'], tibetData['flux_error_lo'], chandraData['flux_error_lo']))
relic_sed_err_hi = np.concatenate((hawcData['flux_error_hi'],hessData[hessData['is_ul']==False]['flux_error_hi'], fermiData[fermiData['is_ul']==False]['flux_error_hi'], fpmaData[fpmaData['is_ul']==False]['flux_error_hi'], km2aData['flux_error_hi'], tibetData['flux_error_hi'], chandraData['flux_error_hi']))

pwn_x = relic_eref
pwn_y = relic_sed
pwn_yerr_lo = relic_sed_err_lo
pwn_yerr_hi = relic_sed_err_hi


pwn_fit_parameters = ['theta','b_now','P0','ecut','alpha',]
pwn_labels = ["Theta (fraction)", "B(Now)(G)", "P0 (s)", "log10(Ecut/TeV)", "spectral index (wind)"]
pwn_fit = pwn_emission_singlezone(bins, known_properties, model_parameters_pwn)
pwn_pars = []
pwn_lower_bounds = []
pwn_upper_bounds = []
for par in pwn_fit_parameters:
    pwn_pars.append(model_parameters_pwn[par][0]*model_parameters_pwn[par][-1])
    pwn_lower_bounds.append(model_parameters_pwn[par][1]*model_parameters_pwn[par][-1])
    pwn_upper_bounds.append(model_parameters_pwn[par][2]*model_parameters_pwn[par][-1])
pwn_bounds = (pwn_lower_bounds, pwn_upper_bounds)
def proxy(cls_instance):
    return cls_instance.log_prob_pwn


pwn_fit = pwn_emission_singlezone(bins, known_properties, model_parameters_pwn)

print(f"self.P0       = {pwn_fit.P0:.3e} s    (should be ~30e-3)")
print(f"self.P        = {pwn_fit.P:.3e} s    (should be ~38.5e-3)")
print(f"self.P_dot    = {pwn_fit.P_dot:.3e}   (should be ~1.4e-14)")
print(f"self.br_ind   = {pwn_fit.br_ind}")
print(f"self.theta    = {pwn_fit.theta:.3e}")
print(f"self.e_dot    = {pwn_fit.e_dot:.3e} erg/s")
print()
print(f"t0            = {pwn_fit.calculate_t0():.3e} yr")
print(f"true_age      = {pwn_fit.calculate_true_age():.3e} yr")
print(f"br_ind_factor = {pwn_fit.calculate_br_ind_power_factor():.3f}")
print(f"L0            = {pwn_fit.calculate_l0():.3e} erg/s")
print(f"B0            = {pwn_fit.calculate_b0():.3e} G")
print()
# Check luminosity at a few times
for t in [1, 100, 1000, 10000, 43000]:
    print(f"L(t={t:6d} yr) = {pwn_fit.luminosity(t):.3e} erg/s")


print(f"true_age = {pwn_fit.calculate_true_age():.3e} yr")
print(f"twindow_sec = {pwn_fit.calculate_true_age() * yr_to_sec:.3e} s")
# Should be ~26700 yr * 3.156e7 s/yr = ~8.4e11 s

t_yr = np.logspace(0, 5, 10)
e_electron = np.logspace(np.log10(gp.m_e/gp.TeV_to_erg), 4, 50) * gp.TeV_to_erg
e_photon = np.logspace(-6, 15, 50) * gp.eV_to_erg


pwn_fit.theta  = model_parameters_pwn['theta'][0]
pwn_fit.b_now  = model_parameters_pwn['b_now'][0]
pwn_fit.P0     = model_parameters_pwn['P0'][0]
pwn_fit.ecut   = model_parameters_pwn['ecut'][0]
pwn_fit.alpha  = model_parameters_pwn['alpha'][0]


pwn_sed_test, fp_test, fr_test = pwn_fit.model_pwn(t_yr, e_electron, e_photon, 1, False)

sp = np.array(fp_test.GetParticleSpectrum())
total_e = np.trapz(sp[:,1]*sp[:,0], sp[:,0])
print(f"Total electron energy: {total_e:.3e} erg  (should be ~1.6e58)")
print(f"Model SED flux range:  {pwn_sed_test[:,1].min():.3e} to {pwn_sed_test[:,1].max():.3e} erg/cm2/s")
print(f"Data flux range:       {pwn_y.min():.3e} to {pwn_y.max():.3e} erg/cm2/s")


import matplotlib.pyplot as plt

# Get the SED with initial parameters
pwn_sed_relic, fp_pwn, fr_pwn = pwn_fit.get_pwn_sed(
    [model_parameters_pwn[p][0] * model_parameters_pwn[p][-1] 
     for p in pwn_fit_parameters]
)

print("Model SED energy range:", pwn_sed_relic[:,0].min(), "to", pwn_sed_relic[:,0].max(), "erg")
print("Model SED flux range:  ", pwn_sed_relic[:,1].min(), "to", pwn_sed_relic[:,1].max(), "erg/cm2/s")
print()
print("Data energy range:", pwn_x.min(), "to", pwn_x.max(), "erg")
print("Data flux range:  ", pwn_y.min(), "to", pwn_y.max(), "erg/cm2/s")

# Plot both on same axes to see the offset
fig, ax = plt.subplots(figsize=(10,6))
ax.plot(pwn_sed_relic[:,0], pwn_sed_relic[:,1], label='Model')
ax.errorbar(pwn_x, pwn_y, yerr=[pwn_yerr_lo, pwn_yerr_hi], 
            fmt='o', label='Data')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Energy (erg)')
ax.set_ylabel('E²dN/dE (erg/cm²/s)')
ax.legend()
plt.savefig("plots/initial_model_vs_data.png", bbox_inches='tight')
print("Saved to plots/initial_model_vs_data.png")

sp = np.array(fp_pwn.GetParticleSpectrum())
print(f"\nParticle spectrum: {sp.shape}")
print(f"Electron energy range: {sp[:,0].min():.3e} to {sp[:,0].max():.3e} erg")
print(f"Electron dN/dE range:  {sp[:,1].min():.3e} to {sp[:,1].max():.3e}")

# Check total electron energy
total_e = np.trapz(sp[:,1]*sp[:,0], sp[:,0])
print(f"Total electron energy: {total_e:.3e} erg")
print(f"Expected (~theta*Edot*age): {pwn_fit.theta * pwn_fit.e_dot * pwn_fit.calculate_true_age() * gp.yr_to_sec:.3e} erg")
print(f"\nDistance used: {pwn_fit.distance} pc")
print(f"In cm: {pwn_fit.distance * 3.086e18:.3e} cm")


# Check what injection spectrum looks like before MCMC
test_e = np.logspace(np.log10(gp.m_e/gp.TeV_to_erg), 4, 200) * gp.TeV_to_erg
test_spec = pwn_fit.injection_spectrum_pwn(test_e)
print("Injection spectrum range:", np.array(test_spec)[:,1].min(), np.array(test_spec)[:,1].max())

# Check data energy units
print("Energy range of data:", relic_eref.min(), relic_eref.max())
print("Flux range of data:", relic_sed.min(), relic_sed.max())


import pandas as pd
for name, path in [("Chandra","data/chandraData.csv"), 
                   ("HESS","data/hessData.csv"),
                   ("Fermi","data/fermiData.csv"),
                   ("KM2A","data/km2aData.csv"),
                   ("Tibet","data/tibetData.csv"),
                   ("NuSTAR","data/fpmaData.csv"),
                   ("HAWC", "data/hawcData.csv")]:
    df = pd.read_csv(path)
    print(f"{name}: E = {df['energy'].min():.3e} to {df['energy'].max():.3e} erg, "
          f"flux = {df['flux'].min():.3e} to {df['flux'].max():.3e} erg/cm2/s")

print(f"Model SED energy range: {pwn_sed_relic[:,0].min():.3e} to {pwn_sed_relic[:,0].max():.3e}")
print(f"Model SED flux range:   {pwn_sed_relic[:,1].min():.3e} to {pwn_sed_relic[:,1].max():.3e}")
print(f"Data energy range:      {pwn_x.min():.3e} to {pwn_x.max():.3e}")
print(f"Data flux range:        {pwn_y.min():.3e} to {pwn_y.max():.3e}")

print(f"Model SED flux range: {pwn_sed_test[:,1].min():.3e} to {pwn_sed_test[:,1].max():.3e}")
print(f"Data flux range:      {pwn_y.min():.3e} to {pwn_y.max():.3e}")

test_e = np.logspace(np.log10(gp.m_e/gp.TeV_to_erg), 4, 50) * gp.TeV_to_erg
spec_pairs = pwn_fit.injection_spectrum_pwn(test_e)
e_arr   = np.array([p[0] for p in spec_pairs])
sp_arr  = np.array([p[1] for p in spec_pairs])

fu = gp.Utils()
integral = fu.Integrate(list(zip(e_arr, e_arr * sp_arr)))
print(f"Integral(E*spec dE) = {integral:.3e}  (should be 1.0)")
print(f"Total electron energy from GAMERA: {3.098e+57:.3e} erg")
print(f"Expected: {pwn_fit.theta * pwn_fit.e_dot * pwn_fit.calculate_true_age() * yr_to_sec:.3e} erg")
print(f"Ratio (actual/expected): {3.098e57 / (pwn_fit.theta * pwn_fit.e_dot * pwn_fit.calculate_true_age() * yr_to_sec):.3e}")

t_yr  = np.logspace(0, 5, 50)
t_sec = t_yr * yr_to_sec
lum_t = np.array([pwn_fit.luminosity(ti) for ti in t_yr])
total_injected = np.trapz(lum_t, t_sec)
print(f"Integral(L dt) = {total_injected:.3e} erg  (total energy injected over lifetime)")
print(f"theta*L0*t0*(br_factor-1) ~ {pwn_fit.theta * pwn_fit.calculate_l0() * pwn_fit.calculate_t0() * yr_to_sec:.3e} erg")

# The GAMERA example normalizes to 1e37 erg/s and gets flux ~1e-12 erg/cm2/s
# at 1 kpc distance. Your luminosity is ~6e45 erg/s at 7 kpc.
# Expected flux ~ L / (4*pi*d^2)
d_cm = 7e3 * 3.086e18
expected_flux = pwn_fit.luminosity(pwn_fit.calculate_true_age()) / (4 * np.pi * d_cm**2)
print(f"Expected flux order: {expected_flux:.3e} erg/cm2/s")
print(f"Model SED peak:      5.5e-02 erg/cm2/s")
print(f"Ratio:               {5.5e-2 / expected_flux:.3e}")

d_cm = 7e3 * 3.086e18  # 7 kpc in cm
L_now = pwn_fit.luminosity(pwn_fit.calculate_true_age())
expected_flux = L_now / (4 * np.pi * d_cm**2)
print(f"L(true_age)    = {L_now:.3e} erg/s")
print(f"distance       = {d_cm:.3e} cm")
print(f"Expected flux  = {expected_flux:.3e} erg/cm2/s")
print(f"Model SED peak = 5.5e-02 erg/cm2/s")
print(f"Ratio model/expected = {5.5e-2 / expected_flux:.3e}")

t_yr  = np.logspace(0, 5, 50)
t_sec = t_yr * yr_to_sec
lum_t = np.array([pwn_fit.luminosity(ti) for ti in t_yr])

true_age_yr = pwn_fit.calculate_true_age()
mask = t_yr <= true_age_yr
total_within_age = np.trapz(lum_t[mask], t_sec[mask])
total_full       = np.trapz(lum_t, t_sec)

print(f"true_age          = {true_age_yr:.3e} yr")
print(f"t_yr max          = {t_yr.max():.3e} yr  (goes {t_yr.max()/true_age_yr:.1f}x beyond true_age)")
print(f"Integral(L dt) within true_age = {total_within_age:.3e} erg")
print(f"Integral(L dt) full t array    = {total_full:.3e} erg")
print(f"Ratio full/within_age          = {total_full/total_within_age:.3e}")