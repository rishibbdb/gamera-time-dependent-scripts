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
from base_model import *
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

model_parameters_fit_pwn = {
    #fit parameters, lowerbound, upperbound
    'b_now' : [4.6e-6,1e-6,20e-6], #Gauss
    'P0' : [22e-3,18e-3,30e-3], #sec
    'theta' : [0.4469,0.01,1.0], #converstion frac fron pulsar lum to electrons
    'ecut' : [2.5,1,3.2], #log10(E/TeV)
    'alpha' : [2.2,1.1,3.0],
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


relic_eref = np.concatenate((hessData[hessData['is_ul']==False]['energy'], fermiData[fermiData['is_ul']==False]['energy'], fpmaData[fpmaData['is_ul']==False]['energy'], km2aData['energy'], tibetData['energy'], chandraData['energy']))
relic_sed = np.concatenate((hessData[hessData['is_ul']==False]['flux'], fermiData[fermiData['is_ul']==False]['flux'], fpmaData[fpmaData['is_ul']==False]['flux'], km2aData['flux'], tibetData['flux'], chandraData['flux']))
relic_sed_err_lo = np.concatenate((hessData[hessData['is_ul']==False]['flux_error_lo'], fermiData[fermiData['is_ul']==False]['flux_error_lo'], fpmaData[fpmaData['is_ul']==False]['flux_error_lo'], km2aData['flux_error_lo'], tibetData['flux_error_lo'], chandraData['flux_error_lo']))
relic_sed_err_hi = np.concatenate((hessData[hessData['is_ul']==False]['flux_error_hi'], fermiData[fermiData['is_ul']==False]['flux_error_hi'], fpmaData[fpmaData['is_ul']==False]['flux_error_hi'], km2aData['flux_error_hi'], tibetData['flux_error_hi'], chandraData['flux_error_hi']))


pwn_x = relic_eref
pwn_y = relic_sed
pwn_yerr_lo = relic_sed_err_lo
pwn_yerr_hi = relic_sed_err_hi

print("Relic size", relic_eref.size, relic_sed.size, relic_sed_err_lo.size)


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
mcmc_result_pwn = run_mcmc(5,pwn_pars,pwn_fit.log_prob_pwn,pwn_x,pwn_y,pwn_yerr_lo, num_burn_in=10, chain_steps=30)

with open('mcmc_results/hawc_mcmc_result_pwn.npy', 'wb') as f:
    np.save(f, mcmc_result_pwn)

mcmc_pwn_fit_results, mcmc_pwn_fit_results_err_high, mcmc_pwn_fit_results_err_low = mcmc_results("pwn", pwn_labels, pwn_fit_parameters, model_parameters_pwn, mcmc_result_pwn)
plt.show()
plt.savefig("plots/pwn_cornerplot.png", bbox_inches='tight')


pwn_sed_relic, fp_pwn, fr_pwn = pwn_emission_singlezone.get_pwn_sed(pwn_fit, mcmc_pwn_fit_results)

def draw_observations_data2(zoom=False):
    fig, ax = plt.subplots(figsize=(14, 9))
    
    # --- HESS ---
    hess_det = hessData[hessData['is_ul']==False]
    hess_ul  = hessData[hessData['is_ul']==True]
    ax.errorbar(hess_det['energy'], hess_det['flux'],
                yerr=[hess_det['flux_error_lo'], hess_det['flux_error_hi']],
                marker='o', linestyle='None', label='HESS',
                mfc='fuchsia', mec='fuchsia', ecolor='fuchsia', capsize=3, fmt='o', zorder=5)
    if len(hess_ul) > 0:
        ax.errorbar(hess_ul['energy'], hess_ul['flux'],
                    yerr=0.3*hess_ul['flux'], uplims=True,
                    marker='v', linestyle='None',
                    mfc='fuchsia', mec='fuchsia', ecolor='fuchsia', capsize=3, fmt='v', zorder=5)

    # --- Fermi ---
    fermi_det = fermiData[fermiData['is_ul']==False]
    fermi_ul  = fermiData[fermiData['is_ul']==True]
    ax.errorbar(fermi_det['energy'], fermi_det['flux'],
                yerr=[fermi_det['flux_error_lo'], fermi_det['flux_error_hi']],
                marker='s', linestyle='None', label='Fermi',
                mfc='lime', mec='lime', ecolor='lime', capsize=3, fmt='s', zorder=5)
    if len(fermi_ul) > 0:
        ax.errorbar(fermi_ul['energy'], fermi_ul['flux'],
                    yerr=0.3*fermi_ul['flux'], uplims=True,
                    marker='v', linestyle='None',
                    mfc='lime', mec='lime', ecolor='lime', capsize=3, fmt='v', zorder=5)

    # --- FPMA ---
    fpma_det = fpmaData[fpmaData['is_ul']==False]
    fpma_ul  = fpmaData[fpmaData['is_ul']==True]
    ax.errorbar(fpma_det['energy'], fpma_det['flux'],
                yerr=[fpma_det['flux_error_lo'], fpma_det['flux_error_hi']],
                marker='^', linestyle='None', label='FPMA',
                mfc='violet', mec='violet', ecolor='violet', capsize=3, fmt='^', zorder=5)
    if len(fpma_ul) > 0:
        ax.errorbar(fpma_ul['energy'], fpma_ul['flux'],
                    yerr=0.3*fpma_ul['flux'], uplims=True,
                    marker='v', linestyle='None',
                    mfc='violet', mec='violet', ecolor='violet', capsize=3, fmt='v', zorder=5)

    # --- KM2A ---
    ax.errorbar(km2aData['energy'], km2aData['flux'],
                yerr=[km2aData['flux_error_lo'], km2aData['flux_error_hi']],
                marker='D', linestyle='None', label='KM2A',
                mfc='red', mec='red', ecolor='red', capsize=3, fmt='D', zorder=5)

    # --- Tibet ---
    ax.errorbar(tibetData['energy'], tibetData['flux'],
                yerr=[tibetData['flux_error_lo'], tibetData['flux_error_hi']],
                marker='P', linestyle='None', label='Tibet',
                mfc='orange', mec='orange', ecolor='orange', capsize=3, fmt='P', zorder=5)

    # --- Chandra ---
    ax.errorbar(chandraData['energy'], chandraData['flux'],
                yerr=[chandraData['flux_error_lo'], chandraData['flux_error_hi']],
                marker='*', linestyle='None', label='Chandra',
                mfc='deepskyblue', mec='deepskyblue', ecolor='deepskyblue', capsize=3, fmt='*', zorder=5, markersize=10)

    # --- Model ---
    ax.plot(pwn_sed_relic[:,0], pwn_sed_relic[:,1],
            linestyle='-', color='steelblue', linewidth=2, label='Total model', zorder=4)

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_title(r'HESS J1813 $\gamma$-ray Flux Points', size=20, pad=12)
    ax.set_ylabel(r'E$_{\gamma}^{2}\Phi_{\gamma}$ [erg/(cm${}^{2}$ s)]', size=16)
    ax.set_xlabel(r'E$_{\gamma}$ (TeV)', size=16)

    if zoom:
        ax.set_xlim(0.8e-3, 1e3)
        ax.set_ylim(bottom=8e-15)
    else:
        ax.set_xlim(5e-19, 1e3)
        ax.set_ylim(bottom=8e-15)

    leg = ax.legend(loc='upper left', fontsize=13, ncol=2, framealpha=0.9,
                    fancybox=False, edgecolor='grey')
    leg.get_frame().set_linewidth(0.8)

    ax.grid(which='both', axis='both', linestyle='--', linewidth=0.4, alpha=0.6)
    ax.tick_params(axis='both', labelsize=13, which='both', direction='in',
                   top=True, right=True, length=5)
    ax.tick_params(axis='both', which='minor', length=3)

    fig.tight_layout()
    filename = "plots/datapoints_zoomed.png" if zoom else "plots/datapoints.png"
    plt.savefig(filename, bbox_inches='tight')

draw_observations_data2(zoom=False)

