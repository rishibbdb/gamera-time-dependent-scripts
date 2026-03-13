import os
import sys
sys.path.append('/users/ywson/tools/hawc/GAMERA/lib')
import gappa as gp
import numpy as np
import astropy.constants as c
import astropy.units as u
import emcee
import rad_field_richard_tuffs.RADIATION_To_GAMERA as RADIATION_To_GAMERA
from multiprocessing import Pool
from scipy.io import readsav
import corner

yr_to_sec = 365.25 * 24 * 3600
sec_to_yr = 1/yr_to_sec

def run_mcmc(num_pars,opt_pars,func,x,y,yerr,num_threads=16,num_walkers=32,num_burn_in=100, chain_steps=300):
    ######
    # Test using the emcee package and the MCMC chain
    print("Start the MCMC part")
    os.environ["OMP_NUM_THREADS"] = str(num_threads)  # to avoid problems with numpy and Pool.
    np.random.seed(42)
    pos = np.array(opt_pars) + 1e-3 * np.random.randn(num_walkers, num_pars)  # initial position of the walkers
    nwalkers, ndim = pos.shape
    burn_in_steps = num_burn_in
    chain_steps = chain_steps
    with Pool(processes=num_threads) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, func, pool=pool, args=(x,y,yerr))
        state = sampler.run_mcmc(pos, burn_in_steps,
                                 progress=True)  # saves the position of the walkers in the state variable
        print("End burn-in phase")
        sampler.reset()  # resets the chain
        sampler.run_mcmc(state, chain_steps, progress=True)  # start again the chain form after the burn-in
        print("End MCMC")

    flat_samples = sampler.get_chain(
        flat=True)  # the burn in could also be set here, with the discard argument. thin option?
    print(flat_samples.shape)
    return flat_samples

def mcmc_results(name, labels, fit_parameters, model_parameters, mcmc_result): # opt_pars,
    '''
       Plotting
    '''
    #finlally lets print the results
    final_results = []
    final_results_err_high = []
    final_results_err_low = []
    for i,par in enumerate(fit_parameters):
        mcmc = np.percentile(mcmc_result[:, i], [16, 50, 84])
        final_results.append(mcmc[1]) 
        q = np.diff(mcmc)
        final_results_err_high.append(q[1])
        final_results_err_low.append(q[0])
        mcmc = mcmc/model_parameters[par][-1]
        q = q/model_parameters[par][-1]
        
        txt = "{3} = {0:.1e}-{1:.1e}+{2:.1e}"#, curve fit = {4:.1e}"
        txt = txt.format(mcmc[1], q[0], q[1], labels[i])#, curveFit)
        print(txt)

    final_results = np.array(final_results)
    final_results_err_high = np.array(final_results_err_high)
    final_results_err_low = np.array(final_results_err_low)
    fig = corner.corner(mcmc_result, labels=labels, truths=final_results)
    return final_results, final_results_err_high, final_results_err_low

def define_model_parameters_scales(model_parameters):
    for key in model_parameters.keys():
        scale = 100./(model_parameters[key][-1] - model_parameters[key][1])
        model_parameters[key].append(scale) #apply scaling and add to dict

class Leptons:
    def __init__(self, bins_pwn_model, known_properties):

        self.bins_pwn_model = bins_pwn_model
        self.longi = known_properties['longi'] #deg gal
        self.lati = known_properties['lati'] #deg gal
        self.distance = known_properties['distance'] #pc
        
        #pulsar properties
        self.e_dot = known_properties['e_dot'] #erg/sec
        self.char_age = known_properties['char_age'] # yrs
        self.P =  known_properties['P']  #sec  
        self.P_dot = known_properties['P_dot']  #/sec/sec   
        
        #fixed parameters
        self.br_ind = known_properties['br_ind'] #pulsar braking index  
        self.ebreak = known_properties['ebreak'] #TeV
        self.alpha0 = known_properties['alpha0'] #radio component
        self._rad_fields_cache = None

    def load_rad_fields(self):
        if self._rad_fields_cache is None:
            earth_from_GC = 8.5e3
            fa = gp.Astro()
            coord = fa.GetCartesian([self.longi, self.lati, self.distance], [0, earth_from_GC, 0])
            distance_from_the_GC = np.sqrt(coord[0]**2 + coord[1]**2)
            cord_z = abs(coord[2])
            self._rad_fields_cache = RADIATION_To_GAMERA.get_radiation_field(
                readsav('rad_field_richard_tuffs/readurad.xdr'), 
                distance_from_the_GC, cord_z)
        return self._rad_fields_cache
    def calculate_t0(self):
        return self.P0**(self.br_ind-1)*self.P**(2-self.br_ind) / ((self.br_ind-1)*self.P_dot) * sec_to_yr

    def calculate_true_age(self):
        return ((self.P/((self.br_ind-1)*self.P_dot))*(1 - (self.P0/self.P)**(self.br_ind - 1))) * sec_to_yr

    def calculate_br_ind_power_factor(self):
        return (self.br_ind + 1)/(self.br_ind -1) #this is braking index dependent power factor goes in most of the eq

    def calculate_l0(self):
        return self.e_dot * ((1 + self.calculate_true_age()/self.calculate_t0())**self.calculate_br_ind_power_factor()) # initial spin down luminosity

    def calculate_b0(self):
        return  self.b_now*(1 + (self.calculate_true_age()/self.calculate_t0())**0.5) 

    def luminosity(self,t):
        return self.theta * self.calculate_l0() * 1 / ((1 + t/self.calculate_t0())**self.calculate_br_ind_power_factor()) # luminosity vs. time

    def bfield(self,t):
        return (self.calculate_b0() / (1 + (t/self.calculate_t0())**0.5)) # b-field vs time
    
    # def injection_spectrum_pwn(self, e):
    #     ecut = np.power(10,self.ecut) * gp.TeV_to_erg
    #     pwl = (e/gp.TeV_to_erg)**-self.alpha
    #     spec = pwl * np.exp(-e/ecut)
    #     return list(zip(e, spec))
    def injection_spectrum_pwn(self, e):
        # e is in erg — same as GAMERA example
        ecut = np.power(10, self.ecut) * gp.TeV_to_erg   # erg
        e0   = 1.0 * gp.TeV_to_erg                        # 1 TeV in erg
        pwl  = (e / e0) ** -self.alpha
        spec = pwl * np.exp(-e / ecut)
        
        # Normalize to 1 erg/s total, SetLuminosity handles absolute scale
        fu = gp.Utils()
        norm = fu.Integrate(list(zip(e, e * spec)))   # use GAMERA's own integrator
        spec = spec / norm
        return list(zip(e, spec))
    def log_prob_pwn(self, pars, x, y, yerr):
        """
        Compute the total log probability.

        Arguments:
            - pars : list of parameters
            - x, y, yerr : data points with errorbar
        """
        
        lp = self.log_prior_pwn(pars)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood_pwn(pars, x, y, yerr)

class pwn_emission_singlezone(Leptons):
    def __init__(self, bins_pwn_model, known_properties, model_parameters):
        super().__init__(
            bins_pwn_model=bins_pwn_model,
            known_properties=known_properties,
        )
        #fit parameters
        self.b_now = model_parameters['b_now'][0] #Gauss
        self.P0 = model_parameters['P0'][0] #sec
        self.theta = model_parameters['theta'][0] #fraction of edot power to electrons
        self.ecut = model_parameters['ecut'][0] #TeV
        self.alpha = model_parameters['alpha'][0] #wind componnet
        self.density = model_parameters['density'][0] #/cm3
        print("Initialized model parameters: b_now", self.b_now, "P0", self.P0, "theta", self.theta, "ecut", self.ecut, "alpha", self.alpha, "density", self.density)
        #low limit
        self.b_now_low = model_parameters['b_now'][1]
        self.P0_low = model_parameters['P0'][1]
        self.theta_low = model_parameters['theta'][1]
        self.ecut_low = model_parameters['ecut'][1] 
        self.alpha_low = model_parameters['alpha'][1] 
        self.density_low = model_parameters['density'][1]
        #high limit
        self.b_now_high = model_parameters['b_now'][2]
        self.P0_high = model_parameters['P0'][2]
        self.theta_high = model_parameters['theta'][2]
        self.ecut_high = model_parameters['ecut'][2] 
        self.alpha_high = model_parameters['alpha'][2] 
        self.density_high = model_parameters['density'][2]
        #scale
        self.b_now_scale = model_parameters['b_now'][-1]
        self.P0_scale = model_parameters['P0'][-1]
        self.theta_scale = model_parameters['theta'][-1]
        self.ecut_scale = model_parameters['ecut'][-1] 
        self.alpha_scale = model_parameters['alpha'][-1] 
        self.density_scale = model_parameters['density'][-1]        

    def setup_particles(self, fp, e_sed, t_yr, twindow_sec, reverse):
        time = np.logspace(0, 5, 10000)
        lum_t = self.luminosity(time)
        bf_t = self.bfield(time)
        t_sec = t_yr * gp.yr_to_sec
        t_recent_sec = t_sec
        lum_recent = lum_t
        bf_recent = bf_t
        twindow_yr = twindow_sec * sec_to_yr
        fp.AddThermalTargetPhotons(2.7, 0.25 * gp.eV_to_erg)
        fp.AddArbitraryTargetPhotons(self.load_rad_fields())
        fp.SetCustomInjectionSpectrum(e_sed)
        fp.SetLuminosity(list(zip(t_yr, lum_recent)))
        fp.SetBField(list(zip(t_yr, bf_recent)))
        fp.SetAmbientDensity(self.density)
        fp.SetAge(twindow_yr)
        fp.CalculateElectronSpectrum()
        return fp

    def setup_radiation(self, fr, fp, sp, e):
        fr.SetElectrons(sp)
        fr.SetAmbientDensity(self.density)
        fr.SetBField(fp.GetBField())
        fr.AddArbitraryTargetPhotons(fp.GetTargetPhotons())
        fr.SetDistance(self.distance)
        fr.CalculateDifferentialPhotonSpectrum(list(e))

        return fr

    def model_pwn(self, t_yr, e_electron, e_photon, time_frac, reverse):
        e_sed = self.injection_spectrum_pwn(e_electron)
        twindow_sec = self.calculate_true_age() * yr_to_sec
        fp = gp.Particles()
        fp.ToggleQuietMode()
        
        self.setup_particles(fp, e_sed, t_yr, twindow_sec, reverse)
        sp = np.array(fp.GetParticleSpectrum())
        fr = gp.Radiation()
        fr.ToggleQuietMode()
        self.setup_radiation(fr, fp, sp, e_photon)
        model_sed = np.array(fr.GetTotalSED())
        model_sed[:,0] *= gp.TeV_to_erg 
        return model_sed, fp, fr


    def fit_pwn_model(self, e_photon, theta_fit, b_now_fit, P0_fit, ecut_fit, alpha_fit):            
        self.theta = theta_fit/self.theta_scale
        self.b_now = b_now_fit/self.b_now_scale
        self.P0 = P0_fit/self.P0_scale
        self.ecut = ecut_fit/self.ecut_scale
        self.alpha = alpha_fit/self.alpha_scale
        '''
        GAMERA
        '''
        # t = np.logspace(0, 5, 2000)
        t = np.logspace(0, 5, 50)
        e_electron = np.logspace(np.log10(gp.m_e/gp.TeV_to_erg),4,self.bins_pwn_model) * gp.TeV_to_erg 
        e_relic = [float(v) for v in e_photon]
        relic_sed, _, _ = self.model_pwn(t, e_electron, e_relic, 1, False) #with full time history
        model_sed = [relic_sed[:,1]]
        return model_sed

    def get_pwn_sed(self, fit_values): 
        bins = 200
        t = np.logspace(0, 5, 200)
        e_photon = np.logspace(-6, 15, bins) * gp.eV_to_erg
        e_electron = np.logspace(np.log10(gp.m_e/gp.TeV_to_erg), 4, bins) * gp.TeV_to_erg

        self.theta = fit_values[0] / self.theta_scale
        self.b_now = fit_values[1] / self.b_now_scale
        self.P0    = fit_values[2] / self.P0_scale
        self.ecut  = fit_values[3] / self.ecut_scale
        self.alpha = fit_values[4] / self.alpha_scale

        print("Getting SED with parameters: theta", self.theta, 
            "b_now", self.b_now, "P0", self.P0, 
            "ecut", self.ecut, "alpha", self.alpha)

        pwn_sed_relic, fp_pwn, fr_pwn = self.model_pwn(t, e_electron, e_photon, 1, False)
        print(f"Raw SED[:,0] first/last: {pwn_sed_relic[0,0]:.3e} / {pwn_sed_relic[-1,0]:.3e}")
        print(f"Raw SED[:,1] min/max:    {pwn_sed_relic[:,1].min():.3e} / {pwn_sed_relic[:,1].max():.3e}")
        print(f"e_photon first/last:     {e_photon[0]:.3e} / {e_photon[-1]:.3e}")
    
        return pwn_sed_relic, fp_pwn, fr_pwn
    
    ## Auxiliary functions for the MCMC
    def log_prior_pwn(self,pars):
        """
        Uninformative flat prior.
        Needs to be adjusted depending on the chosen parameters

        Arguments:
            - pars : list of parameters
        """

        lim = ([self.theta_low*self.theta_scale,
                self.b_now_low*self.b_now_scale,
                self.P0_low*self.P0_scale,
                self.ecut_low*self.ecut_scale,
                self.alpha_low*self.alpha_scale],
               [self.theta_high*self.theta_scale,
                self.b_now_high*self.b_now_scale,
                self.P0_high*self.P0_scale,
                self.ecut_high*self.ecut_scale,
                self.alpha_high*self.alpha_scale])
        
        a, b, c, d, e = pars  # extract the parameters
        if lim[0][0] < a < lim[1][0] and  lim[0][1] < b < lim[1][1] and  lim[0][2] < c < lim[1][2] and  lim[0][3] < d < lim[1][3] and  lim[0][4] < e < lim[1][4]:
            return 0.0
        return -np.inf


    def log_likelihood_pwn(self, pars, x, y, yerr):
        a, b, c, d, e = pars
        model = self.fit_pwn_model(x, a, b, c, d, e)
        sigma2 = yerr ** 2
        likelihood = -0.5 * np.sum((y - model[0]) ** 2 / sigma2 + np.log(sigma2))
        return likelihood