import sys
import os
sys.path.append('/Users/rishi/Documents/Analysis/J1813/GAMERA/lib') #gamera lib installation
import gappa as gp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
# import idlsave
# from xdrlib import Unpacker
from scipy.interpolate import RegularGridInterpolator


def get_radiation_field(data,r,z):

    # data cosmetics for easier handling
    data.urad_sparse_arr = np.swapaxes(data.urad_sparse_arr,0,1)
    data.urad_sparse_arr = np.swapaxes(data.urad_sparse_arr,1,2)
    starlight = data.urad_sparse_arr

    data.urad_out_arr_dust = np.swapaxes(data.urad_out_arr_dust,0,1)
    data.urad_out_arr_dust = np.swapaxes(data.urad_out_arr_dust,1,2)
    dust = data.urad_out_arr_dust

    # make 3D interpolation functions    
    dust_interpolator = RegularGridInterpolator((data.zz,data.rr, data.lambda_dust), dust)
    starlight_interpolator = RegularGridInterpolator((data.zz,data.rr, data.lambda_sparse), starlight)


    # make a set of points where to evaluate the interpolation functions
    dust_coord = np.array(list(zip(z+np.zeros(len(data.lambda_dust)),r+np.zeros(len(data.lambda_dust)),data.lambda_dust)))
    starlight_coord = np.array(list(zip(z+np.zeros(len(data.lambda_sparse)),r+np.zeros(len(data.lambda_sparse)),data.lambda_sparse)))

    dust_sed = dust_interpolator(dust_coord)
    starlight_sed = starlight_interpolator(starlight_coord)
    
    # interpolate
    total_range = np.linspace(np.log10(data.lambda_sparse[0]),np.log10(data.lambda_dust[-1]),1000)
    total_sed = 10**np.interp(total_range,np.log10(data.lambda_sparse),np.log10(starlight_sed))
    total_sed[total_range>np.log10(data.lambda_sparse[-1])] = starlight_sed[-1] * (10**total_range[total_range>np.log10(data.lambda_sparse[-1])]/data.lambda_sparse[-1])**-4
    total_sed += 10**np.interp(total_range,np.log10(data.lambda_dust),np.log10(dust_sed))
    total_range = 10**total_range


    # convert to GAMERA format
    total_range_cgs = gp.hp * gp.c_speed / (total_range * 1e-4)
    total_edens_cgs = total_sed  * 1e8 * gp.hp * gp.c_speed / total_range_cgs**3 / gp.pc_to_cm**3

    gam_form_e = np.flip(total_range_cgs,0)
    gam_form_edens = np.flip(total_edens_cgs,0)

    return np.array(list(zip(gam_form_e,gam_form_edens)))

# if __name__ == "__main__":

#     # read in Richard's data
#     # radiation_data = idlsave.read(sys.argv[1])
#     # radiation_data=Unpacker(sys.argv[1])
#     #extract radiation field at (r,z) and convert to GAMERA input format
#     #(for the function 'AddArbitraryTargetPhotons'). The unit here is
#     #x:erg, y: 1/erg/cm^3
#     r = 8200
#     z = 51.5
#     rad_field = get_radiation_field(radiation_data,r,z)
    
#     # for show:
#     # plot now the _SED_, therefore multiplying the y-axis with x**2 and showing
#     # in units of eV
#     f = plt.figure(figsize=(4,4))
#     plt.loglog(rad_field[:,0]/gp.eV_to_erg,rad_field[:,1]*rad_field[:,0]**2/gp.eV_to_erg)
#     plt.xlabel("E (eV)")
#     plt.ylabel("E^2 dN/dE (eV/cm^3)")
#     plt.grid()
#     f.savefig("tst.png",bbox_inches="tight")


#     # to check that everything works fine, set up the radiation field in GAMERA,
#     # extract it and then plot it
#     ra = gp.Radiation()
#     ra.AddArbitraryTargetPhotons(rad_field)
    
#     field = np.array(ra.GetTargetPhotons())
#     f = plt.figure(figsize=(4,4))
#     plt.loglog(field[:,0]/gp.eV_to_erg,field[:,1]*field[:,0]**2/gp.eV_to_erg)
#     plt.xlabel("E (eV)")
#     plt.ylabel("E^2 dN/dE (eV/cm^3)")
#     plt.grid()
#     f.savefig("tst2.png",bbox_inches="tight")
    


