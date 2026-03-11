#!/usr/local/bin/python

import sys
import os
import numpy as np
import idlsave
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator


'''
    little script to read Richard Tuff's model for galactic radiation fields.
    Usage: python ReadRichardsModel.py readurad.xdr <r (pc)> <z (pc)>
    This will create a plot, a data file and some terminal output.
    You need to have 'idlsave' and 'scipy' installed.
'''
if __name__ == "__main__":
    idl_file = sys.argv[1]
    r = float(sys.argv[2])
    z = float(sys.argv[3])
    data = idlsave.read(idl_file)

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
    dust_coord = np.array(zip(z+np.zeros(len(data.lambda_dust)),r+np.zeros(len(data.lambda_dust)),data.lambda_dust))
    starlight_coord = np.array(zip(z+np.zeros(len(data.lambda_sparse)),r+np.zeros(len(data.lambda_sparse)),data.lambda_sparse))


    dust_sed = dust_interpolator(dust_coord)
    starlight_sed = starlight_interpolator(starlight_coord)
        
    
    total_range = np.linspace(np.log10(data.lambda_sparse[0]),np.log10(data.lambda_dust[-1]),10)
    total_sed = 10**np.interp(total_range,np.log10(data.lambda_sparse),np.log10(starlight_sed))
    total_sed[total_range>np.log10(data.lambda_sparse[-1])] = starlight_sed[-1] * (10**total_range[total_range>np.log10(data.lambda_sparse[-1])]/data.lambda_sparse[-1])**-4
    total_sed += 10**np.interp(total_range,np.log10(data.lambda_dust),np.log10(dust_sed))
    total_range = 10**total_range

    #for the integral energy density
    total_range_ang = 1e4*total_range #converting from micrometer to angstrom
    total_energy_density = np.trapz(total_sed,total_range_ang)/(3.086e18**3) #in erg/cm3 1 pc = 3.086e18
    dust_energy_density = np.trapz(dust_sed,1e4*data.lambda_dust)/(3.086e18**3)
    starlight_energy_density = np.trapz(starlight_sed,1e4*data.lambda_sparse)/(3.086e18**3)
    print(total_energy_density, dust_energy_density, starlight_energy_density)

    # make a plot
    f = plt.figure(figsize = (5,5))    
    plt.loglog(data.lambda_dust,dust_sed,label="dust",alpha=1)
    plt.loglog(data.lambda_sparse,starlight_sed,label="starlight",alpha=1)
    plt.loglog(total_range,total_sed,label="total,\nextrapolated starlight",alpha=0.5)
    plt.legend()
    plt.grid()
    plt.ylabel(r"$\lambda F_\lambda (erg / pc^3 / \AA)$")
    plt.xlabel(r"$\lambda (\mu m)$")
    f.savefig("rad_field_richard_r"+str(r)+"_z"+str(z)+".png",bbox_inches='tight')

    #terminal and data output
    # outfile = open("radiation_field_r"+str(r)+"_z"+str(z)+".dat",'w')
    # outfile.write("#emission at  r = "+str(r)+",pc z = "+str(z)+" pc\n")
    # outfile.write("#lambda (microns), SED value (erg / pc^3 / A)\n")
    # # print("emission at  r = "+str(r)+",pc z = "+str(z)+" pc")
    # # print("..............")
    # # print("lambda (microns), SED value (erg / pc^3 / A)")
    # for l,d in zip(total_range,total_sed):
    #     #print(str(l)+" "+str(d))
    #     outfile.write(str(l)+" "+str(d)+"\n")
    # outfile.close()

    outfile = open("gamera_format_output_for_r"+str(r)+"_z"+str(z)+".txt",'w')
    outfile.write("#Energy (eV), Energy Density (eV / cm^3)\n")
    for l,d in zip(total_range,total_sed):
        energy = 4.135e-15*3e14/l # plank const = 4.135e-15 eV/s, c = 3e14 micorns/s
        energy_density = l * 1e4 * d * 6.242e11 / (3.085678e18)**3 # erg to eV = 6.242e11 , 1pc =3.085678e18 cm3 
        outfile.write(str(energy)+" "+str(energy_density)+"\n")
    outfile.close()

