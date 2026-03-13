import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('Agg')
import pandas as pd
# from data import DataDict
import time
import argparse
import sys
import os
current_directory = os.getcwd()
sys.path.append(current_directory)
print(current_directory)
from astropy.table import Table, Column
from astropy.io import fits
from astropy.constants import c
import astropy.units as u
from astropy.modeling.powerlaws import PowerLaw1D, LogParabola1D

from gammapy.catalog import SourceCatalog4FGL, SourceCatalogHGPS

TeV_to_erg = 1.60218

datadir = current_directory+"/data/"

hawcData = Table.read(datadir+"hawcData.csv", format="csv")
hawcData["energy"].unit=u.erg
hawcData["flux"].unit=1*(u.erg)/(u.cm**2*u.s)
hawcData["flux_error_lo"].unit=1*(u.erg)/(u.cm**2*u.s)
hawcData["flux_error_hi"].unit=1*(u.erg)/(u.cm**2*u.s)

hessData = Table.read(datadir+"hessData.csv", format="csv")
hessData["energy"].unit=u.erg
hessData["flux"].unit=1*(u.erg)/(u.cm**2*u.s)
hessData["flux_error_lo"].unit=1*(u.erg)/(u.cm**2*u.s)
hessData["flux_error_hi"].unit=1*(u.erg)/(u.cm**2*u.s)

fermiData = Table.read(datadir+"fermiData.csv", format="csv")
fermiData["energy"].unit=u.erg
fermiData["flux"].unit=1*(u.erg)/(u.cm**2*u.s)
fermiData["flux_error_lo"].unit=1*(u.erg)/(u.cm**2*u.s)
fermiData["flux_error_hi"].unit=1*(u.erg)/(u.cm**2*u.s)

chandraData = Table.read(datadir+"chandraData.csv", format="csv")
chandraData["energy"].unit=u.erg
chandraData["flux"].unit=1*(u.erg)/(u.cm**2*u.s)
chandraData["flux_error_lo"].unit=1*(u.erg)/(u.cm**2*u.s)
chandraData["flux_error_hi"].unit=1*(u.erg)/(u.cm**2*u.s)

fpmaData = Table.read(datadir+"fpmaData.csv", format="csv")
fpmaData["energy"].unit=u.erg
fpmaData["flux"].unit=1*(u.erg)/(u.cm**2*u.s)
fpmaData["flux_error_lo"].unit=1*(u.erg)/(u.cm**2*u.s)
fpmaData["flux_error_hi"].unit=1*(u.erg)/(u.cm**2*u.s)

km2aData = Table.read(datadir+"km2aData.csv", format="csv")
km2aData["energy"].unit=u.erg
km2aData["flux"].unit=1*(u.erg)/(u.cm**2*u.s)
km2aData["flux_error_lo"].unit=1*(u.erg)/(u.cm**2*u.s)
km2aData["flux_error_hi"].unit=1*(u.erg)/(u.cm**2*u.s)

tibetData = Table.read(datadir+"tibetData.csv", format="csv")
tibetData["energy"].unit=u.erg
tibetData["flux"].unit=1*(u.erg)/(u.cm**2*u.s)
tibetData["flux_error_lo"].unit=1*(u.erg)/(u.cm**2*u.s)
tibetData["flux_error_hi"].unit=1*(u.erg)/(u.cm**2*u.s)

km2aSED = Table.read(datadir+"km2aSED.csv", format="csv")
km2aSED["energy"].unit=u.erg
km2aSED["flux"].unit=1*(u.erg)/(u.cm**2*u.s)
km2aSED["flux_error_lo"].unit=1*(u.erg)/(u.cm**2*u.s)
km2aSED["flux_error_hi"].unit=1*(u.erg)/(u.cm**2*u.s)



DataDict = {
    "Chandra" : chandraData,
    "HESS" : hessData,
    "Fermi" : fermiData,
    "NuSTAR" : fpmaData,
    "KM2A" : km2aData,
    "Tibet" : tibetData,
    "HAWC" : hawcData,
}

SEDDict = {
    "KM2A": km2aSED,
}
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import astropy.units as u
    for inst in DataDict:
        if inst != "KM2A":
            print(f"{inst} data:")
            plt.errorbar(DataDict[inst]["energy"]*u.erg.to(u.TeV),
                DataDict[inst]["flux"],
                yerr=[DataDict[inst]["flux_error_lo"], DataDict[inst]["flux_error_hi"]],
                fmt="*",
                label=inst,
                )
    for inst in SEDDict:
        plt.plot(SEDDict[inst]["energy"]*u.erg.to(u.TeV),
                SEDDict[inst]["flux"],
                linestyle='dashed',
                label=inst+" SED")
        plt.fill_between(
                SEDDict[inst]["energy"]*u.erg.to(u.TeV),
                SEDDict[inst]["flux"]-SEDDict[inst]["flux_error_lo"],
                SEDDict[inst]["flux"]+SEDDict[inst]["flux_error_hi"],
                alpha=0.5)

    plt.ylim(1e-14, 1e-11)
    plt.loglog()
    plt.legend(ncol=2)
    plt.savefig(f"datapoints.pdf")