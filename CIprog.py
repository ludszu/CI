#%%
import numpy as np
from matplotlib import pyplot as plt
import CIcalculation as CI
from scipy import sparse as sp

#%% PROBLEM SETUP

Neltot=2 # total number of electrons
Sz=0 # total Sz of the system
nShell=4 # number of SP energy shells to consider (2D Qoscillator like)

# number of SP energy levels follow from shells
NeneU=np.sum(range(1,nShell+1))*2
NeneD=np.sum(range(1,nShell+1))*2
# set maximum number of excitations for full CI
maxX=Neltot

screen=1#2.5 # dielectric constant
folder="R200V300" # working folder
nEne=6 # number of eigenenergies sought from sparse diagonalisation
Eshift=0 # shift SP energies

#%% PARAMETERS

# input files
files=np.empty((5,1),dtype='S17')
files[0]=folder+"\EmaU.dat" # SP energies UP
files[1]=folder+"\EmaD.dat" # SP energies DN
files[2]=folder+"\VSOU.dat" # CME for spin UU
files[3]=folder+"\VSOD.dat" # CME for spin DD
files[4]=folder+"\VSUD.dat" # CME for spin UD
sparseLeng=200 # number of anticipated non0 elements in each row of Hamiltonian

# setup environment
paramsCI=np.array([Neltot,Sz,NeneU,NeneD,maxX])
params=np.array([Eshift,screen,nEne,sparseLeng])
CI1=CI.CIcalculation(files,paramsCI,params,False)

#%% DIAGONALISE PROBLEM

ifSparse=False
e,v=CI1.diagCIham(nEne,ifSparse)
print("diag done")
print(e)

#%%

# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 10:26:25 2021

@author: ludka
"""

