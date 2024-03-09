#%% 
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg as la
from scipy import interpolate as itp
from scipy import sparse as sp
from scipy import special as sc
import scipy.sparse.linalg as spla

#%%

def combs(n,k,nX):
    # creates k-electron combinations on n SP levels, 
    # for maximum nX of excitations
    hm=sc.comb(n,k,exact=True)
    co=np.zeros((hm,k),dtype=int)
    tags=np.ones(hm,dtype=int)
    Xs=np.zeros(hm,dtype=int)
    
    for i in range(1,hm+1):
        ix=i-1
        kx=k
        for s in range(1,n+1):
            if (kx==0):
                break
            else:
                t=sc.comb(n-s,kx-1,exact=True)
                if (ix<t):
                    co[i-1,k-kx]=s
                    kx=kx-1
                else:
                    ix=ix-t
        if (k>nX):
            if (co[i-1,k-nX-1]>k): tags[i-1]=0
        
        Xs[i-1]=sum(co[i-1,:]>k)
        
    return co,tags,Xs
    
    
    
class CIcalculation:
    def __init__(self,files,paramsCI,params,ifJupiter):
    
        '''
        INPUTS: files: string array (fixed length strings) (5,1):
                        - SP energies U
                        - SP energies D
                        - CME tensor for spin UU
                        - CME tensor for spin DD
                        - CME tensor for spin UD
                paramsCI: [Neltotal,Sz,NESPU,NESPD,maxX]
                        - total electron number
                        - total Sz of the system
                        - number of SP energy levels spin U
                        - number of SP energy levels spin D
                        - maximum number of excitations
                params: [Eshift,screen,nEne*,sparseLeng*,omega*] *optional or used for sparse diagonalisation
                        - SP energy shift (if negative, shifts by EmatU[0])
                        - dielectric constant
                        - number of energies to obtain from sparse matrix diagonalisation
                        - estimated number of nonzero elements in the row of Hamiltonian matrix
                        - unitless oscillator frequency for a 2D Qwell problem
                ifJupiter: reading files commands depend on whether program is run on Jupiter or in Spyder
        '''
    
        self.Neltotal=int(paramsCI[0]) # total electron number
        self.Sz=paramsCI[1] # total Sz of electron system	
        self.NESPU=int(paramsCI[2]) # number of SP energy levels spin U
        self.NESPD=int(paramsCI[3]) # number of SP energy levels spin D
        self.NeltotalU=int(self.Neltotal/2+self.Sz) # total spin U electron number 
        self.NeltotalD=int(self.Neltotal/2-self.Sz) # total spin D electron number 
        
        print("Neltot U: ",self.NeltotalU)
        print("Neltot D: ",self.NeltotalD)
        
        self.NstVU=self.NeltotalU # number of filled VB SP states spin U
        self.NstVD=self.NeltotalD # number of filled VB SP states spin D
        self.NstCU=self.NESPU-self.NstVU # number of empty CB SP states spin U
        self.NstCD=self.NESPD-self.NstVD # number of empty CB SP states spin D
        # max allowed number of excitations, if =NEtotal it's full CI
        self.maxX=paramsCI[4] 
        self.NfreezU=self.NeltotalU-self.NstVU # number of frozen electrons in VB spin U
        self.NfreezD=self.NeltotalD-self.NstVD # number of frozen electrons in VB spin D
        self.NelU=self.NstVU # number of electrons promoted up spin U
        self.NelD=self.NstVD # number of electrons promoted up spin D
        
        self.VmatUU=[] # Coulomb matrix element tensor spin UU
        self.VmatDD=[] # Coulomb matrix element tensor spin DD
        self.VmatUD=[] # Coulomb matrix element tensor spin UD
        self.EmatU=[] # SP energies spin U
        self.EmatD=[] # SP energies spin D

        self.configsU=[] # configurations spin U
        self.configsD=[] # configurations spin D
        self.configsU01=[] # occupation spin U
        self.configsD01=[] # occupation spin D
        self.allConfs=[] # Hamiltonian basis of all configurations U&D
        
        self.noConfU='' # number of spin U configurations
        self.noConfD='' # number of spin D configurations
        self.sizHam='' # Hamiltonian matrix size
        
        self.hampos="" # sparse format Hamiltonian: positions of non0 elements in a row
        self.hamval="" # sparse format Hamiltonian: values of non0 elements in a row
        
        self.Eshift=params[0] # SP energy shift (if negative, shifts by EmatU[0])
        self.screen=params[1] # dielectric constant
        nEne=int(params[2]) # number of eigenenergies to seek from sparse diagonalisation
        
        # if no sparseLeng, assume small Hamiltonian with 10 non0 elements in each row
        if (len(params)<4):
            self.sparseLeng=10
        else:
            self.sparseLeng=int(params[3])
            
        if (len(params)<5):
            self.omega=1 # unitless oscillator frequency for a 2D Qwell problem
            self.switch=1 # strength of the Coulomb interaction
        else:
            self.omega=params[4] # unitless oscillator frequency for a 2D Qwell problem
            self.switch=np.sqrt(np.pi*self.omega) # strength of the Coulomb interaction
                 
        self.makeConfigs()
        self.loadInputs(files,ifJupiter,self.Eshift)
    
    def makeConfigs(self):
        # number of filled states has to be lower or equal to number of total electrons
        if ((self.NstVU<=self.NeltotalU) and (self.NstVD<=self.NeltotalD)):
            NU=self.NstVU+self.NstCU
            ND=self.NstVD+self.NstCD
            
            # create all combinations for spins U and D separately
            if (self.NelU>0): confU,tagU,XsU=combs(NU,self.NelU,self.maxX)
            if (self.NelD>0): confD,tagD,XsD=combs(ND,self.NelD,self.maxX)

            if (self.NelU>0): sU=confU.shape[0]
            if (self.NelD>0): sD=confD.shape[0]
            
            # check which combinations have the required number of excitations
            self.noConfU=int(sum(tagU)) if (self.NelU>0) else 0
            self.noConfD=int(sum(tagD)) if (self.NelD>0) else 0

            if (self.NelU>0): print("# confu",self.noConfU)
            if (self.NelD>0): print("# confd",self.noConfD)

            if (self.NelU>0):
                tempU=np.zeros((sU,self.NelU+1),dtype=int)
                indU=np.zeros(self.noConfU,dtype=int)
                tempU[:,0:self.NelU]=confU
                tempU[:,self.NelU]=XsU
                nums=np.arange(0,sU)
                indU=nums[tagU==1]
                tempU2=tempU[indU,:]
                ix=np.lexsort([np.zeros(self.noConfU),tempU2[:,self.NelU]])
                confUtru=tempU2[ix,:]
            if (self.NelD>0):
                tempD=np.zeros((sD,self.NelD+1),dtype=int)
                indD=np.zeros(self.noConfD,dtype=int)
                tempD[:,0:self.NelD]=confD
                tempD[:,self.NelD]=XsD
                nums=np.arange(0,sD)
                indD=nums[tagD==1]
                tempD2=tempD[indD,:]
                ix=np.lexsort([np.zeros(self.noConfD),tempD2[:,self.NelD]])
                confDtru=tempD2[ix,:]

            if (self.NelU>0): self.configsU=confUtru[:,0:self.NelU]+self.NfreezU-1
            if (self.NelD>0): self.configsD=confDtru[:,0:self.NelD]+self.NfreezD-1

            # store configurations (1,3,5) and ocupations (1,0,1,0,1)
            if (self.NelU>0): self.configsU01=np.zeros((self.noConfU,NU))
            if (self.NelD>0): self.configsD01=np.zeros((self.noConfD,ND))

            if (self.NelU>0): print("configsU:",self.configsU)
            if (self.NelD>0): print("configsD:",self.configsD)

            if (self.NelU>0):
                for i in range(0,self.noConfU):
                    self.configsU01[i,confUtru[i,0:self.NelU]-1]=1

            if (self.NelD>0):
                for i in range(0,self.noConfD):
                    self.configsD01[i,confDtru[i,0:self.NelD]-1]=1
                    
            if ((self.NelU>0) and (self.NelD>0)):
                self.sizHam=self.noConfU*self.noConfD
            elif (self.NelU>0):
                self.sizHam=self.noConfU
            elif (self.NelD>0):
                self.sizHam=self.noConfD
        
    def loadInputs(self,files,ifJupiter,Eshift):
        # INPUTS: as in __init__        
        
        filEU=str(files[0],encoding='ascii')
        filED=str(files[1],encoding='ascii')
        filVUU=str(files[2],encoding='ascii')
        filVDD=str(files[3],encoding='ascii')
        filVUD=str(files[4],encoding='ascii')
                
        # READ SP ENERGIES
        if(ifJupiter):
            with open(filEU,'r') as f1:
                EmatU=np.loadtxt(f1)
            with open(filED,'r') as f2:
                EmatD=np.loadtxt(f2)
        else:
            EmatU=np.loadtxt(filEU)
            EmatD=np.loadtxt(filED)        
            
        if (Eshift<0):
            Eshift=EmatU[0]
        
        EmatU=(EmatU-Eshift)*self.omega
        EmatD=(EmatD-Eshift)*self.omega
                
        # READ CME for 2 electrons of spin UU, DD, UD
        fuu=open(filVUU, "r")
        fdd=open(filVDD, "r")
        fud=open(filVUD, "r")
        luu=fuu.readlines()
        ldd=fdd.readlines()
        lud=fud.readlines()
        
        vecNEU=np.array([self.NESPU,self.NESPU,self.NESPU,self.NESPU])
        vecNED=np.array([self.NESPD,self.NESPD,self.NESPD,self.NESPD])
        vecNE=np.array([self.NESPU,self.NESPD,self.NESPD,self.NESPU])
        VmatUU=self.insertVelem(luu,vecNEU,True)
        VmatDD=self.insertVelem(ldd,vecNED,True)
        VmatUD=self.insertVelem(lud,vecNE,False)
               
        self.VmatUU=VmatUU
        self.VmatDD=VmatDD
        self.VmatUD=VmatUD
        self.EmatU=EmatU
        self.EmatD=EmatD
        
        print("inputs loaded")
        
    def insertVelem(self,lines,noVec,ifSameSpin):
        '''
        reading CME tensor for one spin pair
        INPUTS: - lines from open file
                - dimensions in 4D
                - if spin polarised?
        OUTPUT: CME tensor type complex
        '''
        
        Vmat=np.zeros((self.NESPU,self.NESPU,self.NESPU,self.NESPU),dtype=np.complex_)
        for line in lines:
            ix=0
            ii=np.zeros(4,dtype=int)
            y=[x for x in line.strip().split(' ') if x]
            for x in y:
                if (ix<4):
                    ii[ix]=int(x)-1
                elif(ix==4):                    
                    Vr=float(x)
                elif(ix==5): 
                    Vi=float(x)
                ix=ix+1
            
            if (all(ii<noVec)):
                Vmat[ii[0],ii[1],ii[2],ii[3]]=(Vr+Vi*1j)/self.screen*self.switch
                # insert a flipped element if spins are the same
                if (ifSameSpin):
                    Vmat[ii[1],ii[0],ii[3],ii[2]]=(Vr+Vi*1j)/self.screen*self.switch
                
        return Vmat
    
    def diagCIham(self,nEne,ifSparse):
        '''
        Diagonalises Hamiltonian matrix
        INPUTS: - number of eigenenergies sought from diagonalisation
                - is diagonalisation sparse?
        OUTPUTS: - eigenenergies array
                 - eigenvectors matrix
        '''        
        
        # make Hamiltonian matrix
        if (self.NeltotalU>0 and self.NeltotalD>0):
             hamval,hampos=self.makeCIham()
        elif(self.NeltotalU>0):
             hamval,hampos=self.makeCIhamU()
        elif(self.NeltotalD>0):
             hamval,hampos=self.makeCIhamD()
    
        siz=hampos.shape[0]
        nnz=np.sum(hampos>-1)
        if (nEne>siz):
            nE=siz-2
        else:
            nE=nEne
            
        self.sizHam=siz
        self.hampos=hampos
        self.hamval=hamval
             
        # build sparse CSR format
        indptr=np.zeros(siz+1,dtype=np.int32)
        indic=np.zeros(nnz,dtype=np.int32)
        data=np.zeros(nnz,dtype=np.complex_)
                
        indptr[0]=0
        inds=np.where((hampos==-1).any(axis=1),(hampos==-1).argmax(axis=1),-1)
        last=0
        for i in range(0,siz):
            last=last+inds[i]
            indptr[i+1]=last
            indic[indptr[i]:indptr[i+1]]=hampos[i,0:inds[i]]
            data[indptr[i]:indptr[i+1]]=hamval[i,0:inds[i]]
        
        ham=sp.csr_matrix((data,indic,indptr))
        emin=0
        
        # diagonalise
        if (ifSparse):
            e,v=spla.eigsh(ham,k=nE,sigma=emin)
        else:
            e,v=la.eigh(ham.todense())
        
        return e,v
        
    @staticmethod
    def compareConfigs(co101,co201):
        '''
        conpares two electron configurations: 
        finds number of electrons on different SP states 
        and indices of these states
        IPUTS: two configurations
        OUTPUTS:- ile - number of different electron states
                - diff - different electron states (size varies based on ile)
        '''
        
        n1=co101.shape[0]
        n2=co201.shape[0]
        
        if (n1==n2):
            codiff=np.zeros(n1,dtype=int)
            codiff=co101-co201
            ile=int(sum(codiff!=0)/2)
            if (ile==0):
                diff=np.zeros((1,1),dtype=int)
            elif (ile==1):
                diff=np.zeros((2,1),dtype=int)
                nums=np.arange(0,n1)
                diff[0,:]=nums[codiff>0]
                diff[1,:]=nums[codiff<0]
            elif(ile==2):
                diff=np.zeros((2,2),dtype=int)
                nums=np.arange(0,n1)
                diff[0,:]=nums[codiff>0]
                diff[1,:]=nums[codiff<0]
            else:
                diff=np.zeros((1,1),dtype=int)
        else:
            ile=-1
        
        return ile, diff
        
    def diagSPfreeze(self):
        '''
        Calculates diagonal SP Hamiltonian matrix element
        originating in frozen electrons
        INPUTS: configuration spin U & D
        OUTPUT: Hamiltonian matrix element
        '''
        
        Eu=0
        Ed=0
        if (self.NfreezU>0): Eu=sum(self.EmatU[0:self.NfreezU])
        if (self.NfreezD>0): Ed=sum(self.EmatD[0:self.NfreezD])
        
        E=Eu+Ed
        return E
    
    def diagFreeze(self):
        '''
        Calculates diagonal Hamiltonian matrix element due to Coulomb interaction 
        originating in interaction of frozen electrons among themselves
        it's self energy and vertex correction part, but non converged
        INPUTS: configuration spin U & D
        OUTPUT: Hamiltonian matrix element
        '''
        V=0+0j
        for i in range(0,self.NfreezU):
            for j in range(0,self.NfreezeU):
                if (i!=j):
                    Vd=self.VmatUU[i,j,j,i]
                    Vx=self.VmatUU[i,j,i,j]
                    V=V+(Vd-Vx)/2
            for j in range(0,self.NfreezD):
                Vd=self.VmatUD[i,j,j,i]
                V=V+Vd
        
        for i in range(0,self.NfreezD):
            for j in range(0,self.NfreezD):
                if (i!=j):
                    Vd=self.VmatDD[i,j,j,i]
                    Vx=self.VmatDD[i,j,i,j]
                    V=V+(Vd-Vx)/2
                    
        return V
        
    def diagCofreeze(self,coU,coD):
        '''
        Calculates diagonal Hamiltonian matrix element due to Coulomb interaction 
        originating in interaction with frozen electrons
        it's self energy and vertex correction part, but non converged
        INPUTS: configuration spin U & D
        OUTPUT: Hamiltonian matrix element
        '''
        
        V=0+0j
        for i in range(0,self.NelU):
            for j in range(0,self.NfreezU):
                Vd=self.VmatUU[coU[i],j,j,coU[i]]
                Vx=self.VmatUU[coU[i],j,coU[i],j]
                V=V+(Vd-Vx)
            for j in range(0,self.NfreezD):
                Vd=self.VmatUD[coU[i],j,j,coU[i]]
                V=V+Vd
        
        for i in range(0,self.NelD):
            for j in range(0,self.NfreezD):
                Vd=self.VmatDD[coD[i],j,j,coD[i]]
                Vx=self.VmatDD[coD[i],j,coD[i],j]
                V=V+(Vd-Vx)
            for j in range(0,self.NfreezU):
                Vd=self.VmatUD[j,coD[i],coD[i],j]
                V=V+Vd
        return V
        
    def diagSPCo(self,coU,coD):
        '''
        Calculates diagonal SP Hamiltonian matrix element
        INPUTS: configuration spin U & D
        OUTPUT: Hamiltonian matrix element
        '''
        
        Eu=0
        Ed=0
        
        if (self.noConfU>0): Eu=sum(self.EmatU[coU])
        if (self.noConfD>0): Ed=sum(self.EmatD[coD])
        
        E=Eu+Ed
        return E
    
    def diagCo(self,coU,coD):
        '''
        Calculates diagonal Hamiltonian matrix element due to Coulomb interaction 
        it's self energy and vertex correction part, but non converged
        INPUTS: configuration spin U & D
        OUTPUT: Hamiltonian matrix element
        '''
        
        V=0+0j
        
        for i in range(0,self.NelU):
            for j in range(0,self.NelU):
                if (i!=j):
                    Vd=self.VmatUU[coU[i],coU[j],coU[j],coU[i]]
                    Vx=self.VmatUU[coU[i],coU[j],coU[i],coU[j]]
                    V=V+(Vd-Vx)/2
            for j in range(0,self.NelD):
                Vd=self.VmatUD[coU[i],coD[j],coD[j],coU[i]]
                V=V+Vd
        
        
        for i in range(0,self.NelD):
            for j in range(0,self.NelD):
                if (i!=j):
                    Vd=self.VmatDD[coD[i],coD[j],coD[j],coD[i]]
                    Vx=self.VmatDD[coD[i],coD[j],coD[i],coD[j]]
                    V=V+(Vd-Vx)/2            
        return V
        
    def offDiagOneCoFreeze(self,coU,coD,diff,UorD):
        '''
        calculates the Hamiltonian matrix element 
        for configurations differing by 1 electron state
        originating in interaction with frozen electrons
        INPUTS: - left configuration (bra), 
                - difference between initial and final configuration, 
                - spin U or D
        OUTPUTS: Hamiltonian matrix element
        '''
        il=diff[0,0]
        ir=diff[1,0]
        
        V=0+0j
        for i in range(0,self.NfreezU):
            if (UorD and (i!=il)):
                Vd=self.VmatUU[i,il,ir,i]
                Vx=self.VmatUU[i,il,i,ir]
                V=V+Vd-Vx
            elif (not UorD):
                Vd=self.VmatUD[i,il,ir,i]
                V=V+Vd
        
        for i in range(0,self.NfreezD):
            if ((not UorD) and (i!=il)):
                Vd=self.VmatDD[i,il,ir,i]
                Vx=self.VmatDD[i,il,i,ir]
                V=V+Vd-Vx
            elif (UorD):
                Vd=self.VmatUD[il,i,i,ir]
                V=V+Vd
        return V        
        
    def offDiagOneCo(self,coU,coD,diff,UorD):
        '''
        calculates the Hamiltonian matrix element 
        for configurations differing by 1 electron state
        INPUTS: - left configuration (bra), 
                - difference between initial and final configuration, 
                - spin U or D
        OUTPUTS: Hamiltonian matrix element
        '''
                
        il=diff[0,0]
        ir=diff[1,0]
        
        V=0+0j
        
        # working out the occupied states in the bra and ket for spin U
        ilel=sum(coU>il)
        iler=sum(np.logical_and(coU>ir,coU!=il))
        # sign factor based on the odd or even number of operator swaps 
        # in the bra and ket 
        if ((ilel%2)==(iler%2)):
            facu=1
        else:
            facu=-1
        
        # working out the occupied states in the bra and ket for spin U
        ilel=sum(coD>il)
        iler=sum(np.logical_and(coD>ir,coD!=il))
        # sign factor based on the odd or even number of operator swaps 
        # in the bra and ket 
        if ((ilel%2)==(iler%2)):
            facd=1
        else:
            facd=-1
        
        # for 1 electron difference must sum over all unchanged occupied states
        for i in range(0,self.NelU):
            if (UorD and (coU[i]!=il)):
                Vd=self.VmatUU[coU[i],il,ir,coU[i]]
                Vx=self.VmatUU[coU[i],il,coU[i],ir]
                V=V+facu*(Vd-Vx)
            elif (not UorD):
                Vd=self.VmatUD[coU[i],il,ir,coU[i]]
                V=V+facd*Vd
        
        
        for i in range(0,self.NelD):
            if ((not UorD) and (coD[i]!=il)):
                Vd=self.VmatDD[coD[i],il,ir,coD[i]]
                Vx=self.VmatDD[coD[i],il,coD[i],ir]
                V=V+facd*(Vd-Vx)
            elif (UorD):
                Vd=self.VmatUD[il,coD[i],coD[i],ir]
                V=V+facu*Vd
                
        return V
        
    def offDiagTwoCo(self,coU,coD,diffU,diffD,UorD):
        '''
        calculates the Hamiltonian matrix element 
        for configurations differing by 2 electron states
        INPUTS: - left configuration (bra), 
                - difference between initial and final configuration, 
                - spin U or D
        OUTPUTS: Hamiltonian matrix element
        '''
        
        V=0+0j
        
        if (UorD==-1):
            # sorting indices to work out the sign
            diffl=np.sort(diffD[0,:])
            diffr=np.sort(diffD[1,:])
            
            Vd=self.VmatDD[diffl[0],diffl[1],diffr[1],diffr[0]]
            Vx=self.VmatDD[diffl[0],diffl[1],diffr[0],diffr[1]]
            
            # working out the occupied states in the bra and ket
            if0=(coD!=diffl[0]) & (coD!=diffl[1])
            if1=if0 & (coD>diffl[0]) & (coD<diffl[1])
            if2=if0 & (coD>diffr[0]) & (coD<diffr[1])
            
            ilel=sum(if1)
            iler=sum(if2)
            
            # sign factor based on the odd or even number 
            # of operator swaps in the bra and ket 
            if ((ilel%2)==(iler%2)):
                fac=1
            else:
                fac=-1
                
            V=fac*(Vd-Vx)
            
        elif (UorD==1):
            # sorting indices to work out the sign
            diffl=np.sort(diffU[0,:])
            diffr=np.sort(diffU[1,:])
            
            Vd=self.VmatUU[diffl[0],diffl[1],diffr[1],diffr[0]]
            Vx=self.VmatUU[diffl[0],diffl[1],diffr[0],diffr[1]]
            
            # working out the occupied states in the bra and ket
            if0=(coU!=diffl[0]) & (coU!=diffl[1])
            if1=if0 & (coD>diffl[0]) & (coD<diffl[1])
            if2=if0 & (coD>diffr[0]) & (coD<diffr[1])
            
            ilel=sum(if1)
            iler=sum(if2)
            
            # sign factor based on the odd or even number 
            # of operator swaps in the bra and ket 
            if ((ilel%2)==(iler%2)):
                fac=1
            else:
                fac=-1
                
            V=fac*(Vd-Vx)
            
        elif (UorD==0):
            diffl=np.array([diffU[0,0],diffD[0,0]])
            diffr=np.array([diffU[1,0],diffD[1,0]])
            
            Vd=self.VmatUD[diffl[0],diffl[1],diffr[1],diffr[0]]
            
            # working out the occupied states in the bra and ket
            if1=(coU>diffl[0]) & (coU!=diffl[0])
            if2=(coD<diffl[1]) & (coD!=diffl[1])
            if3=(coU>diffr[0]) & (coU!=diffl[0])
            if4=(coD<diffr[1]) & (coD!=diffl[1])
            
            ilel=sum(if1)+sum(if2)
            iler=sum(if3)+sum(if4)
        
            # sign factor based on the odd or even number 
            # of operator swaps in the bra and ket 
            if ((ilel%2)==(iler%2)):
                fac=1
            else:
                fac=-1
                
            V=fac*Vd
                
        return V
    
    def makeCIham(self):
        '''
        make the Hamiltonian matrix ready to transpose to sparse format
        OUTPUT: - hampos - positions of non0 elements in each row
                - hamval - values of non0 elements in each row
        '''
        
        leng=self.sparseLeng
        # precalculate elements from frozen electrons
        Efreez=self.diagSPfreeze()
        Vfreez=self.diagFreeze()
        
        hamval=np.zeros((self.sizHam,leng),dtype=np.complex_) # positions
        hampos=-1*np.ones((self.sizHam,leng),dtype=int) # values
        counts=np.zeros(self.sizHam,dtype=int) # count elements in a row
        
        confCount=0 # used to store the basis of the Hamiltonian
        allConfs=[] # configuration Hamiltonian basis
        # loop over all configurations, each spin separately
        for i in range(0,self.noConfU):
            co1u=self.configsU[i,:]
            co1u01=self.configsU01[i,:]
            
            for i2 in range(0,self.noConfU):
                co2u=self.configsU[i2,:]
                co2u01=self.configsU01[i2,:]
                
                # compare configurations U
                ileu,diffu=self.compareConfigs(co1u01,co2u01)
                
                # if difference between configurations is ok, proceed
                if (ileu!=-1):
                    
                    for j in range(0,self.noConfD):
                        co1d=self.configsD[j,:]
                        co1d01=self.configsD01[j,:]
                        
                        ind1=i*self.noConfD+j
                        
                        if (i2==0):
                            # store Hamiltonian basis
                            allConfs.append((confCount,co1u,co1d))
                            confCount+=1
                        
                        for j2 in range(0,self.noConfD):
                            co2d=self.configsD[j2,:]
                            co2d01=self.configsD01[j2,:]
                            
                            ind2=i2*self.noConfD+j2
    
                            # compare configurations D
                            iled,diffd=self.compareConfigs(co1d01,co2d01)
    
                            # if difference between configurations is ok, proceed
                            if (iled!=-1):
                                ile=iled+ileu
                                if (ile<3):
                                
                                    # DIAGONAL TERM
                                    if (ile==0):
                                        Ediag=self.diagSPCo(co1u,co1d)+Efreez # SP energy
                                        # self energy and vertex correction
                                        Vdiag=self.diagCo(co1u,co1d)+Vfreez
                                        Vdiag=Vdiag+self.diagCofreeze(co1u,co1d) 
                                        
                                        hampos[ind1,counts[ind1]]=ind2
                                        hamval[ind1,counts[ind1]]=Ediag+Vdiag
                                        counts[ind1]=counts[ind1]+1
        
                                    # OFF DIAGONAL TERM - 1 ELECTRON DIFFERENCE
                                    elif (ile==1):
                                        
                                        if (ileu==1):
                                            UorD=True
                                            Voff1=self.offDiagOneCo(co1u,co1d,diffu,UorD)
                                            Voff1=Voff1+self.offDiagOneCoFreeze(co1u,co1d,diffu,UorD)
                                        elif (iled==1):
                                            UorD=False
                                            Voff1=self.offDiagOneCo(co1u,co1d,diffd,UorD)
                                            Voff1=Voff1+self.offDiagOneCoFreeze(co1u,co1d,diffd,UorD)
                                        else:
                                            Voff1=0+0j
                                            
                                        if (abs(Voff1)>1e-8):
                                            hampos[ind1,counts[ind1]]=ind2
                                            hamval[ind1,counts[ind1]]=Voff1
                                            counts[ind1]=counts[ind1]+1
                                        
                                    # OFF DIAGONAL TERM - 2 ELECTRON DIFFERENCE
                                    elif (ile==2):
                                        
                                        if (ileu==2):
                                            UD=1
                                            Voff2=self.offDiagTwoCo(co1u,co1d,diffu,diffd,UD)
                                            
                                        elif (iled==2):
                                            UD=-1
                                            Voff2=self.offDiagTwoCo(co1u,co1d,diffu,diffd,UD)
                                        elif (ileu==1 and iled==1):
                                            UD=0
                                            Voff2=self.offDiagTwoCo(co1u,co1d,diffu,diffd,UD)
                                        else:
                                            Voff2=0+0j
                                            
                                        if (abs(Voff2)>1e-8):
                                            hampos[ind1,counts[ind1]]=ind2
                                            hamval[ind1,counts[ind1]]=Voff2
                                            counts[ind1]=counts[ind1]+1
                                    
        self.allConfs=allConfs
                                    
        return   hamval,hampos
    
    def makeCIhamU(self):
        '''
        make the Hamiltonian matrix for ready to transpose to sparse format
        spin polarised case for spin U
        OUTPUT: - hampos - positions of non0 elements in each row
                - hamval - values of non0 elements in each row
        '''
        leng=self.sparseLeng
        # precalculate elements from frozen electrons
        Efreez=self.diagSPfreeze()
        Vfreez=self.diagFreeze()
        
        hamval=np.zeros((self.sizHam,leng),dtype=np.complex_)
        hampos=-1*np.ones((self.sizHam,leng),dtype=int)
        counts=np.zeros(self.sizHam,dtype=int)
                
        confCount=0 # used to store the basis of the Hamiltonian
        allConfs=[] # configuration Hamiltonian basis
        # loop over all configurations, each spin separately
        for i in range(0,self.noConfU):
            co1u=self.configsU[i,:]
            co1u01=self.configsU01[i,:]
            ind1=i
            
            allConfs.append((confCount,co1u))
            confCount+=1                        
            
            for i2 in range(0,self.noConfU):
                co2u=self.configsU[i2,:]
                co2u01=self.configsU01[i2,:]
                ind2=i2
                
                # compare configurations U
                ileu,diffu=self.compareConfigs(co1u01,co2u01)
                
                # if difference between configurations is ok, proceed
                if (ileu!=-1):                    
                    
                    ile=ileu
                    if (ile<3):
                        
                        # DIAGONAL TERM
                        if (ile==0):
                            Ediag=self.diagSPCo(co1u,co1u)+Efreez # SP energy
                            # self energy and vertex correction
                            Vdiag=self.diagCo(co1u,co1u)+Vfreez
                            Vdiag=Vdiag+self.diagCofreeze(co1u,co1u)
                            
                            hampos[ind1,counts[ind1]]=ind2
                            hamval[ind1,counts[ind1]]=Ediag+Vdiag
                            counts[ind1]=counts[ind1]+1
    
                        # OFF DIAGONAL TERM - 1 ELECTRON DIFFERENCE
                        elif (ile==1):
                            
                            if (ileu==1):
                                UorD=True
                                Voff1=self.offDiagOneCo(co1u,co1u,diffu,UorD)
                                Voff1=Voff1+self.offDiagOneCoFreeze(co1u,co1u,diffu,UorD)
                            else:
                                Voff1=0+0j
                                
                            if (abs(Voff1)>1e-8):
                                hampos[ind1,counts[ind1]]=ind2
                                hamval[ind1,counts[ind1]]=Voff1
                                counts[ind1]=counts[ind1]+1
    
                        # OFF DIAGONAL TERM - 2 ELECTRON DIFFERENCE
                        elif (ile==2):
                            
                            if (ileu==2):
                                UD=1
                                Voff2=self.offDiagTwoCo(co1u,co1u,diffu,diffu,UD)
                                
                            else:
                                Voff2=0+0j
                                
                            if (abs(Voff2)>1e-8):
                                hampos[ind1,counts[ind1]]=ind2
                                hamval[ind1,counts[ind1]]=Voff2
                                counts[ind1]=counts[ind1]+1
                                 
        self.allConfs=allConfs       
                                    
        return   hamval,hampos
        
    def makeCIhamD(self):
        '''
        make the Hamiltonian matrix for ready to transpose to sparse format
        spin polarised case for spin D
        OUTPUT: - hampos - positions of non0 elements in each row
                - hamval - values of non0 elements in each row
        '''
        leng=self.sparseLeng
        # precalculate elements from frozen electrons
        Efreez=self.diagSPfreeze()
        Vfreez=self.diagFreeze()
        
        hamval=np.zeros((self.sizHam,leng),dtype=np.complex_)
        hampos=-1*np.ones((self.sizHam,leng),dtype=int)
        counts=np.zeros(self.sizHam,dtype=int)
                
        confCount=0 # used to store the basis of the Hamiltonian
        allConfs=[] # configuration Hamiltonian basis
        # loop over all configurations, each spin separately
        for j in range(0,self.noConfD):
            co1d=self.configsD[j,:]
            co1d01=self.configsD01[j,:]
            
            ind1=j
            allConfs.append((confCount,co1d))
            confCount+=1                        
            
            for j2 in range(0,self.noConfD):
                co2d=self.configsD[j2,:]
                co2d01=self.configsD01[j2,:]
                
                ind2=j2

                # compare configurations D
                iled,diffd=self.compareConfigs(co1d01,co2d01)

                # if difference between configurations is ok, proceed
                if (iled!=-1):
                    ile=iled
                    
                    if(ile<3):
                        # DIAGONAL TERM
                        if (ile==0):
                            Ediag=self.diagSPCo(co1d,co1d)+Efreez # SP energy
                            # self energy and vertex correction
                            Vdiag=self.diagCo(co1d,co1d)+Vfreez
                            Vdiag=Vdiag+self.diagCofreeze(co1d,co1d)
                            
                            hampos[ind1,counts[ind1]]=ind2
                            hamval[ind1,counts[ind1]]=Ediag+Vdiag
                            counts[ind1]=counts[ind1]+1
    
                        # OFF DIAGONAL TERM - 1 ELECTRON DIFFERENCE
                        elif (ile==1):
                            
                            if (iled==1):
                                UorD=True
                                Voff1=self.offDiagOneCo(co1d,co1d,diffd,UorD)
                                Voff1=Voff1+self.offDiagOneCoFreeze(co1d,co1d,diffd,UorD)
                            else:
                                Voff1=0+0j
                                
                            if (abs(Voff1)>1e-8):
                                hampos[ind1,counts[ind1]]=ind2
                                hamval[ind1,counts[ind1]]=Voff1
                                counts[ind1]=counts[ind1]+1
    
                        # OFF DIAGONAL TERM - 2 ELECTRON DIFFERENCE
                        elif (ile==2):
                            
                            if (iled==2):
                                UD=-1
                                Voff2=self.offDiagTwoCo(co1d,co1d,diffd,diffd,UD)
                            else:
                                Voff2=0+0j
                                
                            if (abs(Voff2)>1e-8):
                                hampos[ind1,counts[ind1]]=ind2
                                hamval[ind1,counts[ind1]]=Voff2
                                counts[ind1]=counts[ind1]+1
                    
        self.allConfs=allConfs
        
                                    
        return   hamval,hampos
    
    
                                    
                                    
                                    
                                    
                                    
                                    
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 11:02:36 2020

@author: ludka szulakowska
"""

