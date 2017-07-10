# Alternate UHF code separating alpha and beta spin orbitals

import psi4
import numpy as np
import configparser 
import scipy.linalg as la
import integrals 
class UHF:
    def __init__(self, filename = 'Options.ini'):
        config = configparser.ConfigParser()
        config.read(filename)
        self.mol = psi4.geometry(config['DEFAULT']['molecule'])
        self.mol.update_geometry()
        self.basis = psi4.core.BasisSet.build(self.mol, 'BASIS', config['DEFAULT']['basis'],puream=0)
        mints = psi4.core.MintsHelper(self.basis)
        
        self.max_iter = int(config['SCF']['max_iter'])
        self.nalpha = int(config['DEFAULT']['nalpha'])
        self.nbeta = int(config['DEFAULT']['nbeta'])
        self.nelec = -self.mol.molecular_charge()
        self.diis = int(config['SCF']['diis'])
        self.nvector = int(config['SCF']['diis_nvector'])
        self.start = int(config['SCF']['diis_start'])

        for A in range(self.mol.natom()):
            self.nelec += int(self.mol.Z(A))
        self.nocc = self.nelec 
        self.ntot = mints.basisset().nbf()
        
        self.mu_nuc = psi4.core.nuclear_dipole(self.mol).to_array()        
        self.S = integrals.int_overlap(self.mol,self.basis)
        print(self.S)
        T = integrals.int_kinetic(self.mol,self.basis)
        V = mints.ao_potential().to_array()
        self.I = mints.ao_eri().to_array() 
        
        self.H = T + V
        self.C = np.zeros_like((len(self.H)*2,len(self.H)*2))
        self.C_a = np.zeros_like(self.H)
        self.C_b = np.zeros_like(self.H)
        self.Dtot = np.zeros_like(self.H)
        self.ea = np.zeros(len(self.H))
        self.eb = np.zeros(len(self.H))
        self.e = np.zeros(len(self.H)*2)
        self.A = np.matrix(la.inv(la.sqrtm(self.S)))
        

    def get_energy(self):
        S,mol, max_iter, nalpha,nbeta,I, H, A, C_a, C_b, ea, eb,Dtot,diis, nvector,start =\
        self.S,self.mol, self.max_iter, self.nalpha, self.nbeta,self.I, self.H, self.A, self.C_a, self.C_b, self.ea, self.eb, self.Dtot,\
        self.diis, self.nvector, self.start
        Fa = H
        Fb = H 
        E_old = 0.0

        EM_a = []
        EM_b = []
        FA = []
        FB = []
        for iteration in range(1, self.max_iter+1):

            Ft_a = A.dot(Fa).dot(A)
            Ft_b = A.dot(Fb).dot(A)
            ea, C_a = np.linalg.eigh(Ft_a)
            eb, C_b = np.linalg.eigh(Ft_b)
            C_a = A.dot(C_a)
            C_b = A.dot(C_b)
            Ca = C_a[:,:nalpha]
            Cb = C_b[:,:nbeta]
            Da = np.einsum('pi, qi->pq', Ca, Ca)
            Db = np.einsum('pi, qi->pq', Cb, Cb)
            Dtot = Da + Db
            
            Ja = np.einsum('pqrs, rs->pq', I, Da)
            Jb = np.einsum('pqrs, rs->pq', I, Db)
            Ka = np.einsum('prqs, rs->pq', I, Da)
            Kb = np.einsum('prqs, rs->pq', I, Db)
            Fa = H + Ja - Ka + Jb
            Fb = H + Jb - Kb + Ja
        
            if diis==1 and iteration >= start:
                em_a = Fa.dot(Da).dot(S) - S.dot(Da).dot(Fa)
                em_b = Fb.dot(Db).dot(S) - S.dot(Db).dot(Fb)
        #     print(em_a)
            #  print(Da)
                if len(EM_a) < nvector:
                    EM_a.append(em_a)
                    FA.append(Fa)
                    EM_b.append(em_b)
                    FB.append(Fb)
                else:
                    EM_a.pop(0)
                    EM_a.append(em_a)
                    FA.pop(0)
                    FA.append(Fa)
                
                    EM_b.pop(0)
                    EM_b.append(em_b)
                    FB.pop(0)
                    FB.append(Fb)

                n = len(EM_a)
                m = len(EM_b)
                p = np.zeros((n,n))
                if n>=2:
                    for i in range(n):
                        for j in range(n):
                            p[i,j] = np.vdot(EM_a[i],EM_a[j])+np.vdot(EM_b[i],EM_b[j])
                            
                    ff = np.array([0.]*n+[-1.])
                    P = -np.ones((n+1,n+1))
                    P[:n,:n] = p
                    P[n,n] = 0
                    q = np.linalg.solve(P,ff)
                    Fa = sum(q[i]*FA[i] for i in range(n))
                    Fb = sum(q[i]*FB[i] for i in range(m))



            E_SCF = (1/2)*(np.einsum('pq, pq->', Fa+H, Da) + np.einsum('pq,pq->',Fb+H, Db)) +mol.nuclear_repulsion_energy()
            print('UHF iteration {:3d}: energy {:20.14f} dE {:1.5E}'.format(iteration, E_SCF, (E_SCF - E_old)))

            if (abs(E_SCF - E_old) < 1.e-10):
                break
             
            E_old = E_SCF
            
        self.C_a, self.C_b = C_a, C_b 
        self.ea, self.eb = ea, eb
        self.Dtot = Dtot
    
    def get_dipole(self):
        Dtot,mu_nuc = self.Dtot, self.mu_nuc 
        dx,dy,dz = integrals.int_dipole(self.mol,self.basis)    
        mux = -np.einsum('pq,pq->', Dtot, dx) + mu_nuc[0]   
        muy = -np.einsum('pq,pq->', Dtot, dy) + mu_nuc[1]  
        muz = -np.einsum('pq,pq->', Dtot, dz) + mu_nuc[2]
        mu = np.sqrt(mux**2 + muy**2 + muz**2)
        print('The dipole moment is {:20.14f}'.format(mu))
        return mu  

if __name__=='__main__':

    uhf = UHF('Options.ini')
    uhf.get_energy() 
    uhf.get_dipole()
