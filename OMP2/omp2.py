import psi4
import numpy as np
import scipy.linalg as la
from uhf import UHF

class OMP2:

    def __init__(self, uhf):
        #self.e = np.sort(uhf.e)
        self.C = uhf.C[:,uhf.e.argsort()]
        self.nocc = uhf.nocc
        self.max_iter = uhf.max_iter
        self.ntot = uhf.ntot
        self.nvir = self.ntot-self.nocc
        g = spin_block_tei(uhf.I) 
        self.hao = spin_block_oei(uhf.H)
        self.gao = g.transpose(0,2,1,3)-g.transpose(0,2,3,1) 

    def get_energy(self): 
        C, gao, hao,nocc, ntot, nvir = self.C, self.gao,self.hao, self.nocc, self.ntot, self.nvir
    
        o = slice(None,nocc)
        v = slice(nocc,None)
        x = np.newaxis
        odm = np.zeros((ntot,ntot))
        tdm = np.zeros((ntot,ntot,ntot,ntot)) 
        odmref = np.zeros((ntot,ntot))
        odmref[o,o] = np.identity(nocc)
        X = np.zeros((ntot,ntot))        

        hmo_old = int_trans_oei(hao,C)
        gmo_old = int_trans(gao,C)
        t_old = np.zeros((nocc,nocc,nvir,nvir))
        E_OMP2_old = 0.0

        for iteration in range(self.max_iter):
            f = hmo_old + np.einsum('piqi -> pq',gmo_old[:,o,:,o])
            #off diagonal Fock Matrix
            fprime = f.copy()
            np.fill_diagonal(fprime, 0)
            #print(iteration)
            #print(fprime)

            #updated orbital energies 
            e = f.diagonal()
 
            # t amplitudes
            t2 = np.einsum('ac,ijcb -> ijab',fprime[v,v],t_old)
            t3 = np.einsum('ki,kjab -> ijab', fprime[o,o],t_old)
            t = (gmo_old[o,o,v,v] + t2 - t2.transpose((0,1,3,2)) - t3 + t3.transpose((1,0,2,3)))
            t /= (e[o,x,x,x]+e[x,o,x,x]-e[x,x,v,x]-e[x,x,x,v])
            #one and two particle density matrices 
            odm[v,v] = (1/2)*np.einsum('ijac,ijbc -> ab',t,t)            
            odm[o,o] = -(1/2)*np.einsum('jkab,ikab -> ij',t,t)
            tdm[v,v,o,o] = t.T
            tdm[o,o,v,v] = t
            tdm2 = np.einsum('pr,qs -> pqrs', odm,odmref)
            tdm3 = np.einsum('pr,qs->pqrs', odmref,odmref)
            odm_gen = odm + odmref
            tdm_gen = tdm + tdm2 - tdm2.transpose((1,0,2,3))-tdm2.transpose((0,1,3,2))+tdm2.transpose((1,0,3,2)) + tdm3 - tdm3.transpose((0,1,3,2))
            #print(np.trace(odm_gen))        
            #Newton-Raphson
            F = np.einsum('pr,rq->pq',hmo_old,odm_gen)+(1/2)*np.einsum('prst,qrst -> pq',gmo_old,tdm_gen)
            X[o,v] = ((F-F.T)[o,v])/(e[o,x]-e[x,v])
            #rotate coefficients
            U = la.expm(X-X.T)
            C = C.dot(U) 
            #transform integrals
            hmo = int_trans_oei(hao,C)
            gmo = int_trans(gao,C)
            # get energy
            E_OMP2 = uhf.mol.nuclear_repulsion_energy() + np.einsum('pq,pq ->',hmo,odm_gen) + (1/4)*np.einsum('pqrs,pqrs ->',gmo,tdm_gen)
            print('OMP2 iteration{:3d}: energy {:20.14f} dE {:2.5E}'.format(iteration,E_OMP2,(E_OMP2-E_OMP2_old)))

            if (abs(E_OMP2-E_OMP2_old)) < 1.e-10:
                break

            #updating values
            gmo_old = gmo
            hmo_old = hmo
            t_old = t
            E_OMP2_old = E_OMP2
        print('The final OMP2 energy is {:20.14f}'.format(E_OMP2))
            
def spin_block_tei(gao):
    I = np.eye(2)
    gao = np.kron(I, gao)
    return np.kron(I, gao.T)

def spin_block_oei(hao):
    hao = la.block_diag(hao,hao)
    return hao

def int_trans_oei(hao, C):

    return np.einsum('pQ, pP -> PQ',
           np.einsum('pq, qQ -> pQ', hao, C),C)

def int_trans(gao, C):

    return np.einsum('pQRS, pP -> PQRS',
           np.einsum('pqRS, qQ -> pQRS',
           np.einsum('pqrS, rR -> pqRS',
           np.einsum('pqrs, sS -> pqrS', gao, C),C),C),C)


if __name__ == '__main__':
    uhf = UHF('Options.ini')
    uhf.get_energy()
    omp2 = OMP2(uhf)
    omp2.get_energy()
    psi4.set_options({'basis':'sto-3g',
                        'scf_type': 'pk',
                        'MP2_type' : 'conv',
                        'puream' : False,
                        'reference': 'uhf',
                        'guess' : 'core',
                        'e_convergence' : 1e-10})
    #psi4.energy('omp2')
