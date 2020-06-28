# -*- coding: utf-8 -*-
"""
CIGRE microgrid benchmark class
"""

import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix
from matplotlib import pyplot as plt

def Wind_Sce(Pnom, dist, K1, K2):
  ''' 
  Wind Generation scenario assumming a normal, Weibull or beta distributions, 
  with parameters K1 and K2.
  '''
  weibull = 1.0
  beta = 2.0
  normal = 3.0
  if (dist == weibull):    
    v = (-K1*np.log(1.-np.random.rand()))**(1.0/K2)        
  
  if (dist == beta):
    v  = np.nan
  
  if (dist == normal):
     v = (K1+np.random.randn()*K2)  
     
  P = 0. 
  if (v<=12.):
      P = Pnom*(v/12.)**3
  else:
      P = Pnom
  return P

def Solar_Sce(Pnom, dist, K1, K2):
  ''' 
  Solar Generation scenario assumming a normal, Weibull or beta distributions, 
  with parameters K1 and K2.  Nominal radiation scenario 1000 W/m^2  
  which represents a nomila power Pnom.
  '''
  weibull = 1.0
  beta = 2.0
  normal = 3.0
  P = 0.
  if (dist == weibull):    
    v = (-K1*np.log(1.-np.random.rand()))**(1.0/K2)    
    if (v<=12.):
        P = Pnom*(v/12.)**3
    else:
        P = Pnom
  if (dist == beta):
    P  = np.nan

  if (dist == normal):
      P = (K1+np.random.randn()*K2)/1000*Pnom  
  return P


class cigre(object):
    def __init__(self):
        Vnom = 400 # Line to Line voltage in volts
        wnom = 2.*np.pi*60. # Nominal angular velocity
        Pnom = 100000 # Nominal power in watts
        # N1 N2 L(m) type 
        Lines = np.array([[0,1,35,0],
                       [1,2,35,0],
                       [2,3,35,0],
                       [3,4,35,0],
                       [4,5,35,0],
                       [5,6,35,0],
                       [6,7,35,0],
                       [7,8,35,0],
                       [8,9,35,0],
                       [2,10,30,1],
                       [3,11,30,2],
                       [5,12,30,3],
                       [9,13,30,2],
                       [3,14,35,0],
                       [14,15,35,0],
                       [15,16,35,0],
                       [16,17,30,0],
                       [8,18,30,1]])
        # Rph Xph Ro Xo(ohm/km) Cap(uF/km)
        Impedances = np.array([[0.284,0.083,1.136,0.417,0.38], #(1) OL-Twisted cable 4x120 mm2
                               [3.690,0.094,13.64,0.472,0.05], #(2) SC - 4x6 mm2 Cu
                               [1.380,0.082,5.520,0.418,0.18], #(3) SC - 4x16 mm2 Cu
                               [0.871,0.081,3.480,0.409,0.22]])#(4) SC - 4x25 mm2 Cu   
        
        Demand = np.array([[10,15000.0],
                        [12,55000.0],
                        [13,47000.0],
                        [17,72000.0],
                        [18,15000.0]])
        solar = 1.0
        wind = 2.0
        weibull = 1.0
        beta = 2.0
        normal = 3.0
        # node power Kp Kq Tau type dist K1 K2
        Converters = np.array([[11,30000,0.05,0.04,0.32E-3,solar,normal,900,40],
                            [12,10000,0.08,0.09,0.38E-3,solar,normal,900,40],
                            [13,10000,0.10,0.09,0.41E-3,wind,weibull,11,1.2],
                            [17,30000,0.09,0.10,0.31E-3,solar,normal,900,40],
                            [18,10000,0.08,0.08,0.34E-3,wind,weibull,11,1.2]])
        #Organizar la estructura
        NumN = np.max([np.max(Lines[:,0]),np.max(Lines[:,1])])+1
        NumL = np.size(Lines[:,0])
        NumD = np.size(Demand[:,0]) 
        NumC = np.size(Converters[:,0])
        Zbase = Vnom*Vnom/Pnom
        Y = 1j*np.zeros((NumN,NumN))
        Yc = 1j*np.zeros((NumN,1))
        mg = 1j*np.zeros((NumL,4))
        zlin = 1j*np.zeros((NumL,1))
        A = np.zeros((NumL,NumN))
        for k in range(NumL):
            n1 = Lines[k,0]
            n2 = Lines[k,1]
            t = Lines[k,3]
            z = (Impedances[t,0]+1j*Impedances[t,1])/1000/Zbase*Lines[k,2]
            b = 1j*Impedances[t,4]*wnom/1000*Lines[k,2]*Zbase*1E-6
            mg[k,0:4] = np.array([n1,n2,z,b])
            Y[n1,n1] = Y[n1,n1] + 1.0/z + b
            Y[n1,n2] = Y[n1,n2] - 1.0/z
            Y[n2,n1] = Y[n2,n1] - 1.0/z
            Y[n2,n2] = Y[n2,n2] + 1.0/z + b
            A[k,n1] =  1.0
            A[k,n2] = -1.0
            zlin[k] = z
            Yc[n1] = Yc[n1] + b
            Yc[n2] = Yc[n2] + b
            
        # Include the demand as constant impedances
        Demand[:,1] = Demand[:,1]/Pnom
        Yd = np.zeros((NumN,1))
        for k in range(NumD):
            n1 = np.int(Demand[k,0])
            g = Demand[k,1]
            Y[n1,n1] = Y[n1,n1] + g
            Yd[n1] = Yd[n1] + g
            
        # Load flow in grid mode
        Converters[:,1] = Converters[:,1]/Pnom
        s = np.zeros((NumN,))
        for k in range(NumC):
            n1 = np.int(Converters[k,0])
            s[n1] = Converters[k,1]
            
        Yn = csr_matrix(Y[1:,1:])
        Y0 = Y[1:,0]
        V  = 1j*np.ones((NumN,))
        er = 1.0
        while er>1E-8:
            V[1:] = spsolve(Yn,np.conj(s[1:]/V[1:])-Y0*V[0])
            In = Y@V
            sc = V*np.conj(In)
            er = np.linalg.norm(s[1:]-sc[1:])
            
        self.NumN = NumN  # Numbe of nodes
        self.NumL = NumL  # Number of lines
        self.NumD = NumD  # Number of loads
        self.NumC = NumC  # Number of converters
        self.lines = mg
        self.demand = Demand
        self.converters = Converters
        self.Ybus = Y
        self.V = V
        self.A = A
        self.zlin = zlin
        self.Yd = Yd
        self.Yc = Yc
        self.MC = lambda:0 # Easy way to create an object
        self.MC.nd = 100 # Number of samples by default

    def Newton_Freq(self, convs):
        # initialization
        w = 1.0
        Vn = np.ones((5,1))
        an = np.zeros((5,1))

        xi = np.expand_dims(convs[:,2], axis=1)
        zita = np.expand_dims(convs[:,3], axis=1)
        N = np.int_(convs[:,0])
        S = np.setdiff1d(np.arange(self.NumN),N)
        n = np.size(N)
        Pref = np.expand_dims(convs[:,1], axis=1)
        Qref = np.zeros((n,1))
        #er = 10
        for ite in range(50):
            ylin = np.diag( 1./(np.real(self.zlin)+w*1j*np.imag(self.zlin))[:,0] )
            Llin = np.diag( np.imag(self.zlin)[:,0] )
            Yd2 = np.real(self.Yd) + w*1j*np.imag(self.Yd)  # modelo las cargas como admitancias
            Yb = (self.A.T @ ylin @ self.A) + np.diag(Yd2[:,0]) + np.diag(self.Yc[:,0])*w
            Db = -(self.A.T @ ylin)@(1j*Llin)@(ylin @ self.A) + 1j*np.imag(Yd2) + np.diag(self.Yc[:,0])
            Y = Yb[np.ix_(N, N)] - Yb[np.ix_(N, S)]@(np.linalg.inv(Yb[np.ix_(S, S)])@Yb[np.ix_(S, N)])
            D = Db[np.ix_(N, N)] - Db[np.ix_(N, S)]@(np.linalg.inv(Db[np.ix_(S, S)])@Db[np.ix_(S, N)])
            Dr = np.real(D)
            Di = np.imag(D)
            G = np.real(Y)
            B = np.imag(Y)
            Vs = Vn*np.exp(1j*an)
            I = Y@Vs
            Sn = Vs*np.conj(I)
            P = np.real(Sn)
            Q = np.imag(Sn)
            dP = (Pref-P)+(1.-w)/xi
            dQ = (Qref-Q)+(1.-Vn)/zita
            dT = (np.sum(an/xi)/np.sum(1./xi))
            er = np.linalg.norm(np.array([np.linalg.norm(dP),np.linalg.norm(dQ),np.abs(dT)]))
            Dpt = np.zeros((n,n))
            Dpv = np.zeros((n,n))
            Dpw = np.zeros((n,1))
            Dqt = np.zeros((n,n))
            Dqv = np.zeros((n,n))
            Dqw = np.zeros((n,1))   
            for k in range(n):
                Dpw[k] = 0.
                Dqw[k] = 0.
                for m in range(n):
                    akm = an[k]-an[m]
                    if k is m:
                        Dpt[k,k] = -B[k,k]*Vn[k]**2-Q[k]
                        Dpv[k,k] =  G[k,k]*Vn[k]+P[k]/Vn[k]
                        Dqt[k,k] = -G[k,k]*Vn[k]**2+P[k]
                        Dqv[k,k] = -B[k,k]*Vn[k]+Q[k]/Vn[k]
                    else:                
                        Dpv[k,m] = G[k,m]*Vn[k]*np.cos(akm)+B[k,m]*Vn[k]*np.sin(akm)
                        Dqv[k,m] = -(B[k,m]*Vn[k]*np.cos(akm)-G[k,m]*Vn[k]*np.sin(akm))  # TODO: check this term
                        Dpt[k,m] =  Dqv[k,m]*Vn[m]                                 #TODO: check this too!
                        Dqt[k,m] = -Dpv[k,m]*Vn[m]

                    Dpw[k] = Dpw[k] + Dr[k,m]*Vn[k]*Vn[m]*np.cos(akm)+Di[k,m]*Vn[k]*Vn[m]*np.sin(akm)
                    Dqw[k] = Dqw[k] - Di[k,m]*Vn[k]*Vn[m]*np.cos(akm)+Dr[k,m]*Vn[k]*Vn[m]*np.sin(akm)
            

            Jac = np.block([ [Dpt, Dpv, Dpw+1./xi],
                   [Dqt, Dqv+np.diag((1./zita)[:,0]), Dqw],
                   [-1./xi.T, np.zeros((1,n+1))] ])
            dX = np.linalg.solve(Jac,np.block([[dP],[dQ],[dT]]))
            an = an + dX[:n]
            Vn = Vn + dX[n:2*n]
            w  = w + dX[-1]
            if er<1E-8:
                #print(w)
                break
        #print(w)
        return w,Vs,P,Q

    def Montecarlo(self,xi,zita,graficar=False, flagres=False):
        # Frequency variation given by p and q
        nd = self.MC.nd  # Monte Carlo number of iterations
        w = np.zeros((nd,1))
        dv = np.zeros((nd,2))
        dpq = np.zeros((nd,2)) # caching the diference between p_ref and p
        convs = self.converters.copy()
        convs[:,2] = xi 
        convs[:,3] = zita
        solar = 1.0
        wind = 2.0
        #demand = self.demand.copy()
        for k in range(nd):
            #demand[:,1]= self.demand[:,1]*np.random.rand(self.NumD)*10  # Demands are uniform distributed
            for c in range(self.NumC):
                Pnom = self.converters[c,1]
                dist = self.converters[c,6]
                K1 = self.converters[c,7]
                K2 = self.converters[c,8]
                if (self.converters[c,5] == solar):
                    convs[c,1] = Solar_Sce(Pnom,dist,K1,K2)
                    #print('Solar: ',convs[c,1])
                    
                if (self.converters[c,5] == wind):
                    convs[c,1] = Wind_Sce(Pnom,dist,K1,K2)
                    #print('Wind: ',convs[c,1])
                    
            w[k],v,p,q = self.Newton_Freq(convs)
            dv[k,0] = np.min(np.abs(v))
            dv[k,1] = np.max(np.abs(v))
            dpq[k,0] = np.max(np.abs(p[:,0]-convs[:,1]))
            dpq[k,1] = np.max(np.abs(q))
        
        if graficar is True:
            fig01 = plt.figure()
            ax01 = fig01.add_subplot(1, 1, 1) 
            plt.hist(w,30)
            ax01.grid()
            ax01.set_title('$\omega$')
            fig02, (ax021, ax022) = plt.subplots(nrows=2)
            ax021.hist(dpq[:,0],30)
            ax021.grid()
            ax021.set_title('$\Delta$ p')
            ax022.hist(dpq[:,1],30)
            ax022.grid()
            ax022.set_title('$\Delta$ q')
            fig03, (ax031, ax032) = plt.subplots(nrows=2)
            ax031.hist(dv[:,0],30)
            ax031.grid()
            ax031.set_title('$v_{min}$')
            ax032.hist(dv[:,1],30)
            ax032.grid()
            ax032.set_title('$v_{max}$')


        # results on w
        self.MC.mu_w = np.mean(w)
        self.MC.sigma_w = np.std(w)
    
        a = np.where(w>1.05)
        self.MC.var_pos_w = np.size(a)/nd
    
        a = np.where(w<0.95)
        self.MC.var_neg_w = np.size(a)/nd
      
        # results on vmax
        a = np.where(dv[:,1]>1.05)
        self.MC.var_pos_v = np.size(a)/nd
    
        # results on vmin
        a = np.where(dv[:,0]<0.95)
        self.MC.var_neg_v = np.size(a)/nd
    
        # results on delta_p  % deviations larger than 0.5
        a = np.where(dpq[:,0]>0.5)
        self.MC.var_dp = np.size(a)/nd
    
        # results on delta_q  % deviations larger than 0.5
        a = np.where(dpq[:,1]>0.5)
        self.MC.var_dq = np.size(a)/nd
    
        self.resTol = self.MC.var_pos_w+self.MC.var_neg_w+self.MC.var_pos_v+\
                      self.MC.var_neg_v+self.MC.var_dq+self.MC.var_dp 

        # Standar Deviation
        self.resDev = np.sum(np.square(w-1.0))/(w.size-1.) + np.sum(np.square(dv[:,0]-1.0))/(dv.shape[0]-1.) + \
                   np.sum(np.square(dv[:,1]-1.0))/(dv.shape[0]-1.) + np.sum(np.square(dpq[:,0]))/(dpq.shape[0]-1.) + \
                   np.sum(np.square(dpq[:,1]))/(dpq.shape[0]-1.)

        #  self.res = np.max(w, initial = 1.0)-np.min(w,initial = 1.0) +\
        #             np.max(dv[:,1], initial = 1.0)-np.min(dv[:,0],initial = 1.0) +\
        #             np.max(dpq[:,0],initial = 0.5)+np.max(dpq[:,1],initial = 0.5)-1.0

        if flagres:
            self.res = self.resTol
        else:
            self.res = self.resDev



if __name__ == "__main__":
    mg = cigre()
    #xi = 0.05 + 0.15*np.ones((1,5)) #np.random.rand(1,5)
    #zita = 0.05 + 0.15*np.ones((1,5)) #np.random.rand(1,5)
    xi = np.array([[0.0636,0.0806,0.0639,0.0762,0.1425]])
    zita = np.array([[0.4596,0.2963,0.2338,0.4481,0.4753]])
    mg.MC.nd = np.int(1e4) #Sample Size
    mg.Montecarlo(xi, zita, True, True)