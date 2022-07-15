import numpy as np
import matplotlib.pyplot as plt


#### Using Defrise note on half pixel shift
def CoeffShannonInterpolationRect(x_j, x_i, W):
    x = (x_j-x_i)
    return (1-np.cos(2*W*np.pi*x))/(np.pi*x)

def CoeffShannonInterpolationHann(x_j, x_i, W):
    x = (x_j-x_i)
    return ((1-np.cos(np.pi*x*2*W))/(2*x) + (1+np.cos(2*W*np.pi*x))*(1/(x+1/(2*W)) + 1/(x-1/(2*W)))/4)/np.pi

def DefriseIntegrationCgtVar(gamma_s, gamma,g,v,xe,D,window):
    g_c = D*g/np.sqrt(D**2+v**2)
    t = -xe[2]*np.tan(gamma)
    t_new = np.linspace(-t[-1],-t[0],len(gamma))
    dt = np.abs(t_new[1]-t_new[0])
    g_tilde = np.interp(np.arctan(-t_new/xe[2]),gamma,g_c)/np.sqrt(xe[2]**2 + t_new**2)
    if window == "Rect":
        coeffs = CoeffShannonInterpolationRect(xe[0],t_new)
    elif window == "Hann":
        coeffs = CoeffShannonInterpolationHann(xe[0],t_new)
    moment, norm = dt*np.sum(g_tilde*coeffs), dt*np.sum(np.abs(g_tilde*coeffs))
    return moment, norm

def DefriseIntegrationHilbertKernel(gamma_s, gamma, g, v, xe, D, window, W_frac):
    g_c = D*g/np.sqrt(D**2+v**2)
    g_tilde= g_c/np.sinc((gamma_s-gamma)/np.pi)
    dgamma = np.abs(gamma[1]-gamma[0])
    W = 1/(2*dgamma)
    if window == "Rect":
        coeffs = CoeffShannonInterpolationRect(gamma_s, gamma, W/W_frac)
    elif window == "Hann":
        coeffs = CoeffShannonInterpolationHann(gamma_s, gamma, W/W_frac)
    moment = dgamma*np.sum(g_tilde*coeffs)/np.sqrt(xe[0]**2+xe[2]**2)
    return np.sign(gamma_s)*moment, coeffs


def DefriseIntegrationHilbertKernelVec(gamma_s, gamma, g, v, abs_cosalpha, D, window, W_frac):
    g_c = (1/abs_cosalpha)*(np.pi*D/np.sqrt(D**2+v**2))*g/np.sinc((gamma_s-np.array([gamma]*g.shape[1]).T)/np.pi)
    dgamma = np.abs(gamma[1]-gamma[0])
    W = 1/(2*dgamma)
    if window == "Rect":
        coeffs = CoeffShannonInterpolationRect(gamma_s, gamma, W/W_frac)
    elif window == "Hann":
        coeffs = CoeffShannonInterpolationHann(gamma_s, gamma, W/W_frac)
    moment = dgamma*np.sum(g_c*np.array([coeffs]*g.shape[1]).T, axis=0)
    return np.sign(gamma_s)*moment, coeffs



# Trapezoidale rule + adding one sample + king singularity estimation
def TestForSingularity(angle, xs):  # Find the indices of the angles surrounding the singularity
    a = 0
    b = 0
    if len(np.where(angle >= xs)[0]) != len(angle) and len(np.where(angle <= xs)[0]) != len(angle):
        if angle[1]-angle[0] >0:
            a = np.where(angle <= xs)[-1][-1]
            b = np.where(angle >= xs)[-1][0]
        else: 
            a = np.where(angle <= xs)[-1][0]
            b = np.where(angle >= xs)[-1][-1]
    return a, b


def TrapIntegrationAndKingModelSmallInterval(g, gamma, v, gamma_s, eb, D):
    g_tilde = np.sign(gamma_s)*D*g/(np.sqrt(eb[0]**2+eb[2]**2)*np.sqrt(v**2+D**2))
    dgamma = np.abs(gamma[1]-gamma[0])
    ii, iii = TestForSingularity(gamma, gamma_s)

    mom = np.zeros(g.shape[1])
    for j in range(len(mom)):
        if np.abs(np.abs(gamma[ii]-gamma_s)-np.abs(gamma[iii]-gamma_s)) <= 10**(-10):
            I1 = dgamma*(np.sum(g_tilde[:ii, j]/np.sin(gamma_s-gamma[:ii])) + 0.5*g_tilde[ii, j]/np.sin(gamma_s-gamma[ii]))
            I2 = -(g_tilde[iii, j]-g_tilde[ii, j])
            I3 = 0
            I4 = dgamma*(np.sum(g_tilde[iii+1:, j]/np.sin(gamma_s-gamma[iii+1:])) + 0.5*g_tilde[iii, j]/np.sin(gamma_s-gamma[iii]))
        else:
            if np.abs(gamma[ii]-gamma_s) < np.abs(gamma[iii]-gamma_s):
                h = np.abs(gamma[ii]-gamma_s)
                I1 = dgamma*(np.sum(g_tilde[:ii, j]/np.sin(gamma_s-gamma[:ii])) + 0.5*g_tilde[ii, j]/np.sin(gamma_s-gamma[ii]))
                gamma_a = gamma_s + h
                ga = np.interp(gamma_a, gamma, g[:,j])
                va = np.interp(gamma_a, gamma, v[:,j])
                g_tilde_a = np.sign(gamma_s)*D*ga/(np.sqrt(eb[0]**2+eb[2]**2)*np.sqrt(va**2+D**2))
                I2 = -(g_tilde_a-g_tilde[ii, j])
                I3 = (gamma[iii]-gamma_a)*(g_tilde_a/np.sin(gamma_s-gamma_a)+g_tilde[iii, j]/np.sin(gamma_s-gamma[iii]))/2
                I4 = dgamma*(np.sum(g_tilde[iii+1:, j]/np.sin(gamma_s-gamma[iii+1:])) + 0.5*g_tilde[iii, j]/np.sin(gamma_s-gamma[iii]))
            elif np.abs(gamma[ii]-gamma_s) > np.abs(gamma[iii]-gamma_s):
                h = np.abs(gamma[iii]-gamma_s)
                I1 = dgamma*(np.sum(g_tilde[:ii, j]/np.sin(gamma_s-gamma[:ii])) + 0.5*g_tilde[ii, j]/np.sin(gamma_s-gamma[ii]))
                gamma_a = gamma_s - h
                ga = np.interp(gamma_a, gamma, g[:, j])
                va = np.interp(gamma_a, gamma, v[:, j])
                g_tilde_a = np.sign(gamma_s)*D*ga/(np.sqrt(eb[0]**2+eb[2]**2)*np.sqrt(va**2+D**2))
                I2 = (gamma_a-gamma[ii])*(g_tilde_a/np.sin(gamma_s-gamma_a)+g_tilde[ii, j]/np.sin(gamma_s-gamma[ii]))/2
                I3 = -(g_tilde[iii, j]-g_tilde_a)
                I4 = dgamma*(np.sum(g_tilde[iii+1:, j]/np.sin(gamma_s-gamma[iii+1:])) + 0.5*g_tilde[iii, j]/np.sin(gamma_s-gamma[iii]))
        mom[j] = I1+I2+I3+I4
    return mom


def TrapIntegrationAndKingModelLargeInterval(eb, ee, en, gamma_e, gamma_s, gamma, g, v, vp, D):
    A_gamma = -D*(eb[2]*np.sin(np.array([gamma]*g.shape[1]).T)+eb[0]*np.cos(np.array([gamma]*g.shape[1]).T))/(en[:,1]*np.sqrt(D**2+v**2))
    Ap_gamma1 = (-eb[0]*np.sin(np.array([gamma]*g.shape[1]).T)+eb[2]*np.cos(np.array([gamma]*g.shape[1]).T))*(D**2+v**2)
    Ap_gamma2 = (eb[2]*np.sin(np.array([gamma]*g.shape[1]).T)+eb[0]*np.cos(np.array([gamma]*g.shape[1]).T))*v*vp
    Ap_gamma = -D*(Ap_gamma1-Ap_gamma2)/(en[:,1]*(D**2+v**2)**1.5)
    B_gamma = D*(ee[:, 0]*np.cos(np.array([gamma]*g.shape[1]).T) + ee[:, 2]*np.sin(np.array([gamma]*g.shape[1]).T))/en[:,1]
    Bp_gamma = D*(-ee[:, 0]*np.sin(np.array([gamma]*g.shape[1]).T) + ee[:, 2]*np.cos(np.array([gamma]*g.shape[1]).T))/en[:,1]
    C_gamma = -g*en[:, 1]*np.sqrt(D**2+v**2)*2*Bp_gamma*np.arccos(A_gamma)/(D*(eb[2]*np.sin(np.array([gamma]*g.shape[1]).T)+eb[0]*np.cos(np.array([gamma]*g.shape[1]).T)))
    w_gamma = en[:, 1]*np.sqrt(D**2+v**2)*np.sign(B_gamma)*(Ap_gamma/np.sqrt(1-A_gamma**2))/D
    I1 = np.zeros(g.shape[1])
    for j in range(g.shape[1]):
        I1[j] += (np.interp(gamma_e[j], gamma, C_gamma[:, j], left=0., right=0.)/np.abs(np.interp(gamma_e[j], gamma, Bp_gamma[:, j], left=1., right=1.)))
    gw = g*w_gamma*np.sign(gamma_s)*np.sqrt(eb[0]**2+eb[2]**2)
    Gw = gw/np.sin(gamma_s-np.array([gamma]*g.shape[1]).T)
    dgamma = np.abs(gamma[1]-gamma[0])
    ii, iii = TestForSingularity(gamma, gamma_s)
    summ = np.zeros(g.shape[1])
    rest = np.zeros(g.shape[1])
    for j in range(g.shape[1]):
        if np.abs(np.abs(gamma[ii]-gamma_s)-np.abs(gamma[iii]-gamma_s)) <= 10**(-10):
            rest[j] = dgamma*(np.interp(gamma_s+dgamma, gamma, gw[:, j])-np.interp(gamma_s-dgamma, gamma, gw[:, j]))/2
            summ[j] = dgamma*(np.sum(Gw[0:ii, j]) + Gw[ii, j]/2 + np.sum(Gw[iii+1:, j]) + Gw[iii, j]/2)
        elif np.abs(gamma[ii]-gamma_s) < np.abs(gamma[iii]-gamma_s) and np.abs(np.abs(gamma[ii]-gamma_s)-np.abs(gamma[iii]-gamma_s)) >= 10**(-10):
            gamma_a = gamma_s - 2*np.abs(gamma[ii]-gamma_s)
            Gwa = np.interp(gamma_a, gamma, gw[:, j])/np.sin(gamma_s-gamma_a)
            summ[j] = dgamma*(np.sum(Gw[0:ii-1, j]) + Gw[ii-1, j]/2 + np.sum(Gw[iii+1:, j]) + Gw[iii, j]/2) + (gamma_a-gamma[ii-1])*(Gw[ii-1, j]+Gwa)/2
            rest[j] = (gw[iii, j]-np.interp(gamma_a, gamma, gw[:, j]))/2
        elif np.abs(gamma[ii]-gamma_s) > np.abs(gamma[iii]-gamma_s) and np.abs(np.abs(gamma[ii]-gamma_s)-np.abs(gamma[iii]-gamma_s)) >= 10**(-10):
            gamma_a = gamma_s + 2*np.abs(gamma[iii]-gamma_s)
            Gwa = np.interp(gamma_a, gamma, gw[:, j])/np.sin(gamma_s-gamma_a)
            summ[j] = dgamma*(np.sum(Gw[0:ii, j]) + Gw[ii, j]/2 + np.sum(Gw[iii+2:, j]) + Gw[iii+1, j]/2) + (gamma[iii+1] -gamma_a)*(Gw[iii+1, j]+Gwa)/2
            rest[j] = (np.interp(gamma_a, gamma, gw[:, j])-gw[ii, j])/2
    return summ+I1+rest, rest, summ

"""
### Trap integration + estimation de la partie de la singularité avec la methode de michel defrise
def TrapIntegrationAndDefriseModelLargeInterval(gamma_s, gamma, g, v, xe, D):
    dgamma = np.abs(gamma[1]-gamma[0])
    g_cw = D*g/(np.sqrt(D**2+v**2)*(np.cos(gamma)*xe[0]+np.sin(gamma)*xe[2]))
    ii, iii = TestForSingularity(gamma, gamma_s)
    g_c = D*g/np.sqrt(D**2+v**2)
    
    if np.abs(np.abs(gamma[ii]-gamma_s)-np.abs(gamma[iii]-gamma_s)) <= 10**(-10):
        rest = dgamma*(np.interp(gamma_s+dgamma,gamma,g_c)-np.interp(gamma_s-dgamma,gamma,g_c))/2
        summ = dgamma*(np.sum(g_cw[0:ii]) + g_cw[ii]/2 + np.sum(g_cw[iii+1:]) + g_cw[iii]/2)
        norm = np.abs(rest)+np.abs(dgamma*(np.sum(g_cw[0:ii]) + g_cw[ii]/2)) + np.abs(dgamma*(np.sum(g_cw[iii+1:]) + g_cw[iii]/2))
    
    elif np.abs(gamma[ii]-gamma_s)<np.abs(gamma[iii]-gamma_s) and np.abs(np.abs(gamma[ii]-gamma_s)-np.abs(gamma[iii]-gamma_s)) >= 10**(-10):
        gamma_a = gamma_s - np.abs(gamma[iii]-gamma_s)
        g_cw_gamma_a = np.interp(gamma_a,gamma,g_c)/(np.cos(gamma_a)*xe[0]+np.sin(gamma_a)*xe[2])
        summ = dgamma*(np.sum(g_cw[0:ii-1]) + g_cw[ii-1]/2 + np.sum(g_cw[iii+1:]) + g_cw[iii]/2) + (gamma_a-gamma[ii-1])*(g_cw[ii-1]+g_cw_gamma_a)/2
        rest = np.sum(np.array([np.interp(gamma_a,gamma,g_c),g_c[iii]])*CoeffSplineInterpolation(gamma_s,np.array([gamma_a,gamma[iii]])))
        norm = np.abs(rest) + np.abs(dgamma*(np.sum(g_cw[0:ii-1]) + g_cw[ii-1]/2)) + np.abs(dgamma*(np.sum(g_cw[iii+1:]) + g_cw[iii]/2)) + np.abs((gamma_a-gamma[ii-1])*(g_cw[ii-1]+g_cw_gamma_a)/2)
        
    elif np.abs(gamma[ii]-gamma_s)>np.abs(gamma[iii]-gamma_s) and np.abs(np.abs(gamma[ii]-gamma_s)-np.abs(gamma[iii]-gamma_s)) >= 10**(-10):
        gamma_a = gamma_s + np.abs(gamma[ii]-gamma_s)
        g_cw_gamma_a = np.interp(gamma_a,gamma,g_c)/(np.cos(gamma_a)*xe[0]+np.sin(gamma_a)*xe[2])
        summ = dgamma*(np.sum(g_cw[0:ii]) + g_cw[ii]/2 + np.sum(g_cw[iii+2:]) + g_cw[iii+1]/2) + (gamma[iii+1]-gamma_a)*(g_cw[iii+1] + g_cw_gamma_a)/2
        rest = np.sum(np.array([g_c[ii],np.interp(gamma_a,gamma,g_c)])*CoeffSplineInterpolation(gamma_s,np.array([gamma[ii],gamma_a])))
        norm = np.abs(rest) + np.abs(dgamma*(np.sum(g_cw[iii+2:]) + g_cw[iii+1]/2)) + np.abs(dgamma*(np.sum(g_cw[0:ii]) + g_cw[ii]/2)) + np.abs((gamma[iii+1]-gamma_a)*(g_cw[iii+1] + g_cw_gamma_a)/2) 
        
    else:
        print("error")
        
    if gamma_s > 0:
            rest*=-1
        
    return summ+rest, norm

### Implementation of piessens 1970 for the evaluation of cauchy pv
def PiessensIntegrationSingularityNegative(gamma_s, gamma, dgamma, g_c, zk, wk):
    
    if np.abs(gamma[0]-gamma_s)<np.abs(gamma[-1]-gamma_s) and np.abs(gamma_s-(gamma[0]+gamma[-1])/2) >= 10**(-10):
        new_gamma_part1 = np.linspace(gamma[0],2*gamma_s-gamma[0],int(((2*gamma_s-gamma[0])-gamma[0])/dgamma)+1)
        new_g_c_part1 = np.interp(new_gamma_part1,gamma,g_c)
        Gamma = np.sin(new_gamma_part1-gamma_s)/np.sin(gamma_s-gamma[0])
        f_piessens = np.interp(np.arcsin(Gamma*np.sin(gamma_s-gamma[0]))+gamma_s,new_gamma_part1,new_g_c_part1)/np.cos(np.arcsin(Gamma*np.sin(gamma_s-gamma[0])))
        quad = np.sum(wk*(np.interp(zk,Gamma,f_piessens)-np.interp(-zk,Gamma,f_piessens))/zk)
        new_gamma_part2 = np.linspace(2*gamma_s-gamma[0],gamma[-1],int((gamma[-1]-(2*gamma_s-gamma[0]))/dgamma)+1)
        if len(new_gamma_part2)==1:
            rest = 0
        else :
            new_g_c_part2 = np.interp(new_gamma_part2,gamma,g_c)
            rest = (new_gamma_part2[1]-new_gamma_part2[0])*(np.sum(new_g_c_part2[1:-1]/np.sin(new_gamma_part2[1:-1]-gamma_s))+0.5*(new_g_c_part2[0]/np.sin(new_gamma_part2[0]-gamma_s)+new_g_c_part2[-1]/np.sin(new_gamma_part2[-1]-gamma_s)))
        norm = np.abs(rest)+np.abs(np.sum(wk*(np.interp(zk,Gamma,f_piessens)/zk))) + np.abs(np.sum(wk*(np.interp(-zk,Gamma,f_piessens)/zk)))
        
    elif np.abs(gamma_s-(gamma[0]+gamma[-1])/2) <= 10**(-10):
        Gamma = np.sin(gamma-gamma_s)/np.sin(gamma[-1]-gamma_s)
        f_piessens = np.interp(gamma_s + np.arcsin(Gamma*np.sin(gamma[-1]-gamma_s)),gamma,g_c)/np.cos(np.arcsin(Gamma*np.sin(gamma[-1]-gamma_s)))
        quad = np.sum(wk*(np.interp(zk,Gamma,f_piessens)-np.interp(-zk,Gamma,f_piessens))/zk)
        rest = 0
        norm = np.abs(rest)+np.abs(np.sum(wk*(np.interp(zk,Gamma,f_piessens)/zk))) + np.abs(np.sum(wk*(np.interp(-zk,Gamma,f_piessens)/zk)))
        
    elif np.abs(gamma[0]-gamma_s)>np.abs(gamma[-1]-gamma_s) and np.abs(gamma_s-(gamma[0]+gamma[-1])/2) >= 10**(-10):
        new_gamma_part1 = np.linspace(2*gamma_s-gamma[-1],gamma[-1],int((gamma[-1]-(2*gamma_s-gamma[-1]))/dgamma)+1)
        new_g_c_part1 = np.interp(new_gamma_part1,gamma,g_c)
        Gamma = np.sin(new_gamma_part1-gamma_s)/np.sin(gamma[-1]-gamma_s)
        f_piessens = np.interp(np.arcsin(Gamma*np.sin(gamma[-1]-gamma_s))+gamma_s,new_gamma_part1,new_g_c_part1)/np.cos(np.arcsin(Gamma*np.sin(gamma[-1]-gamma_s)))
        quad = np.sum(wk*(np.interp(zk,Gamma,f_piessens)-np.interp(-zk,Gamma,f_piessens))/zk)
        new_gamma_part2 = np.linspace(gamma[0],2*gamma_s-gamma[-1],int(((2*gamma_s-gamma[-1])-gamma[0])/dgamma)+1)
        if len(new_gamma_part2)==1:
            rest = 0
        else :
            new_g_c_part2 = np.interp(new_gamma_part2,gamma,g_c)
            rest = (new_gamma_part2[1]-new_gamma_part2[0])*(np.sum(new_g_c_part2[1:-1]/np.sin(new_gamma_part2[1:-1]-gamma_s))+0.5*(new_g_c_part2[0]/np.sin(new_gamma_part2[0]-gamma_s)+new_g_c_part2[-1]/np.sin(new_gamma_part2[-1]-gamma_s)))
        norm = np.abs(rest)+np.abs(np.sum(wk*(np.interp(zk,Gamma,f_piessens)/zk))) + np.abs(np.sum(wk*(np.interp(-zk,Gamma,f_piessens)/zk)))
        
    else:
        print("error")
        
    return quad+rest,norm


def PiessensIntegrationSingularityPositive(gamma_s, gamma, dgamma, g_c, zk, wk):
    if np.abs(gamma[0]-gamma_s)<np.abs(gamma[-1]-gamma_s) and np.abs(gamma_s-(gamma[0]+gamma[-1])/2) >= 10**(-10):
        new_gamma_part1 = np.linspace(gamma[0],2*gamma_s-gamma[0],int(((2*gamma_s-gamma[0])-gamma[0])/dgamma)+1)
        new_g_c_part1 = np.interp(new_gamma_part1,gamma,g_c)
        Gamma = np.sin(gamma_s-new_gamma_part1)/np.sin(gamma[0]-gamma_s)
        f_piessens = np.interp(gamma_s-np.arcsin(Gamma*np.sin(gamma[0]-gamma_s)),new_gamma_part1,new_g_c_part1)/(-np.cos(np.arcsin(Gamma*np.sin(gamma[0]-gamma_s))))
        quad = np.sum(wk*(np.interp(zk,Gamma,f_piessens)-np.interp(-zk,Gamma,f_piessens))/zk)
        new_gamma_part2 = np.linspace(2*gamma_s-gamma[0],gamma[-1],int((gamma[-1]-(2*gamma_s-gamma[0]))/dgamma)+1)
        if len(new_gamma_part2)==1:
            rest = 0
        else :
            new_g_c_part2 = np.interp(new_gamma_part2,gamma,g_c)
            rest = (new_gamma_part2[1]-new_gamma_part2[0])*(np.sum(new_g_c_part2[1:-1]/np.sin(gamma_s-new_gamma_part2[1:-1]))+0.5*(new_g_c_part2[0]/np.sin(gamma_s-new_gamma_part2[0])+new_g_c_part2[-1]/np.sin(gamma_s-new_gamma_part2[-1])))
        norm = np.abs(rest)+np.abs(np.sum(wk*(np.interp(zk,Gamma,f_piessens)/zk))) + np.abs(np.sum(wk*(np.interp(-zk,Gamma,f_piessens)/zk)))
                                                                                         
    elif np.abs(gamma_s-(gamma[0]+gamma[-1])/2) <= 10**(-10):
        Gamma = np.sin(gamma_s-gamma)/np.sin(gamma_s-gamma[-1])
        f_piessens = np.interp(gamma_s - np.arcsin(Gamma*np.sin(gamma_s-gamma[-1])),gamma,g_c)/(-np.cos(np.arcsin(Gamma*np.sin(gamma_s-gamma[-1]))))
        quad = np.sum(wk*(np.interp(zk,Gamma,f_piessens)-np.interp(-zk,Gamma,f_piessens))/zk)
        rest = 0
        norm = np.abs(rest)+np.abs(np.sum(wk*(np.interp(zk,Gamma,f_piessens)/zk))) + np.abs(np.sum(wk*(np.interp(-zk,Gamma,f_piessens)/zk)))
            
    elif np.abs(gamma[0]-gamma_s)>np.abs(gamma[-1]-gamma_s) and np.abs(gamma_s-(gamma[0]+gamma[-1])/2) >= 10**(-10):
        new_gamma_part1 = np.linspace(2*gamma_s-gamma[-1],gamma[-1],int((gamma[-1]-(2*gamma_s-gamma[-1]))/dgamma)+1)
        new_g_c_part1 = np.interp(new_gamma_part1,gamma,g_c)
        Gamma = np.sin(gamma_s-new_gamma_part1)/np.sin(gamma_s-gamma[-1])
        f_piessens = np.interp(gamma_s-np.arcsin(Gamma*np.sin(gamma_s-gamma[-1])),new_gamma_part1,new_g_c_part1)/(-np.cos(np.arcsin(Gamma*np.sin(gamma_s-gamma[-1]))))
        quad = np.sum(wk*(np.interp(zk,Gamma,f_piessens)-np.interp(-zk,Gamma,f_piessens))/zk)
        new_gamma_part2 = np.linspace(gamma[0],2*gamma_s-gamma[-1],int(((2*gamma_s-gamma[-1])-gamma[0])/dgamma)+1)
        if len(new_gamma_part2)==1:
            rest = 0
        else :
            new_g_c_part2 = np.interp(new_gamma_part2,gamma,g_c)
            rest = (new_gamma_part2[1]-new_gamma_part2[0])*(np.sum(new_g_c_part2[1:-1]/np.sin(gamma_s-new_gamma_part2[1:-1]))+0.5*(new_g_c_part2[0]/np.sin(gamma_s-new_gamma_part2[0])+new_g_c_part2[-1]/np.sin(gamma_s-new_gamma_part2[-1])))
        norm = np.abs(rest)+np.abs(np.sum(wk*(np.interp(zk,Gamma,f_piessens)/zk))) + np.abs(np.sum(wk*(np.interp(-zk,Gamma,f_piessens)/zk)))
        
    else: 
        print("error")
        
    return quad+rest,norm


def PiessensIntegration(gamma_s, gamma, g, v, D):
    g_c = D*g/np.sqrt(D**2+v**2)
    dgamma = np.abs(gamma[1]-gamma[0])
    
    z, w =np.polynomial.legendre.leggauss(100)
    zk = z[np.where(z>=0)]
    wk = w[np.where(z>=0)]
    
    if gamma_s < 0:
        moment, norm = PiessensIntegrationSingularityNegative(gamma_s, gamma, dgamma, g_c, zk, wk)
    elif gamma_s > 0: 
        moment, norm = PiessensIntegrationSingularityPositive(gamma_s, gamma, dgamma, g_c, zk, wk)
        
    return moment, norm


def PiessensApplyToSingularity(gamma_s,gamma,g_c):
    z, w =np.polynomial.legendre.leggauss(100)
    zk = z[np.where(z>=0)]
    wk = w[np.where(z>=0)]
    
    if gamma_s < 0:
        Gamma = np.sin(gamma-gamma_s)/np.sin(gamma[-1]-gamma_s)
        f_piessens = np.interp(gamma_s + np.arcsin(Gamma*np.sin(gamma[-1]-gamma_s)),gamma,g_c)/np.cos(np.arcsin(Gamma*np.sin(gamma[-1]-gamma_s)))
        quad = np.sum(wk*(np.interp(zk,Gamma,f_piessens)-np.interp(-zk,Gamma,f_piessens))/zk)
    elif gamma_s >0:
        Gamma = np.sin(gamma_s-gamma)/np.sin(gamma_s-gamma[-1])
        f_piessens = np.interp(gamma_s - np.arcsin(Gamma*np.sin(gamma_s-gamma[-1])),gamma,g_c)/(-np.cos(np.arcsin(Gamma*np.sin(gamma_s-gamma[-1]))))
        quad = np.sum(wk*(np.interp(zk,Gamma,f_piessens)-np.interp(-zk,Gamma,f_piessens))/zk)
                
    return quad """
