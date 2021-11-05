import numpy as np
import matplotlib.pyplot as plt
import time

def TestForSingularity(angle, xs):  #Find the indices of the angles surrounding the singularity
    a = 0
    b = 0
    if len(np.where(angle >= xs)[0]) != len(angle) and len(np.where(angle <= xs)[0]) != len(angle):
        if angle[1]-angle[0] >0:
            a = np.where(angle <= xs)[-1][-1]
            b = np.where(angle >= xs)[-1][0]
        else : 
            a = np.where(angle <= xs)[-1][0]
            b = np.where(angle >= xs)[-1][-1]
    return a, b


### Trapezoidale rule + linear model
def TrapIntegrationAndLinearModel(gamma_s, gamma, g, v, xe, D):  #Perform numerical integration using trapezoidale rule
    dgamma = np.abs(gamma[1]-gamma[0])
    g_cw = D*g/(np.sqrt(D**2+v**2)*(np.cos(gamma)*xe[0]+np.sin(gamma)*xe[2]))
    ii, iii = TestForSingularity(gamma, gamma_s)
    g_c = D*g/np.sqrt(D**2+v**2)
    
    if np.abs(np.abs(gamma[ii]-gamma_s)-np.abs(gamma[iii]-gamma_s)) <= 10**(-10):
        summ = dgamma*(np.sum(g_cw[1:ii]) + g_cw[0]*3/4 + g_cw[ii]/2 + np.sum(g_cw[iii+1:-1]) + g_cw[-1]*3/4 + g_cw[iii]/2)
        a = (g_c[iii]-g_c[ii])/(gamma[iii]-gamma[ii])
        b = g_c[iii]-a*gamma[iii]
        c = ((np.cos(gamma[iii])*xe[0]+np.sin(gamma[iii])*xe[2])-(np.cos(gamma[ii])*xe[0]+np.sin(gamma[ii])*xe[2]))/(gamma[iii]-gamma[ii])
        d = (np.cos(gamma[iii])*xe[0]+np.sin(gamma[iii])*xe[2]) - c*gamma[iii]
        rest_ii = (a*(c*gamma[ii]+d)+(b*c-a*d)*np.log(np.abs(c*gamma[ii]+d)))
        rest_iii = (a*(c*gamma[iii]+d)+(b*c-a*d)*np.log(np.abs(c*gamma[iii]+d)))
        rest = (rest_iii-rest_ii)/c**2
        norm = np.abs(rest)+np.abs(dgamma*(np.sum(g_cw[1:ii]) + g_cw[0]*3/4 + g_cw[ii]/2)) + np.abs(dgamma*(np.sum(g_cw[iii+1:-1]) + g_cw[-1]*3/4 + g_cw[iii]/2))
        
    elif np.abs(gamma[ii]-gamma_s)<np.abs(gamma[iii]-gamma_s) and np.abs(np.abs(gamma[ii]-gamma_s)-np.abs(gamma[iii]-gamma_s)) >= 10**(-10):
        summ = dgamma*(np.sum(g_cw[1:ii-1])+g_cw[0]*3/4 + g_cw[ii-1]/2 + np.sum(g_cw[iii+1:-1]) + g_cw[-1]*3/4 + g_cw[iii]/2)
        a = (g_c[iii]-g_c[ii-1])/(gamma[iii]-gamma[ii-1])
        b = g_c[iii]-a*gamma[iii]
        c = ((np.cos(gamma[iii])*xe[0]+np.sin(gamma[iii])*xe[2])-(np.cos(gamma[ii-1])*xe[0]+np.sin(gamma[ii-1])*xe[2]))/(gamma[iii]-gamma[ii-1])
        d = (np.cos(gamma[iii])*xe[0]+np.sin(gamma[iii])*xe[2]) - c*gamma[iii]
        rest_ii = (a*(c*gamma[ii-1]+d)+(b*c-a*d)*np.log(np.abs(c*gamma[ii-1]+d)))
        rest_iii = (a*(c*gamma[iii]+d)+(b*c-a*d)*np.log(np.abs(c*gamma[iii]+d)))
        rest = (rest_iii-rest_ii)/c**2
        norm = np.abs(dgamma*(np.sum(g_cw[1:ii-1])+g_cw[0]*3/4 + g_cw[ii-1]/2))+np.abs(dgamma*(np.sum(g_cw[iii+1:-1]) + g_cw[-1]*3/4 + g_cw[iii]/2)) + np.abs(rest)
        
    elif np.abs(gamma[ii]-gamma_s)>np.abs(gamma[iii]-gamma_s) and np.abs(np.abs(gamma[ii]-gamma_s)-np.abs(gamma[iii]-gamma_s)) >= 10**(-10):
        summ = dgamma*(np.sum(g_cw[1:ii])+g_cw[0]*3/4 + g_cw[ii]/2 + np.sum(g_cw[iii+2:-1]) + g_cw[-1]*3/4 + g_cw[iii+1]/2)
        a = (g_c[iii+1]-g_c[ii])/(gamma[iii+1]-gamma[ii])
        b = g_c[ii]-a*gamma[ii]
        c = ((np.cos(gamma[iii+1])*xe[0]+np.sin(gamma[iii+1])*xe[2])-(np.cos(gamma[ii])*xe[0]+np.sin(gamma[ii])*xe[2]))/(gamma[iii+1]-gamma[ii])
        d = (np.cos(gamma[ii])*xe[0]+np.sin(gamma[ii])*xe[2]) - c*gamma[ii]
        rest_ii = (a*(c*gamma[ii]+d)+(b*c-a*d)*np.log(np.abs(c*gamma[ii]+d)))
        rest_iii = (a*(c*gamma[iii+1]+d)+(b*c-a*d)*np.log(np.abs(c*gamma[iii+1]+d)))
        rest = (rest_iii-rest_ii)/c**2
        norm = np.abs(dgamma*(np.sum(g_cw[1:ii])+g_cw[0]*3/4 + g_cw[ii]/2)) + np.abs(dgamma*(np.sum(g_cw[iii+2:-1]) + g_cw[-1]*3/4 + g_cw[iii+1]/2)) + np.abs(rest)
        
    else:
        print('error')
    
    return summ+rest, norm


### Trapezoidale rule + adding one sample + king singularity estimation
def TrapIntegrationAndKingModelSmallInterval(gamma_s, gamma, g, v, xe, D):
    dgamma = np.abs(gamma[1]-gamma[0])
    g_cw = D*g/(np.sqrt(D**2+v**2)*(np.cos(gamma)*xe[0]+np.sin(gamma)*xe[2]))
    ii, iii = TestForSingularity(gamma, gamma_s)
    g_c = D*g/np.sqrt(D**2+v**2)
    
    if np.abs(np.abs(gamma[ii]-gamma_s)-np.abs(gamma[iii]-gamma_s)) <= 10**(-10):
        rest = dgamma*(np.interp(gamma_s+dgamma,gamma,g_c)-np.interp(gamma_s-dgamma,gamma,g_c))/2
        summ = dgamma*(np.sum(g_cw[0:ii]) + g_cw[ii]/2 + np.sum(g_cw[iii+1:]) + g_cw[iii]/2)
        norm = np.abs(rest)+np.abs(dgamma*(np.sum(g_cw[0:ii]) + g_cw[ii]/2)) + np.abs(dgamma*(np.sum(g_cw[iii+1:]) + g_cw[iii]/2))
    
    elif np.abs(gamma[ii]-gamma_s)<np.abs(gamma[iii]-gamma_s) and np.abs(np.abs(gamma[ii]-gamma_s)-np.abs(gamma[iii]-gamma_s)) >= 10**(-10):
        gamma_a = gamma_s + np.abs(gamma[ii]-gamma_s)
        g_cw_gamma_a = np.interp(gamma_a,gamma,g_c)/(np.cos(gamma_a)*xe[0]+np.sin(gamma_a)*xe[2])
        summ = dgamma*(np.sum(g_cw[0:ii]) + g_cw[ii]/2 + np.sum(g_cw[iii+1:]) + g_cw[iii]/2) + (gamma[iii]-gamma_a)*(g_cw[iii]+g_cw_gamma_a)/2
        rest = (np.interp(gamma_a,gamma,g_c) - g_c[ii])
        norm = np.abs(rest) + np.abs(dgamma*(np.sum(g_cw[0:ii]) + g_cw[ii]/2)) + np.abs(dgamma * (np.sum(g_cw[iii+1:]) + g_cw[iii]/2)) + np.abs((gamma[iii]-gamma_a)*(g_cw[iii]+g_cw_gamma_a)/2)
        
    elif np.abs(gamma[ii]-gamma_s)>np.abs(gamma[iii]-gamma_s) and np.abs(np.abs(gamma[ii]-gamma_s)-np.abs(gamma[iii]-gamma_s)) >= 10**(-10):
        gamma_a = gamma_s - np.abs(gamma[iii]-gamma_s)
        g_cw_gamma_a = np.interp(gamma_a,gamma,g_c)/(np.cos(gamma_a)*xe[0]+np.sin(gamma_a)*xe[2])
        summ = dgamma*(np.sum(g_cw[0:ii]) + g_cw[ii]/2 + np.sum(g_cw[iii+1:]) + g_cw[iii]/2) + (gamma_a-gamma[ii])*(g_cw[ii] + g_cw_gamma_a)/2
        rest = (g_c[iii] - np.interp(gamma_a,gamma,g_c))
        norm = np.abs(rest) + np.abs(dgamma*(np.sum(g_cw[0:ii]) + g_cw[ii]/2)) + np.abs(dgamma*(np.sum(g_cw[iii+1:]) + g_cw[iii]/2)) + np.abs((gamma_a-gamma[ii])*(g_cw[ii] + g_cw_gamma_a)/2) 
        
    else:
        print("error")
        
    if gamma_s > 0:
            rest*=-1
        
    return summ+rest, norm, summ, rest


### Trapezoidale rule + adding one sample + king singularity estimation
def TrapIntegrationAndKingModelLargeInterval(gamma_s, gamma, g, v, xe, D):
    dgamma = np.abs(gamma[1]-gamma[0])
    ii, iii = TestForSingularity(gamma, gamma_s)
    g_c = D*g/np.sqrt(D**2+v**2)
    g_cw = g_c/(np.cos(gamma)*xe[0]+np.sin(gamma)*xe[2])
    
    if np.abs(np.abs(gamma[ii]-gamma_s)-np.abs(gamma[iii]-gamma_s)) <= 10**(-10):
        rest = dgamma*(np.interp(gamma_s+dgamma,gamma,g_c)-np.interp(gamma_s-dgamma,gamma,g_c))/2
        summ = dgamma*(np.sum(g_cw[0:ii]) + g_cw[ii]/2 + np.sum(g_cw[iii+1:]) + g_cw[iii]/2)
        norm = np.abs(rest)+np.abs(dgamma*(np.sum(g_cw[0:ii]) + g_cw[ii]/2)) + np.abs(dgamma*(np.sum(g_cw[iii+1:]) + g_cw[iii]/2))
    
    elif np.abs(gamma[ii]-gamma_s)<np.abs(gamma[iii]-gamma_s) and np.abs(np.abs(gamma[ii]-gamma_s)-np.abs(gamma[iii]-gamma_s)) >= 10**(-10):
        gamma_a = gamma_s - np.abs(gamma[iii]-gamma_s)
        g_cw_gamma_a = np.interp(gamma_a,gamma,g_c)/(np.cos(gamma_a)*xe[0]+np.sin(gamma_a)*xe[2])
        summ = dgamma*(np.sum(g_cw[0:ii-1]) + g_cw[ii-1]/2 + np.sum(g_cw[iii+1:]) + g_cw[iii]/2) + (gamma_a-gamma[ii-1])*(g_cw[ii-1]+g_cw_gamma_a)/2
#         rest = (np.interp(2*gamma_s-gamma[ii],gamma,g_c) - g_c[ii])*(gamma[iii]-gamma_s)/(2*(gamma_s-gamma[ii]))
        rest = (g_c[iii]-np.interp(gamma_a,gamma,g_c))
        norm = np.abs(rest) + np.abs(dgamma*(np.sum(g_cw[0:ii-1]) + g_cw[ii-1]/2)) + np.abs(dgamma*(np.sum(g_cw[iii+1:]) + g_cw[iii]/2)) + np.abs((gamma_a-gamma[ii-1])*(g_cw[ii-1]+g_cw_gamma_a)/2)
        
#         plt.figure()
#         plt.plot(gamma[ii-5:iii+6],g_cw[ii-5:iii+6])
#         plt.plot(gamma[ii],0,'+',label='j')
#         plt.plot(gamma[iii],0,'+',label='j+1')
#         plt.plot(gamma_a,0,'+',label='a')
#         plt.axhline(y=0,color='k')
#         plt.legend()
#         plt.show()
        
        
    elif np.abs(gamma[ii]-gamma_s)>np.abs(gamma[iii]-gamma_s) and np.abs(np.abs(gamma[ii]-gamma_s)-np.abs(gamma[iii]-gamma_s)) >= 10**(-10):
        gamma_a = gamma_s + np.abs(gamma[ii]-gamma_s)
        g_cw_gamma_a = np.interp(gamma_a,gamma,g_c)/(np.cos(gamma_a)*xe[0]+np.sin(gamma_a)*xe[2])
        summ = dgamma*(np.sum(g_cw[0:ii]) + g_cw[ii]/2 + np.sum(g_cw[iii+2:]) + g_cw[iii+1]/2) + (gamma[iii+1]-gamma_a)*(g_cw[iii+1] + g_cw_gamma_a)/2
#         rest = (g_c[iii] - np.interp(2*gamma_s-gamma[iii],gamma,g_c))*(gamma_s-gamma[ii])/(2*(gamma[iii]-gamma_s))
        rest = (np.interp(gamma_a,gamma,g_c)-g_c[ii])
        norm = np.abs(rest) + np.abs(dgamma*(np.sum(g_cw[iii+2:]) + g_cw[iii+1]/2)) + np.abs(dgamma*(np.sum(g_cw[0:ii]) + g_cw[ii]/2)) + np.abs((gamma[iii+1]-gamma_a)*(g_cw[iii+1] + g_cw_gamma_a)/2) 
#         plt.figure()
#         plt.plot(gamma[ii-5:iii+6],g_cw[ii-5:iii+6])
#         plt.plot(gamma[ii],0,'+',label='j')
#         plt.plot(gamma[iii],0,'+',label='j+1')
#         plt.plot(gamma_a,0,'+',label='a')
#         plt.axhline(y=0,color='k')
#         plt.legend()
#         plt.show()
    else:
        print("error")
        
    if gamma_s > 0:
            rest*=-1
        
    return summ+rest, norm


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
                
    return quad
        

#### Using Defrise note on half pixel shift
def CoeffShannonInterpolationRect(x_j, x_i, W):
    x = (x_j-x_i)
    return (1-np.cos(2*W*np.pi*x))/(x)

def CoeffShannonInterpolationHann(x_j, x_i, W):
    x = (x_j-x_i)
    return (1-np.cos(np.pi*x*2*W))/(2*x) + (1+np.cos(2*W*np.pi*x))*(1/(x+1/(2*W)) + 1/(x-1/(2*W)))/4

def CoeffShannonInterpolationBlackmann(x_j, x_i, W):
    x = (x_j-x_i)
    I1 = 0.42*(1-np.cos(2*np.pi*W*x))/x
    I2 = (1+np.cos(2*W*np.pi*x))*(1/(x+1/(2*W)) + 1/(x-1/(2*W)))/4
    I3 = 0.04*(1-np.cos(2*np.pi*W*x))*(1/(x+1/W) + 1/(x-1/W))
    return  I1 + I2 + I3

from scipy import special

def CoeffShannonInterpolationGauss(x_j, x_i, W):
    x = (x_j-x_i)
    I= 0.5*np.sqrt(np.pi/W)*np.exp(-(np.pi*x)**2/W)*(special.erfi((np.pi*x-1j*W**2)/np.sqrt(W))+special.erfi((np.pi*x+1j*W**2)/np.sqrt(W))-2*special.erfi(np.pi*x/np.sqrt(W)))
    return -np.pi*I.real

def CoeffSplineInterpolation(x_j,x_i):
    x = (x_j-x_i)
    return ((1+x)*np.log(np.abs((x+1)/x))-(1-x)*np.log(np.abs((x-1)/x)))
                                        
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
    g_tilde= g_c/np.sinc((gamma-gamma_s)/np.pi)
    dgamma = np.abs(gamma[1]-gamma[0])
    W = 1/(2*dgamma)
    if window == "Rect":
        coeffs = CoeffShannonInterpolationRect(gamma_s, gamma, W/W_frac)
    elif window == "Hann":
        coeffs = CoeffShannonInterpolationHann(gamma_s, gamma, W/W_frac)
    elif window == "Blackmann":
        coeffs = CoeffShannonInterpolationGauss(gamma_s, gamma, W/W_frac)
    elif window == "Gauss":
        print("ici")
        coeffs = CoeffShannonInterpolationBlackmann(gamma_s, gamma, W/W_frac)
    moment, norm = dgamma*np.sum(g_tilde*coeffs), dgamma*np.sum(np.abs(g_tilde*coeffs))
    return -np.sign(gamma_s)*moment, norm
    