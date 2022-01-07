#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 13:39:28 2021

@author: mmouchet
"""
import numpy as np
import matplotlib.pyplot as plt

#plots of the intersection between virtual detector plane and epipolar planes
def ComputePlanesIntersection(nb, ne, de):
    uy = 1/(-(nb[1]+nb[2]*(ne[0]*nb[1]-ne[1]*nb[0])/(nb[0]*ne[2]-ne[0]*nb[2]))/nb[0])
    uz = 1/(-(nb[2]+nb[1]*(ne[0]*nb[2]-ne[2]*nb[0])/(nb[0]*ne[1]-ne[0]*nb[1]))/nb[0])
    ux = 1.
    ay = uy * (-nb[2]*de)/(nb[0]*ne[2]-ne[0]*nb[2])
    az = uz * (-nb[1]*de)/(nb[0]*ne[1]-ne[0]*nb[1])
    ax = 0.
    return np.array([ux,uy,uz]), np.array([ax, ay, az])


def PlotPairGeometry(pair_e, pair_bp):
    # plot of the virtual detector
    vo0bis = np.dot(pair_bp.volDirectionT0.T,pair_bp.volOrigin0)
    voc0bis = np.dot(pair_bp.volDirectionT0.T,pair_bp.volOtherCorner0)
    vo1bis = np.dot(pair_bp.volDirectionT1.T,pair_bp.volOrigin1)
    voc1bis = np.dot(pair_bp.volDirectionT1.T,pair_bp.volOtherCorner1)

    c0v = vo0bis
    c1v = np.array([vo0bis[0],voc0bis[1],vo0bis[2]])
    c2v = voc0bis
    c3v = np.array([voc0bis[0],vo0bis[1],voc0bis[2]])

    c0r = np.dot(pair_bp.volDirectionT0,c0v)
    c1r = np.dot(pair_bp.volDirectionT0,c1v)
    c2r = np.dot(pair_bp.volDirectionT0,c2v)
    c3r = np.dot(pair_bp.volDirectionT0,c3v)

    lines_virt_det0 = np.zeros((2,len(pair_bp.vk0),3))
    lines_real_det0 = np.zeros((2,len(pair_bp.vk0),3))
    lines_virt_det0[0,:,0] += np.ones(len(pair_bp.vk0))*vo0bis[0]
    lines_virt_det0[1,:,0] += np.ones(len(pair_bp.vk0))*voc0bis[0]
    lines_virt_det0[0,:,1] += pair_bp.vk0
    lines_virt_det0[1,:,1] += pair_bp.vk0

    for i in range(lines_real_det0.shape[0]):
        for j in range(len(pair_bp.vk0)):
            lines_real_det0[i,j,:] += np.dot(pair_bp.volDirectionT0,lines_virt_det0[i,j,:]+np.array([0.,pair_bp.spw0[1],0.]))

    plt.figure()
    plt.subplot(121)
    plt.plot([vo0bis[0],vo0bis[0],voc0bis[0],voc0bis[0],vo0bis[0]],[vo0bis[1],voc0bis[1],voc0bis[1],vo0bis[1],vo0bis[1]],'r')
    for i in range(len(pair_bp.vk0)):
        plt.axhline(y=pair_bp.vk0[i]+pair_bp.spw0[1],color='k')
    plt.subplot(122)
    plt.plot([c0r[0],c1r[0],c2r[0],c3r[0],c0r[0]],[c0r[1],c1r[1],c2r[1],c3r[1],c0r[1]],'r')
    for i in range(len(pair_bp.vk0)):
        plt.plot(lines_real_det0[:,i,0],lines_real_det0[:,i,1],color='k')
    plt.show()

    plt.figure()
    plt.subplot(221)
    ee = np.cross(pair_e.eb, pair_e.en[0])
    plt.plot(250*np.cos(np.linspace(0,2*np.pi)), 250*np.sin(np.linspace(0,2*np.pi)), 'k--', linewidth=0.5)
    #epipolar drawings
    plt.plot(pair_e.s0[2], pair_e.s0[0], '.', color='indigo')
    plt.plot(pair_e.Det0[:, 0, 2], pair_e.Det0[:, 0, 0],color='indigo')
    plt.plot(pair_e.s1[2], pair_e.s1[0], '.', color='darkorange')
    plt.plot(pair_e.Det1[:, 0, 2], pair_e.Det1[:, 0, 0], color='darkorange')
    plt.plot(pair_e.m_points[2][len(pair_e.m_points[1])//2], pair_e.m_points[0][len(pair_e.m_points[1])//2], 'r+')
    plt.plot([pair_e.s0[2], pair_e.s0[2]+ 100*pair_e.eb[2]], [pair_e.s0[0], pair_e.s0[0]+100*pair_e.eb[0]],color='lightblue')
    plt.plot([pair_e.s0[2], pair_e.s0[2]+pair_e.en[0][2]], [pair_e.s0[0], pair_e.s0[0]+ pair_e.en[0][0]],color='darkblue')
    plt.plot([pair_e.s0[2], pair_e.s0[2]+100*ee[2]], [pair_e.s0[0], pair_e.s0[0] +100*ee[0]],color='blue')
    plt.plot([pair_e.s0[2], pair_e.s0[2]+ 100*pair_e.volDir0[0,2]], [pair_e.s0[0], pair_e.s0[0]+100*pair_e.volDir0[0,0]],color='lightgreen')
    plt.plot([pair_e.s0[2], pair_e.s0[2]+ pair_e.volDir0[1,2]], [pair_e.s0[0], pair_e.s0[0]+pair_e.volDir0[1,0]],color='darkgreen')
    plt.plot([pair_e.s0[2], pair_e.s0[2]+ 100*pair_e.volDir0[2,2]], [pair_e.s0[0], pair_e.s0[0]+100*pair_e.volDir0[2,0]],color='green')
    #bp drawings
    plt.plot(lines_real_det0[:,0,2], lines_real_det0[:,0,0], color="r" )
    plt.plot([pair_bp.CornersDet0[0][2], pair_bp.CornersDet0[1][2]],[pair_bp.CornersDet0[0][0], pair_bp.CornersDet0[1][0]], color="indigo", linestyle="--")
    plt.plot([pair_bp.CornersDet1[0][2], pair_bp.CornersDet1[1][2]],[pair_bp.CornersDet1[0][0], pair_bp.CornersDet1[1][0]], color="darkorange", linestyle="--")
    plt.xlabel("z_rtk")
    plt.ylabel("x_rtk")
    plt.axis("equal")
    plt.subplot(222)
    plt.plot(pair_e.s0[2], pair_e.s0[1], '.', color='indigo')
    plt.plot(pair_e.Det0[pair_e.Det0.shape[0]//2, :, 2], pair_e.Det0[pair_e.Det0.shape[0]//2, :, 1], color='indigo')
    plt.plot(pair_e.s1[2], pair_e.s1[1], '.', color='darkorange')
    plt.plot(pair_e.Det1[pair_e.Det0.shape[0]//2, :, 2], pair_e.Det1[pair_e.Det0.shape[0]//2, :, 1], color='darkorange')
    plt.plot([pair_e.s0[2], pair_e.s0[2]+ 100*pair_e.eb[2]], [pair_e.s0[1], pair_e.s0[1]+100*pair_e.eb[1]],color='lightblue')
    plt.plot([pair_e.s0[2], pair_e.s0[2]+pair_e.en[0][2]], [pair_e.s0[1], pair_e.s0[1]+ pair_e.en[0][1]],color='darkblue')
    plt.plot([pair_e.s0[2], pair_e.s0[2]+100*ee[2]], [pair_e.s0[1], pair_e.s0[1] +100*ee[1]],color='blue')
    plt.plot([pair_e.s0[2], pair_e.s0[2]+ 100*pair_e.volDir0[0,2]], [pair_e.s0[1], pair_e.s0[1]+100*pair_e.volDir0[0,1]],color='lightgreen')
    plt.plot([pair_e.s0[2], pair_e.s0[2]+ pair_e.volDir0[1,2]], [pair_e.s0[1], pair_e.s0[1]+pair_e.volDir0[1,1]],color='darkgreen')
    plt.plot([pair_e.s0[2], pair_e.s0[2]+ 100*pair_e.volDir0[2,2]], [pair_e.s0[1], pair_e.s0[1]+100*pair_e.volDir0[2,1]],color='green')
    plt.plot(lines_real_det0[:,0,2], lines_real_det0[:,0,1], color="r" )
    plt.plot([pair_bp.CornersDet0[0][2], pair_bp.CornersDet0[2][2]],[pair_bp.CornersDet0[0][1], pair_bp.CornersDet0[2][1]], color="indigo", linestyle="--")
    plt.plot([pair_bp.CornersDet1[0][2], pair_bp.CornersDet1[2][2]],[pair_bp.CornersDet1[0][1], pair_bp.CornersDet1[2][1]], color="darkorange", linestyle="--")
    plt.xlabel("z_rtk")
    plt.ylabel("y_rtk")
#     plt.axis("equal")
    plt.subplot(223)
    plt.imshow(pair_e.p0, cmap = "gray", extent=(pair_e.gamma0[0], pair_e.gamma0[-1], pair_e.v_det0[0], pair_e.v_det0[-1]), origin="lower",aspect="auto")
    plt.plot(pair_e.gamma0, pair_e.v_det0[0]*np.ones(len(pair_e.gamma0)), color='indigo')
    plt.plot(pair_e.gamma0, pair_e.v_det0[-1]*np.ones(len(pair_e.gamma0)), color='indigo')
    plt.plot(pair_e.gamma0[0]*np.ones(len(pair_e.v_det0)), pair_e.v_det0, color='indigo')
    plt.plot(pair_e.gamma0[-1]*np.ones(len(pair_e.v_det0)), pair_e.v_det0, color='indigo')
    for j in range(pair_e.v0.shape[1]):
        plt.plot(pair_e.gamma0, pair_e.v0[:, j], 'r', linewidth=0.5)
    plt.subplot(224)
    plt.imshow(pair_e.p1, cmap = "gray", extent=(pair_e.gamma1[0], pair_e.gamma1[-1], pair_e.v_det1[0], pair_e.v_det1[-1]), origin="lower",aspect="auto")
    plt.plot(pair_e.gamma1, pair_e.v_det1[0]*np.ones(len(pair_e.gamma1)), color='darkorange')
    plt.plot(pair_e.gamma1, pair_e.v_det1[-1]*np.ones(len(pair_e.gamma1)), color='darkorange')
    plt.plot(pair_e.gamma1[0]*np.ones(len(pair_e.v_det1)), pair_e.v_det1, color='darkorange')
    plt.plot(pair_e.gamma1[-1]*np.ones(len(pair_e.v_det1)), pair_e.v_det1, color='darkorange')
    for j in range(pair_e.v1.shape[1]):
        plt.plot(pair_e.gamma1, pair_e.v1[:, j], 'r', linewidth=0.5)
    plt.show()
    
    vecdir = np.zeros(pair_e.m_points_accepted.shape)
    point  = np.zeros(pair_e.m_points_accepted.shape)

    for j in range(pair_e.m_points_accepted.shape[1]):
        Mpoint = np.array([pair_e.m_points_accepted[0][j],pair_e.m_points_accepted[1][j],pair_e.m_points_accepted[2][j]])
        plane = ComputePlaneEquation(pair_e.s0, pair_e.s1, Mpoint)
        #intersection bp and epipolar
        vecdir[:,j], point[:,j] = ComputePlanesIntersection(pair_bp.volDirectionT0.T[2,:], plane[0], plane[1])
    plt.figure()
    plt.plot([c0r[0],c1r[0],c2r[0],c3r[0],c0r[0]],[c0r[1],c1r[1],c2r[1],c3r[1],c0r[1]],'k',linewidth=1.5)
    plt.plot(lines_real_det0[:,0,0],lines_real_det0[:,0,1],color='k',label='virt. det. lines, K = %d' %(len(pair_bp.vk0)))
    for i in range(1,len(pair_bp.vk0)):
        plt.plot(lines_real_det0[:,i,0],lines_real_det0[:,i,1],color='k')
    dint=np.array([point[:,0]+t*vecdir[:,0] for t in np.arange(np.round(c0r[0]),np.round(c3r[0]))])
    plt.plot(dint[:,0],dint[:,1],'--',color = 'lightgray',label=r"$\Pi_k^{\lambda,\lambda'} \cap virt. det.$, K = %d" %(pair_e.m_points_accepted.shape[1]))
    for j in range(1,pair_e.m_points_accepted.shape[1]):
        dint=np.array([point[:,j]+t*vecdir[:,j] for t in np.arange(np.round(c0r[0]),np.round(c3r[0]))])
        plt.plot(dint[:,0],dint[:,1],'--',color = 'lightgray')
    plt.xlim((min([c0r[0],c1r[0],c2r[0],c3r[0],c0r[0]])-100,max([c0r[0],c1r[0],c2r[0],c3r[0],c0r[0]])+100))
    plt.ylim((min([c0r[1],c1r[1],c2r[1],c3r[1],c0r[1]])-1,max([c0r[1],c1r[1],c2r[1],c3r[1],c0r[1]])+1))
    plt.legend()
    plt.xlabel('y (mm)')
    plt.ylabel('z (mm)')
    plt.show()
    return 0


def ComputePlaneEquation(A, B, C):  # compute the cartesian equation of the plane formed by the three point ABC
    AB = A-B
    if (np.dot(AB, np.array([1., 0., 0.])) < 0):
            AB *= -1
    AC = C-A
    normal = np.cross(AB, AC)
    normal /= np.linalg.norm(normal)
    d = -1*np.dot(normal, C)
    return normal, d