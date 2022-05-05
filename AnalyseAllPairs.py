#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 13:19:56 2021

@author: mmouchet
"""

from TextFileSaving2 import *
import numpy as np
import matplotlib.pyplot as plt
from RTKToArrayConversion import *
from scipy import signal
from scipy.fft import fft, fftfreq


def SetLowPassFilter(n_proj_per_rotation, Trot):
    # Fréquence d'échantillonnage
    fe = n_proj_per_rotation/Trot  # Hz
    # Fréquence de nyquist
    f_nyq = fe / 2.  # Hz
    # Fréquence de coupure
    fc = 0.025*fe  # Hz
    # Préparation du filtre de Butterworth en passe-bas
    b, a = signal.butter(4, fc/f_nyq, 'low', analog=False)
    return b, a

def moving_average(x, y, w):
    ma = np.zeros(x.shape)
    weights = np.zeros(x.shape)
    for i in range(x.shape[0]):
#         print(len(np.where(np.abs(x-x[i]) <= w//2)[0]))
        ma[i] += np.mean(y[np.where(np.abs(x-x[i]) <= w//2)])
#         weights[i] += len(np.where(np.abs(x-x[i]) <= w//2)[0])
    return ma

# n42_42
# n65_20

plt.close("all")

# files_dir = "/home/mmouchet/Documents/SIEMENSDATA/GO.SIM/4D_patients/MP/donneesBrutes/sans_comp/n42_42/stats/"
# files_dir = "/home/mmouchet/Documents/SIEMENSDATA/GO.SIM/phantom/14_12_21/4D/stats_report2/"

#some geometries
xmlreader = rtk.ThreeDCircularProjectionGeometryXMLFileReader.New()
xmlreader.SetFilename("/home/mmouchet/Documents/SIEMENSDATA/GO.SIM/phantom/14_01_22/4D/sans_table/donneesBrutes/P007_Trot035_Tresp0/sub288_geometry_865_980.xml")
xmlreader.GenerateOutputInformation()
geo_0 = xmlreader.GetOutputObject()
xmlreader = rtk.ThreeDCircularProjectionGeometryXMLFileReader.New()
xmlreader.SetFilename("/home/mmouchet/Documents/SIEMENSDATA/GO.SIM/phantom/14_01_22/4D/sans_table/donneesBrutes/P007_Trot035_Tresp4/sub288_geometry_865_980.xml")
xmlreader.GenerateOutputInformation()
geo_1 = xmlreader.GetOutputObject()


geo_ar_0 =RTKtoNP(geo_0)
geo_ar_1 =RTKtoNP(geo_1)

# ar0 = itk.GetArrayFromImage(itk.imread("/home/mmouchet/Documents/SIEMENSDATA/GO.SIM/phantom/14_12_21/4D/donneesBrutes/P007_Trot035_Tresp0/sub288_corrected_proj_n375_n290.mha")).squeeze()
# ar1 = itk.GetArrayFromImage(itk.imread("/home/mmouchet/Documents/SIEMENSDATA/GO.SIM/phantom/14_12_21/4D/donneesBrutes/P007_Trot035_Tresp4/sub288_corrected_proj_n375_n290.mha")).squeeze()

# reader = itk.ImageFileReader[itk.Image[itk.F, 3]].New()
# reader.SetFileName("/home/mmouchet/Documents/SIEMENSDATA/GO.SIM/phantom/14_12_21/4D/images/mhd/P007_Trot035_Tresp0/average_CT.mha")
# recon0 = reader.GetOutput()
# recon0ar = itk.GetArrayFromImage(recon0).squeeze()
# reader = itk.ImageFileReader[itk.Image[itk.F, 3]].New()
# reader.SetFileName("/home/mmouchet/Documents/SIEMENSDATA/GO.SIM/phantom/14_12_21/4D/images/mhd/P007_Trot035_Tresp4/averageCT.mha")
# recon1 = reader.GetOutput()
# recon1ar = itk.GetArrayFromImage(reader.GetOutput()).squeeze()
################################ One rotation ###################################################################

# idx0, data0 = ReadMoments(files_dir +"Tresp0_corr_on29001_n33199_proj37282_HannWover10.csv")
# idx0_fov, data0_fov = ReadMoments(files_dir +"Tresp0_corr_on29001_n33199_proj37282_HannWover10_FOV.csv")
# # idx0_bp, data0_bp = ReadMoments(files_dir +"Tresp0_sub288_corr_on29001_n33199_proj9320_BP_FOV")
# idx1, data1 = ReadMoments(files_dir +"Tresp4_corr_on29001_n33199_proj37290_HannWover10.csv")
# idx1_fov, data1_fov = ReadMoments(files_dir +"Tresp4_corr_on29001_n33199_proj37290_HannWover10_FOV.csv")
# # idx1_bp, data1_bp = ReadMoments(files_dir +"Tresp4_sub288_corr_on29001_n33199_proj9322_BP_FOV")

# T_proj = 0.35/1152
# # ref0 = 32840
# # ref1 = 37290

# plt.figure(figsize = (8,16))
# plt.subplot(3,1,1)
# plt.title("All pairs")
# # plt.scatter(idx0*T_proj, [np.sum(data0[0][i]-data0[1][i])/len(data0[0][i]) for i in range(len(idx0))], color="blue", marker = 's', label="static")
# plt.scatter(idx1*T_proj, [np.sum(data1[0][i])/len(data1[0][i]) for i in range(len(idx1))], color='orange', marker = '.', label="breathing")
# plt.axhline(y=0, color='k',linewidth=0.5)
# plt.legend()
# plt.xticks([])
# plt.subplot(3,1,2)
# plt.title("No FOV pairs")
# # plt.scatter(idx0_fov*T_proj, [np.sum(data0_fov[0][i]-data0_fov[1][i])/len(data0_fov[0][i]) for i in range(len(idx0_fov))], color = "green", marker = 's', label="static")
# plt.scatter(idx1_fov*T_proj, [np.sum(data1_fov[0][i])/len(data1_fov[0][i]) for i in range(len(idx1_fov))], color="red", marker = '.', label="breathing")
# plt.axhline(y=0, color='k',linewidth=0.5)
# plt.legend()
# plt.xticks([])
# # plt.subplot(4,1,3)
# # plt.scatter(idx0_bp, [np.sum(data0_bp[0][i]-data0_bp[1][i])/len(data0_bp[0][i]) for i in range(len(idx0_bp))], marker = 's',  label="static")
# # plt.scatter(idx1_bp, [np.sum(data1_bp[0][i]-data1_bp[1][i])/len(data1_bp[0][i]) for i in range(len(idx1_bp))], marker = '.',  label="breathing")
# # plt.axhline(y=0, color='k',linewidth=0.5)
# plt.subplot(3,1,3)
# plt.plot(idx0*T_proj, moving_average(idx0, np.array([np.sum(data0[0][i]-data0[1][i])/len(data0[0][i]) for i in range(len(idx0))]), 1152), color="blue", label="static - all pairs XDCC")
# plt.plot(idx0_fov*T_proj, moving_average(idx0_fov, np.array([np.sum(data0_fov[0][i]-data0_fov[1][i])/len(data0_fov[0][i]) for i in range(len(idx0_fov))]), 1152), '--', color='orange', label="static - no FOV pairs XDCC")
# # plt.plot(idx0_bp, moving_average(idx0_bp, np.array([np.sum(data0_bp[0][i]-data0_bp[1][i])/len(data0_bp[0][i]) for i in range(len(idx0_bp))]), 288), label="static - no FOV pair BP")
# plt.plot(idx1*T_proj, moving_average(idx1, np.array([np.sum(data1[0][i]-data1[1][i])/len(data1[0][i]) for i in range(len(idx1))]), 1152), color = "green", label="breathing - all pairs XDCC")
# plt.plot(idx1_fov*T_proj, moving_average(idx1_fov, np.array([np.sum(data1_fov[0][i]-data1_fov[1][i])/len(data1_fov[0][i]) for i in range(len(idx1_fov))]), 1152), '--', color="red", label="breathing - no FOV pairs XDCC")
# # plt.plot(idx1_bp, moving_average(idx1_bp, np.array([np.sum(data1_bp[0][i]-data1_bp[1][i])/len(data1_bp[0][i]) for i in range(len(idx1_bp))]), 288), label="breathing - no FOV pair BP")
# plt.axhline(y=0, color='k',linewidth=0.5)
# plt.legend()
# plt.show()

# plt.figure(figsize = (16,8))
# plt.scatter((idx0)*T_proj, [np.sum((data0[0][i]-data0[1][i]))/len(data0[0][i]) for i in range(len(idx0))], color="blue", marker = 's', s=10, label="all")
# plt.scatter(idx0_fov*T_proj, [np.sum((data0_fov[0][i]-data0_fov[1][i]))/len(data0_fov[0][i]) for i in range(len(idx0_fov))], color = "green", marker = 's', s=10, label="fov")
# plt.axhline(y=0, color='k',linewidth=0.5)
# plt.legend()
# plt.xlabel("Pair gantry angles difference")
# plt.ylabel(r"$E_{\lambda, \lambda'}$")
# plt.show()


# x0 = np.linspace(recon0.GetOrigin()[0], recon0.GetOrigin()[0]+(recon0ar.shape[2]-1)*recon0.GetSpacing()[0])
# y0 = np.linspace(recon0.GetOrigin()[1], recon0.GetOrigin()[1]+(recon0ar.shape[1]-1)*recon0.GetSpacing()[1])
# z0 = np.linspace(recon0.GetOrigin()[2], recon0.GetOrigin()[2]+(recon0ar.shape[0]-1)*recon0.GetSpacing()[2])
# x1 = np.linspace(recon1.GetOrigin()[0], recon1.GetOrigin()[0]+(recon1ar.shape[2]-1)*recon1.GetSpacing()[0])
# y1 = np.linspace(recon1.GetOrigin()[1], recon1.GetOrigin()[1]+(recon1ar.shape[1]-1)*recon1.GetSpacing()[1])
# z1 = np.linspace(recon1.GetOrigin()[2], recon1.GetOrigin()[2]+(recon1ar.shape[0]-1)*recon1.GetSpacing()[2])

# fig = plt.figure( figsize=(16,8))
# plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9,hspace = 0.6)
# ax = fig.add_gridspec(4, 4)
# ax1 = fig.add_subplot(ax[0:2, 0:2])
# ax1.imshow(recon0ar[39,:,:], cmap = "gray", clim = [-1000,1000], extent=(x0[0], x0[-1], y0[-1], y0[0]))
# ax1.set_xlabel("x (mm)")
# ax1.set_ylabel("y (mm)")
# ax1 = fig.add_subplot(ax[0, 2:4])
# plt.imshow(recon0ar[:,261,:], cmap = "gray", clim = [-1000,1000], origin = "lower", extent=(x0[0], x0[-1], z0[0], z0[-1]))
# plt.axhline(y=geo_ar_0[8, idx0[0]], color='r')
# plt.axhline(y=geo_ar_0[8, 9320], color='y', linestyle = '--')
# plt.axhline(y=geo_ar_0[8, idx0[-1]], color='r')
# ax1.set_xlabel("x (mm)")
# ax1.set_ylabel("z (mm)")
# ax1 = fig.add_subplot(ax[1, 2:4])
# plt.imshow(recon0ar[:,:,329], cmap = "gray", clim = [-1000,1000], origin = "lower", extent=(y0[0], y0[-1], z0[0], z0[-1]))
# plt.axhline(y=geo_ar_0[8, idx0[0]], color='r')
# plt.axhline(y=geo_ar_0[8, 9320], color='y', linestyle = '--')
# plt.axhline(y=geo_ar_0[8, idx0[-1]], color='r')
# ax1.set_xlabel("y (mm)")
# ax1.set_ylabel("z (mm)")
# ax1 = fig.add_subplot(ax[2:4, 0:2])
# ax1.imshow(recon1ar[39,:,:], cmap = "gray", clim = [-1000,1000], extent=(x1[0], x1[-1], y1[-1], y1[0]))
# ax1.set_xlabel("x (mm)")
# ax1.set_ylabel("y (mm)")
# ax1 = fig.add_subplot(ax[2, 2:4])
# plt.imshow(recon1ar[:,261,:], cmap = "gray", clim = [-1000,1000],  origin = "lower", extent=(x1[0], x1[-1], z1[0], z1[-1]))
# plt.axhline(y=geo_ar_1[8, idx1[0]], color='r')
# plt.axhline(y=geo_ar_1[8, 9322], color='y', linestyle = '--')
# plt.axhline(y=geo_ar_1[8, idx1[-1]], color='r')
# ax1.set_xlabel("x (mm)")
# ax1.set_ylabel("z (mm)")
# ax1 = fig.add_subplot(ax[3, 2:4])
# plt.imshow(recon1ar[:,:,329], cmap = "gray", clim = [-1000,1000], origin = "lower", extent=(y1[0], y1[-1], z1[0], z1[-1]))
# plt.axhline(y=geo_ar_1[8, idx1[0]], color='r')
# plt.axhline(y=geo_ar_1[8, 9322], color='y', linestyle = '--')
# plt.axhline(y=geo_ar_1[8, idx1[-1]], color='r')
# ax1.set_xlabel("y (mm)")
# ax1.set_ylabel("z (mm)")
# plt.show()



#################################################### Medians ##############

#Diff
refs0, idx0, res0 = ReadAllPairsErrorsFile("/home/mmouchet/Documents/SIEMENSDATA/GO.SIM/phantom/14_01_22/4D/sans_table/donneesBrutes/P007_Trot035_Tresp0/Tresp0_sub288_angle0")
refs1, idx1, res1 = ReadAllPairsErrorsFile("/home/mmouchet/Documents/SIEMENSDATA/GO.SIM/phantom/14_01_22/4D/sans_table/donneesBrutes/P007_Trot035_Tresp4/Tresp4_sub288_angle0")

# protocol_a = SetLowPassFilter(72, 0.35)
# protocol_b = SetLowPassFilter(72, 0.35)

T_proj0 = 0.35/288
T_proj1 = 0.35/288

# filtered0 = signal.filtfilt(protocol_a[0], protocol_a[1], [np.mean(res0[i]) for i in range(len(refs0))])
# filtered1 = signal.filtfilt(protocol_a[0], protocol_a[1], [np.mean(res1[i]) for i in range(len(refs1))])



# plt.figure()
# plt.subplot(121)
# plt.title("angle")
# plt.plot(refs0, geo_ar_0[2, refs0])
# plt.plot(refs1, geo_ar_1[2, refs1])
# plt.subplot(122)
# plt.title("sy")
# plt.plot(refs0, geo_ar_0[8, refs0],'.')
# plt.plot(refs1, geo_ar_1[8, refs1],'.')
# plt.show()

t0 = 18.0753-2.20
proj1 = t0//T_proj1
# print(proj1)
# plt.figure()
# plt.plot(np.arange(geo_ar_0.shape[1])*T_proj1, np.sin(2*np.pi*(np.arange(geo_ar_0.shape[1])-proj1)*T_proj1/4))
# plt.plot(idx1[51]*T_proj1, np.sin(2*np.pi*(idx1[51]*T_proj1-t0)/4))
# plt.plot(np.arange(88)*0.35, np.sin(2*np.pi*(np.arange(88)-45.15)*0.35/4),'k--')
# plt.axhline(y=np.sin(2*np.pi*(refs1[51]*T_proj1-t0)/4))
# plt.axhline(y=0, color='k',linewidth =0.5)
# plt.axvline(x=refs1[51]*T_proj1, color='r')
# plt.axvline(x=refs1[51]*T_proj1+1.5886, color='r')
# plt.axvline(x=refs1[51]*T_proj1+4, color='r')
# plt.show()


ref_idx = 49

fig = plt.figure()
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9,hspace = 0.6)
ax = fig.add_gridspec(3, 2)
ax1 = fig.add_subplot(ax[0:2, 0:2])
plt.scatter(idx0[ref_idx]*T_proj0, res0[ref_idx], color="blue", marker = 's', s=10, label="static")
plt.scatter(idx1[ref_idx]*T_proj1, res1[ref_idx], color="darkorange", marker = 'x', s=10, label="moving")
# plt.scatter(idx1[ref_idx]*T_proj1, moving_average(idx1[ref_idx], res1[ref_idx], 288))
plt.axvline(x=refs1[ref_idx]*T_proj1, color='y', linestyle = '--')
plt.axvline(x=4 + refs1[ref_idx]*T_proj1, color='y', linestyle = '--')
plt.axvline(x=-4 + refs1[ref_idx]*T_proj1, color='y', linestyle = '--')
plt.legend()
plt.axhline(y=0, color='k', linewidth=0.5)
ax1 = fig.add_subplot(ax[2, 0:2])
plt.plot(idx1[ref_idx]*T_proj1, np.sin(2*np.pi*(idx1[ref_idx]*T_proj1-t0)/4),'k')
plt.axvline(x=refs1[ref_idx]*T_proj1, color='y', linestyle = '--')
plt.axvline(x=4 + refs1[ref_idx]*T_proj1, color='y', linestyle = '--')
plt.axvline(x=-4 + refs1[ref_idx]*T_proj1, color='y', linestyle = '--')
plt.show()

fig = plt.figure()
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9,hspace = 0.6)
ax = fig.add_gridspec(7, 2)
ax1 = fig.add_subplot(ax[0:2, 0:2])
plt.title(r"(a) Errors for all pairs computed from the reference $\lambda$ with no FOV condition")
plt.scatter(idx0[ref_idx], res0[ref_idx], color="blue", marker = 's', s=10, label="static")
plt.scatter(idx1[ref_idx], res1[ref_idx], color="darkorange", marker = 'x', s=10, label="moving")
plt.axvline(x=refs1[ref_idx], color='y', linestyle = '--')
plt.axvline(x=4/T_proj1 + refs1[ref_idx], color='y', linestyle = '--')
plt.axvline(x=-4/T_proj1 + refs1[ref_idx], color='y', linestyle = '--')
plt.xlim([idx1[ref_idx][0]-288, idx1[ref_idx][-1]+288])
plt.xticks([])
plt.legend()
plt.ylabel(r"$E_{\lambda, \lambda'}$")
plt.axhline(y=0, color='k', linewidth=0.5)
ax1 = fig.add_subplot(ax[2:4, 0:2])
plt.title(r"(b) Errors for all pairs computed from the reference $\lambda$ with FOV condition")
plt.scatter(idx0[ref_idx], res0[ref_idx], color="blue", marker = 's', s=10, label="static")
plt.scatter(idx1[ref_idx], res1[ref_idx], color="darkorange", marker = 'x', s=10, label="moving")
plt.axvline(x=refs1[ref_idx], color='y', linestyle = '--')
plt.axvline(x=4/T_proj1 + refs1[ref_idx], color='y', linestyle = '--')
plt.axvline(x=-4/T_proj1 + refs1[ref_idx], color='y', linestyle = '--')
plt.xticks([])
plt.xlim([idx1[ref_idx][0]-288, idx1[ref_idx][-1]+288])
plt.legend()
plt.ylabel(r"$E_{\lambda, \lambda'}$")
plt.axhline(y=0, color='k', linewidth=0.5)
ax1 = fig.add_subplot(ax[4:6, 0:2])
plt.title(r"(c) Moving average of the errors for all pairs computed from the reference $\lambda$ with and without FOV condition")
plt.plot(idx0[ref_idx], moving_average(idx0[ref_idx], res0[ref_idx], 288), color = 'blue', label = "static")
plt.plot(idx1[ref_idx], moving_average(idx1[ref_idx], res1[ref_idx], 288), color = 'darkorange', label = "moving")
plt.axvline(x=refs1[ref_idx], color='y', linestyle = '--')
plt.axvline(x=4/T_proj1 + refs1[ref_idx], color='y', linestyle = '--')
plt.axvline(x=-4/T_proj1 + refs1[ref_idx], color='y', linestyle = '--')
plt.xticks([])
plt.xlim([idx1[ref_idx][0]-288, idx1[ref_idx][-1]+288])
plt.legend()
plt.ylabel(r"$E_{\lambda, \lambda'}$")
plt.axhline(y=0, color='k', linewidth=0.5)
ax1 = fig.add_subplot(ax[6, 0:2])
plt.title(r"(d) Tumor motion")
plt.plot(idx1[ref_idx], 10*np.sin(2*np.pi*(idx1[ref_idx]*T_proj1-t0)/4),'k')
plt.axvline(x=refs1[ref_idx], color='y', linestyle = '--')
plt.axvline(x=4/T_proj1 + refs1[ref_idx], color='y', linestyle = '--')
plt.axvline(x=-4/T_proj1 + refs1[ref_idx], color='y', linestyle = '--')
plt.xlim([idx1[ref_idx][0]-288, idx1[ref_idx][-1]+288])
plt.ylabel("Tumor motion (mm)")
plt.xlabel(r"$\lambda'(rad)$")
plt.xticks(np.arange(-12, 13, 2)*288+refs1[ref_idx], [r"$\lambda-22\pi$",r"$\lambda-20\pi$",r"$\lambda-16\pi$",r"$\lambda-12\pi$",r"$\lambda-8\pi$",r"$\lambda-4\pi$",r"$\lambda$",r"$\lambda+4\pi$",r"$\lambda+8\pi$",r"$\lambda+12\pi$",r"$\lambda+16\pi$",r"$\lambda+20\pi$",r"$\lambda+22\pi$"])
plt.show()




# plt.figure()
# plt.plot(idx0[ref_idx]*T_proj0, moving_average(idx0[ref_idx], res0[ref_idx], 288), color = 'blue', label = "static")
# plt.plot(idx1[ref_idx]*T_proj1, moving_average(idx1[ref_idx], res1[ref_idx], 288), color = 'darkorange', label = "moving")
# plt.legend()
# plt.axhline(y=0, color='k', linewidth=0.5)
# plt.show()




# plt.figure()
# plt.title('Detection of motion in 4D acquisitions with 72 projections per rotation')
# plt.plot(refs0*T_proj0, [np.mean(res0[i]) for i in range(len(refs0))], linewidth=0.5, label="static")
# plt.plot(refs1*T_proj1, [np.mean(res1[i]) for i in range(len(refs1))], linewidth=0.5,label="breathing")
# # plt.plot(refs0*T_proj0, filtered0, '--', label="filtered static")
# # plt.plot(refs1*T_proj1, filtered1, '--', label="filtered breathing")
# plt.axhline(y=0, color='k', linewidth=0.5)
# plt.legend()
# plt.ylabel("Med(E)")
# plt.xlabel("t(s)")
# # ax = plt.twiny()
# # plt.xlim((geo_ar_0[8, refs0[0]]+5,geo_ar_0[8, refs0[-1]]-5))
# # plt.xlabel("z (mm)")
# plt.show()

# # plt.figure()
# # plt.title("Row 16 of the sinogram from the static acquisition")
# # plt.imshow(ar0[refs0,16,:].T,cmap="gray", clim=[-3,3], origin="lower", extent=(refs0[0], refs0[-1], 0, 919))
# # plt.yticks([])
# # # ax = plt.twinx()
# # # ax.plot(refs0, [np.median(res0[i]) for i in range(len(refs0))],'r', linewidth=0.5, label="DCC")
# # # plt.yticks([])
# # # plt.legend()
# # plt.show()

# # plt.figure()
# # plt.title("Row 16 of the sinogram from the breathing acquisition")
# # plt.imshow(ar1[refs1,16,:].T,cmap="gray", origin="lower", extent=(refs1[0], refs1[-1], 0, 919))
# # plt.yticks([])
# # ax = plt.twinx()
# # ax.plot(refs1, [np.median(res1[i]) for i in range(len(refs1))],'r', linewidth=0.5, label="DCC")
# # plt.yticks([])
# # plt.legend()
# # plt.show()

# # plt.figure()
# # plt.suptitle("Error for all possile pairs from ref at t = %.3f" %(refs0[len(refs0)//2]*T_proj0))
# # plt.subplot(211)
# # plt.plot(res0[len(refs0)//2], label='Diff')
# # plt.axhline(y=0, color='k', linewidth=0.5)
# # # plt.ylabel()
# # plt.xlabel("Pair idx")
# # plt.subplot(212)
# # plt.plot(res1[len(refs1)//2], label='AbsDiff')
# # plt.axhline(y=0, color='k', linewidth=0.5)
# # # plt.ylabel()
# # plt.xlabel("Pair idx")
# # plt.show()

##############################################################################"

# plt.figure()
# plt.plot(np.linspace(0,1)*8,np.sin(2*np.pi*np.linspace(0,8)/4))
# plt.axhline(y=0,color='k', linewidth=0.5)
# plt.show()
