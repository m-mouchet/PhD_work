#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 13:19:56 2021

@author: mmouchet
"""

from TextFileSaving import *
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

# n42_42
# n65_20

plt.close("all")

files_dir = "/home/mmouchet/Documents/SIEMENSDATA/GO.SIM/4D_patients/MP/donneesBrutes/sans_comp/n42_42/stats/"
# files_dir2 = "/home/mmouchet/Documents/SIEMENSDATA/GO.SIM/phantom/14_12_21/4D/stats/sub36_1over1/"

#Diff
refs0, res0 = ReadAllPairsErrorsFile(files_dir +  "sub36_n42_42_HannWover10_Diff.csv")
refs1, res1 = ReadAllPairsErrorsFile(files_dir +  "sub36_n42_42_HannWover10_AbsDiff.csv")
# refs2, res2 = ReadAllPairsErrorsFile(files_dir + "P007_Trot050_Tresp4/Hann_Wover50_Diff.csv")
# #Abs diff
# refs0_ad, res0_ad = ReadAllPairsErrorsFile(files_dir2 + "P007_Trot035_Tresp0/Hann_Wover100_Diff.csv")
# refs1_ad, res1_ad = ReadAllPairsErrorsFile(files_dir2 + "P007_Trot035_Tresp4/Hann_Wover100_Diff.csv")
# refs2_ad, res2_ad = ReadAllPairsErrorsFile(files_dir + "P007_Trot050_Tresp4/Hann_Wover50_AbsDiff.csv")

#some geometries
xmlreader = rtk.ThreeDCircularProjectionGeometryXMLFileReader.New()
xmlreader.SetFilename("/home/mmouchet/Documents/SIEMENSDATA/GO.SIM/4D_patients/MP/donneesBrutes/sans_comp/n42_42/sub36_geometry_n42_42.xml")
xmlreader.GenerateOutputInformation()
geo_0 = xmlreader.GetOutputObject()
xmlreader = rtk.ThreeDCircularProjectionGeometryXMLFileReader.New()
xmlreader.SetFilename("/home/mmouchet/Documents/SIEMENSDATA/GO.SIM/4D_patients/MP/donneesBrutes/sans_comp/n42_42/sub36_geometry_n42_42.xml")
xmlreader.GenerateOutputInformation()
geo_1 = xmlreader.GetOutputObject()


geo_ar_0 =RTKtoNP(geo_0)
geo_ar_1 =RTKtoNP(geo_1)


protocol_a = SetLowPassFilter(36, 0.35)
protocol_b = SetLowPassFilter(36, 0.35)

T_proj0 = 0.35/36
T_proj1 = 0.35/36

filtered0 = signal.filtfilt(protocol_a[0], protocol_a[1], [np.median(res0[i]) for i in range(len(refs0))])
filtered1 = signal.filtfilt(protocol_a[0], protocol_a[1], [np.median(res1[i]) for i in range(len(refs1))])
# filtered2 = signal.filtfilt(protocol_b[0], protocol_b[1], [np.median(res2[i]) for i in range(len(refs2))])
# filtered0_ad = signal.filtfilt(protocol_a[0], protocol_a[1], [np.median(res0_ad[i]) for i in range(len(refs0_ad))])
# filtered1_ad = signal.filtfilt(protocol_a[0], protocol_a[1], [np.median(res1_ad[i]) for i in range(len(refs1_ad))])
# filtered2_ad = signal.filtfilt(protocol_b[0], protocol_b[1], [np.median(res2_ad[i]) for i in range(len(refs2_ad))])

# plt.figure()
# plt.subplot(231)
# plt.plot(refs0, [np.median(res0[i]) for i in range(len(refs0))])
# plt.plot(refs0_ad, [np.median(res0_ad[i]) for i in range(len(refs0_ad))])
# # plt.plot(refs0, filtered0)
# plt.subplot(232)
# plt.plot(refs1, [np.median(res1[i]) for i in range(len(refs1))])
# # plt.plot(refs1, filtered1)
# # plt.subplot(233)
# # plt.plot(refs2, [np.median(res2[i]) for i in range(len(refs2))])
# # plt.plot(refs2, filtered2)
# # plt.subplot(234)
# # plt.plot(refs0_ad, [np.median(res0_ad[i]) for i in range(len(refs0_ad))])
# # # plt.plot(refs0_ad, filtered0_ad)
# # plt.subplot(235)
# # plt.plot(refs1_ad, [np.median(res1_ad[i]) for i in range(len(refs1_ad))])
# # plt.plot(refs1_ad, filtered1_ad)
# # plt.subplot(236)
# # plt.plot(refs2_ad, [np.median(res2_ad[i]) for i in range(len(refs2_ad))])
# # plt.plot(refs2_ad, filtered2_ad)
# plt.show()


plt.figure()
plt.subplot(211)
plt.plot(refs0*T_proj0, [np.median(res0[i]) for i in range(len(refs0))], label="Diff")
plt.axhline(y=0, color='k', linewidth=0.5)
plt.legend()
plt.ylabel("Med(E)")
plt.xlabel("t(s)")
# ax = plt.twiny()
# plt.xlim((geo_ar_0[8, refs0[0]]+5,geo_ar_0[8, refs0[-1]]-5))
# plt.xlabel("z (mm)")
plt.subplots_adjust(hspace=0.5)
plt.subplot(212)
# plt.plot(refs0*T_proj0, filtered0, label="sub72")
# plt.plot(refs1*T_proj1, filtered1, label="sub36")
plt.plot(refs1*T_proj1, [np.median(res1[i]) for i in range(len(refs1))] ,label="AbsDiff")
plt.axhline(y=0, color='k', linewidth=0.5)
plt.legend()
plt.ylabel("Med(E)")
plt.xlabel("t(s)")
# ax = plt.twiny()
# plt.xlim((geo_ar_0[8, refs0[0]]+5,geo_ar_0[8, refs0[-1]]-5))
# plt.xlabel("z (mm)")
plt.show()

# plt.figure()
# plt.subplot(211)
# plt.plot(refs0_ad*0.35/36, [np.median(res0_ad[i]) for i in range(len(refs0_ad))])
# plt.plot(refs0_ad*0.35/36, filtered0_ad)
# plt.plot(refs1_ad*0.35/36, [np.median(res1_ad[i]) for i in range(len(refs1_ad))])
# plt.plot(refs1_ad*0.35/36, filtered1_ad)
# plt.axhline(y=0, color='k', linewidth=0.5)
# plt.subplot(212)
# plt.plot(geo_ar_0[8, refs0_ad], [np.median(res0_ad[i]) for i in range(len(refs0_ad))])
# plt.plot(geo_ar_0[8, refs0_ad], filtered0_ad)
# plt.plot(geo_ar_1[8, refs1_ad], [np.median(res1_ad[i]) for i in range(len(refs1_ad))])
# plt.plot(geo_ar_1[8, refs1_ad], filtered1_ad)
# plt.axhline(y=0, color='k', linewidth=0.5)
# plt.xlim((geo_ar_0[8, refs0_ad[0]]+5,geo_ar_0[8, refs0_ad[-1]]-5))
# plt.show()

# yf1_flat = fft(filtered1)
# yf1 = fft([np.median(res1[i]) for i in range(len(refs1))])
# xf1 = fftfreq(len(refs1), T_proj1)[:len(refs1)//2]
# yf0 = fft([np.median(res0[i]) for i in range(len(refs0))])
# xf0 = fftfreq(len(refs0), T_proj0)[:len(refs0)//2]

# plt.figure()
# plt.plot(xf0, 2.0/len(refs0) * np.abs(yf0[0:len(refs0)//2]), label="sub72")
# plt.plot(xf1, 2.0/len(refs1) * np.abs(yf1[0:len(refs1)//2]),'--', label="sub36")
# # plt.plot(xf1, 2.0/len(refs1) * np.abs(yf1_flat[0:len(refs1)//2]),'-')
# plt.grid()
# plt.xlabel("Frequency (Hz)")
# plt.ylabel("F(Med(E))")
# plt.legend()
# plt.show()

plt.figure()
plt.suptitle("Error for all possile pairs from ref at t = %.3f" %(refs0[len(refs0)//2]*T_proj0))
plt.subplot(211)
plt.plot(res0[len(refs0)//2], label='Diff')
plt.axhline(y=0, color='k', linewidth=0.5)
# plt.ylabel()
plt.xlabel("Pair idx")
plt.subplot(212)
plt.plot(res1[len(refs1)//2], label='AbsDiff')
plt.axhline(y=0, color='k', linewidth=0.5)
# plt.ylabel()
plt.xlabel("Pair idx")
plt.show()

