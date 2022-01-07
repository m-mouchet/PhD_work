#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 10:13:56 2021

@author: mmouchet
"""

import matplotlib.pyplot as plt
from RTKToArrayConversion import *
from TextFileSaving import *
from scipy import signal


plt.close("all")

work_dir = "/home/mmouchet/Documents/SIEMENSDATA/GO.SIM/4D_patients/VJL/donneesBrutes/"

label_0 = "avec comp"
label_1 = "sans_comp"
medians_0 = ReadMediansFile(work_dir+"avec_comp/n30_20/avec_comp_n30_20_Hann_Wover5.csv")
medians_1 = ReadMediansFile(work_dir+"sans_comp/n30_20/sans_comp_n30_20_Hann_Wover5.csv")
proj_ar_0 = itk.GetArrayFromImage(itk.imread(work_dir+"avec_comp/n30_20/sub_corrected_proj_n30_20.mha"))
proj_ar_1 = itk.GetArrayFromImage(itk.imread(work_dir+"sans_comp/n30_20/sub_corrected_proj_n30_20.mha"))
xmlreader = rtk.ThreeDCircularProjectionGeometryXMLFileReader.New()
xmlreader.SetFilename(work_dir+"avec_comp/n30_20/sub_geometry_n30_20.xml")
xmlreader.GenerateOutputInformation()
geo_0 = xmlreader.GetOutputObject()
xmlreader = rtk.ThreeDCircularProjectionGeometryXMLFileReader.New()
xmlreader.SetFilename(work_dir+"sans_comp/n30_20/sub_geometry_n30_20.xml")
xmlreader.GenerateOutputInformation()
geo_1 = xmlreader.GetOutputObject()
geo_ar_0 =RTKtoNP(geo_0)
geo_ar_1 =RTKtoNP(geo_1)



plt.figure()
plt.title("Medians of the moment absolute difference as a function of the  reference source axial position")
plt.plot(geo_ar_0[8,medians_0[0]], medians_0[1], label=label_0)
plt.plot(geo_ar_1[8,medians_1[0]], medians_1[1], label=label_1)
plt.legend()
plt.xlim((geo_ar_0[8,medians_0[0][0]]+5, geo_ar_0[8,medians_0[0][-1]]-5))
plt.xlabel(r"$z_{\lambda} (mm)$")
plt.ylabel(r"$\mathrm{Med} \left( \frac{1}{K} \sum_k |M(\lambda)-M(\lambda')| \right)$")
plt.show()

plt.figure()
plt.title("Medians of the moment absolute difference as a function of time")
plt.plot(medians_0[0]*0.5/36, medians_0[1], label=label_0)
plt.plot(medians_1[0]*0.35/36, medians_1[1], label=label_1)
plt.xlabel("Time (s)")
plt.ylabel(r"$\mathrm{Med} \left( \frac{1}{K} \sum_k |M(\lambda)-M(\lambda')| \right)$")
plt.legend()
plt.show()
