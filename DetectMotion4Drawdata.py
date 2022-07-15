#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 19:11:40 2021

@author: mmouchet
"""

from ConeBeamDCCWithBackprojectionPlane import ExtractSlice
from RTKToArrayConversion import *
from ExtendedConeBeamDCC import *
from TextFileSaving import *
from AllAcquisitionCD_Beta_Class import *
import sys

if len ( sys.argv ) < 7:
    print( "Usage: DetectMotion4DAcqui <dir> <current_file> <delta_proj> <FOV_condition> <variance> <kernel> <bandwidth> <res_file>" )
    sys.exit ( 1 )

# reading projections
proj_init = itk.imread(sys.argv[1]+"corrected_proj_1.mha")
# Reading the geometry of the scanner
xmlreader = rtk.ThreeDCircularProjectionGeometryXMLFileReader.New()
xmlreader.SetFilename(sys.argv[1]+"geometry.xml")
xmlreader.GenerateOutputInformation()
geometry = xmlreader.GetOutputObject()
print('nproj = %d'%(len(geometry.GetGantryAngles())))
print(proj_init.GetLargestPossibleRegion().GetSize())

#Convert to array for faster computation
geometry_array = RTKtoNP(geometry)
proj_array = itk.GetArrayFromImage(proj_init)
proj_infos = GetProjectionInformations(proj_init)
source_pos_array = GetSourcePositions(geometry)
rotation_matrices_array = GetRotationMatrices(geometry)
fixed_matrices_array = GetFixedSystemMatrices(geometry)
print(proj_infos)
AcquiDCC = DCCOnCDinAnAcquisition(geometry_array, source_pos_array, rotation_matrices_array, fixed_matrices_array, proj_array, proj_infos)

if current_file == "0":
    Ni = 0
else:
    bt_120 = np.genfromtxt(sys.argv[3], skip_header=1, unpack=True).T
    supp_current = np.genfromtxt(sys.argv[4], skip_header=0, unpack=True, delimiter= ',').T
    AcquiDCC.ComputeInitialNumberOfPhotons(bt_120, supp_current)


# ref_list = np.arange(0, geometry_array.shape[1], int(sys.argv[3]))
ref_list = np.array([0, geometry_array.shape[1]-1])

#Compute dcc from one ref
for ref in ref_list:
    stack_ref = ref//proj_array.shape[0] + 1
    proj_ref = itk.GetArrayFromImage(itk.imread(sys.argv[1]+"corrected_proj_%i.mha" % (stack_ref)))[ref-proj_array.shape[0]*(stack_ref-1), :, :]
    res = []
    idx = []
    for i in range(-AcquiDCC.axial_limit, AcquiDCC.axial_limit):
        if sys.argv[4] == 'True':
            if ref == i+ref or i+ref < 0 or i+ref >= proj_infos[2][2] or np.abs(geometry_array[2, ref]-geometry_array[2, ref+i]) < 10**(-12):
                pass
            elif np.cos(AcquiDCC.geometry[2, ref]-AcquiDCC.geometry[2, i+ref]) > 2*((AcquiDCC.R_fov)/AcquiDCC.geometry[0, 0])**2-1:
                pair_geo = np.array([geometry_array[:, ref], geometry_array[:, i + ref]])
                pair_source_pos = np.array([source_pos_array[ref, :], source_pos_array[i+ref, :]])
                pair_mrot = [rotation_matrices_array[ref], source_pos_array[i+ref]]
                pair_fsm = [fixed_matrices_array[ref], fixed_matrices_array[i+ref]]
                stack_i = (ref+i)//proj_array.shape[0] + 1
                proj_i = itk.GetArrayFromImage(itk.imread(sys.argv[1]+"corrected_proj_%i.mha" % (stack_i)))[ref+i-proj_array.shape[0]*(stack_i-1), :, :]
                pair_proj = np.array([proj_ref, proj_i])
                pair = ProjectionsPairBeta(0, 1, pair_geo, pair_source_pos, pair_mrot, pair_fsm, pair_proj, AcquiDCC.proj_infos, AcquiDCC.Ni, sys.argv[5], sys.argv[6], int(sys.argv[7]))
                pair.ComputeBetaRange()
                if len(pair.beta) >= 1:
                    pair.ComputePairMoments()
                    idx.append(i+ref)
                    if type(pair.m0) == np.float64:
                        res.append([np.array([pair.m0]), np.array([pair.m1]), pair.tot_var0, pair.tot_var1])
                    else:
                        res.append([pair.m0, pair.m1, pair.tot_var0, pair.tot_var1])
        elif sys.argv[4] == 'False':
            if ref == i+ref or i+ref < 0 or i+ref >= proj_infos[2][2] or np.abs(geometry_array[2, ref]-geometry_array[2, ref+i]) < 10**(-12):
                pass
            else:
                pair_geo = np.array([geometry_array[:, ref], geometry_array[:, i + ref]])
                pair_source_pos = np.array([source_pos_array[ref, :], source_pos_array[i+ref, :]])
                pair_mrot = [rotation_matrices_array[ref], source_pos_array[i+ref]]
                pair_fsm = [fixed_matrices_array[ref], fixed_matrices_array[i+ref]]
                stack_i = (ref+i)//proj_array.shape[0] + 1
                proj_i = itk.GetArrayFromImage(itk.imread(sys.argv[1]+"corrected_proj_%i.mha" % (stack_i)))[ref+i-proj_array.shape[0]*(stack_i-1), :, :]
                pair_proj = np.array([proj_ref, proj_i])
                pair = ProjectionsPairBeta(0, 1, pair_geo, pair_source_pos, pair_mrot, pair_fsm, pair_proj, AcquiDCC.proj_infos, AcquiDCC.Ni, sys.argv[5], sys.argv[6], int(sys.argv[7]))
                pair.ComputeBetaRange()
                if len(pair.beta) >= 1:
                    pair.ComputePairMoments()
                    idx.append(i+ref)
                    if type(pair.m0) == np.float64:
                        res.append([np.array([pair.m0]), np.array([pair.m1]), pair.tot_var0, pair.tot_var1])
                    else:
                        res.append([pair.m0, pair.m1, pair.tot_var0, pair.tot_var1])
    WriteMomentsFile(sys.argv[1]+sys.argv[8]+"%i.csv" % (ref), idx, res)

print("yip yip")  
