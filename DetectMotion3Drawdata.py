#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 19:11:40 2021

@author: mmouchet
"""

from RTKToArrayConversion import *
from ExtendedConeBeamDCC import *
from TextFileSaving import *
from AllAcquisitionCDClass import *
import sys

if len ( sys.argv ) < 4:
    print( "Usage: DetectMotion3DAcqui <proj_file> <geometry_file> <down_range> <up_range> <error><errors_file>" )
    sys.exit ( 1 )

# reading projections
proj = itk.imread(sys.argv[1])
# Reading the geometry of the scanner
xmlreader = rtk.ThreeDCircularProjectionGeometryXMLFileReader.New()
xmlreader.SetFilename(sys.argv[2])
xmlreader.GenerateOutputInformation()
geometry = xmlreader.GetOutputObject()
print('nproj = %d'%(len(geometry.GetGantryAngles())))
print(proj.GetLargestPossibleRegion().GetSize())

#Convert to array for faster computation
geometry_array = RTKtoNP(geometry)
proj_array = itk.GetArrayFromImage(proj)
proj_infos = GetProjectionInformations(proj)
source_pos_array = GetSourcePositions(geometry)
rotation_matrices_array = GetRotationMatrices(geometry)
fixed_matrices_array = GetFixedSystemMatrices(geometry)
print(proj_infos)
n_proj_per_rotation = np.where(geometry_array[2, :] == geometry_array[2, 0])[0][1] - np.where(geometry_array[2, :] == geometry_array[2, 0])[0][0]
print(n_proj_per_rotation)

# first_idx = np.where(geometry_array[8,:]>np.float32(sys.argv[4]))[0][-1]
# last_idx = np.where(geometry_array[8,:]>np.float32(sys.argv[3]))[0][-1]
# print(first_idx, np.float32(sys.argv[4]), last_idx, np.float32(sys.argv[3]))
idx_proj_ref = np.arange(proj_infos[2][2])
print(len(idx_proj_ref))

#compute dcc
AcquiDCC = DCCOnCDinAnAcquisition(geometry_array, source_pos_array, rotation_matrices_array, fixed_matrices_array, proj_array, proj_infos)
print(AcquiDCC.axial_limit)
AcquiDCC.ComputeAllPossiblePairsForEachPos(idx_proj_ref)
AcquiDCC.ComputeDCCForEachPos(sys.argv[5])

median_file = sys.argv[6]
WriteAllPairsErrorsFile(median_file, idx_proj_ref, AcquiDCC.res)
print("yip yip")  
