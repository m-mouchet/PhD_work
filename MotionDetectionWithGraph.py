#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 13:34:38 2021

@author: mmouchet
"""

from RTKToArrayConversion import *
from ConeBeamDCCWithBackprojectionPlane import *
from AllAcquisitionBPClass import *
from TextFileSaving import *

#Read projections or geometry
filesdir_ref = "/home/mmouchet/Documents/SIEMENSDATA/GO.SIM/3D_patients/CF82/donneesBrutes/inspi_bloquee/"
# reading projections
proj = itk.imread(filesdir_ref+"sub_corrected_proj.mha")
# Reading the geometry of the scanner
xmlreader = rtk.ThreeDCircularProjectionGeometryXMLFileReader.New()
xmlreader.SetFilename(filesdir_ref+'sub_geometry.xml')
xmlreader.GenerateOutputInformation()
geometry = xmlreader.GetOutputObject()
print('nproj = %d'%(len(geometry.GetGantryAngles())))
print(proj.GetLargestPossibleRegion().GetSize())

# #Convert RTK files into arrays
# geometry_array = RTKtoNP(geometry)
# proj_array = itk.GetArrayFromImage(proj)
# proj_infos = GetProjectionInformations(proj)
# source_pos_array = GetSourcePositions(geometry)
# rotation_matrices_array = GetRotationMatrices(geometry)
# fixed_matrices_array = GetFixedSystemMatrices(geometry)
# print(proj_infos)

# #Compute all possible pairs
# AcquiDCC = DCCWithBPinAnAcquisition(geometry_array, source_pos_array, rotation_matrices_array, fixed_matrices_array, proj_array, proj_infos)
# AcquiDCC.ComputeAllPossiblePairs()
# print(AcquiDCC.d, AcquiDCC.P)
# AcquiDCC.ComputeDCCForAllPairs()

# graph_file = 
# WritePairsFile(graph_file , AcquiDCC.pairs)

