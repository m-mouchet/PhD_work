from RTKToArrayConversion import *
from ExtendedConeBeamDCC import *
from TextFileSaving import *
from ConeBeamDCCWithBackprojectionPlane import *
from AllAcquisitionCDClass import *

import sys

if len(sys.argv) < 6:
    print("Usage: OneRotationDCC <proj_file> <geometry_file> <current_file> <ref> <FOV_condition> <res_file>")
    sys.exit(1)


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
AcquiDCC = DCCOnCDinAnAcquisition(geometry_array, source_pos_array, rotation_matrices_array, fixed_matrices_array, proj_array, proj_infos)
print(proj_infos)
print(AcquiDCC.n_proj_per_rotation)
print(AcquiDCC.axial_limit)

#variance utility
bt_120 = np.genfromtxt("bow_tie_file", skip_header=1, unpack=True).T
supp_current = np.genfromtxt(sys.argv[3], skip_header=0, unpack=True, delimiter= ',').T
AcquiDCC.ComputeInitialNumberOfPhotons(bt_120, supp_current)

#Compute dcc from ref
ref = int(sys.argv[4])
print(ref, geometry_array[8, 0], geometry_array[8, ref])
res = []
idx = []
for i in range(-AcquiDCC.axial_limit, AcquiDCC.axial_limit):
    if sys.argv[5] == 'True':
        if ref == i+ref or i+ref < 0 or i+ref >= proj_infos[2][2] or np.abs(geometry_array[2, ref]-geometry_array[2, ref+i]) < 10**(-12):
            pass
        elif np.cos(AcquiDCC.geometry[2, ref]-AcquiDCC.geometry[2, i+ref]) > 2*((AcquiDCC.R_fov)/AcquiDCC.geometry[0, 0])**2-1:
            pair = ProjectionsPair(ref, i+ref, AcquiDCC.geometry, AcquiDCC.source_pos, AcquiDCC.mrot, AcquiDCC.fsm, AcquiDCC.projections, AcquiDCC.proj_infos, AcquiDCC.Ni)
            pair.ComputeMPoints()
            pair.ComputeEpipolarPlanes()
            if len(np.where(pair.final_cond)[0]) >= 1:
                pair.ComputePairMoments()
                idx.append(i+ref)
                if type(pair.m0) == np.float64:
                    res.append([np.array([pair.m0]), np.array([pair.m1]), pair.tot_var0, pair.tot_var1])
                else:
                    res.append([pair.m0, pair.m1, pair.var0, pair.var1])
    elif sys.argv[5] == 'False':
        if ref == i+ref or i+ref < 0 or i+ref >= proj_infos[2][2] or np.abs(geometry_array[2, ref]-geometry_array[2, ref+i]) < 10**(-12):
            pass
        else:
            pair = ProjectionsPairMpoints(ref, i+ref, AcquiDCC.geometry, AcquiDCC.source_pos, AcquiDCC.mrot, AcquiDCC.fsm, AcquiDCC.projections, AcquiDCC.proj_infos, AcquiDCC.Ni)
            pair.ComputeMPoints()
            pair.ComputeEpipolarPlanes()
            if len(np.where(pair.final_cond)[0]) >= 1:
                pair.ComputePairMoments()
                idx.append(i+ref)
                if type(pair.m0) == np.float64:
                    res.append([np.array([pair.m0]), np.array([pair.m1]), pair.tot_var0, pair.tot_var1])
                else:
                    res.append([pair.m0, pair.m1, pair.var0, pair.var1])

WriteMomentsFile(sys.argv[6], idx, res)
print("yip yip")
