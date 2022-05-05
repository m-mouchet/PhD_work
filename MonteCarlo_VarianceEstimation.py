from RTKToArrayConversion import *
from ExtendedConeBeamDCC import *
from AllAcquisitionCDClass import *
import sys
import numpy as np

if len(sys.argv) < 4:
    print("Usage: MC_VarEst <proj_file> <geometry_file> <number_of_realization> <output_file_names>")
    sys.exit(1)

proj = itk.imread(sys.argv[1])
xmlreader = rtk.ThreeDCircularProjectionGeometryXMLFileReader.New()
xmlreader.SetFilename(sys.argv[2])
xmlreader.GenerateOutputInformation()
geometry = xmlreader.GetOutputObject()

restot0 = []
restot1 = []
rescenter0 = []
rescenter1 = []
resmean = []
for it in range(int(sys.argv[3])):
    I0=10**5
    dH2O=0.01879 #mm^-1 at 75 keV
    if I0!=0:
        new_sino = I0*np.exp(-1.*dH2O*itk.GetArrayFromImage(proj))
        new_sino = np.maximum(np.random.poisson(new_sino), 1)
        new_sino = np.log(I0/new_sino)/dH2O
        proj_rebin_n = itk.GetImageFromArray(new_sino.astype(np.float32)) 
        proj_rebin_n.CopyInformation(proj)
        proj_rebin_n.Update()
    geometry_array = RTKtoNP(geometry)
    proj_array = itk.GetArrayFromImage(proj_rebin_n)
    proj_infos = GetProjectionInformations(proj_rebin_n)
    source_pos_array = GetSourcePositions(geometry)
    rotation_matrices_array = GetRotationMatrices(geometry)
    fixed_matrices_array = GetFixedSystemMatrices(geometry)
    AcquiDCC = DCCOnCDinAnAcquisition(geometry_array, source_pos_array, rotation_matrices_array, fixed_matrices_array, proj_array, proj_infos)
    ref = 720
    res0 = []
    res1 = []
    resc0 = []
    resc1 = []
    resm = []
    idx = []
    for i in range(-2*AcquiDCC.axial_limit,2*AcquiDCC.axial_limit):
        if ref == i+ref or i+ref <0 or i+ref >= proj_infos[2][2] or np.abs(geometry_array[2, ref]-geometry_array[2, ref+i]) <10**(-12):
            pass
        else:
            pair = ProjectionsPairMpoints(ref, i+ref , AcquiDCC.geometry, AcquiDCC.source_pos, AcquiDCC.mrot, AcquiDCC.fsm, AcquiDCC.projections, AcquiDCC.proj_infos, AcquiDCC.Ni)
            pair.ComputeMPoints()
            pair.ComputeEpipolarPlanes()
            if len(np.where(pair.final_cond)[0]) >= 1:
                pair.ComputePairMoments()
                idx.append(i+ref)
                if type(pair.m0) == np.float64:
                    res0.append()
                    res1.append(np.mean(np.array([pair.m1])))
                    resc0.append(pair.m0)
                    resc1.append(pair.m1)
                    resm.append(np.mean(np.array([pair.m0]))-np.mean(np.array([pair.m1])))
                    
                else:
                    res0.append(np.mean(pair.m0))
                    res1.append(np.mean(pair.m1))
                    resc0.append(pair.m0[len(pair.m0)//2])
                    resc1.append(pair.m1[len(pair.m1)//2])
                    resm.append(np.mean(pair.m0)-np.mean(pair.m1))
    restot0.append(res0)
    restot1.append(res1)
    rescenter0.append(resc0)
    rescenter1.append(resc1)
    resmean.append(resm)

restot0ar = np.array(restot0)
restot1ar = np.array(restot1)
rescenter0ar = np.array(rescenter0)
rescenter1ar = np.array(rescenter1)
resmeanar = np.array(resmean)
np.savetxt(sys.argv[4]+"0.txt", np.var(restot0ar, axis=0))
np.savetxt(sys.argv[4]+"1.txt", np.var(restot1ar, axis=0))
np.savetxt(sys.argv[4]+"central_0.txt", np.var(rescenter0ar, axis=0))
np.savetxt(sys.argv[4]+"central_1.txt", np.var(rescenter1ar, axis=0))
np.savetxt(sys.argv[4]+"m0_m1.txt", np.var(resmeanar, axis=0))

print("This program ended successfully")
