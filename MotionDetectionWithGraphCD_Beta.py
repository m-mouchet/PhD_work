from RTKToArrayConversion import *
from ExtendedConeBeamDCC import *
from AllAcquisitionCD_Beta_Class import *
from TextFileSaving import WriteEdgesGraph
import time
import sys


if len(sys.argv) < 4:
    print("Usage: OneRotationDCC <dir> <proj_file> <geometry_file> <res_file> <variance> <kernel> <bandwidth> <current_file> <idx_min>")
    sys.exit(1)


filesdir_ref = sys.argv[1]
# reading projections
proj = itk.imread(filesdir_ref+sys.argv[2])
# Reading the geometry of the scanner
xmlreader = rtk.ThreeDCircularProjectionGeometryXMLFileReader.New()
xmlreader.SetFilename(filesdir_ref+sys.argv[3])
# xmlreader.SetFilename(filesdir_ref+"geometry.xml")
xmlreader.GenerateOutputInformation()
geometry = xmlreader.GetOutputObject()
print('nproj = %d'%(len(geometry.GetGantryAngles())))
print(proj.GetLargestPossibleRegion().GetSize())

geometry_array = RTKtoNP(geometry)
proj_array = itk.GetArrayFromImage(proj)
proj_infos = GetProjectionInformations(proj)
source_pos_array = GetSourcePositions(geometry)
rotation_matrices_array = GetRotationMatrices(geometry)
fixed_matrices_array = GetFixedSystemMatrices(geometry)
print(proj_infos)

AcquiDCC = DCCOnCDinAnAcquisition(geometry_array, source_pos_array, rotation_matrices_array, fixed_matrices_array, proj_array, proj_infos)

##variance utility
#bt_120 = np.genfromtxt("/pbs/home/m/mmouchet/sps/GraphDCC/raw_data/Wedge_120_w1.txt", skip_header=1, unpack=True).T
#supp_current = np.genfromtxt(filesdir_ref+sys.argv[8], skip_header=0, unpack=True, delimiter= ',').T
#AcquiDCC.ComputeInitialNumberOfPhotons(bt_120, supp_current)

idx_min = int(sys.argv[9])
idx_max = AcquiDCC.n_proj
print(idx_min,idx_max, geometry_array[8, idx_min])
ref_list = np.arange(idx_min,idx_max)

start_pairs = time.time()
AcquiDCC.ComputeAllPossiblePairsIdx(ref_list, 'True')
print("Pairs computation took %d s" %(time.time()-start_pairs))

start_dcc = time.time()
AcquiDCC.ComputeDCCForAllPairsIdx(sys.argv[5], sys.argv[6], int(sys.argv[7]))
print("DCC computation took %d s" %(time.time()-start_dcc))

WriteEdgesGraph( filesdir_ref + sys.argv[4], AcquiDCC.tot_pairs_moments)
