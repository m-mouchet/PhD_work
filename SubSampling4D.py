import itk
from itk import RTK as rtk
import numpy as np
import sys
from tqdm import tqdm


def RecupParam(geo, idx):
    # Function that extract the geometric parameters of the projection index idx with geometry geo
    sid = geo.GetSourceToIsocenterDistances()[idx]
    sdd = geo.GetSourceToDetectorDistances()[idx]
    ga = geo.GetGantryAngles()[idx]
    dx = geo.GetProjectionOffsetsX()[idx]
    dy = geo.GetProjectionOffsetsY()[idx]
    oa = geo.GetOutOfPlaneAngles()[idx]
    ia = geo.GetInPlaneAngles()[idx]
    sx = geo.GetSourceOffsetsX()[idx]
    sy = geo.GetSourceOffsetsY()[idx]
    R = geo.GetRadiusCylindricalDetector()
    return sid, sdd, ga, dx, dy, oa, ia, sx, sy, R


def WriteGeometry(geometry, name):
    xmlwriter = rtk.ThreeDCircularProjectionGeometryXMLFileWriter.New()
    xmlwriter.SetObject(geometry)
    xmlwriter.SetFilename(name)
    xmlwriter.WriteFile()


def ReadGeometry(name):
    xmlReader = rtk.ThreeDCircularProjectionGeometryXMLFileReader.New()
    xmlReader.SetFilename(name)
    xmlReader.GenerateOutputInformation()
    return xmlReader.GetOutputObject()


def ExtractSlice(stack, num):
    # Function that extract the projection num in the projections stack stack
    ar = itk.GetArrayFromImage(stack)
    projslicea = ar[num:num+1, :, :]
    projslice = itk.GetImageFromArray(projslicea)
    projslice.CopyInformation(stack)
    projslice.Update()
    return projslice


workdir = "/home/mmouchet/Documents/SIEMENSDATA/GO.SIM/4D_patients/OB2/donneesBrutes/sans_comp_bis/"

proj = itk.imread(workdir+"corrected_proj_1.mha")
proj_array = itk.GetArrayFromImage(proj)
geometry_tot = ReadGeometry(workdir+"geometry.xml")
out_proj_file = workdir+"sub_corrected_proj.mha"
out_geometry_file = workdir+"sub_geometry.xml"
input_current = np.genfromtxt(workdir+"tube_current_sans_comp_bis.txt", skip_header=0, unpack=True, delimiter= ',').T
output_current_file = workdir+"sub_tube_current.txt"

n_rotation = len(np.where(np.abs(np.array(geometry_tot.GetGantryAngles())-geometry_tot.GetGantryAngles()[0]) <= 10**(-10))[0])
n_proj = len(geometry_tot.GetGantryAngles())
if n_rotation == 1:
    n_proj_per_rotation = n_proj
else:
    n_proj_per_rotation = np.where(np.array(geometry_tot.GetGantryAngles()) == geometry_tot.GetGantryAngles()[0])[0][1] - np.where(np.array(geometry_tot.GetGantryAngles()) == geometry_tot.GetGantryAngles()[0])[0][0]
idx_selection_one_rotation = np.arange(0, n_proj_per_rotation, int(0.002*n_proj_per_rotation//0.5))
print(len(idx_selection_one_rotation))
print(n_proj_per_rotation)
idx_selection_acquisition = []
for i in range(n_proj):
    if i % n_proj_per_rotation in idx_selection_one_rotation:
        idx_selection_acquisition.append(int(i))

np.savetxt(output_current_file, input_current[idx_selection_acquisition], delimiter=',')

# idx_selection_acquisition = np.where(np.array(geometry.GetGantryAngles()) < np.abs(geometry.GetGantryAngles()[0]-geometry.GetGantryAngles()[1]))[0]
# print(len(idx_selection_acquisition))

sub_proj_array = np.zeros((len(idx_selection_acquisition), proj_array.shape[1], proj_array.shape[2]))
sub_geometry = rtk.ThreeDCircularProjectionGeometry.New()
sub_geometry.SetRadiusCylindricalDetector(geometry_tot.GetRadiusCylindricalDetector())
for i in tqdm(range(sub_proj_array.shape[0])):
    sid, sdd, ga, dx, dy, oa, ia, sx, sy, R = RecupParam(geometry_tot, idx_selection_acquisition[i])
    # print(sid, sdd, ga, dx, dy, oa, ia, sx, sy, R)
    sub_geometry.AddProjectionInRadians(sid, sdd, ga, dx, dy, oa, ia, sx, sy)
    #find correct stack from all the 20 stack
    stack_id = idx_selection_acquisition[i]//proj_array.shape[0] + 1
    proj_id = idx_selection_acquisition[i]-proj_array.shape[0]*(stack_id-1)
    # test_geo = ReadGeometry(workdir+"geometry_%i.xml" % (stack_id))
    # print(RecupParam(test_geo, proj_id))
    #extract slice
    stack = itk.imread(workdir+"corrected_proj_%i.mha" % (stack_id))
    stack_array = itk.GetArrayFromImage(stack)
    sub_proj_array[i:i+1, :, :] += stack_array[proj_id:proj_id+1, :, :]
sub_proj = itk.GetImageFromArray(np.float32(sub_proj_array))
sub_proj.CopyInformation(proj)
sub_proj.Update()
# print(n_rotation, n_proj)
itk.imwrite(sub_proj, out_proj_file)
WriteGeometry(sub_geometry, out_geometry_file)
