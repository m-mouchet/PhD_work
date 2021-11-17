import itk
from itk import RTK as rtk
import numpy as np
import sys


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


#if len(sys.argv) < 4:
#    print("Usage: subsample <proj> <geometry> <step> <out_proj_name> <out_geometry_name>")
#    sys.exit(1)


#proj = itk.imread(sys.argv[0])
#proj_array = itk.GetArrayFromImage(proj)
#geometry = ReadGeometry(sys.argv[1])
#step = sys.argv[2]
#out_proj_file = sys.argv[3]
#out_geometry_file = sys.argv[4]

proj = itk.imread("")
proj_array = itk.GetArrayFromImage(proj)
geometry = ReadGeometry("")
step = 8
out_proj_file = ""
out_geometry_file = ""

n_rotation = len(np.where(np.abs(np.array(geometry.GetGantryAngles())-geometry.GetGantryAngles()[0]) <= 10**(-10))[0])
n_proj = len(geometry.GetGantryAngles())
if n_rotation == 1:
    n_proj_per_rotation = n_proj
else:
    n_proj_per_rotation = np.where(np.array(geometry.GetGantryAngles()) == geometry.GetGantryAngles()[0])[0][1] - np.where(np.array(geometry.GetGantryAngles()) == geometry.GetGantryAngles()[0])[0][0]
idx_selection_one_rotation = np.arange(0, n_proj_per_rotation, step)
idx_selection_acquisition = []
for i in range(n_proj):
    if i % n_proj_per_rotation in idx_selection_one_rotation:
        idx_selection_acquisition.append(i)
sub_proj_array = np.zeros((len(idx_selection_acquisition), proj_array.shape[1], proj_array.shape[2]))
sub_geometry = rtk.ThreeDCircularProjectionGeometry.New()
sub_geometry.SetRadiusCylindricalDetector(geometry.GetRadiusCylindricalDetector())
for i in range(sub_proj_array.shape[0]):
    sid, sdd, ga, dx, dy, oa, ia, sx, sy, R = RecupParam(geometry, idx_selection_acquisition[i])
    sub_geometry.AddProjectionInRadians(sid, sdd, ga, dx, dy, oa, ia, sx, sy)
    sub_proj_array[i:i+1, :, :] += proj_array[idx_selection_acquisition[i]:idx_selection_acquisition[i]+1, :, :]
sub_proj = itk.GetImageFromArray(np.float32(sub_proj_array))
sub_proj.CopyInformation(proj)
sub_proj.Update()
print(n_rotation, n_proj)
itk.imwrite(sub_proj, out_proj_file)
WriteGeometry(sub_geometry, out_geometry_file)
