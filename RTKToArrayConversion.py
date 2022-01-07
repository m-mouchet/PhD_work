import numpy as np
import itk
from itk import RTK as rtk


def NPtoRTK(geometryArray, i):
    sid, sdd, ga, dx, dy, oa, ia, sx, sy = geometryArray[0, i], geometryArray[1, i], geometryArray[2, i], geometryArray[3, i], geometryArray[4, i], geometryArray[5, i], geometryArray[6, i], geometryArray[7, i], geometryArray[8, i]
    g = rtk.ThreeDCircularProjectionGeometry.New()
    g.SetRadiusCylindricalDetector(geometryArray[9, i])
    g.AddProjectionInRadians(sid, sdd, ga, dx, dy, oa, ia, sx, sy)
    return g


def RTKtoNP(geometryRTK):
    # Extrait les informations de rtk pour en faire des vecteurs
    nproj = len(geometryRTK.GetGantryAngles())
    R_det = geometryRTK.GetRadiusCylindricalDetector()
    geometryArray = np.zeros((10, nproj))
    geometryArray[0, :] = geometryRTK.GetSourceToIsocenterDistances()
    geometryArray[1, :] = geometryRTK.GetSourceToDetectorDistances()
    geometryArray[2, :] = geometryRTK.GetGantryAngles()
    geometryArray[3, :] = geometryRTK.GetProjectionOffsetsX()
    geometryArray[4, :] = geometryRTK.GetProjectionOffsetsY()
    geometryArray[5, :] = geometryRTK.GetOutOfPlaneAngles()
    geometryArray[6, :] = geometryRTK.GetInPlaneAngles()
    geometryArray[7, :] = geometryRTK.GetSourceOffsetsX()
    geometryArray[8, :] = geometryRTK.GetSourceOffsetsY()
    geometryArray[9, :] = np.ones(nproj)*R_det
    return geometryArray


def GetProjectionInformations(projection):
    # Extrait les information des projections pour les convertir en vecteurs
    projSpacing = np.copy(projection.GetSpacing())
    projOrigin = np.copy(projection.GetOrigin())
    projSize = np.copy(projection.GetLargestPossibleRegion().GetSize())
    projDirection = np.copy(itk.GetArrayFromMatrix(projection.GetDirection()))
    return [projSpacing, projOrigin, projSize, projDirection]

def GetSourcePositions(geometryRTK):
    # Extrait les positions des sources d'une geometrie RTK
    nproj = len(geometryRTK.GetGantryAngles())
    sourcePositions = np.zeros((nproj, 3))
    for i in range(nproj):
        pos = geometryRTK.GetSourcePosition(i)
        pos = itk.GetArrayFromVnlVector(pos.GetVnlVector())
        sourcePositions[i, :] = pos[0:3]
    return sourcePositions


def GetRotationMatrices(geometryRTK):
    # Extrait les matrices de rotation d'une geometie RTK
    matrices = []
    nproj = len(geometryRTK.GetGantryAngles())
    for i in range(nproj):
        MatRot = geometryRTK.GetRotationMatrix(i)
        MatRot = itk.GetArrayFromVnlMatrix(MatRot.GetVnlMatrix().as_matrix())
        matrices.append(MatRot)
    return matrices


def GetFixedSystemMatrices(geometryRTK):
    # Extrait les matrices de chaque positions de source d'une geometrie RTK
    matrices = []
    nproj = len(geometryRTK.GetGantryAngles())
    for i in range(nproj):
        fsMat = geometryRTK.GetProjectionCoordinatesToFixedSystemMatrix(i)
        fsMat = itk.GetArrayFromVnlMatrix(fsMat.GetVnlMatrix().as_matrix())
        matrices.append(fsMat)
    return matrices


def ARRAYtoRTK(projArray, projSpacing, projOrigin, projSize, projDirection, i):
    slicea = projArray[i:i+1, :, :]
    projSlice = itk.GetImageFromArray(np.float32(slicea))
    projSlice.SetOrigin(projOrigin)
    projSlice.SetSpacing(projSpacing)
    projDirection = itk.Matrix[itk.D, 3, 3](itk.GetVnlMatrixFromArray(projDirection))
    projSlice.SetDirection(projDirection)
    projSlice.Update()
    return projSlice
