import itk
from itk import RTK as rtk
import numpy as np
from ConeBeamDCCWithBackprojectionPlane import *
from tqdm import tqdm_notebook
import itertools


def Error(m0, m1):
    return np.sum(np.abs(m0-m1)/np.mean((m0, m1)))/len(m0)


def ComputeRetroProj(geometry, projections, idx0, idx1):
    projIdxToCoord0 = geometry.GetProjectionCoordinatesToFixedSystemMatrix(idx0)
    projIdxToCoord0 = itk.GetArrayFromVnlMatrix(projIdxToCoord0.GetVnlMatrix().as_matrix())
    projIdxToCoord1 = geometry.GetProjectionCoordinatesToFixedSystemMatrix(idx1)
    projIdxToCoord1 = itk.GetArrayFromVnlMatrix(projIdxToCoord1.GetVnlMatrix().as_matrix())
    s0 = geometry.GetSourcePosition(idx0)
    s0 = itk.GetArrayFromVnlVector(s0.GetVnlVector())[0:3]
    s1 = geometry.GetSourcePosition(idx1)
    s1 = itk.GetArrayFromVnlVector(s1.GetVnlVector())[0:3]
    sourceDir0 = s0-s1
    sourceDir0 /= np.linalg.norm(sourceDir0)
    if (np.dot(sourceDir0, np.array([1., 0., 0.])) < 0):
        sourceDir0 *= -1
    matRot0 = geometry.GetRotationMatrix(idx0)
    matRot0 = itk.GetArrayFromVnlMatrix(matRot0.GetVnlMatrix().as_matrix())
    matRot1 = geometry.GetRotationMatrix(idx1)
    matRot1 = itk.GetArrayFromVnlMatrix(matRot1.GetVnlMatrix().as_matrix())
    n0 = matRot0[2, 0:3]
    n1 = matRot1[2, 0:3]
    sourceDir2 = 0.5*(n0 + n1)
    sourceDir2 /= np.linalg.norm(sourceDir2)
    #y'direction
    sourceDir1 = np.cross(sourceDir2, sourceDir0)
    sourceDir1 /= np.linalg.norm(sourceDir1)
    #backprojection direction matrix
    volDirection = np.vstack((sourceDir0, sourceDir1, sourceDir2))
    # check for non negative spacing
    directionProj = itk.GetArrayFromMatrix(projections.GetDirection())
    matId = np.identity(3)
    matProd = directionProj * matId != directionProj
    if (np.sum(matProd) != 0):
        print("la matrice a %f element(s) non diagonal(aux)" %(np.sum(matProd)))
    else: 
        size = []
        for i in range(len(projections.GetOrigin())):
            size.append(projections.GetSpacing()[i]*(projections.GetLargestPossibleRegion().GetSize()[i]-1)*directionProj[i, i])
    # Retroprojection des coin sur le plan de retroprojection
    rp0 = []
    rp1 = []
    invMag_List = []
    for j in projections.GetOrigin()[1], projections.GetOrigin()[1]+size[1]:
        for i in projections.GetOrigin()[0], projections.GetOrigin()[0]+size[0]:
            if geometry.GetRadiusCylindricalDetector() == 0:  # flat detector
                u = i
                v = j
                w = 0
            else:  # cylindrical detector
                theta = i/geometry.GetRadiusCylindricalDetector()
                u = geometry.GetRadiusCylindricalDetector()*np.sin(theta)
                v = j
                w = geometry.GetRadiusCylindricalDetector()*(1-np.cos(theta))
            idx = np.array((u, v, w, 1))
            coord0 = projIdxToCoord0.dot(idx)
            coord1 = projIdxToCoord1.dot(idx)
            # Project on the plane direction, compute inverse mag to go to isocenter and compute the source to pixel / plane intersection
            coord0Source = s0-coord0[0:3]
            invMag = np.dot(s0, sourceDir2)/np.dot(coord0Source, sourceDir2)
            invMag_List.append(invMag)
            rp0.append(np.dot(volDirection, s0-invMag*coord0Source))
            coord1Source = s1-coord1[0:3]
            invMag = np.dot(s1, sourceDir2)/np.dot(coord1Source, sourceDir2)
            invMag_List.append(invMag)
            rp1.append(np.dot(volDirection, s1-invMag*coord1Source))

    invMagSpacing = np.mean(invMag_List)
    volSpacing = np.array([projections.GetSpacing()[0]*invMagSpacing, projections.GetSpacing()[1]*invMagSpacing, 1])
    return rp0, rp1, volDirection, volSpacing


def CheckOverlapCondition(geometry, projections, idx0, idx1):
    rp0, rp1, volDirection, volSpacing = ComputeRetroProj(geometry, projections, idx0, idx1)
    sourceDir1 = volDirection[1, :]
    if np.dot(np.array([0, 1, 0]), sourceDir1) < 0:
        for i in range(4):
            rp0[i][1] *= -1
            rp1[i][1] *= -1
    if itk.GetArrayFromMatrix(projections.GetDirection())[1][1] < 0:
        if max(rp0[2][1], rp0[3][1], rp1[2][1], rp1[3][1]) + volSpacing[1] < min(rp0[0][1], rp0[1][1], rp1[0][1], rp1[1][1]):
            return True
        else:
            return False
    elif itk.GetArrayFromMatrix(projections.GetDirection())[1][1] > 0:
        if max(rp0[0][1], rp0[1][1], rp1[0][1], rp1[1][1]) + volSpacing[1] < min(rp0[2][1], rp0[3][1], rp1[2][1], rp1[3][1]):
            return True
        else:
            return False
    else:
        return False


class DCCWithBPinAnAcquisition():
    def __init__(self, geometry, projections):
        self.geometry = geometry
        self.projections = projections
        self.n_rotation = len(np.where(np.array(self.geometry.GetGantryAngles()) == geometry.GetGantryAngles()[0]))
        self.n_proj = len(geometry.GetGantryAngles())
        if self.n_rotation == 1:
            self.n_proj_per_rotation = self.n_proj
            self.d = self.geometry.GetSourceOffsetsY()[self.n_proj-1] - self.geometry.GetSourceOffsetsY()[0]
        else:
            self.n_proj_per_rotation = np.where(np.array(self.geometry.GetGantryAngles()) == geometry.GetGantryAngles()[0])[0][1] - np.where(np.array(self.geometry.GetGantryAngles()) == geometry.GetGantryAngles()[0])[0][0]
            self.d = self.geometry.GetSourceOffsetsY()[self.n_proj_per_rotation-1] - self.geometry.GetSourceOffsetsY()[0]
        self.R_fov = self.geometry.GetSourceToIsocenterDistances()[0]*np.sin(self.projections.GetLargestPossibleRegion().GetSize()[0]*self.projections.GetSpacing()[0]/(2*self.geometry.GetSourceToDetectorDistances()[0]))  # ANGLE EN RADIAN
        self.P = self.d*self.geometry.GetSourceToDetectorDistances()[0]/(self.geometry.GetSourceToIsocenterDistances()[0]*self.projections.GetLargestPossibleRegion().GetSize()[1]*self.projections.GetSpacing()[1])
        self.axial_limit = int(self.n_proj_per_rotation*np.round((1-1/self.projections.GetLargestPossibleRegion().GetSize()[1])/self.P))

    def CheckPairGeometry(self, idx0, idx1):
        # check fov
        if np.cos(self.geometry.GetGantryAngles()[idx1]-self.geometry.GetGantryAngles()[idx0]) > 2*((self.R_fov)/self.geometry.GetSourceToIsocenterDistances()[0])**2-1:
            if CheckOverlapCondition(self.geometry, self.projections, idx0, idx1):
                return idx1-idx0
            else:
                return 0
        else:
            return 0

    def ComputeAllPossiblePairs(self): #retourne les indices de toutes les pairs possible dans une acquisition
        # on fait pour toutes les projections de la rotation centrale
        # puis ensuite on généralise
        # il faut verifier le fov et l'overlapp
        s0L = np.arange(self.n_proj_per_rotation*(self.n_rotation//2), self.n_proj_per_rotation*(self.n_rotation//2+1))
        results = []
        for i in tqdm_notebook(range(len(s0L))):
            results.append([])
            for j in range(s0L[i]-self.axial_limit, s0L[i]+self.axial_limit):
                if s0L[i] == j or j < 0 or j >= self.n_proj:
                    pass
                else:
                    results[i].append(self.CheckPairGeometry(int(s0L[i]), int(j)))
        self.rangeList = []
        for i in range(len(results)):
            self.rangeList.append([k for k in results[i] if k != 0])
            self.rangeList[i].sort()
        pair_idx_temp = []
        for ip0 in range(self.n_proj):
            for ip1 in self.rangeList[ip0 % self.n_proj_per_rotation]:
                pair_idx_temp.append([min(ip0, ip0+ip1), max(ip0, ip0+ip1)])
        pair_idx_temp.sort()
        print(len(pair_idx_temp))
        self.pairs_idx = list(pair_idx_temp for pair_idx_temp,_ in itertools.groupby(pair_idx_temp))
        print(len(self.pairs_idx))
        pair_idx_temp.clear()
        return 0

    def ComputeDCCForAllPairs(self):
        self.pairs = []
        for i in tqdm_notebook(range(len(self.pairs_idx))):
            pair_bp = ProjectionsPairBP(self.pairs_idx[i][0], self.pairs_idx[i][1], self.geometry, self.projections)
            pair_bp.LinesMomentsCorners()
            self.pairs.append((pair_bp.idx0, pair_bp.idx1, Error(pair_bp.m0, pair_bp.m1)))
        return 0
