import itk
from itk import RTK as rtk
import numpy as np
from ConeBeamDCCWithBackprojectionPlane import *
from RTKToArrayConversion import *
import itertools
from tqdm import tqdm_notebook
#from joblib import Parallel, delayed


def Error(m0, m1):
    return np.sqrt(np.sum((m0-m1)**2)/len(m0))


def CheckOverlapCondition(idx0, idx1, R_det, source_pos, fsm, mrot, proj_spacing, proj_size, proj_origin, proj_direction, dx):
    rp0, rp1, volDirection, volSpacing = ComputeRetroProj(idx0, idx1, R_det, source_pos, fsm, mrot, proj_spacing, proj_size, proj_origin, proj_direction, dx)
    sourceDir1 = volDirection[1, :]
    if np.dot(np.array([0, 1, 0]), sourceDir1) < 0:
        for i in range(4):
            rp0[i][1] *= -1
            rp1[i][1] *= -1
    if proj_direction[1][1] < 0:
        if max(rp0[2][1], rp0[3][1], rp1[2][1], rp1[3][1]) + 3*volSpacing[1] < min(rp0[0][1], rp0[1][1], rp1[0][1], rp1[1][1]):
            return True
        else:
            return False
    elif proj_direction[1][1] > 0:
        if max(rp0[0][1], rp0[1][1], rp1[0][1], rp1[1][1]) + 3*volSpacing[1] < min(rp0[2][1], rp0[3][1], rp1[2][1], rp1[3][1]):
            return True
        else:
            return False
    else:
        return False


def ComputeRetroProj(idx0, idx1, R_det, source_pos, fsm, mrot, proj_spacing, proj_size, proj_origin, proj_direction, dx):
    sourceDir0 = source_pos[idx0]-source_pos[idx1]
    sourceDir0 /= np.linalg.norm(sourceDir0)
    if (np.dot(sourceDir0, np.array([1., 0., 0.])) < 0):
        sourceDir0 *= -1
    n0 = mrot[idx0][2, 0:3]
    n1 = mrot[idx1][2, 0:3]
    sourceDir2 = 0.5*(n0 + n1)
    sourceDir2 /= np.linalg.norm(sourceDir2)
    #y'direction
    sourceDir1 = np.cross(sourceDir2, sourceDir0)
    sourceDir1 /= np.linalg.norm(sourceDir1)
    #backprojection direction matrix
    volDirection = np.vstack((sourceDir0, sourceDir1, sourceDir2))
    # check for non negative spacing
    matId = np.identity(3)
    matProd = proj_direction * matId != proj_direction
    if (np.sum(matProd) != 0):
        print("la matrice a %f element(s) non diagonal(aux)" %(np.sum(matProd)))
    else:
        size = []
        for i in range(len(proj_origin)):
            size.append(proj_spacing[i]*(proj_size[i]-1)*proj_direction[i, i])
    # Retroprojection des coin sur le plan de retroprojection
    rp0 = []
    rp1 = []
    invMag_List = []
    for j in proj_origin[1], proj_origin[1]+size[1]:
        for i in proj_origin[0], proj_origin[0]+size[0]:
            if R_det == 0:  # flat detector
                u = i
                v = j
                w = 0
            else:  # cylindrical detector
                theta = (i+dx)/R_det
                u = R_det*np.sin(theta)-dx
                v = j
                w = R_det*(1-np.cos(theta))
            idx = np.array((u, v, w, 1))
            coord0 = fsm[idx0].dot(idx)
            coord1 = fsm[idx1].dot(idx)
            # Project on the plane direction, compute inverse mag to go to isocenter and compute the source to pixel / plane intersection
            coord0Source = source_pos[idx0]-coord0[0:3]
            invMag = np.dot(source_pos[idx0], sourceDir2)/np.dot(coord0Source, sourceDir2)
            invMag_List.append(invMag)
            rp0.append(np.dot(volDirection, source_pos[idx0]-invMag*coord0Source))
            coord1Source = source_pos[idx1]-coord1[0:3]
            invMag = np.dot(source_pos[idx1], sourceDir2)/np.dot(coord1Source, sourceDir2)
            invMag_List.append(invMag)
            rp1.append(np.dot(volDirection, source_pos[idx1]-invMag*coord1Source))

    invMagSpacing = np.mean(invMag_List)
    volSpacing = np.array([proj_spacing[0]*invMagSpacing, proj_spacing[1]*invMagSpacing, 1])
    return rp0, rp1, volDirection, volSpacing


class DCCWithBPinAnAcquisition():
    def __init__(self, geometry_array, source_pos_array,
                 rotation_matrices_array, fixed_matrices_array,
                 proj_array, proj_infos):
        self.geometry = geometry_array
        self.projections = proj_array
        self.proj_spacing = proj_infos[0]
        self.proj_origin = proj_infos[1]
        self.proj_size = proj_infos[2]
        self.proj_direction = proj_infos[3]
        self.source_pos = source_pos_array
        self.fsm = fixed_matrices_array
        self.mrot = rotation_matrices_array
        self.n_rotation = len(np.where(np.abs(self.geometry[2, :] - self.geometry[2, 0]) <= 10**(-10))[0])
        self.n_proj = self.geometry.shape[1]
        if self.n_rotation == 1:
            self.n_proj_per_rotation = self.n_proj
            self.d = np.abs(self.geometry[8, self.n_proj-1] - self.geometry[8, 0])
        else:
            self.n_proj_per_rotation = np.where(self.geometry[2, :] == self.geometry[2, 0])[0][1] - np.where(self.geometry[2, :] == self.geometry[2, 0])[0][0]
            self.d = np.abs(self.geometry[8, self.n_proj_per_rotation-1] - self.geometry[8, 0])
        self.R_fov = self.geometry[0, 0]*np.sin(self.proj_size[0]*self.proj_spacing[0]/(2*self.geometry[1, 0]))  # ANGLE EN RADIAN
        self.P = np.abs(self.d*self.geometry[1, 0]/(self.geometry[0, 0]*self.proj_size[1]))
        self.axial_limit = int(self.n_proj_per_rotation*np.round((1-1/self.proj_size[1])/self.P))

    def CheckPairGeometry(self, idx0, idx1):
        # check fov
        if np.cos(self.geometry[2, idx1]-self.geometry[2, idx0]) > 2*((self.R_fov)/self.geometry[0, 0])**2-1:
            if CheckOverlapCondition(idx0, idx1, self.geometry[9, 0], self.source_pos, self.fsm, self.mrot, self.proj_spacing, self.proj_size, self.proj_origin, self.proj_direction, self.geometry[3, idx0]):
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
            for j in range(-self.axial_limit, self.axial_limit):
                if (s0L[i] == s0L[i] + j) or (s0L[i] + j < 0) or (s0L[i] + j >= self.n_proj):
                    pass
                else:
                    results[i].append(self.CheckPairGeometry(int(s0L[i]), int(s0L[i]+j)))
        self.rangeList = []
        for i in range(len(results)):
            self.rangeList.append([k for k in results[i] if k != 0])
            self.rangeList[i].sort()
        pair_idx_temp = []
        for ip0 in range(self.n_proj):
            for ip1 in self.rangeList[ip0 % self.n_proj_per_rotation]:
                if (ip0 == ip0+ip1) or (ip0+ip1 < 0) or (ip0+ip1 >= self.n_proj):
                    pass
                else:
                    pair_idx_temp.append([min(ip0, ip0+ip1), max(ip0, ip0+ip1)])
        pair_idx_temp.sort()
        print(len(pair_idx_temp))
        self.pairs_idx = list(pair_idx_temp for pair_idx_temp,_ in itertools.groupby(pair_idx_temp))
        print(len(self.pairs_idx))
        pair_idx_temp.clear()
        return 0

    def ComputeDCCForAllPairs(self):
        self.pairs = []
        for idx in tqdm_notebook(range(len(self.pairs_idx))):
            g0 = NPtoRTK(self.geometry, self.pairs_idx[idx][0])
            g1 = NPtoRTK(self.geometry, self.pairs_idx[idx][1])
            p0 = ARRAYtoRTK(self.projections, self.proj_spacing, self.proj_origin, self.proj_size, self.proj_direction, self.pairs_idx[idx][0])
            p1 = ARRAYtoRTK(self.projections, self.proj_spacing, self.proj_origin, self.proj_size, self.proj_direction, self.pairs_idx[idx][1])
            pair_bp = ProjectionsPairBP(self.pairs_idx[idx][0], self.pairs_idx[idx][1], g0, g1, p0, p1)
            pair_bp.LinesMomentsCorners()
            self.pairs.append((pair_bp.idx0, pair_bp.idx1, Error(pair_bp.m0, pair_bp.m1)))
        return 0

    def ComputeDCCForAllPairsPara(self):
        indexes = np.arange(len(self.pairs_idx))
        self.pairs = Parallel(n_jobs=4, backend="threading")(delayed(unwrap_self_ComputeDCCForOnePair)(i) for i in tqdm_notebook(zip([self]*len(indexes), indexes), total=len(indexes)))
        #self.pairs = pool.map(unwrap_self_ComputeDCCForOnePair, zip([self]*len(self.pairs_idx), indexes))
        #self.pairs = []
        #jobs = [pool.apply_async(unwrap_self_ComputeDCCForOnePair, args=(argument,)) for argument in zip([self]*len(self.pairs_idx), indexes)]
        #pool.close()
        #self.pairs = []
        #for job in tqdm_notebook(jobs):
        #    self.pairs.append(job.get())

    def f(self, name):
        return (name, 0, 0)

    def ComputeDCCForOnePair(self, idx):
        g0 = NPtoRTK(self.geometry, self.pairs_idx[idx][0])
        g1 = NPtoRTK(self.geometry, self.pairs_idx[idx][1])
        p0 = ARRAYtoRTK(self.projections, self.proj_spacing, self.proj_origin, self.proj_size, self.proj_direction, self.pairs_idx[idx][0])
        p1 = ARRAYtoRTK(self.projections, self.proj_spacing, self.proj_origin, self.proj_size, self.proj_direction, self.pairs_idx[idx][1])
        pair_bp = ProjectionsPairBP(self.pairs_idx[idx][0], self.pairs_idx[idx][1], g0, g1, p0, p1)
        pair_bp.LinesMomentsCorners()
        return (pair_bp.idx0, pair_bp.idx1, Error(pair_bp.m0, pair_bp.m1))


def unwrap_self_f(arg, **kwarg):
    return DCCWithBPinAnAcquisition.f(*arg, **kwarg)


def unwrap_self_ComputeDCCForOnePair(arg, **kwarg):
    return DCCWithBPinAnAcquisition.ComputeDCCForOnePair(*arg, **kwarg)
