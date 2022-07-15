import itk
from itk import RTK as rtk
import numpy as np
from ExtendedConeBeamDCC import *
from RTKToArrayConversion import *
from tqdm import tqdm_notebook
import itertools
# from joblib import Parallel, delayed


# def ComputeDCCForOnePos(ref_list_idx0, pair_idx0, geometry, source_pos, mrot, fsm, projections, proj_infos):
#         res_temp = []
#         for idx1 in range(len(pair_idx0)):
#             epair = ProjectionsPair(ref_list_idx0, pair_idx0[idx1], geometry, source_pos, mrot, fsm, projections, proj_infos)
#             epair.ComputePairMoments()
#             res_temp.append(np.sum((epair.m0 - epair.m1))/len(epair.m0))
#         return ref_list_idx0, pair_idx0, res_temp


def ComputeDCCForOnePos(fov_condition, ref, axial_limit, geometry, source_pos, mrot, fsm, projections, proj_infos):
    idx_temp = []
    mom0 = []
    mom1 = []
    var0 = []
    var1 = []
    for i in range(-axial_limit, axial_limit):
        if fov_condition == 'True':
            if ref == i+ref or i+ref < 0 or i+ref >= proj_infos[2][2] or np.abs(geometry[2, ref]-geometry[2, ref+i]) <10**(-12) :
                pass
            elif np.cos(geometry[2, ref]-geometry[2, i+ref]) > 2*((AcquiDCC.R_fov)/AcquiDCC.geometry[0, 0])**2-1:
                pair = ProjectionsPairMpoints(ref, i+ref, geometry, source_pos, mrot, fsm, projections, proj_infos)
                pair.ComputeMPoints()
                pair.ComputeEpipolarPlanes()
                if len(np.where(pair.final_cond)[0]) > 1:
                    pair.ComputePairMoments()
                    idx_temp.append(i+ref)
                    mom0.append(pair.m0)
                    mom1.append(pair.m1)
                    var0.append(pair.tot_var0)
                    var1.append(pair.tot_var0)
        elif fov_condition == 'False':
            if ref == i+ref or i+ref < 0 or i+ref >= proj_infos[2][2] or np.abs(geometry[2, ref]-geometry[2, ref+i]) <10**(-12) :
                pass
            else:
                pair = ProjectionsPairMpoints(ref, i+ref, geometry, source_pos, mrot, fsm, projections, proj_infos)
                pair.ComputeMPoints()
                pair.ComputeEpipolarPlanes()
                if len(np.where(pair.final_cond)[0]) > 1:
                    pair.ComputePairMoments()
                    idx_temp.append(i+ref)
                    mom0.append(pair.m0)
                    mom1.append(pair.m1)
                    var0.append(pair.tot_var0)
                    var1.append(pair.tot_var0)
    return ref, idx_temp, mom0, mom1


class DCCOnCDinAnAcquisition():
    def __init__(self, geometry_array, source_pos_array,
                 rotation_matrices_array, fixed_matrices_array,
                 proj_array, proj_infos):
        self.proj_infos = proj_infos
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
            self.d = self.geometry[8, self.n_proj-1] - self.geometry[8, 0]
        else:
            self.n_proj_per_rotation = np.where(self.geometry[2, :] == self.geometry[2, 0])[0][1] - np.where(self.geometry[2, :] == self.geometry[2, 0])[0][0]
            self.d = np.abs(self.geometry[8, self.n_proj_per_rotation-1] - self.geometry[8, 0])
        self.spiral_dir = np.sign(self.geometry[8, -1]-self.geometry[8, 0])
        self.R_det = self.geometry[9, 0]
        self.sid = self.geometry[0, 0]
        self.sdd = self.geometry[1, 0]
        self.gamma = (self.proj_origin[0] + self.geometry[3, 0]-self.geometry[7, 0] + np.arange(self.proj_size[0])*self.proj_spacing[0]*self.proj_direction[0, 0])/self.R_det
        self.R_fov = np.max((np.abs(self.geometry[0, 0]*np.sin(self.gamma[0])), np.abs(self.geometry[0, 0]*np.sin(self.gamma[-1]))))  # ANGLE EN RADIAN
        # print(np.abs(self.geometry[0, 0]*np.sin(self.gamma[0])), np.abs(self.geometry[0, 0]*np.sin(self.gamma[-1])))
        self.P = np.abs(self.d*self.geometry[1, 0]/(self.geometry[0, 0]*self.proj_size[1]))
        self.axial_limit = 2*int(self.n_proj_per_rotation*np.round((1-1/self.proj_size[1])/self.P))
        self.Ni = 0 # Uncomment if simu

    def ComputeInitialNumberOfPhotons(self, bowtie, currents):
        bt_factor = np.interp(self.gamma, bowtie[:, 0]*np.pi/180, bowtie[:, 1])
        self.Ni = np.array([bt_factor]*len(currents)).T*(0.001*currents*120000**2)*0.0000329

    def CheckPairGeometry(self, fov_condition, idx0, idx1):
        if np.abs(self.geometry[2, idx0] - self.geometry[2, idx1]) <= 10**(-12):
                return 0
        else:
            eb = -np.sign(np.cos((self.geometry[2, idx1] - self.geometry[2, idx0])/2))*np.sign(self.geometry[2, idx1] - self.geometry[2, idx0])*(self.source_pos[idx1]-self.source_pos[idx0])/np.linalg.norm(self.source_pos[idx1]-self.source_pos[idx0])
            if np.abs(eb[1]/np.sqrt(eb[0]**2+eb[2]**2))>1:
                return 0
            else:
                alpha = np.arcsin(eb[1]/np.sqrt(eb[0]**2+eb[2]**2))
                v_max = (self.proj_size[1]-1)*self.proj_spacing[1]/2
                if fov_condition == 'False' and np.abs(np.tan(alpha)) < v_max/self.sdd:
                    epair = ProjectionsPairBeta(idx0, idx1, self.geometry, self.source_pos, self.mrot, self.fsm, self.projections, self.proj_infos, self.Ni)
                    epair.ComputeBetaRange()
                    if len(epair.beta) >= 1:
                        return epair
                    else:
                        return 0
                elif fov_condition == 'True' and np.abs(np.tan(alpha)) < v_max/self.sdd and np.cos(self.geometry[2, idx1]-self.geometry[2, idx0]) > 2*((self.R_fov)/self.geometry[0, 0])**2-1:
                    epair = ProjectionsPairBeta(idx0, idx1, self.geometry, self.source_pos, self.mrot, self.fsm, self.projections, self.proj_infos, self.Ni)
                    epair.ComputeBetaRange()
                    if len(epair.beta) >= 1:
                        return epair
                    else:
                        return 0
                else:
                    return 0

    def ComputeAllPossiblePairsForOnePos(self, ref, fov_condition, idx_min, idx_max):
        ref_pairs = []
        for i in range(-self.axial_limit, self.axial_limit):
            if (i == 0) or (i+ref < idx_min) or (i+ref >= idx_max):
                pass
            else:
                cpg = self.CheckPairGeometry(fov_condition, ref, i+ref)
                if cpg == 0:
                    pass
                else:
                    ref_pairs.append(cpg)
        return ref_pairs

    def ComputeAllPossiblePairs(self, ref_list, fov_condition):
        self.ref_list = ref_list
        self.tot_pairs = []
        for i in range(len(self.ref_list)):
            ref_pairs = self.ComputeAllPossiblePairsForOnePos(ref_list[i], fov_condition)
            if len(ref_pairs) >= 1:
                self.tot_pairs.append(ref_pairs)

    def ComputeDCCForOnePos(self, ref):
        for i in range(len(self.tot_pairs[ref])):
            self.tot_pairs[ref][i].ComputePairMoments()

    def ComputeDCCForAllPairs(self):
        for i in range(len(self.ref_list)):
            self.ComputeDCCForOnePos(i)

    def CheckPairGeometryIdx(self, fov_condition, idx0, idx1):
        if np.abs(self.geometry[2, idx0] - self.geometry[2, idx1]) <= 10**(-12):
                return 0
        else:
            eb = -np.sign(np.cos((self.geometry[2, idx1] - self.geometry[2, idx0])/2))*np.sign(self.geometry[2, idx1] - self.geometry[2, idx0])*(self.source_pos[idx1]-self.source_pos[idx0])/np.linalg.norm(self.source_pos[idx1]-self.source_pos[idx0])
            if np.abs(eb[1]/np.sqrt(eb[0]**2+eb[2]**2))>1:
                return 0
            else:
                alpha = np.arcsin(eb[1]/np.sqrt(eb[0]**2+eb[2]**2))
                v_max = (self.proj_size[1]-1)*self.proj_spacing[1]/2
                if fov_condition == 'False' and np.abs(np.tan(alpha)) < v_max/self.sdd:
                    epair = ProjectionsPairBeta(idx0, idx1, self.geometry, self.source_pos, self.mrot, self.fsm, self.projections, self.proj_infos, self.Ni, 0, 0, 0)
                    epair.ComputeBetaRange()
                    if len(epair.beta) >= 1:
                        return idx1 - idx0
                    else:
                        return 0
                elif fov_condition == 'True' and np.abs(np.tan(alpha)) < v_max/self.sdd and np.cos(self.geometry[2, idx1]-self.geometry[2, idx0]) > 2*((self.R_fov)/self.geometry[0, 0])**2-1:
                    epair = ProjectionsPairBeta(idx0, idx1, self.geometry, self.source_pos, self.mrot, self.fsm, self.projections, self.proj_infos, self.Ni, 0, 0, 0)
                    epair.ComputeBetaRange()
                    if len(epair.beta) >= 1:
                        return idx1 - idx0
                    else:
                        return 0
                else:
                    return 0

    def ComputePairsFromCentralRotation(self, fov_condition):
        s0L = np.arange(self.n_proj_per_rotation*(self.n_rotation//2), self.n_proj_per_rotation*(self.n_rotation//2+1))
        results = []
        for i in tqdm_notebook(range(len(s0L))):
        # for i in range(len(s0L)):
            results.append([])
            for j in range(-self.axial_limit, self.axial_limit):
                if (j == 0) or (s0L[i] + j < 0) or (s0L[i] + j >= self.n_proj):
                    pass
                else:
                    results[i].append(self.CheckPairGeometryIdx(fov_condition, int(s0L[i]), int(s0L[i]+j)))
        self.rangeList = []
        for i in range(len(results)):
            self.rangeList.append([k for k in results[i] if k != 0])
            self.rangeList[i].sort()

    def ComputeAllPossiblePairsIdx(self, ref_list, fov_condition, idx_min, idx_max):
        self.ComputePairsFromCentralRotation(fov_condition)
        self.ref_list = ref_list
        self.tot_pairs_temp = []
        # for i in tqdm_notebook(range(len(self.ref_list))):
        for i in range(len(self.ref_list)):
            for j in self.rangeList[self.ref_list[i] % self.n_proj_per_rotation]:
                if (j == 0) or (j+self.ref_list[i] < idx_min) or (self.ref_list[i]+j >= idx_max):
                    pass
                else:
                    res = [np.min((self.ref_list[i], self.ref_list[i] + j)), np.max((self.ref_list[i], self.ref_list[i] + j))]
                    self.tot_pairs_temp.append(res)
        print(len(self.tot_pairs_temp))
        self.tot_pairs_temp.sort()
        self.tot_pairs = list(tot_pairs_temp for tot_pairs_temp,_ in itertools.groupby(self.tot_pairs_temp))
        print(len(self.tot_pairs))
        self.tot_pairs_temp.clear()
        return 0

    def ComputeDCCForAllPairsIdx(self, variance, kernel, bandwidth):
        self.tot_pairs_moments = []
        for i in tqdm_notebook(range(len(self.tot_pairs))):
        # for i in range(len(self.tot_pairs)):
            epair = ProjectionsPairBeta(self.tot_pairs[i][0], self.tot_pairs[i][1], self.geometry, self.source_pos, self.mrot, self.fsm, self.projections, self.proj_infos, self.Ni, variance, kernel, bandwidth)
            epair.ComputeBetaRange()
            epair.ComputePairMoments()
            if variance == 'True':
                self.tot_pairs_moments.append((self.tot_pairs[i][0], self.tot_pairs[i][1], np.mean(np.abs(epair.m0-epair.m1))/np.sqrt(epair.tot_var0**2 + epair.tot_var1**2)))
            else:
                self.tot_pairs_moments.append((self.tot_pairs[i][0], self.tot_pairs[i][1], np.mean(np.abs((epair.m0-epair.m1)))))
                # self.tot_pairs_moments.append((self.tot_pairs[i][0], self.tot_pairs[i][1], np.mean(np.abs((epair.m0-epair.m1)))))

#     def ComputeAllPossiblePairsForEachPos(self, ref_list, fov_condition):
#         self.ComputePairsFromCentralRotation(fov_condition)
#         self.ref_list = ref_list
#         self.pair_idx = []
#         for ip0 in tqdm_notebook(range(len(self.ref_list))):
#             self.pair_idx.append([])
#             for ip1 in self.rangeList[self.ref_list[ip0] % self.n_proj_per_rotation]:
#                 if (self.ref_list[ip0] == self.ref_list[ip0]+ip1) or (self.ref_list[ip0]+ip1 < 0) or (self.ref_list[ip0]+ip1 >= self.n_proj):
#                     pass
#                 else:
#                     self.pair_idx[ip0].append(self.ref_list[ip0]+ip1)
#         #pair_idx_temp.sort()
#         #print(len(pair_idx_temp))
#         #self.pairs_idx = list(pair_idx_temp for pair_idx_temp,_ in itertools.groupby(pair_idx_temp))
#         #print(len(self.pairs_idx))
#         #pair_idx_temp.clear()
#         return 0

#     def ComputeDCCForEachPos(self):
#         self.res = []
#         for idx0 in tqdm_notebook(range(len(self.ref_list))):
#             res_temp = []
#             for idx1 in range(len(self.pair_idx[idx0])):
#                 epair = ProjectionsPairMpoints(self.ref_list[idx0], self.pair_idx[idx0][idx1], self.geometry, self.source_pos, self.mrot, self.fsm, self.projections, self.proj_infos, self.Ni)
#                 epair.ComputePairMoments()
#             self.res.append(res_temp)

#     def ComputeDCCForEachPosPara(self, ref_list, fov_condition):
#         self.ref_list = ref_list
# #         self.res = []
# #         self.norm = []
# #         self.res = Parallel(n_jobs=10)(delayed(ComputeDCCForOnePos)(self.ref_list[idx0], self.pair_idx[idx0], self.geometry, self.source_pos, self.mrot, self.fsm, self.projections, self.Dets, self.proj_infos, self.v_det, self.gamma) for idx0 in tqdm_notebook(range(len(self.ref_list))))
#         self.res = Parallel(n_jobs=10)(delayed(ComputeDCCForOnePos)(fov_condition, self.ref_list[idx0], self.axial_limit, self.geometry, self.source_pos, self.mrot, self.fsm, self.projections, self.proj_infos, self.Ni) for idx0 in tqdm_notebook(range(len(self.ref_list))))

# #         for idx0 in tqdm_notebook(range(len(self.ref_list))):
# #             res_temp, norm_temp = ComputeDCCForOnePos(error_string, self.ref_list[idx0], self.pair_idx[idx0], self.geometry, self.source_pos, self.mrot, self.fsm, self.projections, self.Dets, self.proj_infos, self.v_det, self.gamma)
# #             self.res.append(res_temp)
# #             self.norm.append(norm_temp)
