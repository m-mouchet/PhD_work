import itk
from itk import RTK as rtk
import numpy as np
from ExtendedConeBeamDCC import *
from RTKToArrayConversion import *
from tqdm import tqdm_notebook
from joblib import Parallel, delayed


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
        self.P = np.abs(self.d*self.geometry[1, 0]/(self.geometry[0, 0]*self.proj_size[1]))
        self.axial_limit = 2*int(self.n_proj_per_rotation*np.round((1-1/self.proj_size[1])/self.P))
        self.Ni = 0 # Uncomment if simu

#        self.Dets = np.zeros((self.proj_size[2], self.proj_size[0], self.proj_size[1], 3))
#        self.gammas = (self.proj_origin[0] + self.proj_spacing[0]*self.proj_direction[0, 0]*np.array([(np.arange(self.proj_size[0])*np.ones(self.Dets[:, :, 0, 0].shape)).T]*self.Dets.shape[2]).T + ((self.geometry[3, :]-self.geometry[7, :])*np.ones(self.Dets[:, :, :, 0].shape).T).T)/self.R_det
#        self.gammas /= self.R_det
#        self.v_det = self.proj_origin[1] + self.geometry[4, 0]-self.geometry[8, 0] + np.arange(self.proj_size[1])*self.proj_spacing[1]*self.proj_direction[1, 1]
#        self.v_dets = self.proj_origin[1] + self.proj_spacing[1]*self.proj_direction[1, 1]*np.arange(self.proj_size[1])*np.ones(self.Dets[:, :, :, 1].shape)+((self.geometry[4, :]-self.geometry[8, :])*np.ones(self.Dets[:, :, :, 1].shape).T).T
#        self.Dets[:, :, :, 0] += (self.sid-self.sdd*np.cos((self.proj_origin[0] + self.proj_spacing[0]*self.proj_direction[0, 0]*np.array([(np.arange(self.proj_size[0])*np.ones(self.Dets[:, :, 0, 0].shape)).T]*self.Dets.shape[2]).T + ((self.geometry[3, :]-self.geometry[7, :])*np.ones(self.Dets[:, :, :, 0].shape).T).T)/self.R_det))*np.sin(self.geometry[2, :]*np.ones(self.Dets[:, :, :, 0].T.shape)).T + self.sdd*np.sin((self.proj_origin[0] + self.proj_spacing[0]*self.proj_direction[0, 0]*np.array([(np.arange(self.proj_size[0])*np.ones(self.Dets[:, :, 0, 0].shape)).T]*self.Dets.shape[2]).T + ((self.geometry[3, :]-self.geometry[7, :])*np.ones(self.Dets[:, :, :, 0].shape).T).T)/self.R_det)*np.cos(self.geometry[2, :]*np.ones(self.Dets[:, :, :, 0].T.shape)).T
#        self.Dets[:, :, :, 1] += self.proj_origin[1] + self.proj_spacing[1]*self.proj_direction[1, 1]*np.arange(self.proj_size[1])*np.ones(self.Dets[:, :, :, 1].shape)+((self.geometry[4, :]-self.geometry[8, :])*np.ones(self.Dets[:, :, :, 1].shape).T).T + ((self.geometry[8, :])*np.ones(self.Dets[:, :, :, 1].T.shape)).T
#        self.Dets[:, :, :, 2] += (self.sid-self.sdd*np.cos((self.proj_origin[0] + self.proj_spacing[0]*self.proj_direction[0, 0]*np.array([(np.arange(self.proj_size[0])*np.ones(self.Dets[:, :, 0, 0].shape)).T]*self.Dets.shape[2]).T + ((self.geometry[3, :]-self.geometry[7, :])*np.ones(self.Dets[:, :, :, 0].shape).T).T)/self.R_det))*np.cos(self.geometry[2, :]*np.ones(self.Dets[:, :, :, 0].T.shape)).T - self.sdd*np.sin((self.proj_origin[0] + self.proj_spacing[0]*self.proj_direction[0, 0]*np.array([(np.arange(self.proj_size[0])*np.ones(self.Dets[:, :, 0, 0].shape)).T]*self.Dets.shape[2]).T + ((self.geometry[3, :]-self.geometry[7, :])*np.ones(self.Dets[:, :, :, 0].shape).T).T)/self.R_det)*np.sin(self.geometry[2, :]*np.ones(self.Dets[:, :, :, 0].T.shape)).T

    def ComputeInitialNumberOfPhotons(self, bowtie, currents):
        bt_factor = np.interp(self.gamma, bowtie[:, 0]*np.pi/180, bowtie[:, 1])
        self.Ni = np.array([bt_factor]*len(currents)).T*(0.001*currents*120000**2)*0.0000338

    def CheckPairGeometry(self, fov_condition, idx0, idx1):
        if fov_condition == 'True':
            if np.abs(self.geometry[2, idx0] - self.geometry[2, idx1]) <= 10**(-12):
                return 0
            elif np.cos(self.geometry[2, idx1]-self.geometry[2, idx0]) > 2*((self.R_fov)/self.geometry[0, 0])**2-1:
                epair = ProjectionsPairMpoints(idx0, idx1, self.geometry, self.source_pos, self.mrot, self.fsm, self.projections, self.proj_infos, self.Ni)
                epair.ComputeMPoints()
                epair.ComputeEpipolarPlanes()
                if len(np.where(epair.final_cond)[0]) < 1:
                    return 0
                else:
                    return idx1-idx0
        else:
            if np.abs(self.geometry[2, idx0] - self.geometry[2, idx1]) <= 10**(-12):
                return 0
            else:
                epair = ProjectionsPairMpoints(idx0, idx1, self.geometry, self.source_pos, self.mrot, self.fsm, self.projections, self.proj_infos, self.Ni)
                epair.ComputeMPoints()
                epair.ComputeEpipolarPlanes()
                if len(np.where(epair.final_cond)[0]) < 1:
                    return 0
                else:
                    return idx1-idx0

    def ComputePairsFromCentralRotation(self, fov_condition):
        s0L = np.arange(self.n_proj_per_rotation*(self.n_rotation//2), self.n_proj_per_rotation*(self.n_rotation//2+1))
        results = []
        for i in tqdm_notebook(range(len(s0L))):
            results.append([])
            for j in range(-self.axial_limit, self.axial_limit):
                if (s0L[i] == s0L[i] + j) or (s0L[i] + j < 0) or (s0L[i] + j >= self.n_proj):
                    pass
                else:
                    results[i].append(self.CheckPairGeometry(fov_condition, int(s0L[i]), int(s0L[i]+j)))
        self.rangeList = []
        for i in range(len(results)):
            self.rangeList.append([k for k in results[i] if k != 0])
            self.rangeList[i].sort()

    def ComputeAllPossiblePairsForEachPos(self, ref_list, fov_condition):
        self.ComputePairsFromCentralRotation(fov_condition)
        self.ref_list = ref_list
        self.pair_idx = []
        for ip0 in tqdm_notebook(range(len(self.ref_list))):
            self.pair_idx.append([])
            for ip1 in self.rangeList[self.ref_list[ip0] % self.n_proj_per_rotation]:
                if (self.ref_list[ip0] == self.ref_list[ip0]+ip1) or (self.ref_list[ip0]+ip1 < 0) or (self.ref_list[ip0]+ip1 >= self.n_proj):
                    pass
                else:
                    self.pair_idx[ip0].append(self.ref_list[ip0]+ip1)
        #pair_idx_temp.sort()
        #print(len(pair_idx_temp))
        #self.pairs_idx = list(pair_idx_temp for pair_idx_temp,_ in itertools.groupby(pair_idx_temp))
        #print(len(self.pairs_idx))
        #pair_idx_temp.clear()
        return 0

    def ComputeDCCForEachPos(self):
        self.res = []
        for idx0 in tqdm_notebook(range(len(self.ref_list))):
            res_temp = []
            for idx1 in range(len(self.pair_idx[idx0])):
                epair = ProjectionsPairMpoints(self.ref_list[idx0], self.pair_idx[idx0][idx1], self.geometry, self.source_pos, self.mrot, self.fsm, self.projections, self.proj_infos, self.Ni)
                epair.ComputePairMoments()
            self.res.append(res_temp)

    def ComputeDCCForEachPosPara(self, ref_list, fov_condition):
        self.ref_list = ref_list
#         self.res = []
#         self.norm = []
#         self.res = Parallel(n_jobs=10)(delayed(ComputeDCCForOnePos)(self.ref_list[idx0], self.pair_idx[idx0], self.geometry, self.source_pos, self.mrot, self.fsm, self.projections, self.Dets, self.proj_infos, self.v_det, self.gamma) for idx0 in tqdm_notebook(range(len(self.ref_list))))
        self.res = Parallel(n_jobs=10)(delayed(ComputeDCCForOnePos)(fov_condition, self.ref_list[idx0], self.axial_limit, self.geometry, self.source_pos, self.mrot, self.fsm, self.projections, self.proj_infos, self.Ni) for idx0 in tqdm_notebook(range(len(self.ref_list))))

#         for idx0 in tqdm_notebook(range(len(self.ref_list))):
#             res_temp, norm_temp = ComputeDCCForOnePos(error_string, self.ref_list[idx0], self.pair_idx[idx0], self.geometry, self.source_pos, self.mrot, self.fsm, self.projections, self.Dets, self.proj_infos, self.v_det, self.gamma)
#             self.res.append(res_temp)
#             self.norm.append(norm_temp)