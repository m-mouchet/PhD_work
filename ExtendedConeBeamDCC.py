import numpy as np
import matplotlib.pyplot as plt
from IntegrationMethods import DefriseIntegrationHilbertKernelVec, TrapIntegrationAndKingModelLargeInterval

""" def ComputePlaneEquation(A, B, C):  # compute the cartesian equation of the plane formed by the three point ABC
    AB = A-B
    if (np.dot(AB, np.array([1., 0., 0.])) < 0):
        AB *= -1
    AC = C-A
    normal = np.cross(AB, AC)
    normal /= np.linalg.norm(normal, axis=0)
    d = -1*np.dot(normal, C)
    return normal, d """

""" def ComputeEpipolarPlanes(self):
    # compute eb, eb0, eb1
    self.eb = self.s0 - self.s1
    self.eb /= np.linalg.norm(self.eb)
    if (np.dot(self.eb, np.array([1., 0., 0.])) < 0):
        self.eb *= -1
    self.eb0 = np.dot(self.volDir0, self.eb)
    self.eb1 = np.dot(self.volDir1, self.eb)

    self.en = []
    self.en0 = []
    self.en1 = []
    self.gamma_e0 = []
    self.gamma_e1 = []
    self.m_points_accepted = []

    for j in range(len(self.m_points[1])):
        n, d = ComputePlaneEquation(self.s0, self.s1, np.array([self.m_points[0][j], self.m_points[1][j], self.m_points[2][j]]))
        temp_en0 = np.dot(self.volDir0, n)
        temp_en1 = np.dot(self.volDir1, n)
        temp_gamma_e0 = np.arctan(-temp_en0[0]/temp_en0[2])
        temp_gamma_e1 = np.arctan(-temp_en1[0]/temp_en1[2])

        if CanWeApplyDirectlyTheFormula(self.gamma, temp_gamma_e0):
            x0 = np.array([self.gamma[0], self.gamma[-1]])
        else:
            x0 = np.array([self.gamma[0], temp_gamma_e0, self.gamma[-1]])
        if CanWeApplyDirectlyTheFormula(self.gamma, temp_gamma_e1):
            x1 = np.array([self.gamma[0], self.gamma[-1]])
        else:
            x1 = np.array([self.gamma[0], temp_gamma_e1, self.gamma[-1]])

        temp_v0 = self.sdd*(-np.sin(x0)*temp_en0[0]+np.cos(x0)*temp_en0[2])/temp_en0[1]
        temp_v1 = self.sdd*(-np.sin(x1)*temp_en1[0]+np.cos(x1)*temp_en1[2])/temp_en1[1]

        c0_min = (np.min(self.v_det) < np.min(temp_v0) and np.min(temp_v0) < np.max(self.v_det))
        c0_max = (np.min(self.v_det) < np.max(temp_v0) and np.max(temp_v0) < np.max(self.v_det))
        c1_min = (np.min(self.v_det) < np.min(temp_v1) and np.min(temp_v1) < np.max(self.v_det))
        c1_max = (np.min(self.v_det) < np.max(temp_v1) and np.max(temp_v1) < np.max(self.v_det))
        if (c0_min and c0_max) and (c1_min and c1_max):
            self.m_points_accepted.append(np.array([self.m_points[0][j], self.m_points[1][j], self.m_points[2][j]]))
            self.en.append(n)
            self.en0.append(temp_en0)
            self.en1.append(temp_en1)
            self.gamma_e0.append(temp_gamma_e0)
            self.gamma_e1.append(temp_gamma_e1)

    self.m_points_accepted = np.array(self.m_points_accepted)
    self.en = np.array(self.en)
    self.en0 = np.array(self.en0)
    self.en1 = np.array(self.en1)
    self.gamma_e0 = np.array(self.gamma_e0)
    self.gamma_e1 = np.array(self.gamma_e1) """


def ComputePlaneEquation(A, B, C):
    AB = A-B
    AB[np.where(np.dot(AB, np.array([1., 0., 0.])) < 0)] *= -1
    AC = C.T-A
    normal = np.cross(AB, AC)
    normal /= np.array([np.linalg.norm(normal, axis=1)]*C.shape[0]).T
    d = -1*(normal*C.T).sum(axis=1)
    return normal, d


def CanWeApplyDirectlyTheFormulaVec(angle, xs):  # Check if there is a singularity present in the samples
    a = np.min(angle)
    b = np.max(angle)
    answer = np.zeros(xs.shape)
    if np.sign(a*b) > 0:
        answer[np.where(np.abs(xs)>b)] += 1
    else:
        answer[np.where(np.logical_and(xs<0, xs<a))] +=1
        answer[np.where(np.logical_and(xs>0, xs>b))] +=1
        return answer


def ComputeVarianceAllPlanes(gamma, v, proj_var, proj_interp_var, interpolation_weight, floor_indexes, ceil_indexes, gamma_s, eb, D, coeffs):
    dg = np.abs(gamma[1]-gamma[0])
    bi = (np.pi*D*dg/np.sqrt(eb[0]**2+eb[2]**2))
    simple_var = np.sum((proj_interp_var*np.array([coeffs**2]*v.shape[1]).T)/(np.sqrt(D**2+v**2)*np.sinc((gamma_s-np.array([gamma]*v.shape[1]).T)/np.pi))**2, axis=0)*bi**2
    var_part = np.sum(simple_var)
    cov_part=0
    if v.shape[1] == 1:
        pass
    else:
        for k in range(v.shape[1]):
            for l in range(k+1, v.shape[1]):
                #first term
                ija = floor_indexes[:, k] == floor_indexes[:, l]
                wia = np.sign(gamma_s)*np.pi/np.sqrt(eb[0]**2+eb[2]**2)*dg*D*coeffs[ija]/(np.sinc((gamma_s-gamma[ija])/np.pi)*np.sqrt(D**2+v[ija, k]**2))
                wja = np.sign(gamma_s)*np.pi/np.sqrt(eb[0]**2+eb[2]**2)*dg*D*coeffs[ija]/(np.sinc((gamma_s-gamma[ija])/np.pi)*np.sqrt(D**2+v[ija, l]**2))
                varija = (1-interpolation_weight[ija, k])*(1-interpolation_weight[ija, l])*proj_var[floor_indexes[ija, k], ija]
                cov_part += np.sum(wia*wja*varija)
                #second term
                ijb = floor_indexes[:, k] == ceil_indexes[:, l]
                wib = np.sign(gamma_s)*np.pi/np.sqrt(eb[0]**2+eb[2]**2)*dg*D*coeffs[ijb]/(np.sinc((gamma_s-gamma[ijb])/np.pi)*np.sqrt(D**2+v[ijb, k]**2))
                wjb = np.sign(gamma_s)*np.pi/np.sqrt(eb[0]**2+eb[2]**2)*dg*D*coeffs[ijb]/(np.sinc((gamma_s-gamma[ijb])/np.pi)*np.sqrt(D**2+v[ijb, l]**2))
                varijb = (1-interpolation_weight[ijb, k])*interpolation_weight[ijb, l]*proj_var[floor_indexes[ijb, k], ijb]
                cov_part += np.sum(wib*wjb*varijb)
                #third term
                ijc = ceil_indexes[:, k] == floor_indexes[:, l]
                wic = np.sign(gamma_s)*np.pi/np.sqrt(eb[0]**2+eb[2]**2)*dg*D*coeffs[ijc]/(np.sinc((gamma_s-gamma[ijc])/np.pi)*np.sqrt(D**2+v[ijc, k]**2))
                wjc = np.sign(gamma_s)*np.pi/np.sqrt(eb[0]**2+eb[2]**2)*dg*D*coeffs[ijc]/(np.sinc((gamma_s-gamma[ijc])/np.pi)*np.sqrt(D**2+v[ijc, l]**2))
                varijc = interpolation_weight[ijc, k]*(1-interpolation_weight[ijc, l])*proj_var[ceil_indexes[ijc, k], ijc]
                cov_part += np.sum(wic*wjc*varijc)
                #fourth term
                ijd = ceil_indexes[:, k] == ceil_indexes[:, l]
                wid = np.sign(gamma_s)*np.pi/np.sqrt(eb[0]**2+eb[2]**2)*dg*D*coeffs[ijd]/(np.sinc((gamma_s-gamma[ijd])/np.pi)*np.sqrt(D**2+v[ijd, k]**2))
                wjd = np.sign(gamma_s)*np.pi/np.sqrt(eb[0]**2+eb[2]**2)*dg*D*coeffs[ijd]/(np.sinc((gamma_s-gamma[ijd])/np.pi)*np.sqrt(D**2+v[ijd, l]**2))
                varijd = interpolation_weight[ijd, k]*interpolation_weight[ijd, l]*proj_var[ceil_indexes[ijd, k], ijd]
                cov_part += np.sum(wid*wjd*varijd)
    return simple_var, (var_part+2*cov_part)/len(simple_var)**2


class ProjectionsPairMpoints():
    def __init__(self, idx0, idx1, geometry, source_pos, mrot, fsm, projections, proj_infos, Ni):
        self.idx0 = idx0
        self.idx1 = idx1
        self.g0 = geometry[:, idx0]
        self.g1 = geometry[:, idx1]
        self.s0 = source_pos[idx0, :]
        self.s1 = source_pos[idx1, :]
        self.rm0 = mrot[idx0]
        self.rm1 = mrot[idx1]
        self.fm0 = fsm[idx0]
        self.fm1 = fsm[idx1]
        self.p0 = projections[idx0, :, :]
        self.p1 = projections[idx1, :, :]
        # self.Ni0 = np.array([Ni[:, idx0]]*self.p0.shape[0])
        # self.Ni1 = np.array([Ni[:, idx1]]*self.p1.shape[0])
        # Nibis = np.array([Ni.T]*x.shape[0])
        # Yvar = 2294.5**2/(Nibis*np.exp(-Ybis/2294.5))
        # self.proj_var0 = 2294.5**2/(np.array([Ni[:, idx0]]*self.p0.shape[0])*np.exp(-self.p0/2294.5))
        # self.proj_var1 = 2294.5**2/(np.array([Ni[:, idx1]]*self.p1.shape[0])*np.exp(-self.p1/2294.5))
        self.proj_var0 = 1/(10**5*np.exp(-0.01879*self.p0)*0.01879**2)
        self.proj_var1 = 1/(10**5*np.exp(-0.01879*self.p1)*0.01879**2)

        self.proj_spacing = proj_infos[0]
        self.proj_origin = proj_infos[1]
        self.proj_size = proj_infos[2]
        self.proj_direction = proj_infos[3]

        self.sid = geometry[0, idx0]
        self.sdd = geometry[1, idx0]
        self.R_det = geometry[9, idx0]
        self.v_det0 = self.proj_origin[1] + self.g0[4]-self.g0[8] + np.arange(self.proj_size[1])*self.proj_spacing[1]*self.proj_direction[1, 1]
        self.v_det1 = self.proj_origin[1] + self.g1[4]-self.g1[8] + np.arange(self.proj_size[1])*self.proj_spacing[1]*self.proj_direction[1, 1]
        self.gamma0 = (self.proj_origin[0] + self.g0[3]-self.g0[7] + np.arange(self.proj_size[0])*self.proj_spacing[0]*self.proj_direction[0, 0])/self.R_det
        self.gamma1 = (self.proj_origin[0] + self.g1[3]-self.g1[7] + np.arange(self.proj_size[0])*self.proj_spacing[0]*self.proj_direction[0, 0])/self.R_det
        self.volDir0 = np.vstack((np.array([np.cos(self.g0[2]), 0, -np.sin(self.g0[2])]), np.array([0., 1., 0.]), np.array([np.sin(self.g0[2]), 0, np.cos(self.g0[2])])))
        self.volDir1 = np.vstack((np.array([np.cos(self.g1[2]), 0, -np.sin(self.g1[2])]), np.array([0., 1., 0.]), np.array([np.sin(self.g1[2]), 0, np.cos(self.g1[2])])))
        self.Det0 = np.zeros((self.proj_size[0], self.proj_size[1], 3))
        self.Det1 = np.zeros((self.proj_size[0], self.proj_size[1], 3))
        self.Det0[:, :, 0] += np.array([(self.sid-self.R_det*np.cos(self.gamma0))*np.sin(self.g0[2]) + self.R_det*np.sin(self.gamma0)*np.cos(self.g0[2])]*self.proj_size[1]).T
        self.Det0[:, :, 1] += np.ones(self.Det0[:, :, 1].shape)*(self.v_det0 + self.g0[8])
        self.Det0[:, :, 2] += np.array([(self.sid-self.R_det*np.cos(self.gamma0))*np.cos(self.g0[2]) - self.R_det*np.sin(self.gamma0)*np.sin(self.g0[2])]*self.proj_size[1]).T
        self.Det1[:, :, 0] += np.array([(self.sid-self.R_det*np.cos(self.gamma1))*np.sin(self.g1[2]) + self.R_det*np.sin(self.gamma1)*np.cos(self.g1[2])]*self.proj_size[1]).T
        self.Det1[:, :, 1] += np.ones(self.Det1[:, :, 1].shape)*(self.v_det1 + self.g1[8])
        self.Det1[:, :, 2] += np.array([(self.sid-self.R_det*np.cos(self.gamma1))*np.cos(self.g1[2]) - self.R_det*np.sin(self.gamma1)*np.sin(self.g1[2])]*self.proj_size[1]).T

    def ComputeCylindersIntersection(self):  # Compute the intersection of two cylindrical detectors
        u_dir = (np.array([self.s1[2], self.s1[0]])-np.array([self.s0[2], self.s0[0]]))/np.linalg.norm((np.array([self.s1[2], self.s1[0]])-np.array([self.s0[2], self.s0[0]])))
        v_dir = np.array([-u_dir[1], u_dir[0]])  # perpendiculaire
        mid_point = (self.s0+self.s1)/2
        c = (self.s0[2]-self.s1[2])**2/4 + (self.s0[0]-self.s1[0])**2/4 - self.sdd**2
        roots = np.array([-np.sqrt(np.abs(c)), np.sqrt(np.abs(c))])
        t1 = np.array([roots[0]*v_dir[1]+mid_point[0], 0, roots[0]*v_dir[0]+mid_point[2]])
        t2 = np.array([roots[1]*v_dir[1]+mid_point[0], 0, roots[1]*v_dir[0]+mid_point[2]])
        return t1, t2

    def ComputeMPoints(self):
        y_min = np.max([self.Det0[self.proj_size[0]//2, 0, 1], self.Det1[self.proj_size[0]//2, 0, 1]])
        y_max = np.min([self.Det0[self.proj_size[0]//2, -1, 1], self.Det1[self.proj_size[0]//2, -1, 1]])
        y_dcc = np.arange((y_min+y_max)/2 - 4*np.abs(self.v_det0[-1]-self.v_det0[0]), (y_min+y_max)/2 + 4*np.abs(self.v_det0[-1]-self.v_det0[0]), self.proj_spacing[1])
        # y_dcc = np.arange((y_min+y_max)/2 - np.abs(self.v_det0[-1]-self.v_det0[0]), (y_min+y_max)/2 + np.abs(self.v_det0[-1]-self.v_det0[0]), self.proj_spacing[1])
        # Radial coordinates
        self.ta, self.tb = self.ComputeCylindersIntersection()
        dist0a = np.linalg.norm(np.array([self.ta[0], self.ta[2]])-np.array([self.Det0[self.proj_size[0]//2, 0, 0], self.Det0[self.proj_size[0]//2, 0, 2]]))
        dist0b = np.linalg.norm(np.array([self.tb[0], self.tb[2]])-np.array([self.Det0[self.proj_size[0]//2, 0, 0], self.Det0[self.proj_size[0]//2, 0, 2]]))
        dist1a = np.linalg.norm(np.array([self.ta[0], self.ta[2]])-np.array([self.Det1[self.proj_size[0]//2, 0, 0], self.Det1[self.proj_size[0]//2, 0, 2]]))
        dist1b = np.linalg.norm(np.array([self.tb[0], self.tb[2]])-np.array([self.Det1[self.proj_size[0]//2, 0, 0], self.Det1[self.proj_size[0]//2, 0, 2]]))
        if dist0a <= dist0b and dist1a <= dist1b:
            intersect = self.ta
        else:
            intersect = self.tb
        self.m_points = np.array([intersect[0]*np.ones(len(y_dcc)), y_dcc, intersect[2]*np.ones(len(y_dcc))])

    def ComputeEpipolarPlanes(self):
        # compute eb, eb0, eb1
        self.eb = np.sign(self.s1[1]-self.s0[1])*(self.s1-self.s0)/np.linalg.norm(self.s1-self.s0) # pointing in the highest direction always
        self.eb0 = np.dot(self.volDir0, self.eb)
        self.eb1 = np.dot(self.volDir1, self.eb)

        n, d = ComputePlaneEquation(np.array([self.s0]*self.m_points.shape[1]), np.array([self.s1]*self.m_points.shape[1]), self.m_points)
        temp_en0 = np.dot(self.volDir0, n.T)
        temp_en1 = np.dot(self.volDir1, n.T)
        temp_gamma_n0 = np.arctan(-temp_en0[0, :]/temp_en0[2, :])
        temp_gamma_n1 = np.arctan(-temp_en1[0, :]/temp_en1[2, :])
        self.temp_gamma_n0 = temp_gamma_n0
        self.temp_gamma_n1 = temp_gamma_n1

        x0 = np.array([[self.gamma0[0]]*len(temp_gamma_n0), [0.5*(self.gamma0[0]+self.gamma0[-1])]*len(temp_gamma_n0), [self.gamma0[-1]]*len(temp_gamma_n0)]).T
        answer0 = np.logical_and(np.min(self.gamma0) < temp_gamma_n0, temp_gamma_n0 < np.max(self.gamma0))
        x0[answer0, 1] = temp_gamma_n0[answer0]
        x1 = np.array([[self.gamma1[0]]*len(temp_gamma_n1), [0.5*(self.gamma1[0]+self.gamma1[-1])]*len(temp_gamma_n1), [self.gamma1[-1]]*len(temp_gamma_n1)]).T
        answer1 = np.logical_and(np.min(self.gamma1) < temp_gamma_n1, temp_gamma_n1 < np.max(self.gamma1))
        x1[answer1, 1] = temp_gamma_n1[answer1]

        temp_v0 = self.sdd*(-np.sin(x0)*np.array([temp_en0[0, :]]*3).T+np.cos(x0)*np.array([temp_en0[2, :]]*3).T)/np.array([temp_en0[1, :]]*3).T
        temp_v1 = self.sdd*(-np.sin(x1)*np.array([temp_en1[0, :]]*3).T+np.cos(x1)*np.array([temp_en1[2, :]]*3).T)/np.array([temp_en1[1, :]]*3).T

        c0_min = np.logical_and(np.min(self.v_det0) < np.min(temp_v0, axis=1), np.min(temp_v0, axis=1) < np.max(self.v_det0))
        c0_max = np.logical_and(np.min(self.v_det0) < np.max(temp_v0, axis=1), np.max(temp_v0, axis=1) < np.max(self.v_det0))
        c1_min = np.logical_and(np.min(self.v_det1) < np.min(temp_v1, axis=1), np.min(temp_v1, axis=1) < np.max(self.v_det1))
        c1_max = np.logical_and(np.min(self.v_det1) < np.max(temp_v1, axis=1), np.max(temp_v1, axis=1) < np.max(self.v_det1))
        c0_cond = np.logical_and(c0_min, c0_max)
        c1_cond = np.logical_and(c1_min, c1_max)
        self.final_cond = np.logical_and(c0_cond, c1_cond)

        self.m_points_accepted = self.m_points[:, self.final_cond].squeeze()
        self.en = n[self.final_cond, :].squeeze()
        self.en0 = temp_en0.T[self.final_cond, :].squeeze()
        self.en1 = temp_en1.T[self.final_cond, :].squeeze()
        self.gamma_n0 = temp_gamma_n0[self.final_cond].squeeze()
        self.gamma_n1 = temp_gamma_n1[self.final_cond].squeeze()

    def ComputePairMoments(self):
        # self.ComputeCylindricalDetectorsAndFanAngles()
        # self.ComputeMPoints()
        # self.ComputeEpipolarPlanes()

        if len(self.en.shape) == 1:
            self.en0 = np.array([self.en0])
            self.en1 = np.array([self.en1])

        self.v0 = self.sdd*(-self.en0[:, 0]*np.sin(np.array([self.gamma0]*self.en0.shape[0]).T)+self.en0[:, 2]*np.cos(np.array([self.gamma0]*self.en0.shape[0]).T))/self.en0[:, 1]
        self.v1 = self.sdd*(-self.en1[:, 0]*np.sin(np.array([self.gamma1]*self.en1.shape[0]).T)+self.en1[:, 2]*np.cos(np.array([self.gamma1]*self.en1.shape[0]).T))/self.en1[:, 1]

        dv = self.v_det0[1]-self.v_det0[0]
        floor_indexes0 = np.floor((self.v0-np.min(self.v_det0))/dv).astype(int)
        ceil_indexes0 = floor_indexes0 + 1
        interpolation_weight0 = (self.v0-self.v_det0[floor_indexes0])/dv
        floor_indexes1 = np.floor((self.v1-np.min(self.v_det1))/dv).astype(int)
        ceil_indexes1 = floor_indexes1 + 1
        interpolation_weight1 = (self.v1-self.v_det1[floor_indexes1])/dv
        fixed_indexes = np.tile(np.arange(self.v0.shape[0]),(self.v0.shape[1],1)).T
        self.proj_interp0=self.p0.T[fixed_indexes, floor_indexes0]*(1-interpolation_weight0) + self.p0.T[fixed_indexes, ceil_indexes0]*interpolation_weight0
        self.proj_interp1=self.p1.T[fixed_indexes, floor_indexes1]*(1-interpolation_weight1) + self.p1.T[fixed_indexes, ceil_indexes1]*interpolation_weight1
        self.proj_interp_var0=self.proj_var0.T[fixed_indexes, floor_indexes0]*(1-interpolation_weight0)**2 + self.proj_var0.T[fixed_indexes, ceil_indexes0]*interpolation_weight0**2
        self.proj_interp_var1=self.proj_var1.T[fixed_indexes, floor_indexes1]*(1-interpolation_weight1)**2 + self.proj_var1.T[fixed_indexes, ceil_indexes1]*interpolation_weight1**2

        self.gamma_s0 = np.arctan(-self.eb0[0]/self.eb0[2])
        self.gamma_s1 = np.arctan(-self.eb1[0]/self.eb1[2])
        self.ee0 = np.cross(self.eb0, self.en0)
        self.ee1 = np.cross(self.eb1, self.en1)
        self.gamma_e0 = np.arctan(-self.ee0[:, 0]/self.ee0[:, 2])
        self.gamma_e1 = np.arctan(-self.ee1[:, 0]/self.ee1[:, 2])

        self.m0, self.coeffs0 = DefriseIntegrationHilbertKernelVec(self.gamma_s0, self.gamma0, self.proj_interp0, self.v0, self.eb0, self.sdd, "Hann", 10)
        self.m1, self.coeffs1 = DefriseIntegrationHilbertKernelVec(self.gamma_s1, self.gamma1, self.proj_interp1, self.v1, self.eb1, self.sdd, "Hann", 10)

        #compute variance over all moments
        # self.var0, self.var1, self.tot_var0, self.tot_var1 = 0, 0, 0, 0
        self.var0, self.tot_var0 = ComputeVarianceAllPlanes(self.gamma0, self.proj_direction[1, 1]*self.v0, self.proj_var0, self.proj_interp_var0, interpolation_weight0, floor_indexes0, ceil_indexes0, self.gamma_s0, self.eb0, self.sdd, self.coeffs0)
        self.var1, self.tot_var1 = ComputeVarianceAllPlanes(self.gamma1, self.proj_direction[1, 1]*self.v1, self.proj_var1, self.proj_interp_var1, interpolation_weight1, floor_indexes1, ceil_indexes1, self.gamma_s1, self.eb1, self.sdd, self.coeffs1)

        # self.m0 = TrapIntegrationAndKingModelLargeInterval(self.eb0, self.ee0, self.en0, self.gamma_e0, self.gamma_s0, self.gamma0, self.proj_interp0, self.v0, self.vp0, self.sdd)
        # self.m1 = TrapIntegrationAndKingModelLargeInterval(self.eb1, self.ee1, self.en1, self.gamma_e1, self.gamma_s1, self.gamma1, self.proj_interp1, self.v1, self.vp1, self.sdd)

    def PlotPairMoments(self):
        plt.figure()
        plt.plot(self.m0, label="m0")
        plt.plot(self.m1, '--', label="m1")
        plt.xlabel("K")
        plt.ylabel("Moments (u.a.)")
        plt.legend()
        plt.title("Fbdcc on cylindrical detectors")
        plt.show()

    def PlotProjProfile(self, row):
        plt.figure()
        plt.plot(self.p0[row,:], label="p0")
        plt.plot(self.p1[row,:], '--', label="p1")
        plt.xlabel("col")
        plt.ylabel("I (u.a.)")
        plt.legend()
        plt.title("Proj profile row %d"%row)
        plt.show()

    def PlotWeightProfile(self, row):
        plt.figure()
        plt.plot(self.coeffs0[:, row], label="w0")
        plt.plot(self.coeffs1[:, row], '--', label="w1")
        plt.xlabel("col")
        plt.ylabel("I (u.a.)")
        plt.legend()
        plt.title("Weight profile row %d"%row)
        plt.show()


class ProjectionsPairBeta():
    def __init__(self, idx0, idx1, geometry, source_pos, mrot, fsm, projections, proj_infos, Ni):
        self.idx0 = idx0
        self.idx1 = idx1
        self.g0 = geometry[:, idx0]
        self.g1 = geometry[:, idx1]
        self.s0 = source_pos[idx0, :]
        self.s1 = source_pos[idx1, :]
        self.rm0 = mrot[idx0]
        self.rm1 = mrot[idx1]
        self.fm0 = fsm[idx0]
        self.fm1 = fsm[idx1]
        self.p0 = projections[idx0, :, :]
        self.p1 = projections[idx1, :, :]
        # self.Ni0 = np.array([Ni[:, idx0]]*self.p0.shape[0])
        # self.Ni1 = np.array([Ni[:, idx1]]*self.p1.shape[0])
        # Nibis = np.array([Ni.T]*x.shape[0])
        # Yvar = 2294.5**2/(Nibis*np.exp(-Ybis/2294.5))
        # self.proj_var0 = 2294.5**2/(np.array([Ni[:, idx0]]*self.p0.shape[0])*np.exp(-self.p0/2294.5))
        # self.proj_var1 = 2294.5**2/(np.array([Ni[:, idx1]]*self.p1.shape[0])*np.exp(-self.p1/2294.5))
        self.proj_var0 = 1/(10**5*np.exp(-0.01879*self.p0)*0.01879**2)
        self.proj_var1 = 1/(10**5*np.exp(-0.01879*self.p1)*0.01879**2)

        self.proj_spacing = proj_infos[0]
        self.proj_origin = proj_infos[1]
        self.proj_size = proj_infos[2]
        self.proj_direction = proj_infos[3]

        self.sid = geometry[0, idx0]
        self.sdd = geometry[1, idx0]
        self.R_det = geometry[9, idx0]
        self.v_det0 = self.proj_origin[1] + self.g0[4]-self.g0[8] + np.arange(self.proj_size[1])*self.proj_spacing[1]*self.proj_direction[1, 1]
        self.v_det1 = self.proj_origin[1] + self.g1[4]-self.g1[8] + np.arange(self.proj_size[1])*self.proj_spacing[1]*self.proj_direction[1, 1]
        self.gamma0 = (self.proj_origin[0] + self.g0[3]-self.g0[7] + np.arange(self.proj_size[0])*self.proj_spacing[0]*self.proj_direction[0, 0])/self.R_det
        self.gamma1 = (self.proj_origin[0] + self.g1[3]-self.g1[7] + np.arange(self.proj_size[0])*self.proj_spacing[0]*self.proj_direction[0, 0])/self.R_det
        self.volDir0 = np.vstack((np.array([np.cos(self.g0[2]), 0, -np.sin(self.g0[2])]), np.array([0., 1., 0.]), np.array([np.sin(self.g0[2]), 0, np.cos(self.g0[2])])))
        self.volDir1 = np.vstack((np.array([np.cos(self.g1[2]), 0, -np.sin(self.g1[2])]), np.array([0., 1., 0.]), np.array([np.sin(self.g1[2]), 0, np.cos(self.g1[2])])))
        self.Det0 = np.zeros((self.proj_size[0], self.proj_size[1], 3))
        self.Det1 = np.zeros((self.proj_size[0], self.proj_size[1], 3))
        self.Det0[:, :, 0] += np.array([(self.sid-self.R_det*np.cos(self.gamma0))*np.sin(self.g0[2]) + self.R_det*np.sin(self.gamma0)*np.cos(self.g0[2])]*self.proj_size[1]).T
        self.Det0[:, :, 1] += np.ones(self.Det0[:, :, 1].shape)*(self.v_det0 + self.g0[8])
        self.Det0[:, :, 2] += np.array([(self.sid-self.R_det*np.cos(self.gamma0))*np.cos(self.g0[2]) - self.R_det*np.sin(self.gamma0)*np.sin(self.g0[2])]*self.proj_size[1]).T
        self.Det1[:, :, 0] += np.array([(self.sid-self.R_det*np.cos(self.gamma1))*np.sin(self.g1[2]) + self.R_det*np.sin(self.gamma1)*np.cos(self.g1[2])]*self.proj_size[1]).T
        self.Det1[:, :, 1] += np.ones(self.Det1[:, :, 1].shape)*(self.v_det1 + self.g1[8])
        self.Det1[:, :, 2] += np.array([(self.sid-self.R_det*np.cos(self.gamma1))*np.cos(self.g1[2]) - self.R_det*np.sin(self.gamma1)*np.sin(self.g1[2])]*self.proj_size[1]).T

    def ComputeEpipolarPlanes(self):
        #Compute vector for the centeral plane
        self.eb = np.sign(self.s1[1]-self.s0[1])*(self.s1-self.s0)/np.linalg.norm(self.s1-self.s0) # pointing in the highest direction always
        if ((self.g1[2]-self.g0[2])/2)%(2*np.pi) >= np.pi/2 and ((self.g1[2]-self.g0[2])/2)%(2*np.pi) <= 3*np.pi/2:
            self.lambda_bar = (self.g1[2]+self.g0[2])/2
        else:
            self.lambda_bar = (self.g1[2]+self.g0[2])/2 + np.pi
        self.ee0 = np.cos(self.lambda_bar)*np.array([0., 0., 1.]) + np.sin(self.lambda_bar)*np.array([1., 0., 0.])
        self.en0 = np.cross(self.ee0, self.eb)
        self.alpha = np.arccos(np.dot(self.eb, np.cross(np.array([0., 1., 0.]), self.ee0)))
        self.beta = np.linspace(-np.pi/2+self.proj_spacing[1]/(2*self.R_det), np.pi/2-self.proj_spacing[1]/(2*self.R_det), int(np.pi//(self.proj_spacing[1]/self.R_det)))

        # Compute vectors for all possible planes
        self.ee_temp = np.array([np.sin(self.lambda_bar)*np.cos(self.beta)-np.cos(self.lambda_bar)*np.sin(self.alpha)*np.sin(self.beta), np.cos(self.alpha)*np.sin(self.beta), np.cos(self.lambda_bar)*np.cos(self.beta)+np.sin(self.lambda_bar)*np.sin(self.alpha)*np.sin(self.beta)])
        self.en_temp = np.array([-np.sin(self.lambda_bar)*np.sin(self.beta)-np.cos(self.lambda_bar)*np.sin(self.alpha)*np.cos(self.beta), np.cos(self.alpha)*np.cos(self.beta), -np.cos(self.lambda_bar)*np.sin(self.beta) + np.sin(self.lambda_bar)*np.sin(self.alpha)*np.cos(self.beta)])

        self.gamma_n_temp0 = np.arctan((np.tan(self.g0[2])-self.en_temp[0, :]/self.en_temp[2, :])/(1+self.en_temp[0, :]*np.tan(self.g0[2])/self.en_temp[2, :]))
        self.gamma_n_temp1 = np.arctan((np.tan(self.g1[2])-self.en_temp[0, :]/self.en_temp[2, :])/(1+self.en_temp[0, :]*np.tan(self.g1[2])/self.en_temp[2, :]))

        extremum_in_range0 = np.logical_and(np.min(self.gamma0)<=self.gamma_n_temp0, self.gamma_n_temp0<=np.max(self.gamma0))
        x0 = np.array([[self.gamma0[0]]*len(self.gamma_n_temp0), [0.5*(self.gamma0[0]+self.gamma0[-1])]*len(self.gamma_n_temp0), [self.gamma0[-1]]*len(self.gamma_n_temp0)]).T
        x0[extremum_in_range0, 1] = self.gamma_n_temp0[extremum_in_range0]
        extremum_in_range1 = np.logical_and(np.min(self.gamma1)<=self.gamma_n_temp1, self.gamma_n_temp1<=np.max(self.gamma1))
        x1 = np.array([[self.gamma1[0]]*len(self.gamma_n_temp1), [0.5*(self.gamma1[0]+self.gamma1[-1])]*len(self.gamma_n_temp1), [self.gamma1[-1]]*len(self.gamma_n_temp1)]).T
        x1[extremum_in_range1, 1] = self.gamma_n_temp1[extremum_in_range1]

        temp_v0 = self.sdd*(-np.sin(x0-self.g0[2])*np.array([self.en_temp[0, :]]*3).T+np.cos(x0-self.g0[2])*np.array([self.en_temp[2, :]]*3).T)/np.array([self.en_temp[1, :]]*3).T
        temp_v1 = self.sdd*(-np.sin(x1-self.g1[2])*np.array([self.en_temp[0, :]]*3).T+np.cos(x1-self.g1[2])*np.array([self.en_temp[2, :]]*3).T)/np.array([self.en_temp[1, :]]*3).T

        c0_min = np.logical_and(np.min(self.v_det0) < np.min(temp_v0, axis=1), np.min(temp_v0, axis=1) < np.max(self.v_det0))
        c0_max = np.logical_and(np.min(self.v_det0) < np.max(temp_v0, axis=1), np.max(temp_v0, axis=1) < np.max(self.v_det0))
        c1_min = np.logical_and(np.min(self.v_det1) < np.min(temp_v1, axis=1), np.min(temp_v1, axis=1) < np.max(self.v_det1))
        c1_max = np.logical_and(np.min(self.v_det1) < np.max(temp_v1, axis=1), np.max(temp_v1, axis=1) < np.max(self.v_det1))
        c0_cond = np.logical_and(c0_min, c0_max)
        c1_cond = np.logical_and(c1_min, c1_max)
        self.final_cond = np.logical_and(c0_cond, c1_cond)

        self.beta_accepted = self.beta[self.final_cond].squeeze()
        self.en = self.en_temp[:, self.final_cond].squeeze()
        self.ee = self.ee_temp[:, self.final_cond].squeeze()
        self.gamma_n0 = self.gamma_n_temp0[self.final_cond].squeeze()
        self.gamma_n1 = self.gamma_n_temp1[self.final_cond].squeeze()

    def ComputeBetaRange(self):
        #Compute vector for the centeral plane
        self.eb = np.sign(self.s1[1]-self.s0[1])*(self.s1-self.s0)/np.linalg.norm(self.s1-self.s0) # pointing in the highest direction always
        if ((self.g1[2]-self.g0[2])/2)%(2*np.pi) >= np.pi/2 and ((self.g1[2]-self.g0[2])/2)%(2*np.pi) <= 3*np.pi/2:
            self.lambda_bar = (self.g1[2]+self.g0[2])/2
        else:
            self.lambda_bar = (self.g1[2]+self.g0[2])/2 + np.pi
        self.ee0 = np.cos(self.lambda_bar)*np.array([0., 0., 1.]) + np.sin(self.lambda_bar)*np.array([1., 0., 0.])
        self.en0 = np.cross(self.ee0, self.eb)
        self.alpha = np.arccos(np.dot(self.eb,np.cross(np.array([0., 1., 0.]),self.ee0)))

        self.beta = []
        self.b0 = np.arctan(-(np.max(self.v_det0)/self.sdd+np.tan(self.alpha)*np.sin(self.lambda_bar+self.gamma0-self.g0[2]))*np.cos(self.alpha)/np.cos(self.lambda_bar+self.gamma0-self.g0[2]))
        self.u0 = np.arctan(-(np.min(self.v_det0)/self.sdd+np.tan(self.alpha)*np.sin(self.lambda_bar+self.gamma0-self.g0[2]))*np.cos(self.alpha)/np.cos(self.lambda_bar+self.gamma0-self.g0[2]))
        self.b1 = np.arctan(-(np.max(self.v_det1)/self.sdd+np.tan(self.alpha)*np.sin(self.lambda_bar+self.gamma1-self.g1[2]))*np.cos(self.alpha)/np.cos(self.lambda_bar+self.gamma1-self.g1[2]))
        self.u1 = np.arctan(-(np.min(self.v_det1)/self.sdd+np.tan(self.alpha)*np.sin(self.lambda_bar+self.gamma1-self.g1[2]))*np.cos(self.alpha)/np.cos(self.lambda_bar+self.gamma1-self.g1[2]))
        
        if int(np.abs(np.sum(np.sign(self.b0))))==len(self.b0) and int(np.abs(np.sum(np.sign(self.u0))))==len(self.u0) and self.b0[0]<0:
            beta_low = np.max((self.b0,self.b1))
            beta_up = np.min((self.u0,self.u1))
            if beta_up - beta_low > 0:
                self.beta = np.linspace(beta_low, beta_up, int((beta_up - beta_low)//(self.proj_spacing[1]/self.R_det)))
        elif int(np.abs(np.sum(np.sign(self.b0))))==len(self.b0) and int(np.abs(np.sum(np.sign(self.u0))))==len(self.u0) and self.b0[0]>0:
            beta_low = np.max((self.u0,self.u1))
            beta_up = np.min((self.b0,self.b1))
            if beta_up-beta_low > 0:
                self.beta = np.linspace(beta_low , beta_up, int((beta_up - beta_low)//(self.proj_spacing[1]/self.R_det)))
        elif (int(np.abs(np.sum(np.sign(self.b0)))) != len(self.b0) or int(np.abs(np.sum(np.sign(self.u0)))) != len(self.b0)) and np.sign(self.b0[0])!=np.sign(self.u0[0]):
            beta_low = -np.min((np.abs(self.b0),np.abs(self.b1)))
            beta_up = np.min((np.abs(self.u0),np.abs(self.u1)))
            if beta_up-beta_low > 0:
                self.beta = np.linspace(beta_low , beta_up, int((beta_up - beta_low)//(self.proj_spacing[1]/self.R_det)))
        # else:
            # print("Other cases that I am not aware of")
        
        if len(self.beta)>=1:
            self.ee = np.array([np.sin(self.lambda_bar)*np.cos(self.beta)-np.cos(self.lambda_bar)*np.sin(self.alpha)*np.sin(self.beta), np.cos(self.alpha)*np.sin(self.beta), np.cos(self.lambda_bar)*np.cos(self.beta)+np.sin(self.lambda_bar)*np.sin(self.alpha)*np.sin(self.beta)])
            self.en = np.array([-np.sin(self.lambda_bar)*np.sin(self.beta)-np.cos(self.lambda_bar)*np.sin(self.alpha)*np.cos(self.beta), np.cos(self.alpha)*np.cos(self.beta), -np.cos(self.lambda_bar)*np.sin(self.beta) + np.sin(self.lambda_bar)*np.sin(self.alpha)*np.cos(self.beta)])
    
    def ComputePairMoments(self):
        # self.ComputeEpipolarPlanes()

        if len(self.en.shape) == 1:
            self.en = np.array([self.en]).T

        self.v0 = self.sdd*(-self.en[0, :]*np.sin(np.array([self.gamma0]*self.en.shape[1]).T-self.g0[2])+self.en[2, :]*np.cos(np.array([self.gamma0]*self.en.shape[1]).T-self.g0[2]))/self.en[1, :]
        self.v1 = self.sdd*(-self.en[0, :]*np.sin(np.array([self.gamma1]*self.en.shape[1]).T-self.g1[2])+self.en[2, :]*np.cos(np.array([self.gamma1]*self.en.shape[1]).T-self.g1[2]))/self.en[1, :]

        dv = self.v_det0[1]-self.v_det0[0]
        floor_indexes0 = np.floor((self.v0-np.min(self.v_det0))/dv).astype(int)
        ceil_indexes0 = floor_indexes0 + 1
        interpolation_weight0 = (self.v0-self.v_det0[floor_indexes0])/dv
        floor_indexes1 = np.floor((self.v1-np.min(self.v_det1))/dv).astype(int)
        ceil_indexes1 = floor_indexes1 + 1
        interpolation_weight1 = (self.v1-self.v_det1[floor_indexes1])/dv
        fixed_indexes = np.tile(np.arange(self.v0.shape[0]),(self.v0.shape[1],1)).T
        self.proj_interp0=self.p0.T[fixed_indexes, floor_indexes0]*(1-interpolation_weight0) + self.p0.T[fixed_indexes, ceil_indexes0]*interpolation_weight0
        self.proj_interp1=self.p1.T[fixed_indexes, floor_indexes1]*(1-interpolation_weight1) + self.p1.T[fixed_indexes, ceil_indexes1]*interpolation_weight1
        self.proj_interp_var0=self.proj_var0.T[fixed_indexes, floor_indexes0]*(1-interpolation_weight0)**2 + self.proj_var0.T[fixed_indexes, ceil_indexes0]*interpolation_weight0**2
        self.proj_interp_var1=self.proj_var1.T[fixed_indexes, floor_indexes1]*(1-interpolation_weight1)**2 + self.proj_var1.T[fixed_indexes, ceil_indexes1]*interpolation_weight1**2

        self.gamma_s0 = np.arctan((np.tan(self.g0[2])-self.eb[0]/self.eb[2])/(1+self.eb[0]*np.tan(self.g0[2])/self.eb[2]))
        self.gamma_s1 = np.arctan((np.tan(self.g1[2])-self.eb[0]/self.eb[2])/(1+self.eb[0]*np.tan(self.g1[2])/self.eb[2]))

        self.m0, self.coeffs0 = DefriseIntegrationHilbertKernelVec(self.gamma_s0, self.gamma0, self.proj_interp0, self.v0, self.eb, self.sdd, "Hann", 10)
        self.m1, self.coeffs1 = DefriseIntegrationHilbertKernelVec(self.gamma_s1, self.gamma1, self.proj_interp1, self.v1, self.eb, self.sdd, "Hann", 10)

        #compute variance over all moments
        self.var0, self.var1, self.tot_var0, self.tot_var1 = 0, 0, 0, 0
        self.var0, self.tot_var0 = ComputeVarianceAllPlanes(self.gamma0, self.proj_direction[1, 1]*self.v0, self.proj_var0, self.proj_interp_var0, interpolation_weight0, floor_indexes0, ceil_indexes0, self.gamma_s0, self.eb, self.sdd, self.coeffs0)
        self.var1, self.tot_var1 = ComputeVarianceAllPlanes(self.gamma1, self.proj_direction[1, 1]*self.v1, self.proj_var1, self.proj_interp_var1, interpolation_weight1, floor_indexes1, ceil_indexes1, self.gamma_s1, self.eb, self.sdd, self.coeffs1)