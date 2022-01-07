import numpy as np
import matplotlib.pyplot as plt
from IntegrationMethods import DefriseIntegrationHilbertKernelVec


""" def ComputePlaneEquation(A, B, C):  # compute the cartesian equation of the plane formed by the three point ABC
    AB = A-B
    if (np.dot(AB, np.array([1., 0., 0.])) < 0):
        AB *= -1
    AC = C-A
    normal = np.cross(AB, AC)
    normal /= np.linalg.norm(normal, axis=0)
    d = -1*np.dot(normal, C)
    return normal, d """


def ComputePlaneEquation(A, B, C):
    AB = A-B
    AB[np.where(np.dot(AB, np.array([1., 0., 0.])) < 0)] *= -1
    AC = C.T-A
    normal = np.cross(AB, AC)
    normal /= np.array([np.linalg.norm(normal, axis=1)]*C.shape[0]).T
    d = -1*(normal*C.T).sum(axis=1)
    return normal, d


def CanWeApplyDirectlyTheFormula(angle, xs):  # Check if there is a singularity present in the samples
    a = np.min(angle)
    b = np.max(angle)
    if np.sign(a*b) > 0 and np.abs(xs) > b:
        return True
    else:
        if xs < 0 and xs < a:
            return True
        elif xs > 0 and xs > b:
            return True
        else:
            return False


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


def NormalizedError(m0, m1, norm0):
    return np.sum(np.abs(m0 - m1)/norm0)/len(m0)


def Difference(m0, m1):
    return np.sum((m0 - m1))/len(m0)


class ProjectionsPair():
    def __init__(self, idx0, idx1, geometry, source_pos, mrot, fsm, projections, Dets, proj_infos, v_det, gamma):
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
        self.Det0 = Dets[idx0, :, :, :]
        self.Det1 = Dets[idx1, :, :, :]

        self.proj_spacing = proj_infos[0]
        self.proj_origin = proj_infos[1]
        self.proj_size = proj_infos[2]
        self.proj_direction = proj_infos[3]

        self.sid = geometry[0, idx0]
        self.sdd = geometry[1, idx0]
        self.R_det = geometry[9, idx0]
        self.v_det0 = v_det
        self.v_det1 = v_det
        self.gamma0 = gamma
        self.gamma1 = gamma
        self.volDir0 = np.vstack((np.array([np.cos(self.g0[2]), 0, -np.sin(self.g0[2])]), np.array([0., 1., 0.]), np.array([np.sin(self.g0[2]), 0, np.cos(self.g0[2])])))
        self.volDir1 = np.vstack((np.array([np.cos(self.g1[2]), 0, -np.sin(self.g1[2])]), np.array([0., 1., 0.]), np.array([np.sin(self.g1[2]), 0, np.cos(self.g1[2])])))

    """ def ComputeCylindricalDetectorsAndFanAngles(self):
        # Check for non negative spacing
        matId = np.identity(3)
        matProd = self.proj_direction * matId != self.proj_direction
        if (np.sum(matProd) != 0):
            print("la matrice a %f element(s) non diagonal(aux)" % (np.sum(matProd)))
        else:
            size = []
            for k in range(len(self.proj_origin)):
                size.append(self.proj_spacing[k]*(self.proj_size[k]-1)*self.proj_direction[k, k])

        self.Det0 = np.zeros((self.proj_size[0], self.proj_size[1], 3))
        self.Det1 = np.zeros((self.proj_size[0], self.proj_size[1], 3))
        self.gamma = (self.proj_origin[0] + self.g0[3]-self.g0[7] + np.arange(self.proj_size[0])*self.proj_spacing[0]*self.proj_direction[0, 0])/self.R_det

        self.Det0[:, :, 0] += np.array([(self.sid-self.R_det*np.cos(self.gamma))*np.sin(self.g0[2]) + self.R_det*np.sin(self.gamma)*np.cos(self.g0[2])]*self.proj_size[1]).T
        self.Det0[:, :, 1] += np.ones(self.Det0[:, :, 1].shape)*(self.proj_origin[1] + np.arange(self.proj_size[1])*self.proj_spacing[1]*self.proj_direction[1, 1] + self.g0[4])
        self.Det0[:, :, 2] += np.array([(self.sid-self.R_det*np.cos(self.gamma))*np.cos(self.g0[2]) - self.R_det*np.sin(self.gamma)*np.sin(self.g0[2])]*self.proj_size[1]).T
        self.Det1[:, :, 0] += np.array([(self.sid-self.R_det*np.cos(self.gamma))*np.sin(self.g1[2]) + self.R_det*np.sin(self.gamma)*np.cos(self.g1[2])]*self.proj_size[1]).T
        self.Det1[:, :, 1] += np.ones(self.Det1[:, :, 1].shape)*(self.proj_origin[1] + np.arange(self.proj_size[1])*self.proj_spacing[1]*self.proj_direction[1, 1] + self.g1[4])
        self.Det1[:, :, 2] += np.array([(self.sid-self.R_det*np.cos(self.gamma))*np.cos(self.g1[2]) - self.R_det*np.sin(self.gamma)*np.sin(self.g1[2])]*self.proj_size[1]).T """

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
#         y_dcc = np.linspace(min((y_min,y_max)), max((y_min,y_max)), int(np.floor(np.abs(y_max-y_min)/self.proj_spacing[1])))
        y_dcc = np.arange((y_min+y_max)/2 - np.abs(self.v_det0[-1]-self.v_det0[0]), (y_min+y_max)/2 + np.abs(self.v_det0[-1]-self.v_det0[0]), self.proj_spacing[1])
        
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

    def ComputeEpipolarPlanes(self):
        # compute eb, eb0, eb1
        self.eb = self.s0 - self.s1
        self.eb /= np.linalg.norm(self.eb)
        if (np.dot(self.eb, np.array([1., 0., 0.])) < 0):
            self.eb *= -1
        self.eb0 = np.dot(self.volDir0, self.eb)
        self.eb1 = np.dot(self.volDir1, self.eb)

        n, d = ComputePlaneEquation(np.array([self.s0]*self.m_points.shape[1]), np.array([self.s1]*self.m_points.shape[1]), self.m_points)

        temp_en0 = np.dot(self.volDir0, n.T)
        temp_en1 = np.dot(self.volDir1, n.T)
        temp_gamma_e0 = np.arctan(-temp_en0[0, :]/temp_en0[2, :])
        temp_gamma_e1 = np.arctan(-temp_en1[0, :]/temp_en1[2, :])

        answer0 = CanWeApplyDirectlyTheFormulaVec(self.gamma0, temp_gamma_e0)
        answer1 = CanWeApplyDirectlyTheFormulaVec(self.gamma1, temp_gamma_e1)
        x0 = np.zeros((len(answer0), 3))
        x0[np.where(answer0== 1), :] += np.array([self.gamma0[0], 0.5*(self.gamma0[0]+self.gamma0[-1]), self.gamma0[-1]])
        x0[np.where(answer0== 0), :] += np.array([[self.gamma0[0]]*len(np.where(answer0== 0)[0]), temp_gamma_e0[np.where(answer0== 0)], [self.gamma0[-1]]*len(np.where(answer0== 0)[0])]).T
        x1 = np.zeros((len(answer1), 3))
        x1[np.where(answer1== 1), :] += np.array([self.gamma1[0], 0.5*(self.gamma1[0]+self.gamma1[-1]), self.gamma1[-1]])
        x1[np.where(answer1== 0), :] += np.array([[self.gamma1[0]]*len(np.where(answer1== 0)[0]), temp_gamma_e1[np.where(answer1== 0)], [self.gamma1[-1]]*len(np.where(answer1== 0)[0])]).T

        temp_v0 = self.sdd*(-np.sin(x0)*np.array([temp_en0[0, :]]*3).T+np.cos(x0)*np.array([temp_en0[2, :]]*3).T)/np.array([temp_en0[1, :]]*3).T
        temp_v1 = self.sdd*(-np.sin(x1)*np.array([temp_en1[0, :]]*3).T+np.cos(x1)*np.array([temp_en1[2, :]]*3).T)/np.array([temp_en1[1, :]]*3).T

        c0_min = np.logical_and(np.min(self.v_det0) < np.min(temp_v0, axis=1), np.min(temp_v0, axis=1) < np.max(self.v_det0))
        c0_max = np.logical_and(np.min(self.v_det0) < np.max(temp_v0, axis=1), np.max(temp_v0, axis=1) < np.max(self.v_det0))
        c1_min = np.logical_and(np.min(self.v_det1) < np.min(temp_v1, axis=1), np.min(temp_v1, axis=1) < np.max(self.v_det1))
        c1_max = np.logical_and(np.min(self.v_det1) < np.max(temp_v1, axis=1), np.max(temp_v1, axis=1) < np.max(self.v_det1))
        c0_cond = np.logical_and(c0_min, c0_max)
        c1_cond = np.logical_and(c1_min, c1_max)
        self.final_cond = np.logical_and(c0_cond, c1_cond)

        self.m_points_accepted = self.m_points[:, np.where(self.final_cond)].squeeze()
        self.en = n[np.where(self.final_cond), :].squeeze()
        self.en0 = temp_en0.T[np.where(self.final_cond), :].squeeze()
        self.en1 = temp_en1.T[np.where(self.final_cond), :].squeeze()
        self.gamma_e0 = temp_gamma_e0[np.where(self.final_cond)].squeeze()
        self.gamma_e1 = temp_gamma_e1[np.where(self.final_cond)].squeeze()

    def ComputePairMoments(self):
        # self.ComputeCylindricalDetectorsAndFanAngles()
        self.ComputeMPoints()
        self.ComputeEpipolarPlanes()

        self.proj_interp0 = np.zeros((self.Det0.shape[0], self.en0.shape[0]))
        self.proj_interp1 = np.zeros((self.Det1.shape[0], self.en1.shape[0]))
        self.v0 = self.sdd*(-self.en0[:, 0]*np.sin(np.array([self.gamma0]*self.en0.shape[0]).T)+self.en0[:, 2]*np.cos(np.array([self.gamma0]*self.en0.shape[0]).T))/self.en0[:, 1]
        self.v1 = self.sdd*(-self.en1[:, 0]*np.sin(np.array([self.gamma1]*self.en1.shape[0]).T)+self.en1[:, 2]*np.cos(np.array([self.gamma1]*self.en1.shape[0]).T))/self.en1[:, 1]

        for i in range(self.Det0.shape[0]):
            self.proj_interp0[i, :] += np.interp(self.proj_direction[1,1]*self.v0[i, :], self.proj_direction[1,1]*self.v_det0, self.p0[:, i])
            self.proj_interp1[i, :] += np.interp(self.proj_direction[1,1]*self.v1[i, :], self.proj_direction[1,1]*self.v_det1, self.p1[:, i])

        self.gamma_s0 = np.arctan(-self.eb0[0]/self.eb0[2])
        self.gamma_s1 = np.arctan(-self.eb1[0]/self.eb1[2])

        self.m0, self.norm0, self.coeffs0 = DefriseIntegrationHilbertKernelVec(self.gamma_s0, self.gamma0, self.proj_interp0, self.v0, self.eb0, self.sdd, "Hann", 10)
        self.m1, self.norm1, self.coeffs1 = DefriseIntegrationHilbertKernelVec(self.gamma_s1, self.gamma1, self.proj_interp1, self.v1, self.eb1, self.sdd, "Hann", 10)

        # for j in range(self.m0.shape[0]):
        #     self.m0[j], self.norm0[j], self.coeffs0[:, j] = DefriseIntegrationHilbertKernel(self.gamma_s0, self.gamma, self.proj_interp0[:, j], self.v0[:, j], self.eb0, self.sdd, "Hann", 1)
        #     self.m1[j], self.norm1[j], self.coeffs1[:, j] = DefriseIntegrationHilbertKernel(self.gamma_s1, self.gamma, self.proj_interp1[:, j], self.v1[:, j], self.eb1, self.sdd, "Hann", 1)

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
