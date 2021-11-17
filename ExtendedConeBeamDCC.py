import numpy as np
import matplotlib.pyplot as plt
from IntegrationMethods import DefriseIntegrationHilbertKernel


def ComputePlaneEquation(A, B, C):  # compute the cartesian equation of the plane formed by the three point ABC
    AB = B-A
    AC = C-A
    normal = np.cross(AB, AC)
    normal /= np.linalg.norm(normal)
    d = -1*np.dot(normal, C)
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


def NormalizedError(m0, m1, norm0):
    return np.sum(np.abs(m0 - m1)/norm0)/len(m0)


def Difference(m0, m1):
    return np.sum(np.abs(m0 - m1))/len(m0)


def ComputeDCCsForOnePair(idx0, idx1, geometry_array, source_pos_array, rotation_matrices_array, fixed_matrices_array, proj_array, proj_infos):
    pair = ProjectionsPair(idx0, idx1, geometry_array, source_pos_array, rotation_matrices_array, fixed_matrices_array, proj_array, proj_infos)
    pair.ComputePairMoments()
    if len(pair.m0) == 0 and len(pair.m1) == 0:
        return 0
    else:
        return Difference(pair.m0, pair.m1)


class ProjectionsPair():
    def __init__(self, idx0, idx1, geometry_array, source_pos_array,
                 rotation_matrices_array, fixed_matrices_array,
                 proj_array, proj_infos):
        self.g0 = geometry_array[:, idx0]
        self.g1 = geometry_array[:, idx1]
        self.s0 = source_pos_array[idx0, :]
        self.s1 = source_pos_array[idx1, :]
        self.rm0 = rotation_matrices_array[idx0]
        self.rm1 = rotation_matrices_array[idx1]
        self.fm0 = fixed_matrices_array[idx0]
        self.fm1 = fixed_matrices_array[idx1]
        self.p0 = proj_array[idx0, :, :]
        self.p1 = proj_array[idx1, :, :]

        self.proj_spacing = proj_infos[0]
        self.proj_origin = proj_infos[1]
        self.proj_size = proj_infos[2]
        self.proj_direction = proj_infos[3]

        self.sid = geometry_array[0, idx0]
        self.sdd = geometry_array[1, idx0]
        self.R_det = geometry_array[9, idx0]
        self.v_det = self.proj_origin[1] + self.g0[4]-self.g0[8] + (np.arange(self.proj_size[1])*self.proj_spacing[1])*self.proj_direction[1, 1]
        self.volDir0 = np.vstack((np.array([np.cos(self.g0[2]), 0, -np.sin(self.g0[2])]), np.array([0., 1., 0.]), np.array([np.sin(self.g0[2]), 0, np.cos(self.g0[2])])))
        self.volDir1 = np.vstack((np.array([np.cos(self.g1[2]), 0, -np.sin(self.g1[2])]), np.array([0., 1., 0.]), np.array([np.sin(self.g1[2]), 0, np.cos(self.g1[2])])))

    def ComputeCylindricalDetectorsAndFanAngles(self):
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
        self.gamma = (self.proj_origin[0] +self.g0[3]-self.g0[7] + np.arange(self.proj_size[0])*self.proj_spacing[0]*self.proj_direction[0, 0])/self.R_det
        
        for j in range(self.proj_size[1]):
            self.Det0[:, j, 0] += (self.sid-self.R_det*np.cos(self.gamma))*np.sin(self.g0[2]) + self.R_det*np.sin(self.gamma)*np.cos(self.g0[2])
            self.Det0[:, j, 1] += np.ones(self.proj_size[0])*(self.proj_origin[1] + j*self.proj_spacing[1]*self.proj_direction[1, 1] + self.g0[4])
            self.Det0[:, j, 2] += (self.sid-self.R_det*np.cos(self.gamma))*np.cos(self.g0[2]) - self.R_det*np.sin(self.gamma)*np.sin(self.g0[2])
            self.Det1[:, j, 0] = (self.sid-self.R_det*np.cos(self.gamma))*np.sin(self.g1[2]) + self.R_det*np.sin(self.gamma)*np.cos(self.g1[2])
            self.Det1[:, j, 1] = np.ones(self.proj_size[0])*(self.proj_origin[1] + j*self.proj_spacing[1]*self.proj_direction[1, 1] + self.g1[4])
            self.Det1[:, j, 2] = (self.sid-self.R_det*np.cos(self.gamma))*np.cos(self.g1[2]) - self.R_det*np.sin(self.gamma)*np.sin(self.g1[2])

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
        y_dcc = np.linspace(min((y_min,y_max)), max((y_min,y_max)), int(np.floor(np.abs(y_max-y_min)/self.proj_spacing[1])))
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
        self.gamma_e1 = np.array(self.gamma_e1)

    def ComputePairMoments(self):
        self.ComputeCylindricalDetectorsAndFanAngles()
        self.ComputeMPoints()
        self.ComputeEpipolarPlanes()

        if len(self.m_points_accepted) == 0:
            self.m0, self.m1 = [], []
        else:
            self.proj_interp0 = np.zeros((self.Det0.shape[0], self.en0.shape[0]))
            self.proj_interp1 = np.zeros((self.Det1.shape[0], self.en1.shape[0]))
            self.v0 = np.zeros((self.Det0.shape[0], self.en0.shape[0]))
            self.v1 = np.zeros((self.Det1.shape[0], self.en1.shape[0]))

            for i in range(self.Det0.shape[0]):
                self.v0[i, :] = self.sdd*(-self.en0[:, 0]*np.sin(self.gamma[i]) + self.en0[:, 2]*np.cos(self.gamma[i]))/self.en0[:, 1]
                self.v1[i, :] = self.sdd*(-self.en1[:, 0]*np.sin(self.gamma[i]) + self.en1[:, 2]*np.cos(self.gamma[i]))/self.en1[:, 1]
                self.proj_interp0[i, :] += np.interp(self.v0[i, :], self.v_det, self.p0[:, i])
                self.proj_interp1[i, :] += np.interp(self.v1[i, :], self.v_det, self.p1[:, i])

            self.m0, self.m1 = np.zeros(self.v0.shape[1]), np.zeros(self.v1.shape[1])
            self.norm0, self.norm1 = np.zeros(self.v0.shape[1]), np.zeros(self.v1.shape[1])

            self.gamma_s0 = np.arctan(-self.eb0[0]/self.eb0[2])
            self.gamma_s1 = np.arctan(-self.eb1[0]/self.eb1[2])

            for j in range(self.m0.shape[0]):
                self.m0[j], self.norm0[j] = DefriseIntegrationHilbertKernel(self.gamma_s0, self.gamma, self.proj_interp0[:, j], self.v0[:, j], self.eb0, self.sdd, "Hann", 50)
                self.m1[j], self.norm1[j] = DefriseIntegrationHilbertKernel(self.gamma_s1, self.gamma, self.proj_interp1[:, j], self.v1[:, j], self.eb1, self.sdd, "Hann", 50)
    
    def PlotPairMoments(self):
        plt.figure()
        plt.plot(self.m0, label="m0")
        plt.plot(self.m1, '--', label="m1")
        plt.xlabel("K")
        plt.ylabel("Moments (u.a.)")
        plt.legend()
        plt.title("Fbdcc on cylindrical detectors")
        plt.show()

    def PlotPairGeometry(self):
        plt.figure(figsize=(16,16))
        plt.subplot(221)
        plt.plot(self.s0[2], self.s0[0], '.', color='indigo')
        plt.plot(self.Det0[:, 0, 2], self.Det0[:, 0, 0], color='indigo')
        plt.plot(self.s1[2], self.s1[0], '.', color='darkorange')
        plt.plot(self.Det1[:, 0, 2], self.Det1[:, 0, 0], color='darkorange')
        plt.plot(self.m_points[2][len(self.m_points[1])//2],self.m_points[0][len(self.m_points[1])//2], 'r+')
        plt.xlabel("z_rtk")
        plt.ylabel("x_rtk")
        plt.axis("equal")
        plt.subplot(222)
        plt.plot(self.s0[2], self.s0[1], '.', color='indigo')
        plt.plot(self.Det0[self.Det0.shape[0]//2, :, 2], self.Det0[self.Det0.shape[0]//2, :, 1], color='indigo')
        plt.plot(self.s1[2], self.s1[1], '.', color='darkorange')
        plt.plot(self.Det1[self.Det0.shape[0]//2, :, 2], self.Det1[self.Det0.shape[0]//2, :, 1], color='darkorange')
        plt.xlabel("z_rtk")
        plt.ylabel("y_rtk")
        # plt.axis("equal")
        plt.subplot(223)
        plt.imshow(self.p0, cmap = "gray", extent=(self.gamma[0], self.gamma[-1], self.v_det[0], self.v_det[-1]), aspect='auto')
        plt.plot(self.gamma, self.v_det[0]*np.ones(len(self.gamma)), color='indigo')
        plt.plot(self.gamma, self.v_det[-1]*np.ones(len(self.gamma)), color='indigo')
        plt.plot(self.gamma[0]*np.ones(len(self.v_det)), self.v_det, color='indigo')
        plt.plot(self.gamma[-1]*np.ones(len(self.v_det)), self.v_det, color='indigo')
        for j in range(self.v0.shape[1]):
            plt.plot(self.gamma, self.v0[:, j], 'r', linewidth=0.5)
        plt.subplot(224)
        plt.imshow(self.p1, cmap = "gray", extent=(self.gamma[0], self.gamma[-1], self.v_det[0], self.v_det[-1]), aspect='auto')
        plt.plot(self.gamma, self.v_det[0]*np.ones(len(self.gamma)), color='darkorange')
        plt.plot(self.gamma, self.v_det[-1]*np.ones(len(self.gamma)), color='darkorange')
        plt.plot(self.gamma[0]*np.ones(len(self.v_det)), self.v_det, color='darkorange')
        plt.plot(self.gamma[-1]*np.ones(len(self.v_det)), self.v_det, color='darkorange')
        for j in range(self.v1.shape[1]):
            plt.plot(self.gamma, self.v1[:, j], 'r', linewidth=0.5)
        plt.show()
