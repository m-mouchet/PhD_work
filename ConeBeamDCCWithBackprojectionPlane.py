import numpy as np
import itk
from itk import RTK as rtk
import matplotlib.pyplot as plt


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


def ExtractSlice(stack, num):
    # Function that extract the projection num in the projections stack stack
    ar = itk.GetArrayFromImage(stack)
    projslicea = ar[num:num+1, :, :]
    projslice = itk.GetImageFromArray(projslicea)
    projslice.CopyInformation(stack)
    projslice.Update()
    return projslice


def ExtractSourcePosition(geometry0, geometry1):
    sourcePos0 = geometry0.GetSourcePosition(0)
    sourcePos1 = geometry1.GetSourcePosition(0)
    sourcePos0 = itk.GetArrayFromVnlVector(sourcePos0.GetVnlVector())[0:3]
    sourcePos1 = itk.GetArrayFromVnlVector(sourcePos1.GetVnlVector())[0:3]
    return sourcePos0, sourcePos1


def ExtractRotationMatrice(geometry0, geometry1):
    matRot0 = geometry0.GetRotationMatrix(0)
    matRot1 = geometry1.GetRotationMatrix(0)
    matRot0 = itk.GetArrayFromVnlMatrix(matRot0.GetVnlMatrix().as_matrix())
    matRot1 = itk.GetArrayFromVnlMatrix(matRot1.GetVnlMatrix().as_matrix())
    return matRot0, matRot1


def ExtractProjCoorToFixedSysMatrice(geometry0, geometry1):
    projIdxToCoord0 = geometry0.GetProjectionCoordinatesToFixedSystemMatrix(0)
    projIdxToCoord1 = geometry1.GetProjectionCoordinatesToFixedSystemMatrix(0)
    projIdxToCoord0 = itk.GetArrayFromVnlMatrix(projIdxToCoord0.GetVnlMatrix().as_matrix())
    projIdxToCoord1 = itk.GetArrayFromVnlMatrix(projIdxToCoord1.GetVnlMatrix().as_matrix())
    return projIdxToCoord0, projIdxToCoord1


def ComputeWeightedBackProjection(geometry, projection_i, volDirection, invMagSpacing, origin, otherCorner, sourcePos):
    ImageType = itk.Image[itk.F, 3]
    # Create empty bp plane
    volDirection = volDirection.T.copy()
    volSpacing = np.array([projection_i.GetSpacing()[0]*invMagSpacing, projection_i.GetSpacing()[1]*invMagSpacing, 1])
    for i in range(len(volSpacing)):
        if volSpacing[i] < 0:
            volDirection[i, :] = (-1.0)*volDirection[i, :]
            volSpacing[i] = (-1.0)*volSpacing[i]
    volSize = [int((otherCorner[0]-origin[0])//volSpacing[0]+1), int((otherCorner[1]-origin[1])//volSpacing[1]+1), 1]
    volOrigin = np.dot(volDirection, origin)
    volOtherCorner = np.dot(volDirection, otherCorner)
    constantVolFilter = rtk.ConstantImageSource[ImageType].New()
    constantVolFilter.SetOrigin(volOrigin)
    constantVolFilter.SetSize(volSize)
    constantVolFilter.SetSpacing(volSpacing)
    volDirITK = itk.Matrix[itk.D, 3, 3](itk.GetVnlMatrixFromArray(volDirection))
    constantVolFilter.SetDirection(volDirITK)
    constantVolFilter.SetConstant(0.)
    constantVolFilter.Update()
    vol = constantVolFilter.GetOutput()
    # start back projection
    bp = rtk.BackProjectionImageFilter[ImageType, ImageType].New()
    vol = bp.SetInput(0, vol)
    if geometry.GetProjectionOffsetsX()[0] != 0:
        sid, sdd, ga, dx, dy, oa, ia, sx, sy, R = RecupParam(geometry, 0)
        new_geo0 = rtk.ThreeDCircularProjectionGeometry.New()
        new_geo0.SetRadiusCylindricalDetector(R)
        new_geo0.AddProjectionInRadians(sid, sdd, ga, 0., sy, oa, ia, sx, sy)
        new_proj_ar = None
        new_proj_ar = itk.GetArrayFromImage(projection_i)
        new_proj = itk.GetImageFromArray(np.float32(new_proj_ar))
        new_proj.CopyInformation(projection_i)
        new_proj.SetOrigin([dx-sx, dy-sy, new_proj.GetOrigin()[2]])
        new_proj.Update()
        bp.SetGeometry(new_geo0)
        vol = bp.SetInput(1, new_proj)
    else:
        bp.SetGeometry(geometry)
        vol = bp.SetInput(1, projection_i)
    bp.Update()
    vol = bp.GetOutput()
    vol.DisconnectPipeline()
    bp.Update()
    # Reset plane direction and weight
    vol.SetOrigin(origin)
    identity = itk.Matrix[itk.D,3,3]()
    identity.SetIdentity()
    vol.SetDirection(identity)   
    geoWeight = rtk.ThreeDCircularProjectionGeometry.New()
    sourcePosWeight = np.dot(volDirection.T, sourcePos)
    geoWeight.AddProjection(sourcePosWeight[2], sourcePosWeight[2], 0, -sourcePosWeight[0], -sourcePosWeight[1], 0, 0, 0, 0)
    weightFilter = rtk.FDKWeightProjectionFilter[ImageType].New()
    weightFilter.SetGeometry(geoWeight)
    weightFilter.SetInput(vol)
    weightFilter.Update()
    weight = weightFilter.GetOutput()
    weightarray = itk.GetArrayFromImage(weight)

    ione = rtk.ConstantImageSource[ImageType].New()
    ione.SetInformationFromImage(vol)
    ione.SetConstant(1.)
    ione.Update()
    weightFilter2 = rtk.FDKWeightProjectionFilter[ImageType].New()
    weightFilter2.SetGeometry(geoWeight)
    weightFilter2.SetInput(ione.GetOutput())
    weightFilter2.Update()
    weight2 = weightFilter2.GetOutput()
    weightarray2 = itk.GetArrayFromImage(weight2)

    vk = np.linspace(origin[1]+volSpacing[1]/2, otherCorner[1]-volSpacing[1]/2, volSize[1]) - sourcePosWeight[1]
    for i in range(volSize[1]):
        weightarray[0, i, :] /= np.sqrt(vk[i]**2+sourcePosWeight[2]**2)
        weightarray2[0, i, :] /= np.sqrt(vk[i]**2+sourcePosWeight[2]**2)
    return np.squeeze(vol.GetSpacing()[0]*np.sum(weightarray, axis=2)/(np.pi)), volOrigin, volOtherCorner, volDirection, vk, sourcePosWeight, weightarray2


def ComputeDCCsBPForOnePair(idx0, idx1, geometry, projection):
    if idx0 == idx1:
        return [np.array([0.]), np.array([0.])]
    else:
        pair_bp = ProjectionsPairBP(idx0, idx1, geometry, projection)
        pair_bp.LinesMomentsCorners()
        return [pair_bp.m0, pair_bp.m1]


class ProjectionsPairBP():
    def __init__(self, idx0, idx1, g0, g1, p0, p1):
        self.idx0 = idx0
        self.g0 = g0
        self.sid0, self.sdd0, self.ga0, self.dx0, self.dy0, self.oa0, self.ia0, self.sx0, self.sy0, self.R0 = RecupParam(g0, 0)
        self.p0 = p0

        self.idx1 = idx1
        self.g1 = g1
        self.sid1, self.sdd1, self.ga1, self.dx1, self.dy1, self.oa1, self.ia1, self.sx1, self.sy1, self.R1 = RecupParam(g1, 0)
        self.p1 = p1

    def LinesMomentsCorners(self):
        self.s0, self.s1 = ExtractSourcePosition(self.g0, self.g1)
        self.sourceDir0 = self.s0-self.s1
        self.sourceDir0 /= np.linalg.norm(self.sourceDir0)
        if (np.dot(self.sourceDir0, np.array([1., 0., 0.])) < 0):
            self.sourceDir0 *= -1
        self.matRot0, self.matRot1 = ExtractRotationMatrice(self.g0, self.g1)
        self.n0 = self.matRot0[2, 0:3]
        self.n1 = self.matRot1[2, 0:3]
        self.sourceDir2 = 0.5*(self.n0 + self.n1)
        self.sourceDir2 /= np.linalg.norm(self.sourceDir2)
        #y'direction
        self.sourceDir1 = np.cross(self.sourceDir2, self.sourceDir0)
        self.sourceDir1 /= np.linalg.norm(self.sourceDir1)
        #backprojection direction matrix
        self.volDirection = np.vstack((self.sourceDir0, self.sourceDir1, self.sourceDir2))
        # check for non negative spacing
        directionProj = itk.GetArrayFromMatrix(self.p0.GetDirection())
        matId = np.identity(3)
        matProd = directionProj * matId != directionProj
        if (np.sum(matProd)!=0):
            print("la matrice a %f element(s) non diagonal(aux)" %(np.sum(matProd)))
        else:
            size = []
            for i in range(len(self.p0.GetOrigin())):
                size.append(self.p0.GetSpacing()[i]*(self.p0.GetLargestPossibleRegion().GetSize()[i]-1)*directionProj[i, i])
        # Compute BP plane corners
        self.corners = None
        self.projIdxToCoord0, self.projIdxToCoord1 = ExtractProjCoorToFixedSysMatrice(self.g0, self.g1)
        self.CornersDet0 = []
        self.CornersDet1 = [] 
        invMag_List = []
        for j in self.p0.GetOrigin()[1], self.p0.GetOrigin()[1]+size[1]:
            for i in self.p0.GetOrigin()[0], self.p0.GetOrigin()[0]+size[0]:            
                if self.R0 == 0: # flat detector
                    u = i
                    v = j
                    w = 0
                else:  #cylindrical detector
                    self.theta = (i+self.dx0)/self.R0
                    u = self.R0*np.sin(self.theta)-self.dx0
                    v = j
                    w = self.R0*(1-np.cos(self.theta)) 
                idx = np.array((u, v, w, 1))
                coord0 = self.projIdxToCoord0.dot(idx)
                coord1 = self.projIdxToCoord1.dot(idx)
                # Project on the plane direction, compute inverse mag to go to isocenter and compute the source to pixel / plane intersection
                coord0Source = self.s0-coord0[0:3]
                invMag = np.dot(self.s0, self.sourceDir2)/np.dot(coord0Source, self.sourceDir2)
                invMag_List.append(invMag)
                planePos0 = np.dot(self.volDirection, self.s0-invMag*coord0Source)
                coord1Source = self.s1-coord1[0:3]
                invMag = np.dot(self.s1, self.sourceDir2)/np.dot(coord1Source, self.sourceDir2)
                invMag_List.append(invMag)
                planePos1 = np.dot(self.volDirection, self.s1-invMag*coord1Source)
                self.CornersDet0.append(coord0[0:3])
                self.CornersDet1.append(coord1[0:3])
                if self.corners is None:
                    self.corners = planePos0[0:2]
                else:
                    self.corners = np.vstack((self.corners, planePos0[0:2]))
                self.corners = np.vstack((self.corners, planePos1[0:2]))
        self.invMagSpacing = np.mean(invMag_List)
        # Check order of corners after backprojection for y
        for i in range(4):
            if(self.corners[4+i, 1] < self.corners[i, 1]):
                self.corners[4+i, 1], self.corners[i, 1] = self.corners[i, 1], self.corners[4+i, 1]
        # Find origin and opposite corner
        self.origin = np.array([np.min(self.corners[:, 0]), np.max(self.corners[np.arange(4), 1]), 0.])
        self.otherCorner = np.array([np.max(self.corners[:, 0]), np.min(self.corners[4+np.arange(4), 1]), 0.])
        # Compute moments
        self.m0, self.volOrigin0, self.volOtherCorner0, self.volDirectionT0, self.vk0, self.spw0, self.weight0 = ComputeWeightedBackProjection(self.g0, self.p0, self.volDirection, self.invMagSpacing, self.origin, self.otherCorner, self.s0)
        self.m1, self.volOrigin1, self.volOtherCorner1, self.volDirectionT1, self.vk1, self.spw1, self.weight1 = ComputeWeightedBackProjection(self.g1, self.p1, self.volDirection, self.invMagSpacing, self.origin, self.otherCorner, self.s1)

    def PlotPairMoments(self):
        plt.figure()
        plt.plot(self.m0, label="m0")
        plt.plot(self.m1, '--', label="m1")
        plt.xlabel("K")
        plt.ylabel("Moments (u.a.)")
        plt.legend()
        plt.title("Fbdcc on virtual detector")
        plt.show()
    
    def PlotProjProfile(self, row):
        plt.figure()
        plt.plot(itk.GetArrayFromImage(self.p0)[0,row,:], label="p0")
        plt.plot(itk.GetArrayFromImage(self.p1)[0,row,:], '--', label="p1")
        plt.xlabel("col")
        plt.ylabel("I (u.a.)")
        plt.legend()
        plt.title("Proj profile row %d"%row)
        plt.show()
    
    def PlotWeightProfile(self, row):
        plt.figure()
        plt.plot(self.weight0[0,row,:], label="w0")
        plt.plot(self.weight1[0,row,:], '--', label="w1")
        plt.xlabel("col")
        plt.ylabel("I (u.a.)")
        plt.legend()
        plt.title("Weight profile row %d"%row)
        plt.show()