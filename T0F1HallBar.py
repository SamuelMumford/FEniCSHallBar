from dolfin import *
from mshr import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import math
from skimage import data, img_as_float, color, exposure
from skimage.restoration import unwrap_phase
import timeit
from matplotlib.tri import Triangulation
import random
import scipy

def solver(g, tol, mesh, omega, alpha, W, Dx, Dy, Dz, Dyz, l):
   
    #(ureal, uimag, sigma, mu) = TrialFunctions(W)
    #(v1, v2, v3, v4) = TestFunctions(W)
    (ureal, uimag) = TrialFunctions(W)
    (v1, v2) = TestFunctions(W)

    #a = (dot(grad(uimag),grad(v1)) + omega/alpha*ureal*v1 + omega/alpha*uimag*v2 - dot(grad(ureal),grad(v2)) + sigma*v3 + dot(grad(ureal), grad(v3)) + mu*v4 + dot(grad(uimag), grad(v4)))*dx
   
    def bot(x, on_boundary): return on_boundary and near(x[2], 0, tol)
    def top(x, on_boundary): return on_boundary and near(x[2], l, tol)
    noslip = Constant(0.0)
    Tslip = Expression('x[2] > (l - tol)', degree = 2, l = l, tol = 1E-5)
    bc0 = DirichletBC(W.sub(0), noslip, bot)
    bc1 = DirichletBC(W.sub(1), noslip, bot)
    G = Expression('0', degree = 0)
    def boundary(x, on_boundary):
            return on_boundary and not near(x[2], 0, tol)   
    #bc2 = DirichletBC(W.sub(2), G, boundary)
    #bc3 = DirichletBC(W.sub(3), G, boundary)
    bcs = [bc0, bc1]
    
        # Redefine boundary integration measure
    a = (Dx*uimag.dx(0)*v1.dx(0) + Dy*uimag.dx(1)*v1.dx(1) + Dz*uimag.dx(2)*v1.dx(2) + omega/alpha*ureal*v1 + omega/alpha*uimag*v2 - (Dx*ureal.dx(0)*v2.dx(0) + Dy*ureal.dx(1)*v2.dx(1) + Dz*ureal.dx(2)*v2.dx(2)) )*dx + (Dyz*alpha*uimag.dx(2)*v1.dx(1) - Dyz*alpha*uimag.dx(1)*v1.dx(2) - Dyz*alpha*ureal.dx(2)*v2.dx(1) + Dyz*alpha*ureal.dx(1)*v2.dx(2))*dx
    L =  - g*v2*dx + Tslip*v1*ds + Tslip*v2*ds

    w = Function(W)
    solve(a == L, w, bcs, solver_parameters={'linear_solver':'mumps'})
   
    (ureal, uimag) = w.split()
   
    return ureal, uimag

def scatter3d(xdata,ydata,zdata, udata, colorsMap='jet'):
    cm = plt.get_cmap(colorsMap)
    cNorm = matplotlib.colors.Normalize(vmin=min(udata), vmax=max(udata))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(xdata, ydata, zdata, c=scalarMap.to_rgba(udata))
    scalarMap.set_array(udata)
    fig.colorbar(scalarMap)
    plt.show()
    
def scatter3dCoords(shapeMat):
    xdata = shapeMat[:,0]
    xdata = np.transpose(xdata)
    ydata = shapeMat[:,1]
    ydata = np.transpose(ydata)
    zdata = shapeMat[:,2]
    zdata = np.transpose(zdata)
    
    oneM = np.ones(np.shape(shapeMat))
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(xdata, ydata, zdata)
    ax.set_xlabel('x (u)')
    ax.set_ylabel('y (u)')
    ax.set_zlabel('z (u)')
    #plt.show()
    
def find_closest_vertex(x, y, px, py, edges):
    xedges0 = px[edges[:, 0]]
    yedges0 = py[edges[:, 0]]
    xedges1 = px[edges[:, 1]]
    yedges1 = py[edges[:, 1]]
    d0 = dist(x, y, xedges0, yedges0)
    d1 = dist(x, y, xedges1, yedges1)
    minD0 = min(d0)
    minD1 = min(d1)
    if (minD0 < minD1):
        ind = np.where(d0 == minD0)[0]
        vertInd = edges[ind, 0]
    else:
        ind = np.where(d1 == minD1)[0]
        vertInd = edges[ind, 1]
    return vertInd
    
def find_verts(vertIndex, edgeList):
    hasVertex = (edgeList[:, 0] == vertIndex)
    otherV = edgeList[hasVertex, 1]
    hasVertex2 = (edgeList[:, 1] == vertIndex)
    otherV = np.append(otherV, edgeList[hasVertex2, 0])
    hasVertex = hasVertex + hasVertex2
    searchVerts = edgeList[hasVertex]
    remainVerts = edgeList[~hasVertex]
    return otherV, searchVerts, remainVerts

def tri_unwrap(phase, x_st, y_st, x_coord, y_coord, triang):
    edges = triang.edges
    startIndex = find_closest_vertex(x_st, y_st, x_coord, y_coord, edges)
    startIndex = startIndex[0]
    VtoLook, sVs, rVs = find_verts(startIndex, edges)
    
    look = True
    while(look):
        indLook = VtoLook[0]
        VtoLook = VtoLook[1:]
        newVs, tempsVs, rVs = find_verts(indLook, rVs)
        sVs = np.vstack((sVs, tempsVs))
        VtoLook = np.hstack((VtoLook, newVs))
        if(len(rVs) == 0):
            look = False
            
    hasOffset = [False]*len(projx)
    Offsets = np.zeros(len(projx))
    hasOffset[startIndex] = True
    for i in range(0, len(sVs)):
        if(hasOffset[sVs[i, 0]] == False):
            newPt = sVs[i, 0]
            oldPt = sVs[i, 1]
            Offsets[newPt] = Offsets[oldPt]
            diff = phase[newPt] - phase[oldPt]
            if((diff - 2*np.pi)**2 < (diff**2)):
                Offsets[newPt] -= 2*np.pi
            if((diff + 2*np.pi)**2 < (diff**2)):
                Offsets[newPt] += 2*np.pi
            hasOffset[newPt] = True
        if(hasOffset[sVs[i, 1]] == False):
            newPt = sVs[i, 1]
            oldPt = sVs[i, 0]
            Offsets[newPt] = Offsets[oldPt]
            diff = phase[newPt] - phase[oldPt]
            if((diff - 2*np.pi)**2 < (diff**2)):
                Offsets[newPt] -= 2*np.pi
            if((diff + 2*np.pi)**2 < (diff**2)):
                Offsets[newPt] += 2*np.pi
            hasOffset[newPt] = True
    return (phase + Offsets)

def dist(x0, y0, xd, yd):
    return np.sqrt((x0 - xd)**2 + (y0 - yd)**2)

def proj_2d(xdata, ydata, fdist):
    minDX = fdist*(np.max(xdata) - np.min(xdata))
    minDY = fdist*(np.max(ydata) - np.min(ydata))
    minD = min([minDX, minDY])
    j = 0
    look = True
    xd = xdata
    yd = ydata
    while(look):
        keepList = (dist(xd[j], yd[j], xd, yd) > minD)
        keepList[j] = True
        xd = xd[keepList]
        yd = yd[keepList]
        j += 1
        if(j >= len(xd)):
            look = False
    return xd, yd

def top_surface(xdata, ydata, zdata, projx, projy, fdist, rat):
    top_s = np.zeros(len(projx))
    minDX = fdist*(np.max(xdata) - np.min(xdata))
    minDY = fdist*(np.max(ydata) - np.min(ydata))
    minD = min([minDX, minDY])
    minD = minD*rat
    for i in range(0, len(projx)):
        lookList = (dist(projx[i], projy[i], xdata, ydata) < minD)
        if(len(zdata[lookList]) > 0):
            top_s[i] = np.max(zdata[lookList])
    return(top_s)

def side_surface(xdata, ydata, zdata, projx, projz, fdist, rat, booly):
    top_s = np.zeros(len(projx))
    minDX = fdist*(np.max(xdata) - np.min(xdata))
    minDZ = fdist*(np.max(zdata) - np.min(zdata))
    minD = min([minDX, minDZ])
    minD = minD*rat
    for i in range(0, len(projx)):
        lookList = (dist(projx[i], projz[i], xdata, zdata) < minD)
        if(len(ydata[lookList]) > 0):
            if(booly):
                top_s[i] = np.max(ydata[lookList])
            else:
                top_s[i] = np.min(ydata[lookList])
    return(top_s)

def projFunction2d(ur, ui, px, py, tz, depth, layers):
    ur2d = np.zeros(len(px))
    ui2d = np.zeros(len(px))
    weightSum = np.zeros(len(px))
    Rsheet = np.zeros(len(px))
    Isheet = np.zeros(len(px))
    for i in range(0, layers):
        for j in range(0, len(px)):
            Rsheet[j] = ur(px[j], py[j], tz[j]*(1 - i/layers))
            Isheet[j] = ui(px[j], py[j], tz[j]*(1 - i/layers))
        wtemp = np.exp(-(tz*i)/(depth*layers))
        weightSum += wtemp
        ur2d += wtemp*Rsheet
        ui2d += wtemp*Isheet
    ur2d = ur2d/weightSum
    ui2d = ui2d/weightSum
    return ur2d, ui2d

def projFunctionSide(ur, ui, ps, py1, py2, pz, depth, layers, booly):
    ur2d = np.zeros(len(ps))
    ui2d = np.zeros(len(ps))
    weightSum = np.zeros(len(ps))
    Rsheet = np.zeros(len(ps))
    Isheet = np.zeros(len(ps))
    for i in range(0, layers):
        for j in range(0, len(ps)):
            Rsheet[j] = ur(ps[j], py2[j] + (py1[j] - py2[j])*(1 - i/layers), pz[j])
            Isheet[j] = ui(ps[j], py2[j] + (py1[j] - py2[j])*(1 - i/layers), pz[j])
        if(booly):
            wtemp = np.exp(-((py1 - py2)*i)/(depth*layers))
        else:
            wtemp = np.exp(((py1 - py2)*i)/(depth*layers))
        weightSum += wtemp
        ur2d += wtemp*Rsheet
        ui2d += wtemp*Isheet
    ur2d = ur2d/weightSum
    ui2d = ui2d/weightSum
    return ur2d, ui2d

def projFunctionSideL(ur, ui, ps, py1, py2, zc, depth, layers, booly):
    ur2d = np.zeros(len(ps))
    ui2d = np.zeros(len(ps))
    weightSum = np.zeros(len(ps))
    Rsheet = np.zeros(len(ps))
    Isheet = np.zeros(len(ps))
    for i in range(0, layers):
        for j in range(0, len(ps)):
            Rsheet[j] = ur(ps[j], py2[j] + (py1[j] - py2[j])*(1 - i/layers), zc)
            Isheet[j] = ui(ps[j], py2[j] + (py1[j] - py2[j])*(1 - i/layers), zc)
        if(booly):
            wtemp = np.exp(-((py1 - py2)*i)/(depth*layers))
        else:
            wtemp = np.exp(((py1 - py2)*i)/(depth*layers))
        weightSum += wtemp
        ur2d += wtemp*Rsheet
        ui2d += wtemp*Isheet
    ur2d = ur2d/weightSum
    ui2d = ui2d/weightSum
    return ur2d, ui2d

def projFunction2dT(ur, ui, py, pz, tx, depth, layers):
    ur2d = np.zeros(len(py))
    ui2d = np.zeros(len(py))
    weightSum = np.zeros(len(py))
    Rsheet = np.zeros(len(py))
    Isheet = np.zeros(len(py))
    for i in range(0, layers):
        for j in range(0, len(py)):
            Rsheet[j] = ur(tx[j]*(1 - i/layers), py[j], pz[j])
            Isheet[j] = ui(tx[j]*(1 - i/layers), py[j], pz[j])
        wtemp = np.exp(-(tx*i)/(depth*layers))
        weightSum += wtemp
        ur2d += wtemp*Rsheet
        ui2d += wtemp*Isheet
    ur2d = ur2d/weightSum
    ui2d = ui2d/weightSum
    return ur2d, ui2d

def makeGeo(h, w, l, p, s, n):
    # Define geometry
    box = Box(Point(0, 0, 0), Point(h, w, l))
    geometry = box
    for i in range(0, n):
        cy = w*random.random()
        cz = l*random.random()
        box2 = Box(Point(0, cy-s, cz-s), Point(h, cy+s, cz+s))
        geometry -= box2
    m = generate_mesh(geometry, p)
    return m

def findProj(mesh, fd, rat):
    meshpts = mesh.coordinates()
    xdata = meshpts[:,0]
    xdata = np.transpose(xdata)
    ydata = meshpts[:,1]
    ydata = np.transpose(ydata)
    zdata = meshpts[:,2]
    zdata = np.transpose(zdata)
    
    
    #Rescale mesh geometry
    meshpts[:, 0] *= 1
    meshpts[:, 1] *= 1
    meshpts[:, 2] *= 1
    xdata = meshpts[:, 0]
    ydata = meshpts[:, 1]
    zdata = meshpts[:, 2]
    
    projs, projz = proj_2d(xdata, zdata, fd)
    
    sideS = side_surface(xdata, ydata, zdata, projs, projz, fd, rat, False)
    sideR = side_surface(xdata, ydata, zdata, projs, projz, fd, rat, True)
    
    projx, projy = proj_2d(xdata, ydata, fd)
    topS = top_surface(xdata, ydata, zdata, projx, projy, fd, rat)
    
    projw, projl = proj_2d(ydata, zdata, fd)
    ts = top_surface(ydata, zdata, xdata, projw, projl, fd, rat)
    return sideS, sideR, projs, projz, projx, projy, topS, projw, projl , ts

def doSim(mesh, h, w, l, projx, projy, topS, omega, Dx, Dy, Dz, Dyz, tol, depth, hdepth, projw, projl, ts, den):
    # Center point of beam
    # Center point of beam
    truexc = h/2
    trueyc = w/2
    # Rescaled center point of bem
    xcenter = truexc
    ycenter = trueyc
    xwid = h/2
    ywid = w/2
    xwid = xwid
    ywid = ywid
    
    maskx = np.abs(projx - xcenter) < xwid
    masky = np.abs(projy - ycenter) < ywid
    mask = [a and b for a, b in zip(maskx, masky)]
    zspot = max(topS[mask])
    
    #Define test functions on mesh
    CG1_elem = FiniteElement("CG", mesh.ufl_cell(), 1)
    CG2_elem = FiniteElement("CG", mesh.ufl_cell(), 1)
    
    W_elem = MixedElement([CG1_elem, CG2_elem])
    W = FunctionSpace(mesh, W_elem)
    
    #Change this if you need defects
    qtot = Expression('1', degree = 2)
    alpha1 = qtot
    
    g = Expression('0', degree = 2, top = zspot, wid = xwid, leng = ywid, xcenter = xcenter, ycenter = ycenter, omega = omega, b = 1/hdepth)
    
    ureal, uimag = solver(g, tol, mesh, omega, alpha1, W, Dx, Dy, Dz, Dyz, l)
    
    ureal.set_allow_extrapolation(True)
    uimag.set_allow_extrapolation(True)
    
    #Number of z-layers for phase conversion
    layers = math.ceil(mesh.num_facets()/len(projs))

    projR, projI = projFunctionSide(ureal, uimag, projs, sideS, sideR, projz, depth, layers, False)
    projRB, projIB = projFunctionSide(ureal, uimag, projs, sideS, sideR, projz, depth, layers, True)
    
    st = 0
    pts = np.zeros(0)
    vals = np.zeros(0)
    tol = .1
    look = True
    pred = Dyz*w/(Dz*l*max(projR))
    while(look):
        lp = 100
        xLine = np.linspace(0, h, lp)
        yFr = np.zeros(lp)
        yBa = w*np.ones(lp)
        lR, iR = projFunctionSideL(ureal, uimag, xLine, yFr, yBa, st, depth, layers, False)
        lRB, iRB = projFunctionSideL(ureal, uimag, xLine, yFr, yBa, st, depth, layers, True)
        
        val = np.mean(np.sqrt(lR**2 + iR**2) - np.sqrt(lRB**2 + iRB**2))
        pts = np.append(pts, st)
        vals = np.append(vals, val)
        st += den
        if(st >= l):
            look = False
        if(val < -.05):
            look = False
        if(val > 1.5*pred):
            look = False
    projIT, projRT = projFunction2dT(ureal, uimag, projw, projl, ts, depth, layers)
  
    return ureal, uimag, projR, projI, projRB, projIB, projIT, projRT, pts, vals

h = .2
l = 3

# Penetration depth of the laser beam
depth = .01
hdepth = .02
# Frequency of measurement
omega = 0
tol = 1E-14

runs = 10
p = 150
preds = np.zeros(runs)
values = np.zeros(runs)

s = .005

#Density of the 2d projection and max allowed length of a triangle
fd = .02
max_l = .1
rat = 4

Dx = 1
Dy = 1
Dz = 1
Dyz = .001

den = .02
wmin = .3
wmax = 1.5
fig = plt.figure()
a1 = fig.add_axes([0,0,1,1])
ws = np.linspace(wmin, wmax, runs)
copper = cmx.get_cmap('cool', runs)
for i in range(0, runs):
    print(i)
    w = ws[i]
    mesh = makeGeo(h, w, l, p, s, 0)
    sideS, sideR, projs, projz, projx, projy, topS, projw, projl, ts = findProj(mesh, fd, rat)
    ureal, uimag, projR, projI, projRB, projIB, projIT, projRT, pts, vals = doSim(mesh, h, w, l, projx, projy, topS, omega, Dx, Dy, Dz, Dyz, tol, depth, hdepth, projw, projl, ts, den)
    
    amp = np.sqrt(projI**2 + projR**2)
    ampb = np.sqrt(projIB**2 + projRB**2)
        
    maxi = max(amp)
    amp = amp/maxi
    ampb = ampb/maxi
    vals = vals/maxi
        
    print('New Values for length of ' + str(l))
    pred = Dyz*(w - 2*depth)/(Dz*l)
    
    a1.plot(pts, vals/pred, label = str(np.round(100*w)/100), color = copper(i))
a1.plot(np.linspace(0, l, 100), np.ones(100), color = "black")
a1.set_ylim(0,1.2)
a1.legend()
plt.ylabel('Dxy/Dxy_0')
plt.xlabel('z (u)')
plt.savefig("TySignalNBC.pdf", bbox_inches='tight')
plt.show()
