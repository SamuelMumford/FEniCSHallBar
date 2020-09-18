from dolfin import *
from mshr import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import math
import random

#Call FEniCS solver with diffusivities
def solver(g, tol, mesh, omega, alpha, W, Dx, Dy, Dz, Dyz, l):
   
    #Define the initial guesses and weak form test functions
    (ureal, uimag) = TrialFunctions(W)
    (v1, v2) = TestFunctions(W)

    #Define the base differential equation
    a = (Dx*uimag.dx(0)*v1.dx(0) + Dy*uimag.dx(1)*v1.dx(1) + Dz*uimag.dx(2)*v1.dx(2) + omega/alpha*ureal*v1 + omega/alpha*uimag*v2 - (Dx*ureal.dx(0)*v2.dx(0) + Dy*ureal.dx(1)*v2.dx(1) + Dz*ureal.dx(2)*v2.dx(2)) )*dx + (Dyz*alpha*uimag.dx(2)*v1.dx(1) - Dyz*alpha*uimag.dx(1)*v1.dx(2) - Dyz*alpha*ureal.dx(2)*v2.dx(1) + Dyz*alpha*ureal.dx(1)*v2.dx(2))*dx
    L =  - g/alpha*v2*dx
   
    #Define the top and bottom boundary locations for set temperature
    def bot(x, on_boundary): return on_boundary and near(x[2], 0, tol)
    def top(x, on_boundary): return on_boundary and near(x[2], l, tol)
    noslip = Constant(0.0)
    Tslip = Constant(1.0)
    #Change v1 and v2 weights to fit boundary conditions
    bc0 = DirichletBC(W.sub(0), noslip, bot)
    bc1 = DirichletBC(W.sub(1), noslip, bot)
    bc2 = DirichletBC(W.sub(0), Tslip, top)
    bc3 = DirichletBC(W.sub(1), Tslip, top)
    bcs = [bc0, bc1, bc2, bc3]
    
    #Call FEniCS solver on thte defined geometry, diffEq, and test functions
    w = Function(W)
    solve(a == L, w, bcs, solver_parameters={'linear_solver':'mumps'})
   
    (ureal, uimag) = w.split()
   
    return ureal, uimag

#Plotting code to see the temperature in 3D
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

#Plotting code to see the mesh    
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

#Function to get distances
def dist(x0, y0, xd, yd):
    return np.sqrt((x0 - xd)**2 + (y0 - yd)**2)

#Function to project 3D mesh of x, y, and z data into a top-down 2D plane
def proj_2d(xdata, ydata, fdist):
    #Define how dense the 2D projection should be
    minDX = fdist*(np.max(xdata) - np.min(xdata))
    minDY = fdist*(np.max(ydata) - np.min(ydata))
    minD = min([minDX, minDY])
    #Pick a point in the 3D mesh, get rid of all points that are too close
    #In a 2D projection to that point. Iterate until all points are either in
    #the 2D projection or have been deleted
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

#Use the 2D projection to find the top surface
def top_surface(xdata, ydata, zdata, projx, projy, fdist, rat):
    top_s = np.zeros(len(projx))
    minDX = fdist*(np.max(xdata) - np.min(xdata))
    minDY = fdist*(np.max(ydata) - np.min(ydata))
    minD = min([minDX, minDY])
    minD = minD*rat
    #For each point in the 2D projection, find the point in the 3D mesh with the largest z-value
    for i in range(0, len(projx)):
        lookList = (dist(projx[i], projy[i], xdata, ydata) < minD)
        if(len(zdata[lookList]) > 0):
            top_s[i] = np.max(zdata[lookList])
    return(top_s)

#Use a 2D projection to find side surfaces. Same process, but now can keep the largest or the smallest 
#value of the unprojected coordinate
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

#Perform a weighted projection of the 3D function values onto the 2D projected mesh
def projFunction2d(ur, ui, px, py, tz, depth, layers):
    ur2d = np.zeros(len(px))
    ui2d = np.zeros(len(px))
    weightSum = np.zeros(len(px))
    Rsheet = np.zeros(len(px))
    Isheet = np.zeros(len(px))
    #Divide the sample up into layers-levels ranging in height from z=0 to the top surface
    #Find the value and weight for each point in that layer's z-coordinate and 2D projected cooridate
    for i in range(0, layers):
        for j in range(0, len(px)):
            Rsheet[j] = ur(px[j], py[j], tz[j]*(1 - i/layers))
            Isheet[j] = ui(px[j], py[j], tz[j]*(1 - i/layers))
        #Note that tz can vary for each point
        wtemp = np.exp(-(tz*i)/(depth*layers))
        weightSum += wtemp
        #Perform weighted sum of layer values
        ur2d += wtemp*Rsheet
        ui2d += wtemp*Isheet
    ur2d = ur2d/weightSum
    ui2d = ui2d/weightSum
    return ur2d, ui2d

#Same function as above, but for projection onto front or back side surfaces.
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

#Same process, but instead of projecting onto a full side surface, do so only on one line of points
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

#Same side projection function, but projected onto the x-side surface
#instead of the y or z
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

#Make the mesh, allows for defects at random locations
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

#Call 2D projection functions to get a series of side projection points
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

#Use FEniCS and the 2D projection points for different surfaces to simulate
#and project results into surface observables
def doSim(mesh, h, w, l, projx, projy, topS, omega, Dx, Dy, Dz, Dyz, tol, depth, hdepth, projw, projl, ts, den):
    #Define test functions on mesh
    CG1_elem = FiniteElement("CG", mesh.ufl_cell(), 1)
    CG2_elem = FiniteElement("CG", mesh.ufl_cell(), 1)
    
    W_elem = MixedElement([CG1_elem, CG2_elem])
    W = FunctionSpace(mesh, W_elem)
    
    #Change this if you want defects in D
    qtot = Expression('1', degree = 2)
    alpha1 = qtot
    
    #Change this if you want to add a drive
    g = Expression('0', degree = 2)
    
    #Perform simulation
    ureal, uimag = solver(g, tol, mesh, omega, alpha1, W, Dx, Dy, Dz, Dyz, l)
    ureal.set_allow_extrapolation(True)
    uimag.set_allow_extrapolation(True)
    
    #Number of z-layers for phase conversion
    layers = math.ceil(mesh.num_facets()/len(projs))
    
    #Perform projection
    projR, projI = projFunctionSide(ureal, uimag, projs, sideS, sideR, projz, depth, layers, False)
    projRB, projIB = projFunctionSide(ureal, uimag, projs, sideS, sideR, projz, depth, layers, True)
    
    #Find the mean value of temperature difference between the left and right surfaces as the 
    #z-location of measurement changes
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

#Base sample dimensions
w = .3
h = .2
l = 3

# Penetration depth of the thermometer projection
depth = .01
#Penetration depth of a drive if used
hdepth = .02
# Frequency of measurement
omega = 0
tol = 1E-14

#How many paramters to look at
runs = 10
#Mesh density
p = 130
preds = np.zeros(runs)
values = np.zeros(runs)

#Defect size if used
s = .005

#Density of the 2d projection and max allowed length of a triangle
fd = .02
max_l = .1
rat = 4

#Diffusivities
Dx = 1
Dy = 1
Dz = 1
Dyz = .001

#Density of points to examine temperature at
den = .02
#Minimum and maximum widths simulated
wmin = .3
wmax = 1.5
fig = plt.figure()
a1 = fig.add_axes([0,0,1,1])
ws = np.linspace(wmin, wmax, runs)
copper = cmx.get_cmap('cool', runs)
#Find temperature difference between surfaces based on z for a variety of w
for i in range(0, runs):
    print(i)
    w = ws[i]
    #Make the new mesh
    mesh = makeGeo(h, w, l, p, s, 0)
    #Find 2D projection points 
    sideS, sideR, projs, projz, projx, projy, topS, projw, projl, ts = findProj(mesh, fd, rat)
    #Perform simulation and projection
    ureal, uimag, projR, projI, projRB, projIB, projIT, projRT, pts, vals = doSim(mesh, h, w, l, projx, projy, topS, omega, Dx, Dy, Dz, Dyz, tol, depth, hdepth, projw, projl, ts, den)
    
    #Normalize temperature result
    amp = np.sqrt(projI**2 + projR**2)
    ampb = np.sqrt(projIB**2 + projRB**2)
        
    maxi = max(amp)
    amp = amp/maxi
    ampb = ampb/maxi
    vals = vals/maxi
    
    cutoff = 2
    rati = pts/w
    mask = (rati < cutoff)
    #Plot result 
    print('New Values for length of ' + str(l))
    pred = Dyz*(w - 2*depth)/(Dz*l)
    
    a1.plot(pts, vals/pred, label = str(np.round(100*w)/100), color = copper(i))
a1.plot(np.linspace(0, cutoff, 100), np.ones(100), color = "black")
a1.set_ylim(0,1)
a1.legend()
plt.ylabel('Dxy/Dxy_0')
plt.xlabel('z (u)')
plt.savefig("TySignalLoverW.pdf", bbox_inches='tight')
plt.show()
