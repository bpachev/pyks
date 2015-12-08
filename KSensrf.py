"""Ensemble square-root filters for the 1-d Kuramoto-Sivashinsky eqn"""
import numpy as np
import sys
from KS import KS
from enkf import serial_ensrf, bulk_ensrf, etkf, letkf, etkf_modens,\
                 serial_ensrf_modens, bulk_enkf
np.seterr(all='raise') # raise error when overflow occurs

if len(sys.argv) < 3:
    msg="""python KSensrf.py covinflate covlocal method

all variables are observed, assimilation interval given by dtassim,
nens ensemble members, observation error standard deviation = oberrstdev,
observation operator is smooth_len pt boxcar running mean or gaussian.

time mean error and spread stats printed to standard output.

covlocal:  localization distance (distance at which Gaspari-Cohn polynomial goes
to zero).

method:  =0 for serial Potter method
         =1 for bulk Potter method (all obs at once)
         =2 for ETKF (no localization applied)
         =3 for LETKF (using observation localization)
         =4 for serial Potter method with localization via modulation ensemble
         =5 for ETKF with modulation ensemble
         =6 for ETKF with modulation ensemble and perturbed obs
         =7 for serial Potter method using sqrt of localized Pb ensemble
         =8 for bulk EnKF (all obs at once) with perturbed obs.

covinflate1,covinflate2:  (optional) inflation parameters corresponding
to a and b in Hodyss and Campbell.  If not specified, a=b=1. If covinflate2
<=0, relaxation to prior spread (RTPS) inflation used with a relaxation
coefficient equal to covinflate1."""
    raise SystemExit(msg)

corrl = float(sys.argv[1])
method = int(sys.argv[2])
smooth_len = int(sys.argv[3])
use_gaussian = bool(int(sys.argv[4]))
covinflate1=1.; covinflate2=1.
if len(sys.argv) > 5:
    # if covinflate2 > 0, use Hodyss and Campbell inflation,
    # otherwise use RTPS inflation.
    covinflate1 = float(sys.argv[5])
    covinflate2 = float(sys.argv[6])

ntstart = 1000 # time steps to spin up truth run
ntimes = 21000 # ob times
nens = 10 # ensemble members
oberrstdev = 0.1; oberrvar = oberrstdev**2 # ob error
verbose = False # print error stats every time if True
dtassim = 1  # assimilation interval
# Gaussian or running average smoothing in H.
# for running average, smooth_len is half-width of boxcar.
# for gaussian, smooth_len is standard deviation.
thresh = 0.99 # threshold for modulated ensemble eigenvalue truncation.
# model parameters...
# for truth run
dt = 0.5; npts = 128
diffusion_truth = 1.0; exponent_truth = 4
# for forecast model (same as above for perfect model expt)
# for simplicity, assume dt and npts stay the same.
#diffusion = 0.85
diffusion = diffusion_truth

np.random.seed(42) # fix random seed for reproducibility

# model instance for truth (nature) run
model = KS(N=npts,dt=dt,diffusion=diffusion_truth)
# mode instance for forecast ensemble
ensemble = KS(N=npts,members=nens,dt=dt,diffusion=diffusion)
for nt in range(ntstart): # spinup truth run
    model.advance()

# sample obs from truth, compute climo stats for model.
xx = []; tt = []
for nt in range(ntimes):
    model.advance()
    xx.append(model.x[0]) # single member
    tt.append(float(nt)*model.dt)
xtruth = np.array(xx,np.float)
timetruth = np.array(tt,np.float)
xtruth_mean = xtruth.mean()
xprime = xtruth - xtruth_mean
xvar = np.sum(xprime**2,axis=0)/(ntimes-1)
xtruth_stdev = np.sqrt(xvar.mean())
if verbose:
    print 'climo for truth run:'
    print 'x mean =',xtruth_mean
    print 'x stdev =',xtruth_stdev
# forward operator.
# identity obs.
ndim = ensemble.n
h = np.eye(ndim)
# smoothing in forward operator
# gaussian or heaviside kernel.
if smooth_len > 0:
    for j in range(ndim):
        for i in range(ndim):
            rr = float(i-j)
            if i-j < -(ndim/2): rr = float(ndim-j+i)
            if i-j > (ndim/2): rr = float(i-ndim-j)
            r = np.fabs(rr)/smooth_len
            if use_gaussian:
                h[j,i] = np.exp(-r**2) # Gaussian
            else: # running average (heaviside kernel)
                if r <= 1:
                    h[j,i] = 1.
                else:
                    h[j,i] = 0.
        # normalize H so sum of weight is 1
        h[j,:] = h[j,:]/h[j,:].sum()
obs = np.empty(xtruth.shape, xtruth.dtype)
for nt in range(xtruth.shape[0]):
    obs[nt] = np.dot(h,xtruth[nt])
obs = obs + oberrstdev*np.random.standard_normal(size=obs.shape)

# spinup ensemble
ntot = xtruth.shape[0]
nspinup = ntstart
for n in range(ntstart):
    ensemble.advance()

nsteps = int(np.round(dtassim/model.dt)) # time steps in assimilation interval
if verbose:
    print 'ntstart, nspinup, ntot, nsteps =',ntstart,nspinup,ntot,nsteps
if nsteps % 1  != 0:
    raise ValueError, 'assimilation interval must be an integer number of model time steps'
else:
    nsteps = int(nsteps)

def ensrf(ensemble,xmean,xprime,h,obs,oberrvar,covlocal,method=1,z=None):
    if method == 0: # ensrf with obs one at time
        return serial_ensrf(xmean,xprime,h,obs,oberrvar,covlocal,covlocal)
    elif method == 1: # ensrf with all obs at once
        return bulk_ensrf(xmean,xprime,h,obs,oberrvar,covlocal)
    elif method == 2: # etkf (no localization)
        return etkf(xmean,xprime,h,obs,oberrvar)
    elif method == 3: # letkf
        return letkf(xmean,xprime,h,obs,oberrvar,covlocal)
    elif method == 4: # serial ensrf using 'modulated' ensemble
        return serial_ensrf_modens(xmean,xprime,h,obs,oberrvar,covlocal,z)
    elif method == 5: # etkf using 'modulated' ensemble
        return etkf_modens(xmean,xprime,h,obs,oberrvar,covlocal,z)
    elif method == 6: # etkf using 'modulated' ensemble w/pert obs
        return etkf_modens(xmean,xprime,h,obs,oberrvar,covlocal,z,po=True)
    elif method == 7: # serial ensrf using sqrt of localized Pb
        return serial_ensrf_modens(xmean,xprime,h,obs,oberrvar,covlocal,None)
    elif method == 8: # enkf with perturbed obs all at once
        return bulk_enkf(xmean,xprime,h,obs,oberrvar,covlocal)
    else:
        raise ValueError('illegal value for enkf method flag')

# define localization matrix.
covlocal = np.eye(ndim)
if corrl < 2*ndim:
    for j in range(ndim):
        for i in range(ndim):
            rr = float(i-j)
            if i-j < -(ndim/2): rr = float(ndim-j+i)
            if i-j > (ndim/2): rr = float(i-ndim-j)
            r = np.fabs(rr)/corrl
            #if r < 1.: # Bohman taper
            #    taper = (1.-r)*np.cos(np.pi*r) + np.sin(np.pi*r)/np.pi
            #taper = np.exp(-(r**2/0.15)) # Gaussian
            # Gaspari-Cohn polynomial.
            rr = 2.*r
            taper = 0.
            if r <= 0.5:
                taper = ((( -0.25*rr +0.5 )*rr +0.625 )*rr -5.0/3.0 )*rr**2+1.
            elif r > 0.5 and r < 1.:
                taper = (((( rr/12.0 -0.5 )*rr +0.625 )*rr +5.0/3.0 )*rr -5.0 )*rr \
                        + 4.0 - 2.0 / (3.0 * rr)
            covlocal[j,i]=taper

# compute square root of covlocal
if method in [4,5,6]:
    evals, eigs = np.linalg.eigh(covlocal)
    evals = np.where(evals > 1.e-10, evals, 1.e-10)
    evalsum = evals.sum(); neig = 0
    frac = 0.0
    while frac < thresh:
        frac = evals[ndim-neig-1:ndim].sum()/evalsum
        neig += 1
    #print 'neig = ',neig
    zz = (eigs*np.sqrt(evals/frac)).T
    z = zz[ndim-neig:ndim,:]
    #import matplotlib.pyplot as plt
    #print zz[-1].min(),zz[-1].max(),np.sqrt(ndim)*eigs[:,-1].min(),np.sqrt(ndim)*eigs[:,-1].max()
    #print np.sqrt(evals[-1]/(ndim*frac))
    #scalefact_eig1 = zz[-1].max()
    #plt.plot(zz[-1]) # 1st eigenvector.
    #plt.show()
else:
    neig = 0
    z = None

# run assimilation.
fcsterr = []
fcstsprd = []
analerr = []
analsprd = []
diverged = False
fsprdmean = np.zeros(ndim,np.float)
fsprdobmean = np.zeros(ndim,np.float)
asprdmean = np.zeros(ndim,np.float)
ferrmean = np.zeros(ndim,np.float)
aerrmean = np.zeros(ndim,np.float)
corrmean = np.zeros(ndim,np.float)
corrhmean = np.zeros(ndim,np.float)
for nassim in range(0,ntot,nsteps):
    # assimilate obs
    xmean = ensemble.x.mean(axis=0)
    xmean_b = xmean.copy()
    xprime = ensemble.x - xmean
    # calculate background error, sprd stats.
    ferr = (xmean - xtruth[nassim])**2
    if np.isnan(ferr.mean()):
        diverged = True
        break
    fsprd = (xprime**2).sum(axis=0)/(ensemble.members-1)
    corr = (xprime.T*xprime[:,ndim/2]).sum(axis=1)/float(ensemble.members-1)
    hxprime = np.dot(xprime,h)
    fsprdob = (hxprime**2).sum(axis=0)/(ensemble.members-1)
    corrh = (xprime.T*hxprime[:,ndim/2]).sum(axis=1)/float(ensemble.members-1)
    if nassim >= nspinup:
        fsprdmean = fsprdmean + fsprd
        fsprdobmean = fsprdobmean + fsprdob
        corrmean = corrmean + corr
        corrhmean = corrhmean + corrh
        ferrmean = ferrmean + ferr
        fcsterr.append(ferr.mean()); fcstsprd.append(fsprd.mean())
    # update state estimate.
    xmean,xprime =\
    ensrf(ensemble,xmean,xprime,h,obs[nassim,:],oberrvar,covlocal,method=method,z=z)
    # calculate analysis error, sprd stats.
    aerr = (xmean - xtruth[nassim])**2
    asprd = (xprime**2).sum(axis=0)/(ensemble.members-1)
    if nassim >= nspinup:
        asprdmean = asprdmean + asprd
        aerrmean = aerrmean + aerr
        analerr.append(aerr.mean()); analsprd.append(asprd.mean())
    if verbose:
        print nassim,timetruth[nassim],np.sqrt(ferr.mean()),np.sqrt(fsprd.mean()),np.sqrt(aerr.mean()),np.sqrt(asprd.mean())
    if covinflate2 > 0:
        # Hodyss and Campbell inflation.
        inc = xmean - xmean_b
        inf_fact = np.sqrt(covinflate1 + \
        (asprd/fsprd**2)*((fsprd/ensemble.members) + covinflate2*(2.*inc**2/(ensemble.members-1))))
    else:
        # relaxation to prior spread inflation
        asprd = np.sqrt(asprd); fsprd = np.sqrt(fsprd)
        inf_fact = 1.+covinflate1*(fsprd-asprd)/asprd
    xprime *= inf_fact
    # run forecast model.
    ensemble.x = xmean + xprime
    for n in range(nsteps):
        ensemble.advance()

# print out time mean stats.
# error and spread are normalized by observation error.
if diverged:
    print method,len(fcsterr),corrl,covinflate1,covinflate2,oberrstdev,np.nan,np.nan,np.nan,np.nan,neig
else:
    fcsterr = np.array(fcsterr)
    fcstsprd = np.array(fcstsprd)
    analerr = np.array(analerr)
    analsprd = np.array(analsprd)
    fstdev = np.sqrt(fcstsprd.mean())
    astdev = np.sqrt(analsprd.mean())
    asprdmean = asprdmean/len(fcstsprd)
    aerrmean = aerrmean/len(analerr)
    fstd = np.sqrt(fsprdmean)
    fstdob = np.sqrt(fsprdobmean)
    corrmean = corrmean/(fstd*fstd[ndim/2])
    corrhmean = corrhmean/(fstd*fstdob[ndim/2])
    #import matplotlib.pyplot as plt
    #plt.plot(np.arange(ndim),corrmean,color='k',label='r')
    #plt.plot(np.arange(ndim),corrhmean,color='b',label='r (x vs hx)')
    #plt.plot(np.arange(ndim),h[:,ndim/2]/h.max(),color='r',label='H')
    #plt.plot(np.arange(ndim),covlocal[:,ndim/2],'k:',label='L')
    #plt.xlim(0,ndim)
    #plt.legend()
    #plt.show()
    print method,len(fcsterr),corrl,covinflate1,covinflate2,oberrstdev,np.sqrt(fcsterr.mean()),fstdev,\
          np.sqrt(analerr.mean()),astdev,neig
