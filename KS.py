import numpy as np

class KS(object):
    #
    # Solution of 1-d Kuramoto-Sivashinsky equation, the simplest
    # PDE that exhibits spatio-temporal chaos
    # (https://www.encyclopediaofmath.org/index.php/Kuramoto-Sivashinsky_equation).
    #
    # u_t + u*u_x + u_xx + diffusion*u_xxxx = 0, periodic BCs on [0,2*pi*L].
    # time step dt with N fourier collocation points.
    # energy enters the system at long wavelengths via u_xx,
    # (an unstable diffusion term),
    # cascades to short wavelengths due to the nonlinearity u*u_x, and
    # dissipates via diffusion*u_xxxx.
    #
    def __init__(self,L=16,N=128,dt=0.5,diffusion=1.0,members=1,rs=None):
        self.L = L; self.n = N; self.members = members; self.dt = dt
        self.diffusion = diffusion
        kk = N*np.fft.fftfreq(N)[0:(N/2)+1]  # wave numbers
        self.wavenums = kk
        self.k = k  = kk.astype(np.float)/L
        self.ik    = 1j*k                   # spectral derivative operator
        self.lin   = k**2 - diffusion*k**4  # Fourier multipliers for linear term
        # random noise initial condition.
        if rs is None:
            rs = np.random.RandomState()
        x = 0.01*rs.standard_normal(size=(members,N))
        # remove zonal mean from initial condition.
        self.x = x - x.mean()
        # spectral space variable
        self.xspec = np.fft.rfft(self.x,axis=-1)
    def nlterm(self,xspec):
        # compute tendency from nonlinear term.
        x = np.fft.irfft(xspec,axis=-1)
        return -0.5*self.ik*np.fft.rfft(x**2,axis=-1)

    def advance(self):
        # semi-implicit third-order runge kutta update.
        # ref: http://journals.ametsoc.org/doi/pdf/10.1175/MWR3214.1
        self.xspec = np.fft.rfft(self.x,axis=-1)
        xspec_save = self.xspec.copy()
        for n in range(3):
            dt = self.dt/(3-n)
            # explicit RK3 step for nonlinear term
            self.xspec = xspec_save + dt*self.nlterm(self.xspec)
            # implicit trapezoidal adjustment for linear term
            self.xspec = (self.xspec+0.5*self.lin*dt*xspec_save)/(1.-0.5*self.lin*dt)
        self.x = np.fft.irfft(self.xspec,axis=-1)

def fourier_inner_product(spec1, spec2):
    return np.inner(np.fft.irfft(spec1), np.fft.irfft(spec2)).flatten()[0]

class KSAssim(KS):
    """Perform data assimilation on the PDE level
    
    Given spatially limited observations of a solution with unkown initial state and viscosity, we seek to recover the true viscosity and model state (marching forward in time).
    This requires solving a modified version of the original PDE that directly incorporates the observations from the true state.
    Additionally, the viscosity is initially unknown. We start with an initial guess and update it on the fly.
    """

    def __init__(self, projector, mu=1, update_params=True, **kwargs):
        """mu is the weight by the 
        """

        KS.__init__(self, **kwargs)
        self.mu = mu
        self.projector = projector
        self.target_spec = None
        self.last_target_spec = None
        self.update_params = update_params

    def update_diffusion(self, new_diffusion):
        """Update the diffusion mid-simulation - needed for parameter recovery
        """
        print("Updating diffusion to",new_diffusion)
        self.diffusion = new_diffusion
        self.lin = self.k**2 - self.diffusion*self.k**4

#    def nlterm(self,xspec):
 #       interp = self.projector(xspec)
#        return KS.nlterm(self, xspec) + self.mu * (self.target_spec - np.fft.rfft(interp, axis=-1))

    def interpolate(self, spec):
        return np.fft.rfft(self.projector(spec))

    def advance(self):
        # semi-implicit third-order runge kutta update.
        # ref: http://journals.ametsoc.org/doi/pdf/10.1175/MWR3214.1
        self.xspec = np.fft.rfft(self.x,axis=-1)
        xspec_save = self.xspec.copy()
        for n in range(3):
            dt = self.dt/(3-n)
            # explicit RK3 step for nonlinear term
            self.xspec = xspec_save + dt*self.nlterm(self.xspec)
            # implicit trapezoidal adjustment for linear term
            self.xspec = (self.xspec+0.5*self.lin*dt*xspec_save)/(1.-0.5*self.lin*dt)

        w = (self.target_spec - np.fft.rfft(self.projector(xspec_save)))

        if self.last_target_spec is not None and self.update_params:
            #Need to compute time derivative of the projections of the true model state.
            u_t = (self.target_spec-self.last_target_spec) / self.dt
            nl_contrib = self.interpolate(-self.k**2 * xspec_save-self.nlterm(xspec_save))
            G = self.interpolate(self.k**4 * xspec_save) #Contribution from the linear diffusive term
            num = fourier_inner_product(-w,u_t+nl_contrib)
            denom = fourier_inner_product(w, G)
            print (num, denom, num/denom)
            self.update_diffusion(num/denom)
       
        self.xspec += self.dt * self.mu * w
        self.x = np.fft.irfft(self.xspec,axis=-1)
	
    def set_target(self, target_projection):
        self.last_target_spec = self.target_spec
        self.target_spec = np.fft.rfft(target_projection)

    def error(self, true, kind='l2'):
        errs = true.xspec[0] - self.xspec[0]
        return np.sqrt(np.sum(np.abs(errs)**2))
