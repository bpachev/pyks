from KS import KS, KSAssim
import numpy as np

"""
Author Benjamin Pachev <benjaminpachev@gmail.com> 2020
"""

def fourier_projector(spec, modes=21):
	mod_spec = spec.copy()
	mod_spec[:, modes:] = 0
	return np.fft.irfft(mod_spec, axis=-1)

if __name__ == "__main__":
	#See if the data assimilation works
	true = KS()
	assimilator = KSAssim(fourier_projector, mu=1, diffusion=3, update_params=True)
	max_n = 200
	for n in range(max_n):
		target = fourier_projector(true.xspec)
		assimilator.set_target(target)
		assimilator.advance()
		true.advance()
		print(assimilator.error(true))
