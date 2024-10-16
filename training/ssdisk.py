import numpy as np
from torch.utils.data import Dataset

def f_SS(R, Doppler, R0, slope, Rin):
    return Doppler * (((R0/R) **slope)*(1 - (Rin/R)**0.5)) ** -0.25

def I_SS_single(R, Doppler, Rin, R0, slope):
    shape = R.shape
    R = R.flatten()
    Doppler = Doppler.flatten()
    data = np.zeros_like(R)
    select = R > Rin
    data[select] = Doppler[select] ** 3 / (np.exp(f_SS(R[select], Doppler[select], R0, slope, Rin)) - 1.)
    data[~select] = 0.
    return data.reshape(*shape)

def v2(data, SNR=None):
    
    FT = np.fft.rfft2(data)
    v2data = FT.real ** 2 + FT.imag ** 2

    if SNR is not None:
        v2data += np.random.randn(*v2data.shape) / SNR
    
    return v2data

def ssdisk_single(R0, Rin, Rg, slope, phi0, inc, npix, Rmax=None, USE_BEAM=True, return_v2=False, SNR=None):
    
    assert Rin > Rg
    
    cosi = np.cos(inc)
    sini = np.sin(inc)
    
    cp = np.cos(phi0)
    sp = np.sin(phi0)
    
    if Rmax is None:
        Rmax = 3 * R0
    
    dx = 2*Rmax/npix

    xp = -Rmax + (np.arange(npix)[:,None] + 0.5) * dx
    yp = -Rmax + (np.arange(npix)[None] + 0.5) * dx
    
    x = cp*xp - sp*yp
    y = sp*xp + cp*yp
    x /= cosi
    R = (x**2 + y**2) ** 0.5
    
    if USE_BEAM:
        Doppler = np.ones_like(R)
        select = R > Rg
        v = (Rg/R[select]) ** 0.5
        Doppler[select] = (1-v*v)**0.5 / (1-v*sini*y[select]/R[select])
    else:
        Doppler = np.ones_like(R)
    
    data = I_SS_single(R, Doppler, Rin, R0, slope)

    Sum = np.sum(data)
    assert Sum>0
    data /= Sum
    
    if return_v2:
        return data, v2(data, SNR)
    else:
        return data


def I_SS(R, Doppler, Rin, R0, slope):
    npix = R.shape[-1]
    data = np.zeros_like(R)
    R0 = np.tile(R0, (1,npix,npix))
    slope = np.tile(slope, (1,npix,npix))
    Rin = np.tile(Rin, (1,npix,npix))

    select = R > Rin
    data[select] = Doppler[select] ** 3 / (np.exp(f_SS(R[select], Doppler[select], R0[select], slope[select], Rin[select])) - 1.)
    data[~select] = 0.
    return data

def ssdisk(R0, Rin, Rg, slope, phi0, inc, npix, Rmax=None, USE_BEAM=True, return_v2=False, SNR=None):

    assert (Rin > Rg).all()

    R0 = R0.reshape(-1,1,1)
    Rin = Rin.reshape(-1,1,1)
    Rg = Rg.reshape(-1,1,1)
    slope = slope.reshape(-1,1,1)
    phi0 = phi0.reshape(-1,1,1)
    inc = inc.reshape(-1,1,1)

    cosi = np.cos(inc)
    sini = np.sin(inc)

    cp = np.cos(phi0)
    sp = np.sin(phi0)

    if Rmax is None:
        Rmax = 3 * R0

    dx = 2*Rmax/npix

    xp = -Rmax + (np.arange(npix).reshape(1,-1,1) + 0.5) * dx
    yp = -Rmax + (np.arange(npix).reshape(1,1,-1) + 0.5) * dx

    x = cp*xp - sp*yp
    y = sp*xp + cp*yp
    x /= cosi
    R = (x**2 + y**2) ** 0.5

    if USE_BEAM:
        Doppler = np.ones_like(R)
        select = R > Rg
        v = (np.tile(Rg, (1,npix,npix))[select]/R[select]) ** 0.5
        sini_expand = np.tile(sini, (1,npix,npix))
        Doppler[select] = (1-v*v)**0.5 / (1-v*sini_expand[select]*y[select]/R[select])
    else:
        Doppler = np.ones_like(R)

    data = I_SS(R, Doppler, Rin, R0, slope)

    Sum = np.sum(data, (1,2))
    assert (Sum>0).all()
    data /= Sum.reshape(-1,1,1)

    if return_v2:
        return data, v2(data, SNR)
    else:
        return data


def generate_data(config, Ndata=None):

    if Ndata is None:
        Ndata = config['train_batch_size']
    R0 = np.ones(Ndata) * 50.
    Rg = np.random.rand(Ndata) * (config['prior_Rg'][1] - config['prior_Rg'][0]) + config['prior_Rg'][0]
    Rin = np.random.rand(Ndata) * (config['prior_Rin'][1] - config['prior_Rin'][0]) + config['prior_Rin'][0]
    slope = np.random.rand(Ndata) * (config['prior_slope'][1] - config['prior_slope'][0]) + config['prior_slope'][0]
    phi0 = np.random.rand(Ndata) * (config['prior_phi0'][1] - config['prior_phi0'][0]) + config['prior_phi0'][0]
    cosinc = np.random.rand(Ndata) * (config['prior_cosinc'][1] - config['prior_cosinc'][0]) + config['prior_cosinc'][0]
    inc = np.arccos(cosinc)

    npix = 512
    data = ssdisk(R0, Rin, Rg, slope, phi0, inc, npix=npix, Rmax=None, USE_BEAM=True, return_v2=False, SNR=None).reshape(Ndata, 1, npix, npix) #* npix**2

    while npix > config['image_size']:
        data = np.mean(data.reshape(Ndata, 1, npix//2, 2, npix//2, 2), (3,5))
        npix //= 2

    data = data / np.sum(data, (2,3)).reshape(-1,1,1,1) * 100

    return data.astype(np.float32)


class SSdiskDataset(Dataset):
    """SS Disk dataset."""

    def __init__(self, config, transform=None):
        """
        Arguments:
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.config = config
        self.transform = transform

    def __len__(self):
        return self.config['training_size']

    def __getitem__(self, idx):

        sample = generate_data(self.config, Ndata=1)[0]

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __getitems__(self, indices):

        sample = generate_data(self.config, Ndata=len(indices))

        if self.transform:
            sample = self.transform(sample)

        return sample

