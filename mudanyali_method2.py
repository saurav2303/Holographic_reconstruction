import numpy as np
np.set_printoptions(precision=4, linewidth=150)

import matplotlib.pyplot as plt
import imageio

# papers referred:
# mudanyali: Lab Chip, 2010, 10, 1417–1428


#ImgParm contains all the parameters associated with an image
from dataclasses import dataclass
@dataclass
class ImgParm:      # unit: meters
    #fname: str      # filename of the image, treated as grayscale
    wavelen: float  #illumination wavelength
    z: float        # distance between object and the sensor surface.
    pixelsz: float  # pixelsize of the sensor
    thresh: float   # Threshold for controlling the mask area


oz1 = ImgParm( 0.000000532, 0.026, 0.000002, 1.8)

# see for example @tatiana1 code equation8 , can also be found in propagator.m
def propagator(N, wavelen, z, pixel_sz=0.00000112, n=1):
    #single wavelenth is good for lasers. for diodes, can we
    # use multiple propagators to improve image resolution? how?
    # see DOI: 10.1038/srep03760 Ozcan's 2014 paper in Sci.Rep.
    # To be Implemented.
    xsize, ysize = N[0]*pixel_sz*n, N[1]*pixel_sz*n
    p = np.zeros(N, dtype=complex)
    def e(x, Nx, xsz):return (wavelen*(x - Nx/2)/xsz)**2
    for ii in range(N[0]):
        for jj in range(N[1]):
            c = e(ii, N[0], xsize) + e(jj, N[1], ysize)
            if c <= 1: p[ii][jj] = np.exp(-2j*np.pi*z*n*np.sqrt(1 - c)/wavelen)
    #p = np.angle(propagator((500, 500), 0.000000532, 0.02, pixel_sz=0.000002))
    #plt.imshow(p) #plt.show()
    return p

# these arrays are needed for centering thr FTs.
f1, f2 = None, None
def fillf1f2(Nx, Ny):
    global f1, f2
    a = np.array([[(i+j) for j in range(Ny)] for i in range(Nx)])
    f1 = np.exp(1j*np.pi*a)
    f2 = 1/f1 
# centered FTs, without centering, you have to use fftshift, ifftshift pairs for each call.
def __ft2dc(inary, inv):
    global f1, f2
    Nx, Ny = inary.shape
    if inv==1: return f1*np.fft.fft2(f1*inary)
    elif inv==-1: return f2*np.fft.ifft2(f2*inary)
def ft2dc(inary): return __ft2dc(inary, 1)
def ift2dc(inary): return __ft2dc(inary, -1)

def readgray(imgfile): # @tatiana1 code , can also be found in simulation_hologram_planewave.m
    g = plt.imread(imgfile)
    g = (g - np.min(np.min(g)))/(np.max(np.max(g)) - np.min(np.min(g)))
    return np.exp(-1.6*g), -3*g 

def readamp(imgfile):
    # reads a grayscale image (as intensity on sensor) and returns amplitude
    g = plt.imread(imgfile)
    g = (g - np.min(np.min(g)))/(np.max(np.max(g)) - np.min(np.min(g)))
    #g = np.exp(-1.6*g)
    return np.sqrt(g)

# simulates a hologram, not tested well.
def gen_holo(imgfile, wavelen, z, pixelsz=0.00000112):
    def transmission_function(am, ph): return am*np.exp(1j*ph)
    g, ph = readgray(imgfile)
    plt.imshow(am)
    plt.show()
    p = propagator(g.shape, wavelen, z, pixel_sz=pixelsz)
    t = transmission_function(g, ph)
    U = ift2dc(ft2dc(t)*np.conj(p))
    holo = np.abs(U)**2
    return holo
#plt.imshow(gen_holo('ozcan-500px.jpg', 0.0000005, 0.002))
#plt.show() 

def normalize(h): return (h - np.min(np.min(h)))/(np.max(np.max(h)) - np.min(np.min(h)))
def disp_amp(am): plt.imshow(am, cmap='gray'); plt.show()

# @mudanyali method1
def meth1(img, num_iter):
    beta = 2
    from skimage import filters
    g = readamp(img.fname)
    N = g.shape
    global f1, f2
    fillf1f2(N[0], N[1])
    t, U1 = [np.zeros(N, dtype=complex) for i in range(2)]
    amp, phase, mask, maskb = [np.zeros(N) for i in range(4)]
    mask += 1
    p = propagator(N, img.wavelen, img.z, pixel_sz=img.pixelsz)
    # this propagator goes +- 2*z2.
    p1 = propagator(N, img.wavelen, 2*img.z, pixel_sz=img.pixelsz)
    t = g*np.exp(1j*phase)
    t = ft2dc(ift2dc(t)*np.conj(p))
    amp = normalize(np.abs(t))
    mask[amp < 0.4] = 0 # img.thresh*filters.threshold_otsu(amp)] = 0
    maskb = 1-mask
    U1 = mask*t
    t = U1 + maskb*(np.sum(maskb*t)/np.sum(maskb))
    D = 0j
    for x in range(num_iter):
        # propagate to virtual image
        t = ift2dc(ft2dc(t)*p1)
        amp = np.abs(t)
        # set outside of cells to DC levels, with graduation
        D = (np.sum(mask*t)/np.sum(mask))
        t = mask*(D - (D - t)/beta) + maskb*t
        # propagate to twin image plane
        t = ft2dc(ift2dc(t)*np.conj(p1))
        # set outside to U1
        t = U1 + maskb*t
    disp_amp(amp)

# @mudanyali method2
def meth2(img, holo_in, num_iter):
    
    from skimage import filters
    g = readamp(holo_in)
    N = g.shape   
    global f1, f2
    fillf1f2(N[0], N[1])
    t, D = [np.zeros(N, dtype=complex) for i in range(2)]
    amp, phase, mask = [np.zeros(N) for i in range(3)]
    mask += 1
    p = propagator(N, img.wavelen, img.z, pixel_sz=img.pixelsz)
    m = 0j, 
    for x in range(num_iter):
        t = g*np.exp(1j*phase)
        t = ift2dc(ft2dc(t)*p)
        amp = np.abs(t)
        if x==0:
            amp = normalize(amp)
            mask[amp > img.thresh*filters.threshold_otsu(amp)] = 0
            D = t*mask
        #m = (np.sum(mask*t)/mask_num) / np.mean(D)
        ## this is not entirely correrct, the top should have bg image
        m = np.mean(t*mask) / np.mean(D)   
        t = m*D + (1-mask)*t
        t = ft2dc(ift2dc(t)*np.conj(p))
        phase = np.angle(t)
    disp_amp(amp)
    return amp

for i in range(20):

    oz1.z = 0.02 + 0.002*i  # the best occurs at 2.6 cm
    meth2(oz1, 10)
    
    
