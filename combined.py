import numpy as np
np.set_printoptions(precision=4, linewidth=150)

import matplotlib.pyplot as plt
import imageio

# papers referred:
# mudanyali: Lab Chip, 2010, 10, 1417–1428
# tatiana1: Vol 36, no. 12, J. Opt. Soc. Am. A, D31. Code at: 
#   https://in.mathworks.com/matlabcentral/fileexchange/64143-iterative-twin-image-free-reconstruction-of-in-line-hologram
# tatiana2: arxiv: 1809.04626.pdf
#   https://in.mathworks.com/matlabcentral/fileexchange/68646-examples-of-iterative-phase-retrieval-algorithms?s_tid=prof_contriblnk
# fienup:

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

# see for example @tatiana1, equation 8.
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

def readgray(imgfile): # @tatiana1 code
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

# see @tatiana1 code section. doesn't work as well as ozcan meth1 or meth2.
def tatiana(img, num_iter): #imgfile, wavelen, z, pixelsz=0.00000112):
    g = readamp(img.fname)
    N = g.shape
    global f1, f2
    fillf1f2(N[0], N[1])
    field, field_updated, t = [np.zeros(N, dtype=complex) for i in range(3)]
    amplitude, phase, am, ph, abso = [np.zeros(N) for i in range(5)]
    p = propagator(N, img.wavelen, img.z, pixel_sz=img.pixelsz)
    r = 50
    for x in range(num_iter):
        field = g*np.exp(1j*phase)
        t = ift2dc(ft2dc(field)*p)
        am, ph = np.abs(t), np.angle(t)
        abso = -np.log(am)
        suppb = np.where(abso < 0)
        ph[suppb] = 0.0
        abso[suppb] = 0.0 # if abso[ii][jj] < 0: abso[ii][jj], ph[ii][jj] = 0, 0
        am = np.exp(-abso)
        t = am*np.exp(1j*ph)
        field_updated = ft2dc(ift2dc(t)*np.conj(p))
        phase = np.angle(field_updated)
    disp_amp(am)
    return am, ph

# @tatiana2 code
def hio(img, num_iter):
    beta, threshold = 0.9, 1
    g = readamp(img.fname)
    N = g.shape
    phase = np.zeros(N) #(2*np.random.rand(N[0], N[1]) - 1)*np.pi
    field_detector_0 = g*np.exp(1j*phase)
    global f1, f2
    fillf1f2(N[0], N[1])
    object_0 = ift2dc(field_detector_0)
    gk = np.real(object_0)
    print(np.abs(100000000*object_0))
    plt.imshow(np.abs(1000000000*object_0))
    plt.show()
    gk1 = np.zeros(gk.shape)
    support = np.zeros(N)
    R1, R2 = N[0]/4, N[1]/4
    for ii in range(N[0]):
        for jj in range(N[1]):
            x = N[0]/2 - ii
            y = N[1]/2 - jj
            if ((np.abs(x) < R1) and (np.abs(y) < R2)):
                support[ii][jj] = 1
    for x in range(num_iter):
        field_detector = ft2dc(gk)
        # Replacing updated amplitude for measured amplitude
        field_detector_updated = g*np.exp(1j*np.angle(field_detector))
        # Getting updated object distribution
        gk_prime = np.real(ift2dc(field_detector_updated))
    
        #Object constaint
        for ii in range(N[0]):
            for jj in range(N[1]):
                if (gk_prime[ii][jj] > 0) and (support[ii][jj] > 0.5):
                    gk1[ii][jj] = gk_prime[ii][jj]
                else:
                    gk1[ii][jj] = gk[ii][jj] - beta*gk_prime[ii][jj]
        plt.imshow(gk1)
        plt.show()
        # Threshold constraint, transmission<1 or absorption>0
        for ii in range(N[0]):
            for jj in range(N[1]):
                if gk1[ii][jj] > threshold: gk1[ii][jj] = threshold

        gk = gk1;

    phase = np.angle(field_detector_updated)
    field_detector = np.sqrt(g)*np.exp(1j*phase)
    object1 = np.real(ift2dc(field_detector))
    disp_amp(object1)
    return object1
         
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
    
    
