import numpy as np
np.set_printoptions(precision=4, linewidth=150)
import cv2
import matplotlib.pyplot as plt
import imageio



# papers referred:

# @tatiana1: Vol 36, no. 12, J. Opt. Soc. Am. A, D31. Code at: 
#   https://in.mathworks.com/matlabcentral/fileexchange/64143-iterative-twin-image-free-reconstruction-of-in-line-hologram

#ImgParm contains all the parameters associated with an image
from dataclasses import dataclass
@dataclass
class ImgParm:      # unit: meters
    fname: str      # filename of the image, treated as grayscale
    wavelen: float  #illumination wavelength
    z: float        # distance between object and the sensor surface.
    pixelsz: float  # pixelsize of the sensor
    thresh: float   # Threshold for controlling the mask area



# see for example @tatiana1
def propagator(N, wavelen, z, pixel_sz=0.00000112, n=1):
    
    xsize, ysize = N[0]*pixel_sz*n, N[1]*pixel_sz*n
    p = np.zeros(N, dtype=complex)
    def e(x, Nx, xsz):return (wavelen*(x - Nx/2)/xsz)**2
    for ii in range(N[0]):
        for jj in range(N[1]):
            c = e(ii, N[0], xsize) + e(jj, N[1], ysize)
            if c <= 1: p[ii][jj] = np.exp(-2j*np.pi*z*n*np.sqrt(1 - c)/wavelen)
   
    return p

# centered FTs, without centering, you have to use fftshift, ifftshift pairs for each call.
def __ft2dc(inary, inv):
    
    Nx, Ny = inary.shape
    a = np.array([[(i+j) for j in range(Ny)] for i in range(Nx)])
    f1 = np.exp(1j*np.pi*a)
    f2 = 1/f1 
    
    if inv==1: return f1*np.fft.fft2(f1*inary)
    elif inv==-1: return f2*np.fft.ifft2(f2*inary)
def ft2dc(inary): return __ft2dc(inary, 1)
def ift2dc(inary): return __ft2dc(inary, -1)

def readgray(imgfile):  
    g = cv2.imread(imgfile,0)
    g = (g - np.min(np.min(g)))/(np.max(np.max(g)) - np.min(np.min(g)))
    return np.exp(-1.6*g), -3*g 

def readamp(imgfile):
    # reads a grayscale image (as intensity on sensor) and returns amplitude
    #g = plt.imread(imgfile)
    g= cv2.imread(imgfile)
    g = (g - np.min(np.min(g)))/(np.max(np.max(g)) - np.min(np.min(g)))
    #g = np.exp(-1.6*g)
    return np.sqrt(g)

# simulates a hologram
def gen_holo(imgfile, wavelen, z, pixelsz=0.00000112):
    def transmission_function(am, ph): return am*np.exp(1j*ph)
    g, ph = readgray(imgfile)
    #plt.imshow(am)
    #plt.show()
    p = propagator(g.shape, wavelen, z, pixel_sz=pixelsz)
    t = transmission_function(g, ph)
    U = ift2dc(ft2dc(t)*np.conj(p))
    holo = np.abs(U)**2
    return holo
  
  # accessing and saving file which has to be simulated as hologram
  
import os
path= "F:/2400/"
i=1
for file in os.listdir(path):
    x=gen_holo(file,0.000000532,0.000550)
    imageio.imwrite('F:/hologram/HOLO_{}.jpg'.format(str(i).zfill(3)),x)  #saving string taking care of *enumeration* 
    print("hologram generated :",i)
    i+=1
