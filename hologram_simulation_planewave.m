%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SIMULATION OF IN-LINE HOLOGRAM WITH PLANE WAVES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Citation for this code/algorithm or any of its parts:
% Tatiana Latychevskaia and Hans-Werner Fink
% "Solution to the Twin Image Problem in Holography",
% Physical Review Letters 98, 233901 (2007)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The code is written by Tatiana Latychevskaia, 2007
% The version of Matlab for this code is R2010b

clear all
close all
% addpath('C:/Program Files/MATLAB/R2010b/myfiles');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Parameters

N = 500;                    % number of pixels
lambda = 532*10^(-9);       % wavelength in meter
object_area = 0.002;        % object area sidelength in meter
z = 0.05;                   % object-to-detector distance in meter
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Creating object plane transmission function

object = zeros(N,N);        
    object0 = imread('500x500-100.jpg');   
    object(:,:) = object0(:,:,1);    
    object = (object - min(min(object)))/(max(max(object)) - min(min(object)));    
    figure, imshow(object, []);  
    
am = exp(-1.6*object);    % exp(-1.6)=0.2
ph = - 3*object;

t = zeros(N,N);
t = am.*exp(i*ph);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Simulating hologram

prop = Propagator(N, lambda, object_area, z);
U = IFT2Dc(FT2Dc(t).*conj(prop));
hologram = abs(U).^2;

figure('Name','Simulated hologram','NumberTitle','off')
imshow(hologram, []), colorbar;
colormap(gray)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Saving hologram

fid = fopen(strcat('a_hologram.bin'), 'w');
fwrite(fid, hologram, 'real*4');
fclose(fid);

p = hologram;
p = 255*(p - min(min(p)))/(max(max(p)) - min(min(p)));
imwrite (p, gray, 'a_hologram.bmp');
