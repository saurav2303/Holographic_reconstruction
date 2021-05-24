6%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ITERATIVE TWIN-IMAGE-FREE RECONSTRUCTION OF IN-LINE HOLOGRAM
% ACQUIRED WITH PLANE WAVE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Citation for this code/algorithm or any of its parts:
% Tatiana Latychevskaia and Hans-Werner Fink
% "Solution to the Twin Image Problem in Holography",
% Physical Review Letters 98, 233901 (2007)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The code is written by Tatiana Latychevskaia, 2007
% The version of Matlab for this code is R2010b

close all
clear all

% addpath('C:/Program Files/MATLAB/R2010b/myfiles');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N = 500;                    % number of pixels
Iterations = 50;            % number of iterations

wavelength = 532*10^(-9);   % wavelength in meter
area_size = 0.002;          % area size = object area size, in meter
z = 0.05;                   % object-to-detector distance in meter

p = 0.01;                   % time to pause, otherwise images are not shown
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Reading hologram

    fid = fopen('a_hologram.bin', 'r');
    hologram = fread(fid, [N, N], 'real*4');
    fclose(fid);   
    measured = sqrt(hologram);

% Creating initial complex-valued field distribution in the detector plane
phase = zeros(N,N);
field_detector = measured.*exp(i*phase);

% Creating wave propagation term
prop = Propagator(N, wavelength, area_size, z);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Iterative reconstruction
 
figure('Name','Current amplitude = exp(-absorption) / a.u.','NumberTitle','off')
    
for kk = 1:Iterations

fprintf('Iteration: %d\n', kk)
field_detector = measured.*exp(i*phase);

% Reconstruction of transmission function t
t = IFT2Dc((FT2Dc(field_detector)).*prop);

am = abs(t);
ph = angle(t);
abso = - log(am);

% Constraint in the object domain
for ii = 1:N
    for jj = 1:N
        if ((abso(ii,jj) < 0))
            abso(ii,jj) = 0;
            ph(ii,jj)=0;
        end
    end
end

am = exp(-abso);

imshow(am, [],'colormap', 1-gray)
pause(p);

t = zeros(N,N);
for ii = 1:N
    for jj = 1:N
        t(ii,jj) = complex(am(ii,jj)*cos(ph(ii,jj)), am(ii,jj)*sin(ph(ii,jj))); 
    end
end

% Calculating complex-valued wavefront in the detector plane
field_detector_updated = FT2Dc((IFT2Dc(t)).*conj(prop));

amplitude = abs(field_detector_updated);
phase = angle(field_detector_updated);

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Showing reconsrtucted amplitude
    figure('Name','Reconstructed amplitude = exp(-absorption) / a.u.','NumberTitle','off')
    imshow(am, [],'colormap', 1-gray), colorbar;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Showing reconsrtucted phase
    figure('Name','Reconstructed phase / radian','NumberTitle','off')
    imshow(ph, [],'colormap', 1-gray), colorbar;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Saving reconsrtucted amplitude as jpg image
        h = - am;
        h = (h - min(min(h)))/(max(max(h)) - min(min(h)));
        imwrite (h, 'reconstructed_amplitude.jpg');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Saving reconsrtucted phase as jpg image
        h = - ph;
        h = (h - min(min(h)))/(max(max(h)) - min(min(h)));
        imwrite (h, 'reconstructed_phase.jpg');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
