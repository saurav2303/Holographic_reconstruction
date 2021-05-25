A = imread('5um_350_Holo.jpg');
A1=cast(A,'double');
A1 = imgaussfilt(A1, 2); %Filtered image
ref = imread('5um_350_Holo.jpg'); %Noisy image

ref1=cast(ref,'double');
test =snr(ref1,ref1-A1);


% test will give the SNR value
