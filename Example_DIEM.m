clear, clc, close all
%This is an example code for the use of the DIEM functions
%Latest Version --- November 13th, 2024
%Code prepared by Federico Tessari, PhD
%Newman Laboratory for Biomechanics and Human Rehabiliation, MechE, MIT

%To compute the DIEM between 2 hyper-dimensional points (or a matrix of
%them) you first need to compute the statistical distribution properties.

%This involves setting the following quantities:
%Number of Dimensions
N = 12;
%Maximum and Minimum Values of your measured quantities
minV = 0;
maxV = 1;

%Based on these, you can compute the DIEM center, min, max and orthogonal
%values with the following function:

%Set Figure Flag to '1' if you want to also have  a graphical representation
%of the DIEM distribution
fig_flag = 1;

[exp_center,vard,std_one,orth_med,min_DIEM,max_DIEM] = DIEM_Stat(N,maxV,minV,fig_flag);

%You can use the extracted DIEM statistical values to compute the DIEM
%between any pairs of hyper-dimensional quantities of dimenion 'N', maximum
%'maxV', and minimum 'minV'

%Consider the following uniformly random generated matrices
S1 = rand(N,5)*(maxV-minV)+minV;
S2 = rand(N,5)*(maxV-minV)+minV;

%Use the following code to compute the DIEM between the columsn of the two
%matrices
[DIEM, ax] = getDIEM(S1,S2,maxV,minV,exp_center,vard,'Plot','On','Text','on');