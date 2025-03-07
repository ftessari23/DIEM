% This code reproduces the results presented in the paper: "Suprassing
% Cosine Similarity for Multidimensional Comparisons: Dimension Insensitive
% Euclidean Metric"

% Latest Version --- March 7th, 2025
% Code prepared by Federico Tessari, PhD
% Newman Laboratory for Biomechanics and Human Rehabiliation, MechE, MIT


% Just hit the "Run" button and let it crunch the numbers. Depending on the
% computer, the code might take some time to run (1-3 minutes).

% Sensitivity to number of dimensions
clear, clc, close all
addpath("TextEmbeddings");
fontSize_nr = 10;
N = 2:10:102; % Wider search

vmax = 1;
vmin = 0;

dist = input('Choose a distribution type: (1) Uniform, (2) Gaussian, (3) Uniform on Unit-Sphere: ');
for i = 1:length(N)
    %Euclidian Max-Distance
    dmax_pn(i) = sqrt(N(i))*(vmax-vmin);
    dmin_pn(i) = 0;

    dmax_t(i) = sqrt(N(i))*(2*vmax-vmin);
    dmin_t(i) = 0;

    %Euclidean Distance Expected Values
    %Uniform Random Vectors
    %Positive Real (between 0 and 1)
    ev_ed_p(i) = sqrt(N(i)/6)*(vmax-vmin);
    %Negative Real (between -1 and 0)
    ev_ed_n(i) = sqrt(N(i)/6)*(vmax-vmin);
    %All Real (between -1 and 1)
    vmin = -vmax;
    ev_ed_t(i) = sqrt(N(i)/6)*(vmax-vmin);
    vmin = 0;


    for j = 1:1e4
        switch dist
            case 1
                % Uniform Distribution
                ap{i,j} = vmax*rand(N(i),1);
                an{i,j} = -vmax*rand(N(i),1);
                at{i,j} = 2*vmax*rand(N(i),1)-vmax;
                %Uniform Distribution
                bp{i,j} = vmax*rand(size(ap{i,j}));
                bn{i,j} = -vmax*rand(size(an{i,j}));
                bt{i,j} = 2*vmax*rand(size(at{i,j}))-vmax;
            case 2
                % Gaussian Distribution
                ap{i,j} = 0.3*randn(N(i),1)+vmax/2;
                an{i,j} = 0.3*randn(N(i),1)-vmax/2;
                at{i,j} = 0.6*randn(N(i),1);
                % Gaussian   Distribution
                bp{i,j} = 0.3*randn(N(i),1)+vmax/2;
                bn{i,j} = 0.3*randn(N(i),1)-vmax/2;
                bt{i,j} = 0.6*randn(N(i),1);
            case 3
                % Uniform Distribution on a Sphere
                ap{i,j} = randu_sphere(N(i),1,vmax,vmin);
                an{i,j} = randu_sphere(N(i),1,vmin,-vmax);
                at{i,j} = randu_sphere(N(i),1,vmax,-vmax);
                % Uniform Distribution on a Sphere
                bp{i,j} = randu_sphere(N(i),1,vmax,vmin);
                bn{i,j} = randu_sphere(N(i),1,vmin,-vmax);
                bt{i,j} = randu_sphere(N(i),1,vmax,-vmax);
        end

        %Cosine Similarity
        cs_tot_p(i,j) = cosSim(ap{i,j},bp{i,j});
        cs_tot_n(i,j) = cosSim(an{i,j},bn{i,j});
        cs_tot_t(i,j) = cosSim(at{i,j},bt{i,j});

        %Normalized Eucledian Distance
        d_tot_p_norm(i,j) = pdist2(ap{i}'/vecnorm(ap{i}),bp{i,j}'/vecnorm(bp{i,j}),"euclidean");
        d_tot_n_norm(i,j) = pdist2(an{i}'/vecnorm(an{i}),bn{i,j}'/vecnorm(bn{i,j}),"euclidean");
        d_tot_t_norm(i,j) = pdist2(at{i}'/vecnorm(at{i}),bt{i,j}'/vecnorm(bt{i,j}),"euclidean");

        %Eucledian Distance
        d_tot_p(i,j) = pdist2(ap{i,j}',bp{i,j}',"euclidean");
        d_tot_n(i,j) = pdist2(an{i,j}',bn{i,j}',"euclidean");
        d_tot_t(i,j) = pdist2(at{i,j}',bt{i,j}',"euclidean");

        %Cityblock (1-norm)
        c_tot_p(i,j) = pdist2(ap{i}',bp{i,j}',"cityblock");
        c_tot_n(i,j) = pdist2(an{i}',bn{i,j}',"cityblock");
        c_tot_t(i,j) = pdist2(at{i}',bt{i,j}',"cityblock");

    end
end

%Deterending Euclidian Distance on Median Value
d_tot_p_det = (vmax-vmin)*(d_tot_p - median(d_tot_p,2))./(var(d_tot_p'))';
d_tot_n_det = (vmax-vmin)*(d_tot_n - median(d_tot_n,2))./(var(d_tot_n'))';
vmin = -vmax;
d_tot_t_det = (vmax-vmin)*(d_tot_t - median(d_tot_t,2))./(var(d_tot_t'))';

%Test of Normality/ChiSqaured
for i = 1:length(N)
    for j = 1:3
        %Normality
        if j == 1
            x = d_tot_p(i,:);
            mu = mean(d_tot_p(i,:));
            % mu = ev_ed_p(i);
            sigma = std(d_tot_p(i,:));
        elseif j == 2
            x = d_tot_n(i,:);
            mu = mean(d_tot_n(i,:));
            % mu = ev_ed_n(i);
            sigma = std(d_tot_n(i,:));
        elseif j == 3
            x = d_tot_t(i,:);
            mu = mean(d_tot_t(i,:));
            % mu = ev_ed_t(i);
            sigma = std(d_tot_t(i,:));
        end
        test_cdf = [x',cdf('Normal',x',mu,sigma)];
        [h,p] = kstest(x,'CDF',test_cdf);
        H_norm(i,j) = h;
        Pval_norm(i,j) = p;
        %Chi-Squared
        [h2,p2] = chi2gof(x,'cdf',@(xx)chi2cdf(xx,mu),'nparams',1);
        H_chi(i,j) = h2;
        Pval_chi(i,j) = p2;
    end
end

% Figures
clc
%Cosine Similarity
figure('Renderer','painters')
set(gcf,'Color','white','Units','inches','Position',[3 3 4.5 3])
tt = tiledlayout(1,3);
nexttile()
boxplot(cs_tot_p',N), box off
ylim([0 1])
title('Real Positive','FontName','Times New Roman','FontSize',fontSize_nr)
nexttile()
boxplot(cs_tot_n',N), box off
ylim([0 1])
title('Real Negative','FontName','Times New Roman','FontSize',fontSize_nr)
nexttile()
boxplot(cs_tot_t',N), box off
ylim([0 1])
title('All Real','FontName','Times New Roman','FontSize',fontSize_nr)
xlabel(tt,'Dimensions','FontName','Times New Roman','FontSize',fontSize_nr)
ylabel(tt,'Cosine Similarity','FontName','Times New Roman','FontSize',fontSize_nr)

%Normalized Euclidean Distance
figure('Renderer','painters')
set(gcf,'Color','white','Units','inches','Position',[3 3 4.5 3])
tt = tiledlayout(1,3);
nexttile()
boxplot(d_tot_p_norm',N), box off
hold on
title('Real Positive','FontName','Times New Roman','FontSize',fontSize_nr)
nexttile()
boxplot(d_tot_n_norm',N), box off
hold on
title('Real Negative','FontName','Times New Roman','FontSize',fontSize_nr)
nexttile()
boxplot(d_tot_t_norm',N), box off
hold on
title('All Real','FontName','Times New Roman','FontSize',fontSize_nr)
xlabel(tt,'Dimensions','FontName','Times New Roman','FontSize',fontSize_nr)
ylabel(tt,'Norm. Euclidian Distance','FontName','Times New Roman','FontSize',fontSize_nr)


clc
% EUCLIDIAN DISTANCE ------------------------------------------------------
% Euclidean Distance without upper and lower limit
figure('Renderer','painters')
set(gcf,'Color','white','Units','inches','Position',[3 3 4.5 3])
tt = tiledlayout(1,3);
nexttile()
boxplot(d_tot_p',N), box off
hold on
title('Real Positive','FontName','Times New Roman','FontSize',fontSize_nr)
nexttile()
boxplot(d_tot_n',N), box off
hold on
title('Real Negative','FontName','Times New Roman','FontSize',fontSize_nr)
nexttile()
boxplot(d_tot_t',N), box off
hold on
title('All Real','FontName','Times New Roman','FontSize',fontSize_nr)

xlabel(tt,'Dimensions','FontName','Times New Roman','FontSize',fontSize_nr)
ylabel(tt,'Euclidean Distance','FontName','Times New Roman','FontSize',fontSize_nr)

%Euclidean Distance with upper and lower limit
figure('Renderer','painters')
set(gcf,'Color','white','Units','inches','Position',[3 3 4.5 3])
tt = tiledlayout(1,3);
nexttile()
hold on
boxplot(dmax_pn,N,'PlotStyle','compact'), hold on, boxplot(dmin_pn,N,'PlotStyle','compact'), hold on, boxplot(ev_ed_p,N,'PlotStyle','compact'), hold on
boxplot(d_tot_p',N), box off
title('Real Positive','FontName','Times New Roman','FontSize',fontSize_nr)
nexttile()
hold on
boxplot(dmax_pn,N,'PlotStyle','compact'), hold on, boxplot(dmin_pn,N,'PlotStyle','compact'), hold on, boxplot(ev_ed_n,N,'PlotStyle','compact'), hold on
boxplot(d_tot_n',N), box off
title('Real Negative','FontName','Times New Roman','FontSize',fontSize_nr)
nexttile()
hold on
boxplot(dmax_t,N,'PlotStyle','compact'), hold on, boxplot(dmin_t,N,'PlotStyle','compact'), hold on, boxplot(ev_ed_t,N,'PlotStyle','compact'), hold on
boxplot(d_tot_t',N), box off
title('All Real','FontName','Times New Roman','FontSize',fontSize_nr)
xlabel(tt,'Dimensions','FontName','Times New Roman','FontSize',fontSize_nr)
ylabel(tt,'Euclidian Distance','FontName','Times New Roman','FontSize',fontSize_nr)

clc
%Detrended Euclidean Distance
figure('Renderer','painters')
set(gcf,'Color','white','Units','inches','Position',[3 3 4.5 3])
tt = tiledlayout(1,3);
nexttile()
boxplot(d_tot_p_det',N), box off
ylim([-20 20])
title('Real Positive','FontName','Times New Roman','FontSize',fontSize_nr)
nexttile()
boxplot(d_tot_n_det',N), box off
ylim([-20 20])
title('Real Negative','FontName','Times New Roman','FontSize',fontSize_nr)
nexttile()
boxplot(d_tot_t_det',N), box off
ylim([-20 20])
title('All Real','FontName','Times New Roman','FontSize',fontSize_nr)
xlabel(tt,'Dimensions','FontName','Times New Roman','FontSize',fontSize_nr)
ylabel(tt,'DIEM','FontName','Times New Roman','FontSize',fontSize_nr)

% Histograms at Different Dimensions - Gross Search
clc
figure(),
set(gcf,'Color','white','Units','inches','Position',[3 3 4.5 3])
k = 0.1;
for NN = 1:1:5 %11
    histogram(d_tot_p(NN,:)),
    box off
    hold on
end
xlabel('Euclidean Distance','FontName','Times New Roman','FontSize',fontSize_nr)
ylabel('Frequency','FontName','Times New Roman','FontSize',fontSize_nr)
legend(strcat('n= ',num2str(N(1))),strcat('n= ',num2str(N(2))),strcat('n= ',num2str(N(3))),strcat('n= ',num2str(N(4))),strcat('n= ',num2str(N(5))),...
       strcat('n= ',num2str(N(6))),strcat('n= ',num2str(N(7))),strcat('n= ',num2str(N(8))),strcat('n= ',num2str(N(9))),strcat('n= ',num2str(N(10))),strcat('n= ',num2str(N(11))),'FontSize',fontSize_nr)

clc
%Histograms at Different Dimensions - Detrended
figure(),
set(gcf,'Color','white','Units','inches','Position',[3 3 4.5 3])
for NN = 2:6
    histogram(d_tot_p_det(NN,:)),
    box off
    hold on
end
xlabel('Euclidean Distance','FontName','Times New Roman','FontSize',fontSize_nr)
ylabel('Frequency','FontName','Times New Roman','FontSize',fontSize_nr)
legend(strcat('n= ',num2str(N(2))),strcat('n= ',num2str(N(3))),strcat('n= ',num2str(N(4))),strcat('n= ',num2str(N(5))),strcat('n= ',num2str(N(6))),'FontSize',fontSize_nr)

% Manhattan Distance
figure('Renderer','painters')
set(gcf,'Color','white','Units','inches','Position',[3 3 4.5 3])
tt = tiledlayout(1,3);
nexttile()
boxplot(c_tot_p',N), box off
title('Real Positive','FontName','Times New Roman','FontSize',fontSize_nr)
nexttile()
boxplot(c_tot_n',N), box off
title('Real Negative','FontName','Times New Roman','FontSize',fontSize_nr)
nexttile()
boxplot(c_tot_t',N), box off
title('All Real','FontName','Times New Roman','FontSize',fontSize_nr)

xlabel(tt,'Dimensions','FontName','Times New Roman','FontSize',fontSize_nr)
ylabel(tt,'Manhattan Distance','FontName','Times New Roman','FontSize',fontSize_nr)

% Text-embedding Case-Study
clear, clc
%Optimal Figure Setting to use in any script
set(0,'defaultaxesfontname','Times New Roman');
set(0,'defaulttextfontname','Times New Roman');
set(0,'defaultaxesfontsize',12); % 8 for paper images 12 for normal images
set(0,'defaulttextfontsize',12); % 8 for paper images 12 for normal images

% Data Loading
% Get the current working directory
currentFolder = pwd;
% Construct file paths dynamically
filePath1 = fullfile(currentFolder, 'TextEmbeddings', 'test-00000-of-00001.parquet');
filePath2 = fullfile(currentFolder, 'TextEmbeddings', 'train-00000-of-00001.parquet');
filePath3 = fullfile(currentFolder, 'TextEmbeddings', 'validation-00000-of-00001.parquet');
% Read the Parquet file
data1 = parquetread(filePath1);
data2 = parquetread(filePath2);
data3 = parquetread(filePath3);

dataT = [data1;data2;data3];
score = [data1.score;data2.score;data3.score];

%Load Embeddings - embedding computed with 'all-MiniLM-L6-v2'
emb1 = importdata('embeddings1.csv');
sent1 = emb1(2:end,:);
emb2 = importdata('embeddings2.csv');
sent2 = emb2(2:end,:);

% Similarity Analysis
%Cosine Similarity
cosineSimilarity = getCosineSimilarity(sent1',sent2','Plot','off');
%DIEM Similarity
%This involves setting the following quantities:
%Number of Dimensions
N = 384;
%Maximum and Minimum Values of your measured quantities
minV = min(min(sent1));
maxV = max(max(sent2));
%Based on these, you can compute the DIEM center, min, max and orthogonal
%values with the following function:
%Set Figure Flag to '1' if you want to also have  a graphical representation
%of the DIEM distribution
fig_flag = 0;
[exp_center,vard,std_one,orth_med,min_DIEM,max_DIEM] = DIEM_Stat(N,maxV,minV,fig_flag);

%You can use the extracted DIEM statistical values to compute the DIEM
%between any pairs of hyper-dimensional quantities of dimenion 'N', maximum
%'maxV', and minimum 'minV'

%Use the following code to compute the DIEM between the columsn of the two
%matrices
DIEM = getDIEM(sent1',sent2',maxV,minV,exp_center,vard,'Plot','off','Text','off');

% Take only the rated Similarities
cos_Sim = diag(cosineSimilarity);
DIEMSim = diag(DIEM);

% Text Embedding Figure
clc
figure('Color','white')
tt = tiledlayout(2,2);
ylabel(tt,'Percentage Frequency','FontName','Times New Roman')

nexttile()
histogram(cos_Sim,'Normalization','percentage','EdgeColor','none'), hold on
% histogram(cosineSimilarity(:),'Normalization','percentage'), hold on
plot([0 0],[0 max(histcounts(cos_Sim(:),'Normalization','percentage'))],'k--','LineWidth',2), hold on
plot([1 1],[0 max(histcounts(cos_Sim(:),'Normalization','percentage'))],'k--','LineWidth',2), hold on
% plot([0.75 0.75],[0 max(histcounts(cosineSimilarity(:),'Normalization','percentage'))],'r--','LineWidth',2), hold on
box off
set(gca, 'XDir','reverse')
title('Cosine Similarity')
subtitle('Pair-wise Comparisons')

nexttile()
histogram(DIEMSim,'Normalization','percentage','EdgeColor','none'), hold on
% histogram(DIEM(:),'Normalization','percentage'), hold on
plot([min_DIEM min_DIEM],[0 max(histcounts(DIEMSim(:),'Normalization','percentage'))],'k--','LineWidth',2), hold on
plot([max(DIEM(:)) max(DIEM(:))],[0 max(histcounts(DIEMSim(:),'Normalization','percentage'))],'k--','LineWidth',2), hold on
xlim([1.03*min_DIEM 0.9*max(DIEM(:))])
box off
title('DIEM')
subtitle('Pair-wise Comparisons')

nexttile()
% histogram(cos_Sim,'Normalization','percentage'), hold on
histogram(cosineSimilarity(:),'Normalization','percentage','EdgeColor','none'), hold on
plot([0 0],[0 max(histcounts(cosineSimilarity(:),'Normalization','percentage'))],'k--','LineWidth',2), hold on
plot([1 1],[0 max(histcounts(cosineSimilarity(:),'Normalization','percentage'))],'k--','LineWidth',2), hold on
% plot([0.75 0.75],[0 max(histcounts(cosineSimilarity(:),'Normalization','percentage'))],'r--','LineWidth',2), hold on
set(gca, 'XDir','reverse')
box off
subtitle('All-possible Comparisons')

nexttile()
% histogram(DIEMSim,'Normalization','percentage'), hold on
histogram(DIEM(:),'Normalization','percentage','EdgeColor','none'), hold on
plot([min_DIEM min_DIEM],[0 max(histcounts(DIEM(:),'Normalization','percentage'))],'k--','LineWidth',2), hold on
plot([max(DIEM(:)) max(DIEM(:))],[0 max(histcounts(DIEM(:),'Normalization','percentage'))],'k--','LineWidth',2), hold on
xlim([1.03*min_DIEM 0.9*max(DIEM(:))])
box off
subtitle('All-possible Comparisons')

% Statistical Testing
[h_0_DIEM,p_0_DIEM] = ztest(DIEM(:),0,std_one);


% SUPPLEMENTARY MATERIAL FIGURES

% Cosine Similarity and Euclidian Distance Relationship
a = [2 0]';
b = [0.1 2]';
norm_a = vecnorm(a);
norm_b = vecnorm(b);
cs = cosSim(a,b);
dist = pdist2(a',b',"euclidean");
%Relation Between Eucledian Distance and Cosine Similarity
cs_test = abs(((vecnorm(a)^2+vecnorm(b)^2-dist^2)/2))/(vecnorm(a)*vecnorm(b));

figure()
set(gcf,'Color','white')
d = 0:0.01:2;
cs_d = abs(1-d.^2/2);
plot(d,cs_d,'k','LineWidth',2), box off
xlabel('Euclidian Distance')
ylabel('Cosine Similarity')

% Histograms at Different Dimensions - Finer Search
% Sensitivity to number of dimensions
clear, clc
fontSize_nr = 10;
N = 2:1:12; % Tighter search

vmax = 1;
vmin = 0;

dist = input('Choose a distribution type: (1) Uniform, (2) Gaussian, (3) Uniform on Unit-Sphere: ');
for i = 1:length(N)
    %Euclidian Max-Distance
    dmax_pn(i) = sqrt(N(i))*(vmax-vmin);
    dmin_pn(i) = 0;

    dmax_t(i) = sqrt(N(i))*(2*vmax-vmin);
    dmin_t(i) = 0;

    %Euclidean Distance Expected Values
    %Uniform Random Vectors
    %Positive Real (between 0 and 1)
    ev_ed_p(i) = sqrt(N(i)/6)*(vmax-vmin);
    %Negative Real (between -1 and 0)
    ev_ed_n(i) = sqrt(N(i)/6)*(vmax-vmin);
    %All Real (between -1 and 1)
    vmin = -vmax;
    ev_ed_t(i) = sqrt(N(i)/6)*(vmax-vmin);
    vmin = 0;


    for j = 1:1e4
        switch dist
            case 1
                % Uniform Distribution
                ap{i,j} = vmax*rand(N(i),1);
                an{i,j} = -vmax*rand(N(i),1);
                at{i,j} = 2*vmax*rand(N(i),1)-vmax;
                %Uniform Distribution
                bp{i,j} = vmax*rand(size(ap{i,j}));
                bn{i,j} = -vmax*rand(size(an{i,j}));
                bt{i,j} = 2*vmax*rand(size(at{i,j}))-vmax;
            case 2
                % Gaussian Distribution
                ap{i,j} = 0.3*randn(N(i),1)+vmax/2;
                an{i,j} = 0.3*randn(N(i),1)-vmax/2;
                at{i,j} = 0.6*randn(N(i),1);
                % Gaussian   Distribution
                bp{i,j} = 0.3*randn(N(i),1)+vmax/2;
                bn{i,j} = 0.3*randn(N(i),1)-vmax/2;
                bt{i,j} = 0.6*randn(N(i),1);
            case 3
                % Uniform Distribution on a Sphere
                ap{i,j} = randu_sphere(N(i),1,vmax,vmin);
                an{i,j} = randu_sphere(N(i),1,vmin,-vmax);
                at{i,j} = randu_sphere(N(i),1,vmax,-vmax);
                % Uniform Distribution on a Sphere
                bp{i,j} = randu_sphere(N(i),1,vmax,vmin);
                bn{i,j} = randu_sphere(N(i),1,vmin,-vmax);
                bt{i,j} = randu_sphere(N(i),1,vmax,-vmax);
        end
        %Eucledian Distance
        d_tot_p(i,j) = pdist2(ap{i,j}',bp{i,j}',"euclidean");
        d_tot_n(i,j) = pdist2(an{i,j}',bn{i,j}',"euclidean");
        d_tot_t(i,j) = pdist2(at{i,j}',bt{i,j}',"euclidean");

    end
end

%Deterending Euclidian Distance on Median Value
d_tot_p_det = (vmax-vmin)*(d_tot_p - median(d_tot_p,2))./(var(d_tot_p'))';
d_tot_n_det = (vmax-vmin)*(d_tot_n - median(d_tot_n,2))./(var(d_tot_n'))';
vmin = -vmax;
d_tot_t_det = (vmax-vmin)*(d_tot_t - median(d_tot_t,2))./(var(d_tot_t'))';

figure(),
set(gcf,'Color','white','Units','inches','Position',[3 3 4.5 3])
k = 0.1;
for NN = 1:1:11
    histogram(d_tot_p(NN,:),'FaceColor',[2*k-0.2 0 1],'FaceAlpha',k),
    box off
    hold on
    k = k+0.05;
end
xlabel('Euclidean Distance','FontName','Times New Roman','FontSize',fontSize_nr)
ylabel('Frequency','FontName','Times New Roman','FontSize',fontSize_nr)
legend(strcat('n= ',num2str(N(1))),strcat('n= ',num2str(N(2))),strcat('n= ',num2str(N(3))),strcat('n= ',num2str(N(4))),strcat('n= ',num2str(N(5))),...
       strcat('n= ',num2str(N(6))),strcat('n= ',num2str(N(7))),strcat('n= ',num2str(N(8))),strcat('n= ',num2str(N(9))),strcat('n= ',num2str(N(10))),strcat('n= ',num2str(N(11))),'FontSize',fontSize_nr)

clear, clc
%Comparing 2D points generated from three different distribution
%Number of points
N = 5e2;
%Number of dimension
n = 2;
%Uniform
x_u = 2*rand(N,n)-1;
%Gaussian
x_g = 0.3*randn(N,n);
%Uniform Unitary Sphere
x_us = randu_sphere(n,N,1,-1);

figure(),
set(gcf,'Color','white')
tiledlayout(1,3)
nexttile()
plot(x_u(:,1),x_u(:,2),'.r'), 
axis equal
box off
xlabel('x_1','FontName','Times New Roman')
ylabel('x_2','FontName','Times New Roman')
xlim([-1 1])
ylim([-1 1])
title('Uniform','FontName','Times New Roman')
nexttile()
plot(x_g(:,1),x_g(:,2),'.b'), 
axis equal
box off
xlabel('x_1','FontName','Times New Roman')
ylabel('x_2','FontName','Times New Roman')
xlim([-1 1])
ylim([-1 1])
title('Gaussian','FontName','Times New Roman')
nexttile()
plot(x_us(1,:),x_us(2,:),'.k'), 
axis equal
box off
xlabel('x_1','FontName','Times New Roman')
ylabel('x_2','FontName','Times New Roman')
xlim([-1 1])
ylim([-1 1])
title('Uniform Uni-sphere','FontName','Times New Roman')


function cs = cosSim(a,b) %Vectors should be oriented as columns
    norm_a = vecnorm(a);
    norm_b = vecnorm(b);

    cs = ((a'*b)/(norm_a*norm_b));
end

