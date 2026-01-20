% This code reproduces the results presented in the paper: "Suprassing
% Cosine Similarity for Multidimensional Comparisons: Dimension Insensitive
% Euclidean Metric"

% Latest Version --- January 15th, 2026
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

% Single combined figure: 2x3 (a) Cosine Similarity on top, (b) Norm. Eucl. Distance on bottom
clc

% ---- Readability settings (tune if needed) ----
fontName = 'Times New Roman';
fsAxes   = 8;     % tick labels
fsTitle  = 9;     % tile titles
fsLabel  = 9;     % axis labels / panel labels
lw       = 0.8;   % axes line width

% ---- Figure (width fixed at 3.5 in; compact height) ----
figH = 3.2; % adjust down if still too tall (try 3.0 / 2.8 depending on labels)
figure('Renderer','painters');
set(gcf,'Color','white','Units','inches','Position',[3 3 3.5 figH]);

% Set defaults for this figure
set(gcf,'DefaultAxesFontName',fontName, ...
        'DefaultTextFontName',fontName, ...
        'DefaultAxesFontSize',fsAxes, ...
        'DefaultTextFontSize',fsLabel, ...
        'DefaultAxesLineWidth',lw);

tt = tiledlayout(2,3,'TileSpacing','compact','Padding','compact');

% =========================
% (a) Cosine Similarity (top row)
% =========================
ax1 = nexttile(1);
boxplot(cs_tot_p', N); box off
ylim([0 1])
title('Real Positive','FontSize',fsTitle)
set(ax1,'TickDir','out');

ax2 = nexttile(2);
boxplot(cs_tot_n', N); box off
ylim([0 1])
title('Real Negative','FontSize',fsTitle)
set(ax2,'TickDir','out');

ax3 = nexttile(3);
boxplot(cs_tot_t', N); box off
ylim([0 1])
title('All Real','FontSize',fsTitle)
set(ax3,'TickDir','out');

% Common labels for top row (keep compact: only ylabel on first, xlabel omitted)
ylabel(ax1,'Cosine Similarity','FontSize',fsLabel);

% =========================
% (b) Norm. Euclidean Distance (bottom row)
% =========================
ax4 = nexttile(4);
boxplot(d_tot_p_norm', N); box off
ylim([0 1])
% title('Real Positive','FontSize',fsTitle)
set(ax4,'TickDir','out');

ax5 = nexttile(5);
boxplot(d_tot_n_norm', N); box off
ylim([0 1])
% title('Real Negative','FontSize',fsTitle)
set(ax5,'TickDir','out');

ax6 = nexttile(6);
boxplot(d_tot_t_norm', N); box off
% title('All Real','FontSize',fsTitle)
set(ax6,'TickDir','out');

% Common labels for bottom row (ylabel on first; xlabel on middle to avoid height bloat)
ylabel(ax4,'Norm. Eucl. Distance','FontSize',fsLabel);
xlabel(ax5,'Dimensions','FontSize',fsLabel);

% ---- Panel labels (a) and (b) ----
% Place them just above the leftmost tile in each row
text(ax1, -0.18, 1.10, '(a)', 'Units','normalized', ...
    'FontName',fontName,'FontSize',fsLabel,'FontWeight','bold');
text(ax4, -0.18, 1.10, '(b)', 'Units','normalized', ...
    'FontName',fontName,'FontSize',fsLabel,'FontWeight','bold');

% ---- Optional: reduce clutter if many categories ----
% set([ax4 ax5 ax6],'XTickLabelRotation',45);  % rotate only bottom row if needed
% set([ax1 ax2 ax3],'XTickLabel',[]);          % hide top x tick labels entirely (more compact)

% ---- Export (vector PDF for Overleaf) ----
% exportgraphics(gcf,'ed_dim_1and2.pdf','ContentType','vector');

% EUCLIDEAN DISTANCE
% Single combined figure: 2x3 (a) Euclidean Distance on top, (b) Eucl.
% Distance with Bounds on bottom
clc

% ---- Readability / publication settings ----
fontName = 'Times New Roman';
fsAxes   = 8;
fsTitle  = 9;
fsLabel  = 9;
lw       = 0.8;

% ---- Figure (fixed width, compact height) ----
figH = 3.2;   % try 3.0 if you want it even tighter
figure('Renderer','painters');
set(gcf,'Color','white','Units','inches','Position',[3 3 3.5 figH]);

set(gcf,'DefaultAxesFontName',fontName, ...
        'DefaultTextFontName',fontName, ...
        'DefaultAxesFontSize',fsAxes, ...
        'DefaultTextFontSize',fsLabel, ...
        'DefaultAxesLineWidth',lw);

tt = tiledlayout(2,3,'TileSpacing','compact','Padding','compact');

% =========================================================
% (a) Euclidean Distance – no upper / lower limits (top row)
% =========================================================
ax1 = nexttile(1);
boxplot(d_tot_p',N); box off
title('Real Positive','FontSize',fsTitle)
set(ax1,'TickDir','out');

ax2 = nexttile(2);
boxplot(d_tot_n',N); box off
title('Real Negative','FontSize',fsTitle)
set(ax2,'TickDir','out');

ax3 = nexttile(3);
boxplot(d_tot_t',N); box off
title('All Real','FontSize',fsTitle)
set(ax3,'TickDir','out');

ylabel(ax1,'Euclidean Distance','FontSize',fsLabel);

% Panel label
text(ax1,-0.18,1.10,'(a)','Units','normalized', ...
    'FontSize',fsLabel,'FontWeight','bold');

% =========================================================
% (b) Euclidean Distance – with upper / lower bounds (bottom)
% =========================================================
ax4 = nexttile(4);
hold on
boxplot(dmax_pn,N,'PlotStyle','compact')
boxplot(dmin_pn,N,'PlotStyle','compact')
boxplot(ev_ed_p,N,'PlotStyle','compact')
boxplot(d_tot_p',N)
box off
% title('Real Positive','FontSize',fsTitle)
set(ax4,'TickDir','out');

ax5 = nexttile(5);
hold on
boxplot(dmax_pn,N,'PlotStyle','compact')
boxplot(dmin_pn,N,'PlotStyle','compact')
boxplot(ev_ed_n,N,'PlotStyle','compact')
boxplot(d_tot_n',N)
box off
% title('Real Negative','FontSize',fsTitle)
set(ax5,'TickDir','out');

ax6 = nexttile(6);
hold on
boxplot(dmax_t,N,'PlotStyle','compact')
boxplot(dmin_t,N,'PlotStyle','compact')
boxplot(ev_ed_t,N,'PlotStyle','compact')
boxplot(d_tot_t',N)
box off
% title('All Real','FontSize',fsTitle)
set(ax6,'TickDir','out');

ylabel(ax4,'Euclidean Distance','FontSize',fsLabel);
xlabel(ax5,'Dimensions','FontSize',fsLabel);

% Panel label
text(ax4,-0.18,1.10,'(b)','Units','normalized', ...
    'FontSize',fsLabel,'FontWeight','bold');

% ---- Optional clutter control ----
% set([ax4 ax5 ax6],'XTickLabelRotation',45);
% set([ax1 ax2 ax3],'XTickLabel',[]);

% ---- Export for Overleaf (vector PDF) ----
% exportgraphics(gcf,'ed_dim_1and2.pdf','ContentType','vector');

%
clc
% Detrended Euclidean Distance (DIEM) — readable at 3.5 in width
figure('Renderer','painters')
set(gcf,'Color','white','Units','inches','Position',[3 3 3.5 2.4])   % width fixed, compact height

% Fonts / axes style (match other figures)
fontName = 'Times New Roman';
fsAxes   = fontSize_nr - 1;   % slightly smaller tick labels
fsTitle  = fontSize_nr;       % subplot titles
fsLabel  = fontSize_nr;

set(gcf,'DefaultAxesFontName',fontName, ...
        'DefaultTextFontName',fontName, ...
        'DefaultAxesFontSize',fsAxes, ...
        'DefaultAxesLineWidth',0.8);

tt = tiledlayout(1,3,'TileSpacing','compact','Padding','compact');

ax1 = nexttile(1);
boxplot(d_tot_p_det', N); box off
ylim([-20 20])
title('Real Positive','FontSize',fsTitle)
set(ax1,'TickDir','out');

ax2 = nexttile(2);
boxplot(d_tot_n_det', N); box off
ylim([-20 20])
title('Real Negative','FontSize',fsTitle)
set(ax2,'TickDir','out');

ax3 = nexttile(3);
boxplot(d_tot_t_det', N); box off
ylim([-20 20])
title('All Real','FontSize',fsTitle)
set(ax3,'TickDir','out');

% Common labels (outside)
xlabel(tt,'Dimensions','FontSize',fsLabel,'FontName',fontName)
ylabel(tt,'DIEM','FontSize',fsLabel,'FontName',fontName)

% Optional: rotate x tick labels if crowded
% set([ax1 ax2 ax3],'XTickLabelRotation',45);

% Export (vector PDF for Overleaf)
% exportgraphics(gcf,'DIEM_boxplot_v2.pdf','ContentType','vector');


% ---- Figure: Histograms -----
figure('Renderer','painters')
set(gcf,'Color','white','Units','inches','Position',[3 3 3.5 2.8])

tt = tiledlayout(1,2,'TileSpacing','compact','Padding','compact');

% =========================
% Left: Histograms at Different Dimensions - Gross Search
% =========================
ax1 = nexttile(1); hold on; box off
hL = gobjects(0);
for NN = 1:5
    hL(end+1) = histogram(d_tot_p(NN,:), ...
        'EdgeColor','none');   % keep fill, remove bin edges
end
title('(a)','FontName','Times New Roman','FontSize',fontSize_nr)
set(ax1,'TickDir','out')

% =========================
% Right: Histograms at Different Dimensions - Detrended
% =========================
ax2 = nexttile(2); hold on; box off
hR = gobjects(0);
for NN = 2:6
    hR(end+1) = histogram(d_tot_p_det(NN,:), ...
        'EdgeColor','none');   % keep fill, remove bin edges
end
title('(b)','FontName','Times New Roman','FontSize',fontSize_nr)
set(ax2,'TickDir','out')

% IMPORTANT: do NOT link axes (independent x-limits preserved)
% linkaxes([ax1 ax2],'xy');   % <-- intentionally removed

% =========================
% Common labels (outside)
% =========================
xlabel(tt,'Euclidean Distance','FontName','Times New Roman','FontSize',fontSize_nr)
ylabel(tt,'Frequency','FontName','Times New Roman','FontSize',fontSize_nr)

% =========================
% Single shared legend (bottom center)
% =========================
% Use handles from LEFT plot to define colors/order
legStr = strcat('n = ', string(N(1:5)));

lgd = legend(hL, legStr, ...
    'Orientation','horizontal', ...
    'Location','southoutside', ...
    'Box','off', ...
    'FontSize',fontSize_nr-1);

lgd.NumColumns = ceil(numel(hL)/2);   % forces ~2 rows
lgd.Layout.Tile = 'south';

% ---- Export (vector PDF for Overleaf) ----
% exportgraphics(gcf,'hist_ed_DIEM.pdf','ContentType','vector');

% Manhattan Distance
clc

% Manhattan Distance boxplots (readable at 3.5 in width, Overleaf-safe)
figure('Renderer','painters')
set(gcf,'Color','white','Units','inches','Position',[3 3 3.5 2.4])   % width fixed, compact height

% Fonts / axes style
fontName = 'Times New Roman';
fsAxes   = fontSize_nr - 1;   % slightly smaller tick labels
fsTitle  = fontSize_nr;       % keep titles readable
fsLabel  = fontSize_nr;

set(gcf,'DefaultAxesFontName',fontName, ...
        'DefaultTextFontName',fontName, ...
        'DefaultAxesFontSize',fsAxes, ...
        'DefaultAxesLineWidth',0.8);

tt = tiledlayout(1,3,'TileSpacing','compact','Padding','compact');

ax1 = nexttile(1);
boxplot(c_tot_p',N); box off
title('Real Positive','FontSize',fsTitle)
set(ax1,'TickDir','out');

ax2 = nexttile(2);
boxplot(c_tot_n',N); box off
title('Real Negative','FontSize',fsTitle)
set(ax2,'TickDir','out');

ax3 = nexttile(3);
boxplot(c_tot_t',N); box off
title('All Real','FontSize',fsTitle)
set(ax3,'TickDir','out');

% Common labels (outside)
xlabel(tt,'Dimensions','FontSize',fsLabel,'FontName',fontName)
ylabel(tt,'Manhattan Distance','FontSize',fsLabel,'FontName',fontName)

% Optional: if N labels are crowded at 3.5", rotate tick labels
% set([ax1 ax2 ax3],'XTickLabelRotation',45);

% Export (vector PDF for Overleaf)
% exportgraphics(gcf,'manhattan_boxplots.pdf','ContentType','vector');

%
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

% Generate Random Vectors
vmax = maxV;
vmin = minV;

for j = 1:1e4
    % Uniform Distribution
    at(:,j) = (vmax-vmin)*rand(N,1)+vmin;
    %Uniform Distribution
    bt(:,j) = (vmax-vmin)*rand(N,1)+vmin;

    %Cosine Similarity
    cs_rand(j) = cosSim(at(:,j),bt(:,j));
    DIEM_rand(j) = getDIEM(at(:,j),bt(:,j),maxV,minV,exp_center,vard,'Plot','off','Text','off');
end

% Text Embedding Figure
clc

% ---- Figure setup (single-column, Overleaf-safe) ----
figW = 3.5;
figH = 4;   % compact but still readable for 3x2
figure('Renderer','painters')
set(gcf,'Color','white','Units','inches','Position',[3 2 figW figH])

% ---- Fonts & defaults ----
fontName = 'Times New Roman';
fsAxes   = 7.5;
fsTitle  = 9;
fsSub    = 8;
lw       = 1.2;

set(gcf,'DefaultAxesFontName',fontName, ...
        'DefaultTextFontName',fontName, ...
        'DefaultAxesFontSize',fsAxes, ...
        'DefaultAxesLineWidth',0.75);

tt = tiledlayout(3,2,'TileSpacing','compact','Padding','compact');
ylabel(tt,'Percentage Frequency','FontSize',fsTitle,'FontName',fontName)

% =========================
% Row 1 — Pair-wise
% =========================
ax1 = nexttile(); hold on; box off
histogram(cos_Sim,'Normalization','percentage','EdgeColor','none')
ymax = max(histcounts(cos_Sim(:),'Normalization','percentage'));
plot([0 0],[0 ymax],'k--','LineWidth',lw)
plot([1 1],[0 ymax],'k--','LineWidth',lw)
xlim([0 1.1])
set(gca,'XDir','reverse','TickDir','out')
title('Cosine Similarity','FontSize',fsTitle)
subtitle('Pair-wise','FontSize',fsSub)

ax2 = nexttile(); hold on; box off
histogram(DIEMSim,'Normalization','percentage','EdgeColor','none')
ymax = max(histcounts(DIEMSim(:),'Normalization','percentage'));
plot([min_DIEM min_DIEM],[0 ymax],'k--','LineWidth',lw)
plot([max(DIEM_rand(:)) max(DIEM_rand(:))],[0 ymax],'k--','LineWidth',lw)
xlim([1.03*min_DIEM 1.1*max(DIEM_rand(:))])
set(gca,'TickDir','out')
title('DIEM','FontSize',fsTitle)
subtitle('Pair-wise','FontSize',fsSub)

% =========================
% Row 2 — All comparisons
% =========================
ax3 = nexttile(); hold on; box off
histogram(cosineSimilarity(:),'Normalization','percentage','EdgeColor','none')
ymax = max(histcounts(cosineSimilarity(:),'Normalization','percentage'));
plot([0 0],[0 ymax],'k--','LineWidth',lw)
plot([1 1],[0 ymax],'k--','LineWidth',lw)
xlim([0 1.1])
set(gca,'XDir','reverse','TickDir','out')
subtitle('All comparisons','FontSize',fsSub)

ax4 = nexttile(); hold on; box off
histogram(DIEM(:),'Normalization','percentage','EdgeColor','none')
ymax = max(histcounts(DIEM(:),'Normalization','percentage'));
plot([min_DIEM min_DIEM],[0 ymax],'k--','LineWidth',lw)
plot([max(DIEM_rand(:)) max(DIEM_rand(:))],[0 ymax],'k--','LineWidth',lw)
xlim([1.03*min_DIEM 1.1*max(DIEM_rand(:))])
set(gca,'TickDir','out')
subtitle('All comparisons','FontSize',fsSub)

% =========================
% Row 3 — Random
% =========================
ax5 = nexttile(); hold on; box off
histogram(cs_rand(:),'Normalization','percentage','EdgeColor','none')
ymax = max(histcounts(cs_rand(:),'Normalization','percentage'));
plot([0 0],[0 ymax],'k--','LineWidth',lw)
plot([1 1],[0 ymax],'k--','LineWidth',lw)
xlim([0 1.1])
set(gca,'XDir','reverse','TickDir','out')
subtitle('Random','FontSize',fsSub)

ax6 = nexttile(); hold on; box off
histogram(DIEM_rand(:),'Normalization','percentage','EdgeColor','none')
ymax = max(histcounts(DIEM_rand(:),'Normalization','percentage'));
plot([min_DIEM min_DIEM],[0 ymax],'k--','LineWidth',lw)
plot([max(DIEM_rand(:)) max(DIEM_rand(:))],[0 ymax],'k--','LineWidth',lw)
xlim([1.03*min_DIEM 1.1*max(DIEM_rand(:))])
set(gca,'TickDir','out')
subtitle('Random','FontSize',fsSub)

% ---- Common x-label (bottom only) ----
xlabel(tt,'Similarity / Distance','FontSize',fsTitle,'FontName',fontName)

% ---- Export (vector, Overleaf) ----
% exportgraphics(gcf,'cossim_vs_DIEM_LLM_v4.pdf','ContentType','vector');


% Statistical Testing
[h_0_DIEM,p_0_DIEM] = ztest(DIEM(:),0,std_one);


% SUPPLEMENTARY MATERIAL FIGURES

clear, clc

% % Cosine Similarity and Euclidian Distance Relationship
% a = [2 0]';
% b = [0.1 2]';
% norm_a = vecnorm(a);
% norm_b = vecnorm(b);
% cs = cosSim(a,b);
% dist = pdist2(a',b',"euclidean");
% %Relation Between Eucledian Distance and Cosine Similarity
% cs_test = abs(((vecnorm(a)^2+vecnorm(b)^2-dist^2)/2))/(vecnorm(a)*vecnorm(b));
% 
% clc
% 
% % Readable single-column figure (3.5 in wide, Overleaf-safe)
% figW = 3.5;
% figH = 2.2;   % compact height
% 
% figure('Renderer','painters');
% set(gcf,'Color','white','Units','inches','Position',[3 3 figW figH]);
% 
% % Fonts / axes style (match your other figures)
% fontName = 'Times New Roman';
% fsAxes   = 8;
% fsLabel  = 9;
% lwAx     = 0.8;
% 
% set(gcf,'DefaultAxesFontName',fontName, ...
%         'DefaultTextFontName',fontName, ...
%         'DefaultAxesFontSize',fsAxes, ...
%         'DefaultAxesLineWidth',lwAx);
% 
% % Data
% d    = 0:0.01:2;
% cs_d = abs(1 - d.^2/2);
% 
% % Plot
% plot(d, cs_d, 'k', 'LineWidth', 2);
% box off
% set(gca,'TickDir','out');
% 
% % Limits (optional but helps readability / framing)
% xlim([0 2]);
% ylim([0 1]);
% 
% % Labels
% xlabel('Euclidean Distance','FontSize',fsLabel)
% ylabel('Cosine Similarity','FontSize',fsLabel)
% 
% % Export (vector PDF for Overleaf)
% % exportgraphics(gcf,'cs_vs_ed.pdf','ContentType','vector');


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

% ---- Figure setup (Overleaf-safe) ----
figure('Renderer','painters')
set(gcf,'Color','white','Units','inches','Position',[3 3 3.5 3])  % a bit taller for 3-line legend

% ---- Fonts ----
fontName = 'Times New Roman';
fsAxes  = fontSize_nr - 1;
fsLabel = fontSize_nr;

set(gcf,'DefaultAxesFontName',fontName, ...
        'DefaultTextFontName',fontName, ...
        'DefaultAxesFontSize',fsAxes);

ax = axes; hold(ax,'on'); box(ax,'off');
set(ax,'TickDir','out');

% ---- Histograms ----
k = 0.10;
h = gobjects(11,1);

for NN = 1:11
    h(NN) = histogram(d_tot_p(NN,:), ...
        'FaceColor',[2*k-0.2 0 1], ...
        'FaceAlpha',k, ...
        'EdgeColor','none');   % IMPORTANT for readability with overlays
    k = k + 0.05;
end

xlabel('Euclidean Distance','FontSize',fsLabel)
ylabel('Frequency','FontSize',fsLabel)

% ---- Legend strings ----
legStr = arrayfun(@(x) sprintf('n=%g', x), N(1:11), 'UniformOutput', false);

% ---- Legend (bottom, ~3 lines) ----
% 11 items -> 4 columns gives 3 rows (4+4+3)
lgd = legend(ax, h, legStr, ...
    'Location','southoutside', ...
    'Orientation','horizontal', ...
    'Box','off', ...
    'FontSize',fsAxes);

lgd.NumColumns = 4;   % forces ~3 rows for 11 entries

% ---- Optional: give more breathing room for the legend (robust) ----
% Move axes up a bit to make room for the legend, without relying on tiledlayout
ax.Position(2) = ax.Position(2) + 0.06;
ax.Position(4) = ax.Position(4) - 0.06;

% ---- Export (vector PDF for Overleaf) ----
% exportgraphics(gcf,'hist_ed_DIEM_fine.pdf','ContentType','vector');


%
% clear, clc
% %Comparing 2D points generated from three different distribution
% %Number of points
% N = 5e2;
% %Number of dimension
% n = 2;
% %Uniform
% x_u = 2*rand(N,n)-1;
% %Gaussian
% x_g = 0.3*randn(N,n);
% %Uniform Unitary Sphere
% x_us = randu_sphere(n,N,1,-1);
% 
% figure(),
% set(gcf,'Color','white')
% tiledlayout(1,3)
% nexttile()
% plot(x_u(:,1),x_u(:,2),'.r'), 
% axis equal
% box off
% xlabel('x_1','FontName','Times New Roman')
% ylabel('x_2','FontName','Times New Roman')
% xlim([-1 1])
% ylim([-1 1])
% title('Uniform','FontName','Times New Roman')
% nexttile()
% plot(x_g(:,1),x_g(:,2),'.b'), 
% axis equal
% box off
% xlabel('x_1','FontName','Times New Roman')
% ylabel('x_2','FontName','Times New Roman')
% xlim([-1 1])
% ylim([-1 1])
% title('Gaussian','FontName','Times New Roman')
% nexttile()
% plot(x_us(1,:),x_us(2,:),'.k'), 
% axis equal
% box off
% xlabel('x_1','FontName','Times New Roman')
% ylabel('x_2','FontName','Times New Roman')
% xlim([-1 1])
% ylim([-1 1])
% title('Uniform Uni-sphere','FontName','Times New Roman')


function cs = cosSim(a,b) %Vectors should be oriented as columns
    norm_a = vecnorm(a);
    norm_b = vecnorm(b);

    cs = ((a'*b)/(norm_a*norm_b));
end

