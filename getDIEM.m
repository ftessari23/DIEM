function [DIEM, ax] = getDIEM(synMat1,synMat2,maxV,minV,exp_center,vard,varargin)
% This function will obtain the Matrix of DIEM between two
% vectors or groups of vectors

% INPUTS:
%   synMat1 - a column-wise vector (or matrix) where each row represents
%       the a specific feature.
%   synMat2 - a column-wise vector (or matrix) where each row represents
%       the a specific feature.
% (optional Name-Value Arguments):
%   'Plot' - accepted values 'on' or 'off' (default) determine whether or
%       not to plot the DIEM.
%   'Text' - accepted values 'on' or 'off' (default) determine whether or
%       not to plot the DIEM similarity values on the plot.
%   'TextSize' - integer option for the fontsize of the text of the DIEM
%       similarity values that are plotted.
% OUTPUTS: 
%   DIEM - matrix of DIEM similarities for matrix values.
%   ax - axes of the DIEM plot

%% name value argument
if isempty( varargin ) % set default values
    plotToggle = 'off'; %default value for 'Plot'
    textToggle = 'off'; %default value for 'Text'
    textSize = 10; %default value for 'TextSize'
else % Parsing the varargin
    %%% 'Plot' %%%
    idx = find( strcmpi( 'plot', varargin ) );
    if ~isempty( idx ) % If there is a value, then set it
        plotToggle = varargin{ idx + 1 };
    else %if there is no value, set to default.
        plotToggle = 'off'; %default value for 'Plot'
    end
    %%% 'Text' %%%
    idx = find( strcmpi( 'text', varargin ) );
    if ~isempty( idx ) % If there is a value, then set it
        textToggle = varargin{ idx + 1 };
    else %if there is no value, set to default.
        textToggle = 'off'; %default value for 'Text'
    end
    %%% 'TextSize' %%%
    idx = find( strcmpi( 'textsize', varargin ) );
    if ~isempty( idx ) % If there is a value, then set it
        plotToggle = varargin{ idx + 1 };
    else %if there is no value, set to default.
        textSize = 10; %default value for 'TextSize'
    end
end

%% get DIEM similarities.

DIEM = (maxV-minV)*((pdist2(synMat1',synMat2',"euclidean")-exp_center))./vard;

if DIEM == DIEM' %if symmetric
    DIEM = triu(DIEM); %just show upper triangular matrix
    DIEM(find(DIEM==0)) = NaN;
end

if strcmpi(plotToggle, 'on') %if plot is on, use plotDIEM function to plot the DIEM Similarities.
    ax = plotDIEM(DIEM,textToggle,textSize,1.1*min(min(DIEM)),max(max(DIEM)));
end

end

function ax = plotDIEM(DIEM,textToggle,textSize,minD,maxD)
% This embedded function will plot magnitude (absolute value) of the DIEM
% Similarity using imagesc.
MITred = [163,31,52]./255;

[m,n] = size(DIEM);

figure();
%plot using imageSC function
imagesc(DIEM,[minD maxD]);
colormap([1,1,1; flipud(autumn)]); c = colorbar; %c.Label.String = '|C|'; c.Label.FontSize = 10;
ax = gca;
ax.XTick = 1:1:n; ax.YTick = 1:1:m;
ax.XTickLabelRotation = 0; ax.YTickLabelRotation = 0; %Not sure why this is neccessary

%add text to image sc plot
if strcmpi(textToggle, 'on') %if text is on, plot the DIEM similarity values on the imagesc graph.
    for i = 1:m
        for j = 1:n
            if ~isnan(DIEM(i,j))
                text(j,i,num2str(round(DIEM(i,j),0)),'HorizontalAlignment','center','FontWeight','bold','FontName','Times New Roman','FontSize',textSize,'Color','black');
            end
        end
    end
end

set(gcf,'Color','White');
end