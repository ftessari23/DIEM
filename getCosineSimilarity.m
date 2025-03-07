function [cosineSimilarity, ax] = getCosineSimilarity(synMat1,synMat2,varargin)
% This function will obtain the Matrix of Cosine Similarity's between two
% vectors or groups of vectors

% INPUTS:
%   synMat1 - a column-wise vector (or matrix) where each row represents
%       the a specific feature.
%   synMat2 - a column-wise vector (or matrix) where each row represents
%       the a specific feature.
% (optional Name-Value Arguments):
%   'Plot' - accepted values 'on' or 'off' (default) determine whether or
%       not to plot the cosineSimilarity.
%   'Text' - accepted values 'on' or 'off' (default) determine whether or
%       not to plot the cosine similarity values on the plot.
%   'TextSize' - integer option for the fontsize of the text of the cosine
%       similarity values that are plotted.
% OUTPUTS: 
%   cosineSimilarity - matrix of cosine similarities for matrix values.
%   ax - axes of the cosineSimilarity plot

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

%% get cosine similarities.
%normalize the columns of the input vectors or matrices
synMat1 = normc(synMat1);
synMat2 = normc(synMat2);

cosineSimilarity = abs(synMat1'*synMat2);
if cosineSimilarity == cosineSimilarity' %if symmetric
    cosineSimilarity = triu(cosineSimilarity); %just show upper triangular matrix
    cosineSimilarity(find(cosineSimilarity==0)) = NaN;
end

if strcmpi(plotToggle, 'on') %if plot is on, use plotCosineSimilarity function to plot the Cosine Similarities.
    ax = plotCosineSimilarity(cosineSimilarity,textToggle,textSize);
end

end

function ax = plotCosineSimilarity(cosineSimilarity,textToggle,textSize)
% This embedded function will plot magnitude (absolute value) of the cosine
% Similarity using imagesc.
MITred = [163,31,52]./255;

cosineSimilarity = abs(cosineSimilarity); %convert to absolute value.
[m,n] = size(cosineSimilarity);

%plot using imageSC function
imagesc(cosineSimilarity,[0 1]);
colormap([1,1,1; flipud(autumn)]); c = colorbar; %c.Label.String = '|C|'; c.Label.FontSize = 10;
ax = gca;
ax.XTick = 1:1:n; ax.YTick = 1:1:m;
ax.XTickLabelRotation = 0; ax.YTickLabelRotation = 0; %Not sure why this is neccessary

%add text to image sc plot
if strcmpi(textToggle, 'off') %if text is on, plot the cosine similarity values on the imagesc graph.
    for i = 1:m
        for j = 1:n
            if ~isnan(cosineSimilarity(i,j))
                text(j,i,num2str(round(cosineSimilarity(i,j),2)),'HorizontalAlignment','center','FontWeight','bold','FontName','Times New Roman','FontSize',textSize,'Color','black');
            end
        end
    end
end

set(gcf,'Color','White');
end