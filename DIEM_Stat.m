function [exp_center,vard,std_one,orth_med,min_DIEM,max_DIEM] = DIEM_Stat(N,maxV,minV,fig_flag)
%This Function Generates the Normalizing and Detrending values to correctly
%compute the DIEM Similarity
for j = 1:1e5
    %Uniform Distribution
    a{j} = (maxV-minV)*rand(N,1)+minV;
    b{j} = (maxV-minV)*rand(N,1)+minV;
    tmp = null(a{j}');
    ort{j} = tmp(:,1);
    %Eucledian Distance
    d(j) = pdist2(a{j}',b{j}',"euclidean");
    dort(j) = pdist2(a{j}',ort{j}',"euclidean");
end
%Deterending Euclidian Distance on Median Value
exp_center = median(d);
vard = var(d); %Variance of Euclidean Distirbution
orth_med = (maxV-minV)*(median(dort) - exp_center)./var(d); %Median DIEM of Orthogonal Quantities
adjusted_dist = (maxV-minV)*(d - exp_center)./var(d); %Centering the DIEM distribution to zero and scaling it to have same variance and range
std_one = std(adjusted_dist); %One Standard Deviation of DIEM

min_DIEM = -(maxV-minV)*(exp_center/vard); %Minimum DIEM 
max_DIEM = (maxV-minV)*(sqrt(N)*(maxV-minV)-exp_center)/vard; %Maximum DIEM 

width = 10;

if fig_flag == 1
    %Optimal Figure Setting to use in any script
    set(0,'defaultaxesfontname','Times New Roman');
    set(0,'defaulttextfontname','Times New Roman');
    set(0,'defaultaxesfontsize',12); % 8 for paper images 12 for normal images
    set(0,'defaulttextfontsize',12); % 8 for paper images 12 for normal images

    fig = figure();
    set(gcf,'color','w','Units','inches','Position',[1 1 6 6]);
    fill([1:width fliplr(1:width)],[-std_one*ones(1,width) std_one*ones(1,width)],'r','FaceAlpha',0.2,'EdgeColor','none'), hold on
    fill([1:width fliplr(1:width)],[-2*std_one*ones(1,width) 2*std_one*ones(1,width)],'r','FaceAlpha',0.2,'EdgeColor','none'), hold on
    fill([1:width fliplr(1:width)],[-3*std_one*ones(1,width) 3*std_one*ones(1,width)],'r','FaceAlpha',0.2,'EdgeColor','none'), hold on
    plot(1:width,zeros(1,width),'k--','Linewidth',1), hold on
    plot(1:width,orth_med*ones(1,width),'k-.','Linewidth',1,'MarkerSize',15), hold on
    plot(1:width,-(maxV-minV)*(exp_center/vard)*ones(1,width),'k-.','Linewidth',1,'MarkerSize',15), hold on
    plot(1:width,(maxV-minV)*(sqrt(N)*(maxV-minV)-exp_center)/vard*ones(1,width),'k-.','Linewidth',1), hold on
    ylabel('DIEM','FontName','Times New Roman')
    box off
    h = gca;
    h.XAxis.Visible = 'off';
    
end
end