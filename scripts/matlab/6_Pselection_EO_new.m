clear;
clc;

%---------------------- initialization ----------------------%
% load data and set path
rawdir = 'E:\research_data\Methamphetamine_ERP_EGI\connectome-based analysis\CPM_ZHB\CPM_formal\0718_update\supplement';  % path to raw data
scrdir = fullfile(rawdir,'scripts');     % path to scripts
datadir = fullfile(rawdir,'data');     % path to scripts

addpath(rawdir)
addpath(scrdir)
addpath(datadir)


bandname = {'theta', 'alpha', 'beta', 'gamma'};

% Parameter selection
band =  'gamma'; % theta 1, alpha 2, beta 3, gamma 4
behav_name = "craving";

outdir = fullfile(rawdir, band);          % path to output depending on

if ~exist(outdir)
    mkdir(outdir)
end

band_number = find(strcmp(bandname, band));
%sample_EO_87 = conn_matrix_all(:,:,order_choise,:);
% Here, the last dimention is frequency band
%% loading data
%load('sample144.mat')
load('sampleEO87MUD.mat')

conn_matrix_oneband = sample_EO_87(:,:,:,band_number); 
behav_table = relatedtableEO;
%behav_table = sample144_behavall;
%conn_matrix_oneband = sample144_wplimatrix(:,:,:,band_number); 
%%
%coviates1 = table2array(behav_table(:,'withdraw_day'));
% coviates1 = table2array(behav_table(:,'abestinence'));
%coviates2 = table2array(behav_table(:,'PSQI'));
%coviates3 = table2array(behav_table(:,'BDI'));
%coviates3 = table2array(behav_table(:,'BIS_total'));
%coviates = [coviates1];%, coviates2, coviates3];
craving_all = table2array(behav_table(:,behav_name));
% [r,p] = corr(craving_all,coviates)
%---------------------- CPM construction ----------------------%
%% 1. Select threshold
% ------------ INPUTS -------------------
all_mats  =  conn_matrix_oneband;
all_behav = craving_all;
% threshold for feature selection
%For edge selection, you can try a range of thresholds, p=0.05, 0.01, 0.005, etc. Usually, the performance should not be sensitive to the p threshold. 
%If the model only works at a single threshold, and assuming that you are running the leave-one-out validation scheme, you may also want to run k-fold validation to verify the result.
thresh_all = [0.01,0.02,0.05,0.1,0.2,0.3];
R_all = zeros(15,7);

for n = 1:length(thresh_all)
    
fprintf('\n n = %6.3f',n);

thresh = thresh_all(n); 

% ---------------------------------------

no_sub = size(all_mats,3);
no_node = size(all_mats,1);

behav_pred_pos = zeros(no_sub,1);
behav_pred_neg = zeros(no_sub,1);
behav_pred = zeros(no_sub,1); 

pos_consensus = zeros(no_node, no_node);
neg_consensus = zeros(no_node, no_node);

for leftout = 1:no_sub
    %percentage = (percentage + 1)/(no_sub*length(thresh_all));
    %fprintf('\n percentage = %6.3f',percentage);
    fprintf('\n Leaving out subj # %6.3f',leftout);
    
    % leave out subject from matrices and behavior
    
    train_mats = all_mats;
    train_mats(:,:,leftout) = [];
    train_vcts = reshape(train_mats,[],size(train_mats,3));
    
%    train_coviates = coviates;
%    train_coviates(leftout,:) = [];
    train_behav = all_behav;
    train_behav(leftout) = [];
   
%-----------------------train data standardization----------------------------% 
% if needed    
%    for i = 1:no_node
%       for j = 1:no_node
%           train_edges_mean(i,j) = mean(train_mats(i,j,:),3);
%           train_edges_std(i,j) = std(train_mats(i,j,:));
%           train_mats(i,j,:) = (train_mats(i,j,:) - train_edges_mean(i,j))./train_edges_std(i,j);
%       end 
%    end
   
%    train_mats(isnan(train_mats)) = 1;
%-----------------------------------------------------------------% 

    
%     %-----------------Correlation Choice---------------------------%    

    % correlate all edges with behavior using robust regression
%    edge_no = size(train_vcts,1);
%    r_mat = zeros(1, edge_no);
%    p_mat = zeros(1, edge_no);
    
%    for edge_i = 1: edge_no;
%        [~, stats] = robustfit(train_vcts(edge_i,:)', train_behav);
%        cur_t = stats.t(2);
%        r_mat(edge_i) = sign(cur_t)*sqrt(cur_t^2/(no_sub-1-2+cur_t^2));
%        p_mat(edge_i) = 2*(1-tcdf(abs(cur_t), no_sub-1-2));  %two tailed
%    end
    
%     % correlate all edges with behavior using partial correlation
%     [r_mat, p_mat] = partialcorr(train_vcts', train_behav, train_coviates);
%    [r_mat, p_mat] = partialcorr([train_vcts', train_behav], train_coviates);
%        
     % correlate all edges with behavior using rank correlation
%     [r_mat, p_mat] = corr(train_vcts', train_behav, 'type', 'Spearman');
    
    [r_mat, p_mat] = corr(train_vcts', train_behav);

%     %-----------------Correlation Choice---------------------------%
    r_mat = reshape(r_mat,no_node,no_node);
    p_mat = reshape(p_mat,no_node,no_node);

    % set threshold and define masks 
    pos_mask = zeros(no_node, no_node);
    neg_mask = zeros(no_node, no_node);
    
    pos_edges = find( r_mat >0 & p_mat < thresh);
    neg_edges = find( r_mat <0 & p_mat < thresh);
    
%     %-----------------sigmoidal weighting---------------------------%
%     pos_edges = find(r_mat > 0 );
%     neg_edges = find(r_mat < 0 );
%     
%     % covert p threshold to r threshold
%     T = tinv(thresh/2, no_sub-1-2);
%     R = sqrt(T^2/(no_sub-1-2+T^2));
%     
%     % create a weighted mask using sigmoidal function
%     % weight = 0.5, when correlation = R/3;
%     % weight = 0.88, when correlation = R;
%     pos_mask(pos_edges) = sigmf( r_mat(pos_edges), [3/R, R/3]);
%     neg_mask(neg_edges) = sigmf( r_mat(neg_edges), [-3/R, R/3]);
%     %---------------sigmoidal weighting-----------------------------%
    
     pos_mask(pos_edges) = 1;
     neg_mask(neg_edges) = 1;
     
     pos_consensus = pos_consensus + pos_mask;
     neg_consensus = neg_consensus + neg_mask;
    % get sum of all edges in TRAIN subs (divide by 2 to control for the
    % fact that matrices are symmetric)
  
    train_sumpos = zeros(no_sub-1,1);
    train_sumneg = zeros(no_sub-1,1);
    
    for ss = 1:size(train_sumpos)
        train_sumpos(ss) = sum(sum(train_mats(:,:,ss).*pos_mask))/2;
        train_sumneg(ss) = sum(sum(train_mats(:,:,ss).*neg_mask))/2;
    end
    
    if (sum(train_sumpos) == 0 )&&(sum(train_sumneg) == 0)

           
    behav_pred_pos(leftout) = NaN;
    behav_pred_neg(leftout) = NaN;
    behav_pred(leftout) = NaN;
       
    else    
    % build model on TRAIN subs
    % combining both postive and negative features
%    b = regress(train_behav, [train_sumpos, train_sumneg, ones(no_sub-1,1)]);
%    fit_pos = polyfit(train_sumpos, train_behav,1);
%    fit_neg = polyfit(train_sumneg, train_behav,1);
    model_pos = fitlm(train_sumpos,train_behav,'RobustOpts','on');
    model_neg = fitlm(train_sumneg,train_behav,'RobustOpts','on');
    train_sum_cmb = [train_sumpos, train_sumneg];
    %model_cmb = fitlm(train_sum_cmb,train_behav,'RobustOpts','on');
%    model_data = [train_sum_cmb,train_coviates];
    model_data = train_sum_cmb;
    model_cmb = fitlm(model_data,train_behav,'RobustOpts','on');
    % Get the coefficients of the regression line
    coeffs_pos = model_pos.Coefficients.Estimate;
    coeffs_neg = model_neg.Coefficients.Estimate;
    coeffs_cmb = model_cmb.Coefficients.Estimate;
    
    % run model on TEST sub %%%
    
    test_mat = all_mats(:,:,leftout);
%    test_coviates = coviates(leftout,:);
        %-----------------------test data standardization----------------------------% 
    
%    for i = 1:no_node
%       for j = 1:no_node
%           test_mat(i,j) = (test_mat(i,j) - train_edges_mean(i,j))/train_edges_std(i,j);
%       end 
%    end
   
%    test_mat(isnan(test_mat)) = 1;
%-----------------------------------------------------------------% 
    
    test_sumpos = sum(sum(test_mat.*pos_mask))/2;
    test_sumneg = sum(sum(test_mat.*neg_mask))/2;
    
    
    % combine postive and negative network
   
%    behav_pred_pos(leftout) = fit_pos(1)*test_sumpos + fit_pos(2);
%    behav_pred_neg(leftout) = fit_neg(1)*test_sumneg + fit_neg(2);
%    behav_pred(leftout) = b(1)*test_sumpos + b(2)*test_sumneg + b(3);
    
    behav_pred_pos(leftout) = coeffs_pos(1) + coeffs_pos(2)*test_sumpos;
    behav_pred_neg(leftout) = coeffs_neg(1) + coeffs_neg(2)*test_sumneg;
    behav_pred(leftout) = coeffs_cmb(1) + coeffs_cmb(2)*test_sumpos + coeffs_cmb(3)*test_sumneg; % + coeffs_cmb(4)*test_coviates(:,1);% + coeffs_cmb(5)*test_coviates(:,2) + coeffs_cmb(6)*test_coviates(:,3);
     end
end

percentage_edges = 0.95;
pos_mud = (pos_consensus/no_sub > percentage_edges);
neg_mud = (neg_consensus/no_sub > percentage_edges);
pos_mud = double(pos_mud);
neg_mud = double(neg_mud);

% compare predicted and observed scores
    if sum(isnan(behav_pred)) >= 0.1*length(behav_pred)
        R_pos = -1;
        P_pos = -1;
        R_neg = -1;
        P_neg = -1;
        R_cmb = -1;
        P_cmb = -1;
        R_all(9,n+1) = -1;
        R_all(10,n+1) = -1;
        R_all(11,n+1) = -1;

    else
        [R_pos, P_pos] = corr(behav_pred_pos(~isnan(behav_pred_pos)),all_behav(~isnan(behav_pred_pos)), 'type', 'spearman');
        [R_neg, P_neg] = corr(behav_pred_neg(~isnan(behav_pred_neg)),all_behav(~isnan(behav_pred_neg)), 'type', 'spearman');
        [R_cmb, P_cmb] = corr(behav_pred(~isnan(behav_pred)),all_behav(~isnan(behav_pred)), 'type', 'spearman');
        R_all(9,n+1) = sqrt(sum((behav_pred_pos(~isnan(behav_pred_pos))-all_behav(~isnan(behav_pred_pos))).^2)/(sum(~isnan(behav_pred_pos))-length(coeffs_pos)-1)); %RMSE
        R_all(10,n+1) = sqrt(sum((behav_pred_neg(~isnan(behav_pred_neg))-all_behav(~isnan(behav_pred_neg))).^2)/(sum(~isnan(behav_pred_neg))-length(coeffs_neg)-1));
        R_all(11,n+1) = sqrt(sum((behav_pred(~isnan(behav_pred))-all_behav(~isnan(behav_pred))).^2)/(sum(~isnan(behav_pred))-length(coeffs_cmb)-1));

    end
%mdl_cmb = fitlm(behav_pred,all_behav);
%mdl_cmb.Rsquared.Adjusted;

R_all(1,n+1) = R_pos;
R_all(2,n+1) = P_pos;
R_all(3,n+1) = R_neg;
R_all(4,n+1) = P_neg;
R_all(5,n+1) = R_cmb;
R_all(6,n+1) = P_cmb;
R_all(7,n+1) = sum(sum(pos_mud));
R_all(8,n+1) = sum(sum(neg_mud));
%R_all(9,n+1) = sqrt(sum((behav_pred_pos(~isnan(behav_pred_pos))-all_behav(~isnan(behav_pred_pos))).^2)/(sum(~isnan(behav_pred_pos))-length(coeffs_pos)-1)); %RMSE
%R_all(10,n+1) = sqrt(sum((behav_pred_neg(~isnan(behav_pred_neg))-all_behav(~isnan(behav_pred_neg))).^2)/(sum(~isnan(behav_pred_neg))-length(coeffs_neg)-1));
%R_all(11,n+1) = sqrt(sum((behav_pred(~isnan(behav_pred))-all_behav(~isnan(behav_pred))).^2)/(sum(~isnan(behav_pred))-length(coeffs_cmb)-1));
%R_all(12,n+1) = 1 - R_all(9,n+1) / var(all_behav, 1); %q^2
%R_all(13,n+1) = 1 - R_all(10,n+1) / var(all_behav, 1);
%R_all(14,n+1) = 1 - R_all(11,n+1) / var(all_behav, 1);
R_all(12,n+1) = sum(~isnan(behav_pred))/no_sub; %success_rate
end
R_all(15,[2:7]) = thresh_all;

save(fullfile(outdir,[band,'_cravingtraining_Pchoose_EO.mat']), "R_all")

plot(R_all(15,[2:7]),R_all(5,[2:7]),'-o','LineWidth',2,'Color','k')
xlabel('Threshold')
ylabel('Pearson r')
set(gca,'LooseInset',get(gca,'TightInset'));
savename = 'ThresholdSelection.tif';
saveas(gcf,fullfile(outdir,savename))

%% Threshold number of shared edges
% This shold run after use model build with specific threshold
threshold_edges = [0.7:0.05:1];
edges_test = length(threshold_edges);
edge_n = zeros(4,edges_test);
for i = 1:edges_test
pos_mud = (pos_consensus/no_sub >= threshold_edges(i));
neg_mud = (neg_consensus/no_sub >= threshold_edges(i));
pos_mud = double(pos_mud);
neg_mud = double(neg_mud);
edge_n(1,i) = sum(sum(pos_mud));
edge_n(2,i) = sum(sum(neg_mud));
edge_n(3,i) = sum(sum(pos_mud)) + sum(sum(neg_mud));
end
edge_n(4,:) = threshold_edges;

close all
plot(edge_n(4,[1:7]),edge_n(1,[1:7]),'-o','LineWidth',2,'Color','r')
hold on;
plot(edge_n(4,[1:7]),edge_n(2,[1:7]),'-o','LineWidth',2,'Color','b')
xlabel('Percentage of iterations')
ylabel('Number of Shared Edges')
legend('positive edges','negative edges')
set(gca,'LooseInset',get(gca,'TightInset'));
savename = [num2str(thresh),'_EdgesSelection.tif'];
saveas(gcf,fullfile(outdir,savename))

%R2_pos = 1 - MSE_pos / var(all_behav, 1);

%figure(1); plot(behav_pred_pos,all_behav,'r.'); lsline
%figure(2); plot(behav_pred_neg,all_behav,'b.'); lsline
%figure(3); plot(behav_pred,all_behav,'g.'); lsline

% remove 26
%save("connectivity_craving_matrix_58.mat","abstinence", "craving","rest_1_mats_alpha", "rest_1_mats_beta", "rest_1_mats_delta","rest_1_mats_gamma","rest_1_mats_theta","subjects")

% With this script, there're three files should be generated in subfolder named by frequency band:
% 1. CPM performance in different thresholds
% 2. the plot of relationship between thresholds and correlation r
% 3. the mask of edges (the threshold should be selected)