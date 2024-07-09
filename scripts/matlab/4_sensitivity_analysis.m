% Sensitivity Analyses 
% Input
train_mats_all  =  conn_matrix_oneband;
train_behav_all = phenotype_all;
rawdir = 'E:\research_data\Methamphetamine_ERP_EGI\CPM_EEG_craving\';
no_iterations = 1000;
outdir = fullfile(rawdir,'\results\', band); 

% Steiger’s Z test preperation
% model1_pred = behav_pred;

% construct leasiong matrix
DM_network = [2 3 4 35 36 45 46 100];
SubCor_network = [13 14	57 58 63 64	79 80 81 82 85 97 98 104 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136];
FP_network = [9	10	23	24	31	32	40	42	43	44	47	48	50	71	72	73	74	76	87	88	110	111];
MF_network = [1	5	39	41	49	51	52	75	99	106	112	113	114	115	117];
MS_network = [7	8	55	56	60	83	84	86	91	92	93	94	102	103	108	109	116	118	119];
Salience_network = [6	17	19	20	33	34	59	77	78	95	96	107];
CB_network = [15 16	18	21	22	25	26	27	28	101	105	137	138	139	140	141	142	143	144];
VA_network = [54 65	67	68	89	90];

highdegree_Cerebellum = [21,22,23,24,25,26,27,30,32,137,144];
highdegree_Occipital = [68];
highdegree_Parietal = [90];
highdegree_Temporal = [113];

% virtual lesioning - choose network_retained or network_lesioned and which
% network, change the names of network
network_name = get_var_name(CB_network);
manipulated_network = CB_network;

% prepare lesion
complete_matrix = ones(144,144);
network_lesioned = complete_matrix;
network_lesioned(manipulated_network,:) = 0;
network_lesioned(:,manipulated_network) = 0;

% prepare retain
none_matrix = zeros(144,144);
network_retained  = none_matrix;
network_retained(manipulated_network,:) = 1;
network_retained(:,manipulated_network) = 1;


% choose lesion one network or retain one network (virtual lesioning others)
% Prepare lesion and retain name
%lesion_name= ['lesion_', network_name]; 
lesion_name= ['lesion_', network_name]; ; % set the save name

lesioned_mats_all = train_mats_all.*network_lesioned;
%lesioned_mats_all = train_mats_all.*network_retained; 

% Virtual Lesioning Analysis

%% LOOCV
thresh = 0.02;
save_CPMname = ['CPM_craving_',band,lesion_name, num2str(thresh)];
% ---------------------------------------

no_sub = size(train_mats_all,3);
no_node = size(train_mats_all,1);

behav_pred_pos = zeros(no_sub,1);
behav_pred_neg = zeros(no_sub,1);
behav_pred = zeros(no_sub,1); 

pos_consensus = zeros(no_node, no_node);
neg_consensus = zeros(no_node, no_node);

all_mats  = lesioned_mats_all;
all_behav = train_behav_all;


for leftout = 1:no_sub
    fprintf('\n Leaving out subj # %6.3f',leftout);
    
    % leave out subject from matrices and behavior
    
    train_mats = lesioned_mats_all;
    train_mats(:,:,leftout) = [];
%-----------------------train data train data normalization----------------------------% 
    
%    for i = 1:no_node
%       for j = 1:no_node
%           train_edges_mean(i,j) = mean(train_mats(i,j,:),3);
%           train_edges_std(i,j) = std(train_mats(i,j,:));
%           train_mats(i,j,:) = (train_mats(i,j,:) - train_edges_mean(i,j))./train_edges_std(i,j);
%       end 
%    end

%    train_mats(isnan(train_mats)) = 1;
%-----------------------------------------------------------------% 

    train_vcts = reshape(train_mats,[],size(train_mats,3));
    
    train_behav = train_behav_all;
    train_behav(leftout) = [];
    
    %train_age = co_mats;
    %train_age(leftout) = [];
    

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
%     [r_mat, p_mat] = partialcorr(train_vcts', train_behav, train_age);
%     
%        
     % correlate all edges with behavior using rank correlation
%     [r_mat, p_mat] = corr(train_vcts.', train_behav, 'type', 'Spearman');
    
    [r_mat, p_mat] = corr(train_vcts', train_behav);
%     %-----------------Correlation Choice---------------------------%
r_mat = reshape(r_mat,no_node,no_node);
p_mat = reshape(p_mat,no_node,no_node);

    % set threshold and define masks 
    pos_mask = zeros(no_node, no_node);
    neg_mask = zeros(no_node, no_node);
    
    
    pos_edges = find( r_mat >0 & p_mat < thresh);
    neg_edges = find( r_mat <0 & p_mat < thresh);
    
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
       behav_pred_pos(leftout) = -1;
       behav_pred_neg(leftout) = -1;
       behav_pred(leftout) = -1;
    else    
    % build model on TRAIN subs
    % combining both postive and negative features
%    b = regress(train_behav, [train_sumpos, train_sumneg, ones(no_sub-1,1)]);
%    fit_pos = polyfit(train_sumpos, train_behav,1);
%    fit_neg = polyfit(train_sumneg, train_behav,1);
    model_pos = fitlm(train_sumpos,train_behav,'RobustOpts','on');
    model_neg = fitlm(train_sumneg,train_behav,'RobustOpts','on');
    train_sum_cmb = [train_sumpos, train_sumneg];
    model_cmb = fitlm(train_sum_cmb,train_behav,'RobustOpts','on');
    % Get the coefficients of the regression line
    coeffs_pos = model_pos.Coefficients.Estimate;
    coeffs_neg = model_neg.Coefficients.Estimate;
    coeffs_cmb = model_cmb.Coefficients.Estimate;
    
    % run model on TEST sub %%%
    
    test_mat = all_mats(:,:,leftout);
    
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
    
    %if want to combine postive and negative
   
    behav_pred_pos(leftout) = coeffs_pos(1) + coeffs_pos(2)*test_sumpos;
    behav_pred_neg(leftout) = coeffs_neg(1) + coeffs_neg(2)*test_sumneg;
    behav_pred(leftout) = coeffs_cmb(1) + coeffs_cmb(2)*test_sumpos + coeffs_cmb(3)*test_sumneg;
    end
%save(fullfile(outdir, [save_CPMname '_pos_' num2str(percentage_edges) '.csv']), "pos_mud",'-ascii')
%save(fullfile(outdir, [save_CPMname '_neg_' num2str(percentage_edges) '.csv']), "neg_mud",'-ascii')
% compare predicted and observed scores
end

percentage_edges = 0.95;
pos_mud = (pos_consensus/no_sub >= percentage_edges);
neg_mud = (neg_consensus/no_sub >= percentage_edges);
pos_mud = double(pos_mud);
neg_mud = double(neg_mud);

% compare predicted and observed scores
    if sum(behav_pred==-1) >= 0.1*length(behav_pred)
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
        [R_pos, P_pos] = corr(behav_pred_pos(behav_pred_pos~=-1),all_behav(behav_pred_pos~=-1), 'type', 'spearman');
        [R_neg, P_neg] = corr(behav_pred_neg(behav_pred_neg~=-1),all_behav(behav_pred_neg~=-1), 'type', 'spearman');
        [R_cmb, P_cmb] = corr(behav_pred(behav_pred~=-1),all_behav(behav_pred~=-1), 'type', 'spearman')
        RMSE_pos = sqrt(sum((behav_pred_pos(behav_pred_pos~=-1)-all_behav(behav_pred_pos~=-1)).^2)/(sum(behav_pred_pos~=-1)-length(coeffs_pos)-1)); %RMSE
        RMSE_neg = sqrt(sum((behav_pred_neg(behav_pred_neg~=-1)-all_behav(behav_pred_neg~=-1)).^2)/(sum(behav_pred_neg~=-1)-length(coeffs_neg)-1));
        RMSE_cmb = sqrt(sum((behav_pred(behav_pred~=-1)-all_behav(behav_pred~=-1)).^2)/(sum(behav_pred~=-1)-length(coeffs_cmb)-1))
    end

save(fullfile(outdir,[save_CPMname, '_internal.mat']),"pos_mud","neg_mud","behav_pred_pos","behav_pred_neg","behav_pred")

figure(1); plot(train_behav_all,behav_pred_pos,'r.'); lsline
figure(2); plot(train_behav_all,behav_pred_neg,'b.'); lsline
figure(3); plot(train_behav_all,behav_pred,'g.'); lsline

plotname = {'Positive','Negative','Combined'};
for i = 1:3
figure(i);
xlabel('Observed Craving Score')
ylabel('Predicted Craving Score')
title(['The predictive ability of ',char(plotname(i)), ' CPM'])
set(gca,'LooseInset',get(gca,'TightInset'));
filename = ["InternalValidation_" + num2str(i)+lesion_name+'.tif'];
savename = fullfile(outdir, filename);
saveas(gcf,savename)
end

close all

%% Steiger’s Z test preperation
model2_pred = behav_pred;
%model1_pred = behav_pred;
%[model_r, model_p] = corr(model1_pred,model2_pred)

% Calculate Spearman correlations
[rho12, p12] = corr(phenotype_all, model1_pred, 'Type', 'Spearman');
[rho13, p13] = corr(phenotype_all, model2_pred, 'Type', 'Spearman');
[rho23, p23] = corr(model1_pred, model2_pred, 'Type', 'Spearman');

% Number of observations
n = length(phenotype_all);

% Fisher's Z transformations for Spearman correlations
z12 = 0.5 * log((1 + rho12) / (1 - rho12));
z13 = 0.5 * log((1 + rho13) / (1 - rho13));
z23 = 0.5 * log((1 + rho23) / (1 - rho23));

% Calculate covariance
cov_z = (z12 - z13)^2 + (z12 - z23)^2 + (z13 - z23)^2;

% Compute Steiger's Z
steiger_z = (z12 - z13) / sqrt((2 * (1 - rho23)) / (n - 3));

% Compute the p-value for the test
p_value = 2 * (1 - normcdf(abs(steiger_z)));

% Display results
fprintf('Spearman correlation between behavior and model1: rho = %.3f, p = %.3f\n', rho12, p12);
fprintf('Spearman correlation between behavior and model2: rho = %.3f, p = %.3f\n', rho13, p13);
fprintf('Spearman correlation between model1 and model2: rho = %.3f, p = %.3f\n', rho23, p23);
fprintf('Steiger''s Z test: Z = %.3f, p = %.3f\n', steiger_z, p_value);


%% permutation test
% ------------ INPUTS -------------------
all_mats  = lesioned_mats_all; 
all_behav = train_behav_all;

no_sub = size(all_mats,3);

% calculate the true prediction correlation
[true_prediction_r_pos, true_prediction_r_neg, true_prediction_r_total,RMSE_raw] = predict_behavior(all_mats, all_behav,0.02);

% number of iterations for permutation testing
no_iterations   = 1000; % in general set to 100~10000
%prediction_r    = zeros(no_iterations,2);
prediction_r(1,1) = true_prediction_r_pos;
prediction_r(1,2) = true_prediction_r_neg;
prediction_r(1,3) = true_prediction_r_total;
RMSE_all = zeros(no_iterations,1);
RMSE_all(1,1) = RMSE_raw;
%rng(1234);
% create estimate distribution of the test statistic
% via random shuffles of data lables   

for it= 2:no_iterations %2:no_iterations
    fprintf('\n Performing iteration %d out of %d', it, no_iterations);
    new_behav        = all_behav(randperm(no_sub));
    [prediction_r(it,1), prediction_r(it,2), prediction_r(it,3),RMSE_all(it)] = predict_behavior(all_mats, new_behav,0.02);    
end


% calculate permutation p
%%%% you can also only run the below chunk after load
% permutation_test_[frequency band]_LOOCV_[threshold]_1000.mat to extract the pvalue%%%%%
% lesion_name = 'retain_SubCor_network', band = 'beta';thresh = 0.02;filename = ['permutation_test_', band, '_LOOCV_', num2str(thresh), '_1000_',lesion_name,'.mat']; load(fullfile(outdir, filename))

true_prediction_r_pos = prediction_r(1,1);
sorted_prediction_r_pos = sort(prediction_r(:,1),'descend');
position_pos            = find(sorted_prediction_r_pos(sorted_prediction_r_pos~= -1)==true_prediction_r_pos);
pval_pos                = position_pos(1)/length(sorted_prediction_r_pos(sorted_prediction_r_pos~= -1))

true_prediction_r_neg = prediction_r(1,2);
sorted_prediction_r_neg = sort(prediction_r(:,2),'descend');
position_neg            = find(sorted_prediction_r_neg(sorted_prediction_r_neg~= -1)==true_prediction_r_neg);
pval_neg                = position_neg(1)/length(sorted_prediction_r_neg(sorted_prediction_r_neg~= -1))


true_prediction_r_total = prediction_r(1,3);
sorted_prediction_r_total = sort(prediction_r(:,3),'descend');
position_total            = find(sorted_prediction_r_total(sorted_prediction_r_total~= -1)==true_prediction_r_total);
pval_total                = position_total(1)/length(sorted_prediction_r_total(sorted_prediction_r_total~= -1))

true_RMSE_all = RMSE_all(1,1)
sorted_RMSE_all = sort(RMSE_all);
position_RMSE            = find(sorted_RMSE_all (sorted_RMSE_all ~= -1) == true_RMSE_all);
pval_RMSE                = position_RMSE/length(sorted_RMSE_all(sorted_RMSE_all ~= -1))
success_rate = 1-(sum(prediction_r(:,3) == -1)/1000)
    


filename = ['permutation_test_beta_LOOCV_0.02_1000_' lesion_name '.mat']; 
save(fullfile(outdir,filename), 'prediction_r','RMSE_all')
close all
plotname = {'Positive','Negative','Combined'};
    pval(1) = pval_pos;
    pval(2) = pval_neg;
    pval(3) = pval_total;

    for i = 3 % only plot cmb plot
        figure(i);
        r_all = prediction_r(:, i);
        histogram(r_all(r_all~=-1), 'Normalization', 'probability', 'BinWidth', 0.01, 'FaceAlpha', 0.3, 'EdgeAlpha', 0.3);
        str_r = ["r = " + num2str(round(prediction_r(1, i), 3))];
        str_p = ["p = " + num2str(round(pval(i), 3))];
        str_rp = [str_r newline str_p];
        xline(prediction_r(1, i), 'r', 'Label', str_rp, 'LabelHorizontalAlignment', 'left', 'LabelOrientation', 'horizontal');
        xlabel('Pearson r');
        ylabel('frequency');
        title([char(plotname(i)), ' CPM permutation distribution']);
        set(gca, 'LooseInset', get(gca, 'TightInset'));
        filename = ['distribution_', band, num2str(thresh), '_', num2str(i), '.tif'];
        savename = fullfile(outdir, filename);
        saveas(gcf, savename);
    end

    figure(4);
    histogram(RMSE_all(RMSE_all~=-1), 'Normalization', 'probability', 'BinWidth', 0.1, 'FaceAlpha', 0.3, 'EdgeAlpha', 0.3);
    str_r = ["RMSE = " + num2str(round(RMSE_all(1), 3))];
    str_p = ["p = " + num2str(round(pval_RMSE, 3))];
    str_rp = [str_r newline str_p];
    xline(true_RMSE_all, 'r', 'Label', str_rp, 'LabelHorizontalAlignment', 'left', 'LabelOrientation', 'horizontal');
    xlabel('RMSE');
    ylabel('frequency');
    title('RMSE CPM permutation distribution');
    set(gca, 'LooseInset', get(gca, 'TightInset'));
    filename = ['distribution_RMSE', band, num2str(thresh), '.tif'];
    savename = fullfile(outdir, filename);
    saveas(gcf, savename);
end
