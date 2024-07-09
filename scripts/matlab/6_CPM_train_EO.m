% after selecting the appropriate threshold

%% LOOCV
thresh = 0.02;
save_CPMname = ['CPM_craving_',band,'_',num2str(thresh)];
% ---------------------------------------

train_mats_all  =  conn_matrix_oneband;
train_behav_all = craving_all;

no_sub = size(train_mats_all,3);
no_node = size(train_mats_all,1);

behav_pred_pos = zeros(no_sub,1);
behav_pred_neg = zeros(no_sub,1);
behav_pred = zeros(no_sub,1); 

pos_consensus = zeros(no_node, no_node);
neg_consensus = zeros(no_node, no_node);

for leftout = 1:no_sub
    fprintf('\n Leaving out subj # %6.3f',leftout);
    
    % leave out subject from matrices and behavior
    
    train_mats = train_mats_all;
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
    
    % run model on TEST sub
    
    test_mat = train_mats_all(:,:,leftout);
%    for i = 1:no_node
%        for j = 1:no_node
%           test_mat(i,j) = (test_mat(i,j) - train_edges_mean(i,j))./train_edges_std(i,j);
%        end 
%    end
%    test_mat(isnan(test_mat)) = 1;
    
    test_sumpos = sum(sum(test_mat.*pos_mask))/2;
    test_sumneg = sum(sum(test_mat.*neg_mask))/2;
    
    %if want to combine postive and negative
   
%    behav_pred_pos(leftout) = fit_pos(1)*test_sumpos + fit_pos(2);
%    behav_pred_neg(leftout) = fit_neg(1)*test_sumneg + fit_neg(2);    
%    behav_pred(leftout) = b(1)*test_sumpos + b(2)*test_sumneg + b(3);
    behav_pred_pos(leftout) = coeffs_pos(1) + coeffs_pos(2)*test_sumpos;
    behav_pred_neg(leftout) = coeffs_neg(1) + coeffs_neg(2)*test_sumneg;
    behav_pred(leftout) = coeffs_cmb(1) + coeffs_cmb(2)*test_sumpos + coeffs_cmb(3)*test_sumneg;
end

% the percentage of passed iterations in all CV iterations
percentage_edges = 0.95;
pos_mud = (pos_consensus/no_sub >= percentage_edges);
neg_mud = (neg_consensus/no_sub >= percentage_edges);
pos_mud = double(pos_mud);
neg_mud = double(neg_mud);
sum(sum(pos_mud))
sum(sum(neg_mud))

save(fullfile(outdir, [save_CPMname '_pos_' num2str(percentage_edges) '.csv']), "pos_mud",'-ascii')
save(fullfile(outdir, [save_CPMname '_neg_' num2str(percentage_edges) '.csv']), "neg_mud",'-ascii')

% compare predicted and observed scores
[R_pos, P_pos] = corr(behav_pred_pos,train_behav_all,'type','Spearman')
[R_neg, P_neg] = corr(behav_pred_neg,train_behav_all,'type','Spearman')
[R_cmb, P_cmb] = corr(behav_pred,train_behav_all,'type','Spearman')


%RMSE_pos = sqrt((sum((behav_pred_pos - train_behav_all).^2))/(no_sub-length(fit_pos)-1))
%RMSE_neg = sqrt((sum((behav_pred_neg - train_behav_all).^2))/(no_sub-length(fit_neg)-1))
RMSE_cmb = sqrt((sum((behav_pred - train_behav_all).^2))/(no_sub-3-1))

save(fullfile(outdir,[save_CPMname, '_internal.mat']),"pos_mud","neg_mud","behav_pred_pos","behav_pred_neg","behav_pred")

%figure(1); plot(train_behav_all,behav_pred_pos,'r.'); lsline
%figure(2); plot(train_behav_all,behav_pred_neg,'b.'); lsline
figure(3); plot(train_behav_all,behav_pred,'g.'); lsline

plotname = {'Positive','Negative','Combined'};
for i = 1:3
figure(i);
xlabel('Observed Craving Score')
ylabel('Predicted Craving Score')
title(['The predictive ability of ',char(plotname(i)), ' CPM'])
set(gca,'LooseInset',get(gca,'TightInset'));
filename = ["InternalValidation_" + num2str(i)+'.tif'];
savename = fullfile(outdir, filename);
saveas(gcf,savename)
end

close all

%% permutation test
% number of iterations for permutation testing
no_iterations   = 1000; % in general set to 100~10000
for thresh = [0.001,0.002,0.01,0.02,0.05];
all_mats  = train_mats_all; % if alpha is finished, you can change alpha to delta/gamma/theta together with the second last line
all_behav = train_behav_all;

no_sub = size(all_mats,3);

RMSE_all = zeros(no_iterations,1);

% calculate the true prediction correlation
[true_prediction_r_pos, true_prediction_r_neg, true_prediction_r_total,RMSE_raw] = predict_behavior(all_mats, all_behav,thresh);


%prediction_r    = zeros(no_iterations,2);
prediction_r(1,1) = true_prediction_r_pos;
prediction_r(1,2) = true_prediction_r_neg;
prediction_r(1,3) = true_prediction_r_total;
RMSE_all(1,1) = RMSE_raw;
%rng(1234);
% create estimate distribution of the test statistic
% via random shuffles of data lables   

for it= 2:no_iterations %2:no_iterations
    fprintf('\n Performing iteration %d out of %d', it, no_iterations);
    new_behav        = all_behav(randperm(no_sub));
    [prediction_r(it,1), prediction_r(it,2), prediction_r(it,3), RMSE_all(it)] = predict_behavior(all_mats, new_behav,thresh);
    fprintf('\n prediction r is %d \n RMSE is %d ', prediction_r(it,3), RMSE_all(it));
end

filename = ['permutation_test_',band,'_LOOCV_',num2str(thresh),'_1000.mat']; % if alpha is finished, you can change alpha to delta/gamma/theta together with the second last line
%save(filename, 'pval_pos' ,'pval_neg','pval_total','prediction_r')
save(fullfile(outdir,filename), 'prediction_r','RMSE_all')
close all

% calculate permutation p
true_prediction_r_pos = prediction_r(1,1);
sorted_prediction_r_pos = sort(prediction_r(:,1),'descend');
position_pos            = find(sorted_prediction_r_pos==true_prediction_r_pos);
pval_pos                = position_pos(1)/no_iterations;

true_prediction_r_neg = prediction_r(1,2);
sorted_prediction_r_neg = sort(prediction_r(:,2),'descend');
sorted_prediction_r_neg = sorted_prediction_r_neg(all(~isnan(sorted_prediction_r_neg),2),:); 
position_neg            = find(sorted_prediction_r_neg==true_prediction_r_neg);
pval_neg                = position_neg(1)/length(sorted_prediction_r_neg);

true_prediction_r_total = prediction_r(1,3);
sorted_prediction_r_total = sort(prediction_r(:,3),'descend');
position_total            = find(sorted_prediction_r_total(sorted_prediction_r_total~= -1)==true_prediction_r_total);
pval_total                = position_total(1)/length(sorted_prediction_r_total(sorted_prediction_r_total~= -1))

true_RMSE_all = RMSE_all(1,1)
sorted_RMSE_all = sort(RMSE_all);
position_RMSE            = find(sorted_RMSE_all (sorted_RMSE_all ~= -1) == true_RMSE_all);
pval_RMSE                = position_RMSE/length(sorted_RMSE_all(sorted_RMSE_all ~= -1))
success_rate = 1-(sum(prediction_r(:,3) == -1)/1000)

plotname = {'Positive','Negative','Combined'};
pval(1) = pval_pos;
pval(2) = pval_neg;
pval(3) = pval_total;

for i = 1:3
    figure(i); histogram(prediction_r(:,i),'Normalization','probability','BinWidth',0.01,'FaceAlpha',0.3,'EdgeAlpha',0.3)
    str_r = ["r = " + num2str(round(prediction_r(1,i),3))];
    str_p = ["p = " + num2str(round(pval(i),3))];
    str_rp = [str_r newline str_p];
    xline(prediction_r(1,i),'r','Label',str_rp,'LabelHorizontalAlignment','left','LabelOrientation','horizontal')
    %str = [str, num2str(p(i)), 'X\textsuperscript{', num2str(length(p) -i), '} + '];
    xlabel('Pearson r')
    ylabel('frequency')
    title([char(plotname(i)), ' CPM permutation distribution'])
    set(gca,'LooseInset',get(gca,'TightInset'));
    filename = ['distribution_',band,num2str(thresh), '_', num2str(i),'.tif'];
    savename = fullfile(outdir, filename);
    saveas(gcf,savename)
end

% Plot distributino of RMSE
    figure(4); histogram(sorted_RMSE_all (sorted_RMSE_all ~= -1),'Normalization','probability','BinWidth',0.1,'FaceAlpha',0.3,'EdgeAlpha',0.3)
    str_r = ["RMSE = " + num2str(round(RMSE_raw,3))];
    str_p = ["p = " + num2str(round(pval_RMSE,3))];
    str_rp = [str_r newline str_p];
    xline(RMSE_raw,'r','Label',str_rp,'LabelHorizontalAlignment','left','LabelOrientation','horizontal')
    %str = [str, num2str(p(i)), 'X\textsuperscript{', num2str(length(p) -i), '} + '];
    xlabel('RMSE')
    ylabel('frequency')
    title('RMSE CPM permutation distribution')
    set(gca,'LooseInset',get(gca,'TightInset'));
    filename = ['distribution_RMSE',band,num2str(thresh),'.tif'];
    savename = fullfile(outdir, filename);
    saveas(gcf,savename)
end
%% external validation
CPM_betamud_pos = pos_mud;
CPM_betamud_neg = neg_mud;

% Here, the last dimention is frequency band
test_mats_all = sample47_wplimatrix(:,:,:,band_number); 
behav_table_test = sample47_behavall;
test_behav_all = table2array(behav_table_test(:,behav_name));

% Rescale of external test dataset
% 1. calculate mean and std of all traning data
no_node = size(train_mats_all,1);
for i = 1:no_node
    for j = 1:no_node
        train_all_mean(i,j) = mean(train_mats_all(i,j,:),3);
        train_all_std(i,j) = std(train_mats_all(i,j,:));
        train_all_rescale(i,j,:) = (train_mats_all(i,j,:) - train_all_mean(i,j))./train_all_std(i,j);
    end 
end
train_all_rescale(isnan(train_all_rescale)) = 1;
% 2. normalize external test dataset    
for i = 1:no_node
    for j = 1:no_node
        test_all_rescale(i,j,:) = (test_mats_all(i,j,:) - train_all_mean(i,j))/train_all_std(i,j);
    end
end
test_all_rescale(isnan(test_all_rescale)) = 1;

% calculate parameter of model(b,k...)
%train_all_rescale = train_mats_all;
%test_all_rescale = test_mats_all;

no_sub_all = size(train_all_rescale,3);

train_sumpos_all = zeros(no_sub_all,1);
train_sumneg_all = zeros(no_sub_all,1);

for ss = 1:size(train_sumpos_all)
    train_sumpos_all(ss) = sum(sum(train_all_rescale(:,:,ss).*CPM_betamud_pos))/2;
    train_sumneg_all(ss) = sum(sum(train_all_rescale(:,:,ss).*CPM_betamud_neg))/2;
end
    

% combining both postive and negative features
%b_all = regress(train_behav_all, [train_sumpos_all, train_sumneg_all, ones(no_sub_all,1)]);
%fit_pos_all = polyfit(train_sumpos_all, train_behav_all,1);
%fit_neg_all = polyfit(train_sumneg_all, train_behav_all,1);
    
model_pos_all = fitlm(train_sumpos_all,train_behav_all,'RobustOpts','on');
model_neg_all = fitlm(train_sumneg_all,train_behav_all,'RobustOpts','on');
train_sum_cmb_all = [train_sumpos_all, train_sumneg_all];
model_cmb_all = fitlm(train_sum_cmb_all,train_behav_all,'RobustOpts','on');
% Get the coefficients of the regression line
coeffs_pos_all = model_pos_all.Coefficients.Estimate;
coeffs_neg_all = model_neg_all.Coefficients.Estimate;
coeffs_cmb_all = model_cmb_all.Coefficients.Estimate;

% run model on TEST sub


no_test_all = size(test_all_rescale,3);

test_sumpos_all = zeros(no_test_all,1);
test_sumneg_all = zeros(no_test_all,1);

for ss = 1:no_test_all
test_mat = test_all_rescale(:,:,ss);
test_sumpos(ss) = sum(sum(test_mat.*CPM_betamud_pos))/2;
test_sumneg(ss) = sum(sum(test_mat.*CPM_betamud_neg))/2;
end

%behav_pred_pos = fit_pos_all(1)*test_sumpos + fit_pos_all(2);
%behav_pred_neg = fit_neg_all(1)*test_sumneg + fit_neg_all(2);   
%behav_pred = b_all(1)*test_sumpos + b_all(2)*test_sumneg + b_all(3);

behav_pred_pos_all = coeffs_pos_all(1) + coeffs_pos_all(2)*test_sumpos;
behav_pred_neg_all = coeffs_neg_all(1) + coeffs_neg_all(2)*test_sumneg;
behav_pred_all = coeffs_cmb_all(1) + coeffs_cmb_all(2)*test_sumpos + coeffs_cmb_all(3)*test_sumneg;



%test_behav = craving;
[R_pos, P_pos] = corr(behav_pred_pos_all',test_behav_all,'type','Spearman')
[R_neg, P_neg] = corr(behav_pred_neg_all',test_behav_all,'type','Spearman')
[R_cmb, P_cmb] = corr(behav_pred_all',test_behav_all,'type','Spearman')


%MSE_pos = sum((behav_pred_pos'-test_behav_all).^2)/(no_test_all)
%MSE_neg = sum((behav_pred_neg'-test_behav_all).^2)/(no_test_all)
%MSE_cmb = sum((behav_pred'-test_behav_all).^2)/(no_test_all)

%RMSE_pos = sqrt(MSE_pos)
%RMSE_neg = sqrt(MSE_neg)
%RMSE_cmb = sqrt(MSE_cmb)

%q2_pos = 1 - MSE_pos/ var(test_behav_all, 1)
%q2_neg = 1 - MSE_neg/ var(test_behav_all, 1)
%q2_cmb = 1 - MSE_cmb/ var(test_behav_all, 1)


figure(1); plot(test_behav_all,behav_pred_pos_all','r.'); lsline
figure(2); plot(test_behav_all,behav_pred_neg_all','b.'); lsline
figure(3); plot(test_behav_all,behav_pred_all','g.'); lsline

plotname = {'Positive','Negative','Combined'};
for i = 1:3
figure(i);
xlabel('Observed Craving Score')
ylabel('Predicted Craving Score')
title(['External sample validation of ',char(plotname(i)), ' CPM'])
set(gca,'LooseInset',get(gca,'TightInset'));
filename = ["ExternalValidation_" + num2str(i)+'.tif'];
savename = fullfile(outdir, filename);
saveas(gcf,savename)
end

save(fullfile(outdir, [save_CPMname '_external.mat']),"pos_mud","neg_mud","behav_pred_pos_all","behav_pred_neg_all","behav_pred_all")

%% calculate all predicted craving intensity

% load cpm 
CPM_betamud_pos = pos_mud;
CPM_betamud_neg = neg_mud;
%train_mats_all = lesioned_mats_all;
% Rescale of external test dataset
% 1. calculate mean and std of all traning data

% conn_matrix_beta = test_mats_all;
% craving_all = test_behav_all;
no_node = size(conn_matrix_beta,1);
train_all_rescale = conn_matrix_beta;

%train_all_rescale = train_mats_all;
%test_all_rescale = test_mats_all;

no_sub_all = size(conn_matrix_beta,3);

train_sumpos_all = zeros(no_sub_all,1);
train_sumneg_all = zeros(no_sub_all,1);

for ss = 1:size(train_sumpos_all)
    train_sumpos_all(ss) = sum(sum(train_all_rescale(:,:,ss).*CPM_betamud_pos))/2;
    train_sumneg_all(ss) = sum(sum(train_all_rescale(:,:,ss).*CPM_betamud_neg))/2;
end
    

% combining both postive and negative features
%b_all = regress(train_behav_all, [train_sumpos_all, train_sumneg_all, ones(no_sub_all,1)]);
%fit_pos_all = polyfit(train_sumpos_all, train_behav_all,1);
%fit_neg_all = polyfit(train_sumneg_all, train_behav_all,1);
    
model_pos_all = fitlm(train_sumpos_all,craving_all,'RobustOpts','on');
model_neg_all = fitlm(train_sumneg_all,craving_all,'RobustOpts','on');
train_sum_cmb_all = [train_sumpos_all, train_sumneg_all];
model_cmb_all = fitlm(train_sum_cmb_all,craving_all,'RobustOpts','on');
% Get the coefficients of the regression line
coeffs_pos_all = model_pos_all.Coefficients.Estimate;
coeffs_neg_all = model_neg_all.Coefficients.Estimate;
coeffs_cmb_all = model_cmb_all.Coefficients.Estimate;

% run model on TEST sub


no_test_all = size(train_all_rescale,3);

test_sumpos_all = zeros(no_test_all,1);
test_sumneg_all = zeros(no_test_all,1);

for ss = 1:no_test_all
test_mat = train_all_rescale(:,:,ss);
test_sumpos(ss) = sum(sum(test_mat.*CPM_betamud_pos))/2;
test_sumneg(ss) = sum(sum(test_mat.*CPM_betamud_neg))/2;
end

%behav_pred_pos = fit_pos_all(1)*test_sumpos + fit_pos_all(2);
%behav_pred_neg = fit_neg_all(1)*test_sumneg + fit_neg_all(2);   
%behav_pred = b_all(1)*test_sumpos + b_all(2)*test_sumneg + b_all(3);

behav_pred_pos_all = coeffs_pos_all(1) + coeffs_pos_all(2)*test_sumpos;
behav_pred_neg_all = coeffs_neg_all(1) + coeffs_neg_all(2)*test_sumneg;
behav_pred_all = coeffs_cmb_all(1) + coeffs_cmb_all(2)*test_sumpos + coeffs_cmb_all(3)*test_sumneg;


% correlated variable
order_variable = 'craving_score'
order_by_variable = table2array(behav_all(:,order_variable)); 
[out,idx] = sort(order_by_variable);

conn_matrix_beta = conn_matrix_beta_orginal(:,:,[idx]);
craving_all = table2array(behav_all([idx],behav_name));



%test_behav = craving;
[R_pos, P_pos] = corr(behav_pred_pos_all',craving_all,'type','Spearman')
[R_neg, P_neg] = corr(behav_pred_neg_all',craving_all,'type','Spearman')
[R_cmb, P_cmb] = corr(behav_pred_all',craving_all,'type','Spearman')


%MSE_pos = sum((behav_pred_pos'-test_behav_all).^2)/(no_test_all)
%MSE_neg = sum((behav_pred_neg'-test_behav_all).^2)/(no_test_all)
%MSE_cmb = sum((behav_pred'-test_behav_all).^2)/(no_test_all)

%RMSE_pos = sqrt(MSE_pos)
%RMSE_neg = sqrt(MSE_neg)
%RMSE_cmb = sqrt(MSE_cmb)

%q2_pos = 1 - MSE_pos/ var(test_behav_all, 1)
%q2_neg = 1 - MSE_neg/ var(test_behav_all, 1)
%q2_cmb = 1 - MSE_cmb/ var(test_behav_all, 1)


figure(1); plot(test_behav_all,behav_pred_pos_all','r.'); lsline
figure(2); plot(test_behav_all,behav_pred_neg_all','b.'); lsline
figure(3); plot(test_behav_all,behav_pred_all','g.'); lsline

plotname = {'Positive','Negative','Combined'};
for i = 1:3
figure(i);
xlabel('Observed Craving Score')
ylabel('Predicted Craving Score')
title(['External sample validation of ',char(plotname(i)), ' CPM'])
set(gca,'LooseInset',get(gca,'TightInset'));
filename = ["ExternalValidation_" + num2str(i)+'.tif'];
savename = fullfile(outdir, filename);
saveas(gcf,savename)
end

save(fullfile(outdir, [save_CPMname '_external.mat']),"pos_mud","neg_mud","behav_pred_pos_all","behav_pred_neg_all","behav_pred_all")
