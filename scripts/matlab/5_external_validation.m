%% External Validation
CPM_betamud_pos = pos_mud;
CPM_betamud_neg = neg_mud;

% Extract the test matrices for the specified frequency band
test_mats_all = sample47_wplimatrix(:,:,:,band_number); 
behav_table_test = sample47_behavall;
test_behav_all = table2array(behav_table_test(:,behav_name));

% Number of nodes and subjects
no_node = size(train_mats_all, 1);
no_sub_all = size(train_mats_all, 3);

% Preallocate matrices for mean and std calculations
train_all_mean = zeros(no_node, no_node);
train_all_std = zeros(no_node, no_node);
train_all_rescale = zeros(size(train_mats_all));

% Calculate mean and std for each element in the training data
for i = 1:no_node
    for j = 1:no_node
        train_all_mean(i, j) = mean(train_mats_all(i, j, :), 3);
        train_all_std(i, j) = std(train_mats_all(i, j, :), 0, 3);
        train_all_rescale(i, j, :) = (train_mats_all(i, j, :) - train_all_mean(i, j)) ./ train_all_std(i, j);
    end 
end

% Handle NaN values in rescaled training data
train_all_rescale(isnan(train_all_rescale)) = 1;

% Normalize external test dataset
test_all_rescale = zeros(size(test_mats_all));
for i = 1:no_node
    for j = 1:no_node
        test_all_rescale(i, j, :) = (test_mats_all(i, j, :) - train_all_mean(i, j)) / train_all_std(i, j);
    end
end

% Handle NaN values in rescaled test data
test_all_rescale(isnan(test_all_rescale)) = 1;

% Preallocate arrays for sum calculations
train_sumpos_all = zeros(no_sub_all, 1);
train_sumneg_all = zeros(no_sub_all, 1);

% Calculate positive and negative feature sums for the training data
for ss = 1:no_sub_all
    train_sumpos_all(ss) = sum(sum(train_all_rescale(:, :, ss) .* CPM_betamud_pos)) / 2;
    train_sumneg_all(ss) = sum(sum(train_all_rescale(:, :, ss) .* CPM_betamud_neg)) / 2;
end

% Fit linear models
model_pos_all = fitlm(train_sumpos_all, train_behav_all, 'RobustOpts', 'on');
model_neg_all = fitlm(train_sumneg_all, train_behav_all, 'RobustOpts', 'on');
model_cmb_all = fitlm([train_sumpos_all, train_sumneg_all], train_behav_all, 'RobustOpts', 'on');

% Get the coefficients of the regression lines
coeffs_pos_all = model_pos_all.Coefficients.Estimate;
coeffs_neg_all = model_neg_all.Coefficients.Estimate;
coeffs_cmb_all = model_cmb_all.Coefficients.Estimate;

% Number of test subjects
no_test_all = size(test_all_rescale, 3);

% Preallocate arrays for sum calculations
test_sumpos = zeros(no_test_all, 1);
test_sumneg = zeros(no_test_all, 1);

% Calculate positive and negative feature sums for the test data
for ss = 1:no_test_all
    test_mat = test_all_rescale(:, :, ss);
    test_sumpos(ss) = sum(sum(test_mat .* CPM_betamud_pos)) / 2;
    test_sumneg(ss) = sum(sum(test_mat .* CPM_betamud_neg)) / 2;
end

% Predict behaviors
behav_pred_pos_all = coeffs_pos_all(1) + coeffs_pos_all(2) * test_sumpos;
behav_pred_neg_all = coeffs_neg_all(1) + coeffs_neg_all(2) * test_sumneg;
behav_pred_all = coeffs_cmb_all(1) + coeffs_cmb_all(2) * test_sumpos + coeffs_cmb_all(3) * test_sumneg;

% Calculate Spearman correlations
[R_pos, P_pos] = corr(behav_pred_pos_all, test_behav_all, 'Type', 'Spearman');
[R_neg, P_neg] = corr(behav_pred_neg_all, test_behav_all, 'Type', 'Spearman');
[R_cmb, P_cmb] = corr(behav_pred_all, test_behav_all, 'Type', 'Spearman');

% Display results
fprintf('Spearman correlation between predicted and actual (positive features): R = %.5f, P = %.5f\n', R_pos, P_pos);
fprintf('Spearman correlation between predicted and actual (negative features): R = %.5f, P = %.5f\n', R_neg, P_neg);
fprintf('Spearman correlation between predicted and actual (combined features): R = %.5f, P = %.5f\n', R_cmb, P_cmb);

figure(1); plot(test_behav_all,behav_pred_pos_all','r.'); lsline
figure(2); plot(test_behav_all,behav_pred_neg_all','b.'); lsline
figure(3); plot(test_behav_all,behav_pred_all','g.'); lsline

plotname = {'Positive','Negative','Combined'};
for i = 1:3
figure(i);
xlabel('Observed phenotype Score')
ylabel('Predicted phenotype Score')
title(['External sample validation of ',char(plotname(i)), ' CPM'])
set(gca,'LooseInset',get(gca,'TightInset'));
filename = ["ExternalValidation_" + num2str(i)+'.tif'];
savename = fullfile(outdir, filename);
saveas(gcf,savename)
end

save(fullfile(outdir, strcat(save_CPMname, '_external.mat')),"pos_mud","neg_mud","behav_pred_pos_all","behav_pred_neg_all","behav_pred_all")