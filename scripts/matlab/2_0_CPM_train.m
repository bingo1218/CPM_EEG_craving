% After selecting the appropriate threshold, some variables were created
% in 1_Pselection.m script

%% LOOCV
thresh = 0.02;
save_CPMname = strcat('CPM_', behav_name,'_', band, '_', num2str(thresh));

train_mats_all  = conn_matrix_oneband;
train_behav_all = phenotype_all;

no_sub = size(train_mats_all, 3);
no_node = size(train_mats_all, 1);

behav_pred_pos = zeros(no_sub, 1);
behav_pred_neg = zeros(no_sub, 1);
behav_pred = zeros(no_sub, 1); 

pos_consensus = zeros(no_node, no_node);
neg_consensus = zeros(no_node, no_node);

for leftout = 1:no_sub
    fprintf('\n Leaving out subj # %6.3f', leftout);
    
    % Leave out subject from matrices and behavior
    train_mats = train_mats_all;
    train_mats(:, :, leftout) = [];
    
    train_vcts = reshape(train_mats, [], size(train_mats, 3));
    train_behav = train_behav_all;
    train_behav(leftout) = [];
    
    % Correlate all edges with behavior using Pearson correlation
    [r_mat, p_mat] = corr(train_vcts', train_behav);
    r_mat = reshape(r_mat, no_node, no_node);
    p_mat = reshape(p_mat, no_node, no_node);

    % Set threshold and define masks 
    pos_mask = (r_mat > 0) & (p_mat < thresh);
    neg_mask = (r_mat < 0) & (p_mat < thresh);
    
    pos_consensus = pos_consensus + pos_mask;
    neg_consensus = neg_consensus + neg_mask;

    % Get sum of all edges in TRAIN subs (divide by 2 to control for symmetry)
    train_sumpos = squeeze(sum(sum(train_mats .* pos_mask, 1), 2) / 2);
    train_sumneg = squeeze(sum(sum(train_mats .* neg_mask, 1), 2) / 2);

    % Build model on TRAIN subs
    model_pos = fitlm(train_sumpos, train_behav, 'RobustOpts', 'on');
    model_neg = fitlm(train_sumneg, train_behav, 'RobustOpts', 'on');
    model_cmb = fitlm([train_sumpos, train_sumneg], train_behav, 'RobustOpts', 'on');

    % Get the coefficients of the regression line
    coeffs_pos = model_pos.Coefficients.Estimate;
    coeffs_neg = model_neg.Coefficients.Estimate;
    coeffs_cmb = model_cmb.Coefficients.Estimate;

    % Run model on TEST sub
    test_mat = train_mats_all(:, :, leftout);
    test_sumpos = sum(sum(test_mat .* pos_mask)) / 2;
    test_sumneg = sum(sum(test_mat .* neg_mask)) / 2;

    behav_pred_pos(leftout) = coeffs_pos(1) + coeffs_pos(2) * test_sumpos;
    behav_pred_neg(leftout) = coeffs_neg(1) + coeffs_neg(2) * test_sumneg;
    behav_pred(leftout) = coeffs_cmb(1) + coeffs_cmb(2) * test_sumpos + coeffs_cmb(3) * test_sumneg;
end
model1_pred = behav_pred; % prepare for steiger's Z for sensitivity analysis
% Calculate consensus masks
percentage_edges = 0.95;
pos_mud = double(pos_consensus / no_sub >= percentage_edges);
neg_mud = double(neg_consensus / no_sub >= percentage_edges);

% Save masks
save(fullfile(outdir, strcat(save_CPMname, '_pos_', num2str(percentage_edges), '.csv')), "pos_mud", '-ascii')
save(fullfile(outdir, strcat(save_CPMname, '_neg_', num2str(percentage_edges), '.csv')), "neg_mud", '-ascii')

% Compare predicted and observed scores
[R_pos, P_pos] = corr(behav_pred_pos, train_behav_all, 'type', 'Spearman')
[R_neg, P_neg] = corr(behav_pred_neg, train_behav_all, 'type', 'Spearman')
[R_cmb, P_cmb] = corr(behav_pred, train_behav_all, 'type', 'Spearman')

% Calculate RMSE
RMSE_cmb = sqrt(sum((behav_pred - train_behav_all).^2) / (no_sub - 3 - 1))

% Save results
save(fullfile(outdir, strcat(save_CPMname, '_internal.mat')), "pos_mud", "neg_mud", "behav_pred_pos", "behav_pred_neg", "behav_pred");

% Plot results
figure(1); plot(train_behav_all, behav_pred_pos, 'r.'); lsline
figure(2); plot(train_behav_all, behav_pred_neg, 'b.'); lsline
figure(3); plot(train_behav_all, behav_pred, 'g.'); lsline

plotname = {'Positive', 'Negative', 'Combined'};
for i = 1:3
    figure(i);
    xlabel('Observed phenotype Score');
    ylabel('Predicted phenotype Score');
    title(['The predictive ability of ', char(plotname(i)), ' CPM']);
    set(gca, 'LooseInset', get(gca, 'TightInset'));
    filename = ["InternalValidation_" + num2str(i) + '.tif'];
    savename = fullfile(outdir, filename);
    saveas(gcf, savename);
end

close all;