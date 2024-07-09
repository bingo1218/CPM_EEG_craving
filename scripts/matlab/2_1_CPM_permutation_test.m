
%% Permutation test

rawdir = 'E:\research_data\Methamphetamine_ERP_EGI\CPM_EEG_craving\';
no_iterations = 1000;
outdir = fullfile(rawdir,'\results\', band); 
for thresh = [0.02]
    all_mats = train_mats_all;
    all_behav = train_behav_all;

    % Calculate the true prediction correlation
    [true_prediction_r_pos, true_prediction_r_neg, true_prediction_r_total, RMSE_raw] = predict_behavior(all_mats, all_behav, thresh);

    prediction_r = zeros(no_iterations, 3);
    RMSE_all = zeros(no_iterations, 1);
    prediction_r(1, :) = [true_prediction_r_pos, true_prediction_r_neg, true_prediction_r_total];
    RMSE_all(1) = RMSE_raw;

    for it = 2:no_iterations
        fprintf('\n Performing iteration %d out of %d', it, no_iterations);
        new_behav = all_behav(randperm(no_sub));
        [prediction_r(it, 1), prediction_r(it, 2), prediction_r(it, 3), RMSE_all(it)] = predict_behavior(all_mats, new_behav, thresh);
    end

    filename = ['permutation_test_', band, '_LOOCV_', num2str(thresh), '_1000.mat'];
    save(fullfile(outdir, filename), 'prediction_r', 'RMSE_all');
    
    % calculate permutation p
    %%%% you can also only run the below chunk after load
    % permutation_test_[frequency band]_LOOCV_[threshold]_1000.mat to extract the pvalue%%%%%
    % band = 'beta';thresh = 0.05;filename = ['permutation_test_', band, '_LOOCV_', num2str(thresh), '_1000.mat']; load(fullfile(outdir, filename))
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
