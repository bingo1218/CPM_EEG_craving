%% prediction all other factors with CPM
rawdir = 'E:\research_data\Methamphetamine_ERP_EGI\connectome-based analysis\CPM_ZHB\CPM_formal\0718_update\supplement\';  % path to raw data
scrdir = fullfile(rawdir,'scripts');     % path to scripts
datadir = fullfile(rawdir,'data');     % path to scripts

addpath(rawdir)
addpath(scrdir)
addpath(datadir)

load('sampleEO87MUD.mat')
bandname = {'theta', 'alpha', 'beta', 'gamma'};
threshold_all = [0.2];%0.01, 0.02, 0.05, 0.1, 0.2];
outdir = fullfile(rawdir, 'otherCPM');          % path to output depending on
    if ~exist(outdir)
        mkdir(outdir)
    end
percentage_edges = 0.95;
no_iterations = 1000; %permutation_test number

for band_i = [4] %:4] % band name: theta 1, alpha 2, beta 3, gamma 4
    band = bandname{band_i};
    for phenotype_name = {'PSQI',  'BIS'}%{'addiction', 'abestinence', 'dose', 'craving', 'PSQI',  'BIS'} BDI
        fprintf('\nBand: %s, Phenotype: %s', band, phenotype_name{1});
        % Here, the last dimention is frequency band
            conn_matrix_oneband = sample_EO_87(:,:,:,band_i);
            no_sub = size(conn_matrix_oneband,3);
            no_node = size(conn_matrix_oneband,1);
            behav_table = relatedtableEO;
            phenotype = table2array(behav_table(:,phenotype_name));
            
            % renew stored parameters
            parameters = NaN(length(threshold_all),9);
             
        for thresholdi = 1:length(threshold_all)
            prediction_r = NaN(length(no_iterations),1);
            RMSE_all = NaN(length(no_iterations),1);
            threshold = threshold_all(thresholdi);
            
            % run CPM
            [R_cmb,P_cmb, RMSE_cmb, pos_consensus, neg_consensus] = CPM_train(conn_matrix_oneband,  phenotype, threshold);
            prediction_r(1) = R_cmb;
            RMSE_all(1) = RMSE_cmb;
            
            % run permutation test
            for it= 2:no_iterations %2:no_iterations
            fprintf('\n Performing iteration %d out of %d', it, no_iterations);
            new_phenotype        = phenotype(randperm(no_sub));
            [prediction_r(it), RMSE_all(it,1), ~, ~] = CPM_train(conn_matrix_oneband, new_phenotype, threshold);
            end
            
            true_prediction_r_total = prediction_r(1);
            sorted_prediction_r_total = sort(prediction_r(:),'descend');
            if true_prediction_r_total ~= -1
            position_total            = find(sorted_prediction_r_total(sorted_prediction_r_total~= -1)==true_prediction_r_total);
            pval_total                = position_total(1)/length(sorted_prediction_r_total(sorted_prediction_r_total~= -1))
            else 
                pval_total = -1;
            end
            true_RMSE_all = RMSE_all(1)
            sorted_RMSE_all = sort(RMSE_all,'descend');
            if true_RMSE_all ~= -1;
            position_RMSE            = find(sorted_RMSE_all (sorted_RMSE_all ~= -1) == true_RMSE_all);
                        pval_RMSE                = position_RMSE/length(sorted_RMSE_all(sorted_RMSE_all ~= -1))

            else
                pval_RMSE = -1;
            end
            success_rate = 1-(sum(prediction_r(:) == -1)/1000)
            
            pos_mud = (pos_consensus/no_sub >= percentage_edges);
            neg_mud = (neg_consensus/no_sub >= percentage_edges);
            pos_mud = double(pos_mud);
            neg_mud = double(neg_mud);
            
            parameters(thresholdi,1) = R_cmb;
            parameters(thresholdi,2) = P_cmb;
            parameters(thresholdi,3) = RMSE_cmb;
            parameters(thresholdi,4) = sum(sum(pos_mud));
            parameters(thresholdi,5) = sum(sum(neg_mud));
            parameters(thresholdi,6) = threshold;3
            parameters(thresholdi,7) = pval_total;
            parameters(thresholdi,8) = pval_RMSE;
            parameters(thresholdi,9) = success_rate;
        end
       % outputname
       savename  = fullfile(outdir,[band,'_',phenotype_name{:}, '_EO', '.mat']);
       save(savename,'parameters');
    end
end