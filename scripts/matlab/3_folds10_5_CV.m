%% Distribution of K-fold Cross-Validation
clear;
clc;

%---------------------- initialization ----------------------%
% load data and set path
rawdir = 'E:\research_data\Methamphetamine_ERP_EGI\CPM_EEG_craving\';  % parent path to data and scripts
scrdir = fullfile(rawdir,'scripts');     % path to scripts
datadir = fullfile(rawdir,'data');     % path to data
addpath(rawdir)
addpath(scrdir)
addpath(datadir)

load('sample144.mat')
load('sample47.mat')
bandname = {'theta', 'alpha', 'beta', 'gamma'};

% phenotype and frequency band of brain activity selection
band = 'beta'; % theta 1, alpha 2, beta 3, gamma 4
behav_name = "craving_score"; % craving_scroe, withdraw_day

outdir = fullfile(rawdir,'/results/', band);          % path to output depending on

if ~exist(outdir)
    mkdir(outdir)
end

band_number = find(strcmp(bandname, band));

% Here, the last dimention of sample*_wplimatrix is frequency band
conn_matrix_oneband = sample144_wplimatrix(:,:,:,band_number); 
behav_table = sample144_behavall;
phenotype_all = table2array(behav_table(:,behav_name));

% ------------ INPUTS -------------------

all_mats  = conn_matrix_oneband;
all_behav = phenotype_all;

% threshold for feature selection
%For edge selection, you can try a range of thresholds, p=0.05, 0.01, 0.005, etc. Usually, the performance should not be sensitive to the p threshold. 
%If the model only works at a single threshold, and assuming that you are running the leave-one-out validation scheme, you may also want to run k-fold validation to verify the result.
%thresh_all = 0.001:0.001:0.02;
%R_all = zeros(14,5);
thresh_all = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05];
kfolds = 5
no_iterations   = 100;

for i = 1:length(thresh_all)
thresh = thresh_all(i);

%thresh = 0.01;
R_pos = zeros(no_iterations,1);
R_neg = zeros(no_iterations,1);
R_cmb = zeros(no_iterations,1);
P_pos = zeros(no_iterations,1);
p_neg = zeros(no_iterations,1);
p_cmb = zeros(no_iterations,1);
%for n = 1:length(thresh_all)

%fprintf('\n n = %6.3f',n);
    for it = 1:no_iterations
    fprintf('\n# Running over %1.0f iterations.\nPerforming fold no. ',it);


    % ---------------------------------------

    no_sub = size(all_mats,3);
    no_node = size(all_mats,1);

    pos_consensus = zeros(no_node, no_node);
    neg_consensus = zeros(no_node, no_node);

    randinds=randperm(no_sub);
    %ksample=floor(no_sub/kfolds);


    behav_pred_pos = zeros(no_sub,1);
    behav_pred_neg = zeros(no_sub,1);
    behav_pred = zeros(no_sub,1);

    for leftout = 1:kfolds
    fprintf('%1.0f ',leftout);
    
    % leave out subject from matrices and behavior
    
    if kfolds == no_sub % doing leave-one-out
        testinds=randinds(leftout);
        traininds=setdiff(randinds,testinds);
    else
        if leftout <= rem(no_sub,kfolds)
        si=1+((leftout-1)*(fix(no_sub/kfolds)+1));
        fi=si+(fix(no_sub/kfolds)+1)-1;
        
        testinds=randinds(si:fi);
        traininds=setdiff(randinds,testinds);
        else 
        si= fi+1;
        fi=si+fix(no_sub/kfolds)-1;
        
        testinds=randinds(si:fi);
        traininds=setdiff(randinds,testinds);    
        end
    end
    
    
    % Assign x and y data to train and test groups 
    train_mats = all_mats;
    train_mats = train_mats(:,:,traininds);
    train_vcts = reshape(train_mats,[],size(train_mats,3));
    
    test_mats = all_mats;
    test_mats = test_mats(:,:,testinds);
    test_vcts = reshape(test_mats,[],size(test_mats,3));
    
    train_behav = all_behav(traininds);
    
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
%     [r_mat, p_mat] = partialcorr(train_vcts', train_behav, age);
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
    
    train_sumpos = zeros(length(traininds),1);
    train_sumneg = zeros(length(traininds),1);
    
    for ss = 1:size(train_sumpos)
        train_sumpos(ss) = sum(sum(train_mats(:,:,ss).*pos_mask))/2;
        train_sumneg(ss) = sum(sum(train_mats(:,:,ss).*neg_mask))/2;
    end
    
    if (sum(train_sumpos) == 0 )&&(sum(train_sumneg) == 0)

       for testind = 1:length(testinds)
           
           behav_pred_pos(testinds(testind)) = NaN;
           behav_pred_neg(testinds(testind)) = NaN;
           behav_pred(testinds(testind)) = NaN;
           
       end
       
    else    
        % build model on TRAIN subs
        % combining both postive and negative features
%        b = regress(train_behav, [train_sumpos, train_sumneg, ones(length(traininds),1)]);
%        fit_pos = polyfit(train_sumpos, train_behav,1);
%        fit_neg = polyfit(train_sumneg, train_behav,1);
    
    model_pos = fitlm(train_sumpos,train_behav,'RobustOpts','on');
    model_neg = fitlm(train_sumneg,train_behav,'RobustOpts','on');
    train_sum_cmb = [train_sumpos, train_sumneg];
    model_cmb = fitlm(train_sum_cmb,train_behav,'RobustOpts','on');
    % Get the coefficients of the regression line
    coeffs_pos = model_pos.Coefficients.Estimate;
    coeffs_neg = model_neg.Coefficients.Estimate;
    coeffs_cmb = model_cmb.Coefficients.Estimate;
    
        % run model on TEST sub

        for testind = 1:length(testinds)
            test_mat = test_mats(:,:,testind);
            test_sumpos = sum(sum(test_mat.*pos_mask))/2;
            test_sumneg = sum(sum(test_mat.*neg_mask))/2;

            %if want to combine postive and negative

    %    behav_pred_pos(leftout) = fit_pos(1)*test_sumpos + fit_pos(2);
    %    behav_pred_neg(leftout) = fit_neg(1)*test_sumneg + fit_neg(2);    
    %    behav_pred(leftout) = b(1)*test_sumpos + b(2)*test_sumneg + b(3);
        behav_pred_pos(testinds(testind)) = coeffs_pos(1) + coeffs_pos(2)*test_sumpos;
        behav_pred_neg(testinds(testind)) = coeffs_neg(1) + coeffs_neg(2)*test_sumneg;
        behav_pred(testinds(testind)) = coeffs_cmb(1) + coeffs_cmb(2)*test_sumpos + coeffs_cmb(3)*test_sumneg;
        end
    end
end
% compare predicted and observed scores
    if sum(isnan(behav_pred)) >= 0.1*length(behav_pred)
        R_pos(it) = -1;
        R_neg(it) = -1;
        R_cmb(it) = -1;
    else
        [R_pos(it), P_pos(it)] = corr(behav_pred_pos(~isnan(behav_pred_pos)),all_behav(~isnan(behav_pred_pos)), 'type', 'Spearman');
        [R_neg(it), P_neg(it)] = corr(behav_pred_neg(~isnan(behav_pred_neg)),all_behav(~isnan(behav_pred_neg)), 'type', 'Spearman');
        [R_cmb(it), P_cmb(it)] = corr(behav_pred(~isnan(behav_pred)),all_behav(~isnan(behav_pred)), 'type', 'Spearman');
    end
    
    end
%    R_CV_pos(i,:) = [mean(R_neg()),median(R_neg),std(R_neg)]
%    R_CV_neg(i,:) = [mean(R_pos),median(R_pos),std(R_pos)]
    R_CV_cmb(i,:) = [mean(R_cmb(R_cmb ~= -1)),median(R_cmb(R_cmb ~= -1)),std(R_cmb(R_cmb ~= -1))]

% plot distribution
R_all = zeros(no_iterations,3);
%R_all(:,1) = R_pos;
%R_all(:,2) = R_neg;
R_all(:,3) = R_cmb;
%pval(1) = pval_pos;
%pval(2) = pval_neg;
%pval(3) = pval_total;
plotname = {'Positive','Negative','Combined'};
%tiledlayout(3,1)
for plot_i = 3 %only plot CMB CPM, if want others, select 1:3
    figure(plot_i); 
    %nexttile
    histogram(R_all(:,plot_i),'Normalization','probability','BinWidth',0.01,'FaceAlpha',0.3,'EdgeAlpha',0.3)
    medianRtxt = ['r = ',num2str(round(median(R_cmb(R_cmb ~= -1)),3)),newline,'(',num2str(round(std(R_cmb(R_cmb ~= -1)),3)),')'];
    xline(median(R_all(:,plot_i)),'r','LineWidth',1,'Label',medianRtxt,'LabelOrientation','horizontal')
    xlabel('Pearson r')
    ylabel('Frequency')
    title([char(plotname(plot_i)), ' CPM ', num2str(kfolds), '-fold-CV distribution'])
    set(gca,'LooseInset',get(gca,'TightInset'));
    filename = ['distribution_',band,'_', num2str(kfolds),'CV', num2str(thresh), '_', num2str(plot_i),'.tif'];
    savename = fullfile(outdir,filename);
    saveas(gcf,savename)
end

save(fullfile(outdir,[band, num2str(kfolds),'fold_', num2str(thresh) ,'.mat']),"R_pos","R_neg","R_cmb")
close all
end

save(fullfile(outdir,[band, num2str(kfolds),'fold_CV' ,'.mat']),"R_CV_cmb")