%% predict_behavior function
% divide to positive and negative correlation 
function [R_pos, R_neg, R_total,RMSE_cmb] = predict_behavior(rest_1_mats,  behavior, threshold)
% ------------ INPUTS -------------------

all_mats  = rest_1_mats;
all_behav = behavior;

% threshold for feature selection
thresh = threshold;

% ---------------------------------------

no_sub = size(all_mats,3);
no_node = size(all_mats,1);

behav_pred_pos = zeros(no_sub,1);
behav_pred_neg = zeros(no_sub,1);
behav_pred = zeros(no_sub,1);

for leftout = 1:no_sub
%    fprintf('\n Leaving out subj # %6.3f',leftout);
    
    % leave out subject from matrices and behavior
    
    train_mats = all_mats; 
    train_mats(:,:,leftout) = [];
    train_vcts = reshape(train_mats,[],size(train_mats,3));
    
    train_behav = all_behav;
    train_behav(leftout) = [];
    
    %-----------------------train data standardization----------------------------% 
    
%    for i = 1:no_node
%       for j = 1:no_node
%           train_edges_mean(i,j) = mean(train_mats(i,j,:),3);
%           train_edges_std(i,j) = std(train_mats(i,j,:));
%           train_mats(i,j,:) = (train_mats(i,j,:) - train_edges_mean(i,j))./train_edges_std(i,j);
%       end 
%    end
   
%    train_mats(isnan(train_mats)) = 1;
%-----------------------------------------------------------------% 
    
%    [r_mat, p_mat] = corr(train_vcts', train_behav); 
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
    %[r_mat, p_mat] = corr(train_vcts', train_behav, 'type', 'Spearman');
%    r_mat = reshape(r_mat,no_node,no_node);
%    p_mat = reshape(p_mat,no_node,no_node);
    
%     % correlate all edges with behavior using partial correlation
%     [r_mat, p_mat] = partialcorr(train_vcts', train_behav, age);
%     
%        
%     % correlate all edges with behavior using rank correlation
%     [r_mat, p_mat] = corr(train_vcts', train_behav, 'type', 'Spearman');
     [r_mat, p_mat] = corr(train_vcts', train_behav);
        

    
%    % set threshold and define masks 
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
    
    % get sum of all edges in TRAIN subs (divide by 2 to control for the
    % fact that matrices are symmetric)
    
    train_sumpos = zeros(no_sub-1,1);
    train_sumneg = zeros(no_sub-1,1);
    
    for ss = 1:size(train_sumpos)
        train_sumpos(ss) = sum(sum(train_mats(:,:,ss).*pos_mask))/2;
        train_sumneg(ss) = sum(sum(train_mats(:,:,ss).*neg_mask))/2;
    end
    
    if (sum(train_sumpos) == 0) &&(sum(train_sumneg) == 0)
        behav_pred(leftout) = NaN;
        behav_pred_pos(leftout) = NaN;
        behav_pred_neg(leftout) = NaN;
    else
    
    % build model on TRAIN subs
    % combining both postive and negative features
 %   b = regress(train_behav, [train_sumpos, train_sumneg, ones(no_sub-1,1)]);
 %   fit_pos = polyfit(train_sumpos, train_behav,1);
 %   fit_neg = polyfit(train_sumneg, train_behav,1);
    
    model_pos = fitlm(train_sumpos,train_behav,'RobustOpts','on');
    model_neg = fitlm(train_sumneg,train_behav,'RobustOpts','on');
    train_sum_cmb = [train_sumpos, train_sumneg];
    model_cmb = fitlm(train_sum_cmb,train_behav,'RobustOpts','on');
    % Get the coefficients of the regression line
    coeffs_pos = model_pos.Coefficients.Estimate;
    coeffs_neg = model_neg.Coefficients.Estimate;
    coeffs_cmb = model_cmb.Coefficients.Estimate;
    
    warning('off','all');
    
    % run model on leave sub
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

%    behav_pred_pos(leftout) = fit_pos(1)*test_sumpos + fit_pos(2);
%    behav_pred_neg(leftout) = fit_neg(1)*test_sumneg + fit_neg(2);    
%    behav_pred(leftout) = b(1)*test_sumpos + b(2)*test_sumneg + b(3);
    behav_pred_pos(leftout) = coeffs_pos(1) + coeffs_pos(2)*test_sumpos;
    behav_pred_neg(leftout) = coeffs_neg(1) + coeffs_neg(2)*test_sumneg;
    behav_pred(leftout) = coeffs_cmb(1) + coeffs_cmb(2)*test_sumpos + coeffs_cmb(3)*test_sumneg;
    
    end
end
% compare predicted and observed scores
    
    if sum(sum(isnan(behav_pred))) >= 0.02 * length(behav_pred)
        R_pos = -1;
        R_neg = -1;
        R_total = -1;
        RMSE_cmb = -1;
    else
        [R_pos, ~] = corr(behav_pred_pos(~isnan(behav_pred_pos)),all_behav(~isnan(behav_pred_pos)), 'type', 'Spearman');
        [R_neg, ~] = corr(behav_pred_neg(~isnan(behav_pred_neg)),all_behav(~isnan(behav_pred_neg)), 'type', 'Spearman');
        [R_total, ~] = corr(behav_pred(~isnan(behav_pred)),all_behav(~isnan(behav_pred)), 'type', 'Spearman');
        RMSE_cmb = sqrt(sum((behav_pred(~isnan(behav_pred))-all_behav(~isnan(behav_pred))).^2)/(sum(~isnan(behav_pred))-3-1));
    end

end