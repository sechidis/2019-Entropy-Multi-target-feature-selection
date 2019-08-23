function [selectedFeatures] = Group_JMI_Rand(X_data,Y_labels, topK, distance)
% Summary
%    Group_JMI_Rand algorithm for feature selection in multi-target problems
% Inputs
%    X_data: n x d matrix X, with categorical values for n examples and d features
%    Y_labels: n x q matrix with the labels
%    topK: Number of features to be selected
%    distance: the distance measure that will be used for clustering the output space,
%              for example 'hamming' for multi-label, and 'euclidean'for multivariate regression problems


num_features = size(X_data,2);
num_labels = size(Y_labels,2);
num_ensemble = num_labels;

% Create the cluster ensebmle, as a first approach I will generate the same
% number of targets as the initial, numLabels
for index_label = 1:num_ensemble
    
    Y_labels_new(:,index_label) = kmedoids(Y_labels(:, datasample(1:num_labels,randi([ceil(num_labels/4) floor(3*num_labels/4)]),'Replace',false)),randi([4 16]), 'Distance', distance);
    
end


%%%% Proceed to the BR
score_per_feature = zeros(1,num_features);
for index_feature = 1:num_features
    for index_label = 1:num_ensemble
        score_per_feature(index_feature) = score_per_feature(index_feature) + mi(X_data(:,index_feature),Y_labels_new(:,index_label))/sqrt(h(X_data(:,index_feature)) *h(Y_labels_new(:,index_label)));
    end
end
[val_max,selectedFeatures(1)]= max(score_per_feature);
not_selected_features = setdiff(1:num_features,selectedFeatures);

%%% Efficient implementation of the second step, at this point I will store
%%% the score of each feature. Whenever I select a feature I put NaN score
score_per_feature = zeros(1,num_features);
score_per_feature(selectedFeatures(1)) = NaN;
count = 2;

while count<=topK
    
    for index_feature_ns = 1:length(not_selected_features)
        
        for index_label = 1:num_ensemble
            
            score_per_feature(not_selected_features(index_feature_ns)) = score_per_feature(not_selected_features(index_feature_ns))+mi([X_data(:,not_selected_features(index_feature_ns)),X_data(:, selectedFeatures(count-1))], Y_labels_new(:,index_label))/sqrt(h([X_data(:,not_selected_features(index_feature_ns)),X_data(:, selectedFeatures(count-1))])*h(Y_labels_new(:,index_label)));
            
        end
    end
    
    [val_max,selectedFeatures(count)]= nanmax(score_per_feature);
    
    
    score_per_feature(selectedFeatures(count)) = NaN;
    not_selected_features = setdiff(1:num_features,selectedFeatures);
    count = count+1;
end


