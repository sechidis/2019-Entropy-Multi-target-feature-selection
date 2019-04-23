function selectedFeatures = Group_JMI(X_inputs,Y_targets, topK, num_ensemble, PoT, NoC)
% Summary
%    JMI_Group algorithm for feature selection
% Inputs
%    X_inputs: n x d matrix X, with categorical values for n examples and d features
%    Y_targets: n x q matrix with the labels
%    topK: Number of features to be selected


numFeatures = size(X_inputs,2);
num_labels = size(Y_targets,2);


%%% Do the grouping and the transforming of the output space through clustering 
for index_label = 1:num_ensemble
    Groups(index_label,:) = datasample(1:num_labels,ceil(PoT*num_labels),'Replace',false);
    Y_transformed(:,index_label) = kmedoids(Y_targets(:, Groups(index_label,:)), NoC, 'Distance', 'hamming'); 
end


%%% First step, calculate the mutual information between each feature and Y_transformed
score_per_feature = zeros(1,numFeatures);
for index_feature = 1:numFeatures
    for index_label = 1:num_ensemble
        score_per_feature(index_feature) = score_per_feature(index_feature) + mi(X_inputs(:,index_feature),Y_transformed(:,index_label))/sqrt(h(X_inputs(:,index_feature)) *h(Y_transformed(:,index_label)));
    end
end
[val_max,selectedFeatures(1)]= max(score_per_feature);
not_selected_features = setdiff(1:numFeatures,selectedFeatures);

%%% Efficient implementation of the second step, at this point we will store
%%% the score of each feature. Whenever we select a feature we add a NaN score
score_per_feature = zeros(1,numFeatures);
score_per_feature(selectedFeatures(1)) = NaN;
count = 2;

while count<=topK   

    for index_feature_ns = 1:length(not_selected_features)
        
        for index_label = 1:num_ensemble
            
            score_per_feature(not_selected_features(index_feature_ns)) = score_per_feature(not_selected_features(index_feature_ns))+mi([X_inputs(:,not_selected_features(index_feature_ns)),X_inputs(:, selectedFeatures(count-1))], Y_transformed(:,index_label))/sqrt(h([X_inputs(:,not_selected_features(index_feature_ns)),X_inputs(:, selectedFeatures(count-1))])*h(Y_transformed(:,index_label)));
            
        end
    end
    
    [val_max,selectedFeatures(count)]= nanmax(score_per_feature);
    
    
    score_per_feature(selectedFeatures(count)) = NaN;
    not_selected_features = setdiff(1:numFeatures,selectedFeatures);
    count = count+1;
end

