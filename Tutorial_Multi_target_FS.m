% Load one of the provided dataset, e.g.
load('./Datasets/emotions.mat')

% Discretize input-space using 5-bins equal width strategy
bins = 5;
X_inputs_disc = disc_dataset_equalwidth( inputs, bins );


% Now we will select the topK features with our criterion using the
% following parameters
topK = 10 % number of selected to be selected
num_ensemble = size(labels, 2) % Size of the transformed input-space, in this case we use the same as the number of labels
PoT = 0.50 %  Proportion of Targets
NoC = 4 % Number of Clusters
Selected_with_Group_JMI = Group_JMI(X_inputs_disc, labels, topK, num_ensemble, PoT, NoC);
disp('Selected features using Group-JMI:')
disp(Selected_with_Group_JMI)