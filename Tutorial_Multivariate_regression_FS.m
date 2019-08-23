
% For feature selection in multivariate regression data, load one of the provided dataset, e.g.
load('./Datasets/atp1d.mat')

% Discretize input-space using 5-bins equal width strategy
bins = 5;
X_inputs_disc = disc_dataset_equalwidth( inputs, bins );

% Now we will select the topK features with our criterion using the
% following parameters
topK = 10; % number of selected to be selected

Selected_with_Group_JMI = Group_JMI_Rand(X_inputs_disc,labels, topK, 'euclidean');

disp('Selected features using Group-JMI:')
disp(Selected_with_Group_JMI)