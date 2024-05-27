%%
clc
clear 
close all
%% 

dataChest = fullfile('F:\Scene recognition\Data');
imds2 = imageDatastore(dataChest, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

%%  Dividir o conjunto de dados em cada categoria
    [testSet,trainingSet] = splitEachLabel(imds2, 0.2, 'randomize');
%%%%%%%code for resizing
inputSize=[40 40 3];
trainingSet1=augmentedImageDatastore(inputSize, trainingSet);
testSet1=augmentedImageDatastore(inputSize, testSet);

      

lgraph = layerGraph;

tempLayers = imageInputLayer([40 40 3],"Name","Scene Images");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],3,"Name","conv_1","Padding","same")
    maxPooling2dLayer([2 2],"Name","maxpool_1","Padding","same")
    reluLayer("Name","relu_1")
    convolution2dLayer([3 3],3,"Name","conv_3","Padding","same")
    maxPooling2dLayer([2 2],"Name","maxpool_3","Padding","same")
    reluLayer("Name","relu_3")
    convolution2dLayer([3 3],3,"Name","conv_5","Padding","same")
    maxPooling2dLayer([2 2],"Name","maxpool_5","Padding","same")
    reluLayer("Name","relu_5")
    fullyConnectedLayer(1024,"Name","fc_1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],3,"Name","conv_2","Padding","same")
    maxPooling2dLayer([2 2],"Name","maxpool_2","Padding","same")
    reluLayer("Name","relu_2")
    batchNormalizationLayer("Name","batch1");
    convolution2dLayer([3 3],3,"Name","conv_4","Padding","same")
    maxPooling2dLayer([2 2],"Name","maxpool_4","Padding","same")
    reluLayer("Name","relu_4")
    batchNormalizationLayer("Name","batch2")
    convolution2dLayer([3 3],3,"Name","conv_6","Padding","same")
    maxPooling2dLayer([2 2],"Name","maxpool_6","Padding","same")
    reluLayer("Name","relu_6")
    convolution2dLayer([3 3],3,"Name","conv_7","Padding","same")
    maxPooling2dLayer([2 2],"Name","maxpool_7","Padding","same")
    reluLayer("Name","relu_7")
    fullyConnectedLayer(1024,"Name","fc_2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition")
    reluLayer("Name","relu_8")
    fullyConnectedLayer(600,"Name","Feature Selection Layer")
    reluLayer("Name","relu_9")
    dropoutLayer(0.7,"Name","dropout")
    fullyConnectedLayer(6,"Name","fc_3")
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classoutput")];
lgraph = addLayers(lgraph,tempLayers);

lgraph = connectLayers(lgraph,"Scene Images","conv_1");
lgraph = connectLayers(lgraph,"Scene Images","conv_2");
lgraph = connectLayers(lgraph,"fc_1","addition/in2");
lgraph = connectLayers(lgraph,"fc_2","addition/in1");

% figure
%  plot(lgraph)
% %%

train_options = trainingOptions('adam', ...
    'MiniBatchSize',8, ...
    'MaxEpochs',30, ...
    'InitialLearnRate',3.0000000e-04, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.5, ...
    'LearnRateDropPeriod',5, ...
    'Shuffle','every-epoch', ...
    'ValidationData',testSet1, ...
    'ValidationFrequency',87, ...
    'Plots','training-progress', ...
    'Verbose',false);

 net2 = trainNetwork(trainingSet1, lgraph, train_options);
%%
% Extract the features from the fully connected layer after addition

featureVector = net2.Layers(29, 1).Bias;

% Generate a random feature vector as an example (replace with your own data)

% Calculate mutual information between each feature and the rest of the features
mutual_info = zeros(1, numel(featureVector));

for i = 1:numel(featureVector)
    other_features = featureVector([1:i-1, i+1:end]);
    mutual_info(i) = mutualinformation(featureVector(i), other_features);
end

% Sort the features based on their mutual information (in descending order)
[sortedMutualInfo, featureIndices] = sort(mutual_info, 'descend');

% Select the top 400 features with the highest mutual information
selectedFeatureIndices = featureIndices(1:400);

% Extract the selected features from the original feature vector
selectedFeatures = featureVector(selectedFeatureIndices);

% Display the selected feature indices
disp('Selected Feature Indices:');
disp(selectedFeatureIndices);

% Check the size of the selected features
disp(['Size of Selected Features: ' num2str(numel(selectedFeatures))]);

selectedFeatures = ones(400,1);
% Create a new fully connected layer using the optimized features
newFcLayer = fullyConnectedLayer(400, 'Name', 'fc_optimized', 'Bias', selectedFeatures);

% Replace the original fully connected layer with the optimized layer
lgraph = replaceLayer(lgraph, net2.Layers(24, 1).Name  , newFcLayer);

% Define new training options for fine-tuning
fineTuneOptions = trainingOptions('adam', ...
    'MiniBatchSize', 4, ...
    'MaxEpochs', 30, ...
    'InitialLearnRate', 3.0000000e-04, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', testSet1, ...
    'ValidationFrequency', 87, ...
    'Plots', 'training-progress', ...
    'Verbose', false);

% Fine-tune the network with the optimized features
net3 = trainNetwork(trainingSet1, lgraph, fineTuneOptions);
%%

YPred = classify(net3,testSet1);
figure
cm = confusionchart(testSet.Labels,YPred);
accuracy = sum(YPred == testSet.Labels)/numel(testSet.Labels)

save('SceneRecognition.mat','net3');
 analyzeNetwork(net2)