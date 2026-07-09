%% READ WINE DATA

% Include some breaf description here

clear; clc
wine = wineConfig;

%% LINEAR DISCRIMINANT ANALYSIS

% This model uses the leave-one-out approach. It runs 178 models leaving
% one sample out every time. LDA is not affected by the different scales
% the attributes are showing, thus we don't do anything to the original
% data.

wineModel = fitcdiscr(wine.data,categorical(wine.id), "DiscrimType","linear",...
    'Prior','empirical', 'Leaveout','on'); 

% Here he quantify the percentage of missclassifications (performance)
looError = kfoldLoss(wineModel);
looAccuracy = 1 - looError;
fprintf('Leave-one-out accuracy: %.2f%%\n', looAccuracy*100);

% 
[predictedLabels, scores] = kfoldPredict(wineModel);
trueLabels = categorical(wine.id);

figure;
confusionchart(trueLabels, predictedLabels);

%%
% This is the full model without any cross-validation.
wineModelFull = fitcdiscr(wine.data, categorical(wine.id), ...
    "DiscrimType","linear", "Prior","empirical");

[~,~,stats] = manova1(wine.data{:,:}, wine.id);
canonScores = stats.canon;   % proyección: columna 1 = LD1, columna 2 = LD2

% this is important to compare to the LOO case. The full model tends to
% overestimate the perfromance
fullAccuracy = 1 - resubLoss(wineModelFull); 

figure; % Maps each sample into the LDA space. 
gscatter(canonScores(:,1), canonScores(:,2), wine.id);
xlabel('LD1'); ylabel('LD2');

%% CLASSIFYING NEW SAMPLES

% In the case we have a new wine sample, it should be in the same order and
% scale as the data used to train the model. Then we can classify the new
% data and 

newWine = [13.5, 1.8, 2.3, 15, 100, 2.5, 2.8, 0.3, 1.5, 5, 1.0, 3.0, 900];  % mismo orden y escala que wine.data
newWineTab = array2table(newWine, "VariableNames", wine.attribute.fieldNames);
[predictedLabel, score] = predict(wineModelFull, newWineTab);

fprintf('Classification: Cultivar %i with %.2f%% probability\n', ...
    predictedLabel, score(predictedLabel));

% Map the new sample in the LDA space using the eigenvectors
gmean = mean(wine.data{:,:}, 1);
newCanon = (newWine - gmean) * stats.eigenvec;

hold on
plot(newCanon(1), newCanon(2), 'kp', 'MarkerSize', 15, 'MarkerFaceColor', 'y')
hold off

%% PRINCIPAL COMPONENT ANALYSIS
pcaWine = zscore(wine{:,2:end},[],1);

[coeff,score,latent,tsquared,explained] = pca(pcaWine);

wfig(3);

subplot 121
plot3(score(:,1),score(:,2),score(:,3),'.k','MarkerSize',5)
grid on
xlabel 'PCA1'; ylabel 'PCA2'; zlabel 'PCA3'

subplot 122
explVariance = cumsum(explained);
plot(explVariance,'-ok','MarkerFaceColor','auto')
box off; grid on; ylim([0 100])
ylabel 'Explained Variance'
xlabel 'PCA components'