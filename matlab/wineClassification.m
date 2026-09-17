%% READ WINE DATA

% Loads raw attribute data + metadata (names, paths) via wineConfig.

clear; clc
wine = wineConfig;
nSamples = height(wine.data);

%% LINEAR DISCRIMINANT ANALYSIS

% This model uses the leave-one-out approach. It runs 178 models leaving
% one sample out every time. LDA is not affected by the different scales
% the attributes are showing, thus we don't do anything to the original
% data.

wineModel = fitcdiscr(wine.data,categorical(wine.id), ...
    "DiscrimType","linear", 'Prior','empirical', 'Leaveout','on'); 

% Quantify the percentage of misclassifications (performance)
looError = kfoldLoss(wineModel);
looAccuracy = 1 - looError;
fprintf('Leave-one-out accuracy: %.2f%%\n', looAccuracy*100);

% Per-sample predictions from the held-out folds, for the confusion matrix
[predictedLabels, scores] = kfoldPredict(wineModel);
trueLabels = categorical(wine.id);

wfig(1)
confusionchart(trueLabels, predictedLabels);

%%
% This is the full model without any cross-validation.
wineModelFull = fitcdiscr(wine.data, categorical(wine.id), ...
    "DiscrimType","linear", "Prior","empirical");

% this is important to compare to the LOO case. The full model tends to
% overestimate the perfromance
fullAccuracy = 1 - resubLoss(wineModelFull); 

[~,~,stats] = manova1(wine.data{:,:}, wine.id);
canonScores = stats.canon;   % proyección: columna 1 = LD1, columna 2 = LD2

wfig(2); % Maps each sample into the LDA space. 
gscatter(canonScores(:,1), canonScores(:,2), wine.id);
box off
xlabel('LD1'); ylabel('LD2');



%% CLASSIFYING NEW SAMPLES

% In the case we have a new wine sample, it should be in the same order and
% scale as the data used to train the model. Then we can classify the new
% data 

% new wine sample with the same attributes in the same order
newWine = [13.5, 1.8, 2.3, 15, 100, 2.5, 2.8, 0.3, 1.5, 5, 1.0, 3.0, 900];  
newWineTab = array2table(newWine, "VariableNames", wine.attribute.fieldNames);
[predictedLabel, score] = predict(wineModelFull, newWineTab);

predictedNum = double(predictedLabel);
fprintf('Classification: Cultivar %i with %.2f%% probability\n', ...
    predictedNum, score(predictedNum)*100);

% Map the new sample in the LDA space using the eigenvectors
gmean = mean(wine.data{:,:}, 1);
newCanon = (newWine - gmean) * stats.eigenvec;

hold on
plot(newCanon(1), newCanon(2), 'kp', 'MarkerSize', 15, 'MarkerFaceColor', 'y')
hold off

%% RANKING COEFFICIENTS 

attributeStd = std(wine.data{:,:});
attributeRank = nan(4,13);

classPairs = [1 2;1 3;2 3];
classPairsLabels = {...
    'Class1 vs Class2',...
    'Class1 vs Class3',...
    'Class2 vs Class3',...
    'Mean weight'};

% since the range of the attributes are quite different, it is necessary to
% multiply the coefficients by each attribute's standard deviation. This is
% equivalent to having zscored the data from the begining, however, to do
% so would make the classification of a new wine sample less intuitive
% because we would have to normalized first. 
for ipair = 1:3
    compCoeffs = wineModelFull.Coeffs(classPairs(ipair,1),classPairs(ipair,2)).Linear;
    attributeRank(ipair,:) = compCoeffs.*attributeStd(:);
end
attributeRank(4,:) = mean(abs(attributeRank(1:3,:)));
maxWeight = max(abs(attributeRank),[],'all');

phenolicCluster = {'Flavanoids','Total phenols'};
adjunctCluster = {'OD280/OD315', 'Proanthocyanins'};
clusterAttribute = ismember(wine.attribute.labels,phenolicCluster);
adjunctAttribute = ismember(wine.attribute.labels,adjunctCluster);
singleAttribute = ~(clusterAttribute | adjunctAttribute);

organizedRank = [attributeRank(:,clusterAttribute) ...
    attributeRank(:,adjunctAttribute) ...
    attributeRank(:,singleAttribute)];

organizedLabels = [wine.attribute.labels(clusterAttribute) ...
    wine.attribute.labels(adjunctAttribute) ...
    wine.attribute.labels(singleAttribute)];

xCluster = [0.55, 2.5, 2.5, 0.55, 0.55];
yCluster = [0.5, 0.5, 4.5, 4.5, 0.5];

xClusterAdj = [2.55, 4.5, 4.5, 2.55, 2.55];
yClusterAdj = [0.5, 0.5, 4.5, 4.5, 0.5];

wfig(3)
imagesc(organizedRank)
clim([-maxWeight maxWeight])
colormap(gca, 'redbluecmap')   % o cualquier colormap divergente que tengas disponible

hbar = colorbar;
hbar.Label.String = 'Weight';
set(gca,'Box', 'off', 'TickDir', 'out',...
    'ytick',1:4,...
    'YTickLabel',classPairsLabels,...
    'xtick',1:13,...
    'XTickLabel',organizedLabels)
hold on
plot(xCluster, yCluster, 'k-', 'LineWidth', 5);
plot(xClusterAdj, yClusterAdj, 'k--', 'LineWidth', 2);
hold off

%% PRINCIPAL COMPONENT ANALYSIS

pcaWine = zscore(wine.data{:,2:end},[],1);

[~,pcaScore,latent,~,explained] = pca(pcaWine);

wfig(4)
subplot 121
plot(pcaScore(:,1),pcaScore(:,2),'.k','MarkerSize',5)
grid on
xlabel 'PCA1'; ylabel 'PCA2'; 

subplot 122
explVariance = cumsum(explained);
plot(explVariance,'-ok','MarkerFaceColor','auto')
box off; grid on; ylim([0 100])
ylabel 'Explained Variance'
xlabel 'PCA components'

%% USE PCA TO NON SUPERVISED CLUSTERING

kRange = 2:8;
wcss = nan(1,length(kRange));
clusterSilho = nan(nSamples,length(kRange));
clusterId = nan(nSamples,length(kRange));
for kIdx = 1:length(kRange)
    [idx, C, sumd] = kmeans(pcaScore(:,1:2), kRange(kIdx), 'Replicates', 10);
    wcss(kIdx) = sum(sumd);
    clusterSilho(:,kIdx) = silhouette(pcaScore(:,1:2), idx);
    clusterId(:,kIdx) = idx;
end
meanSilho = mean(clusterSilho);
[~, optIdx] = max(meanSilho);
optimalK = kRange(optIdx);

%%
wfig(5); clf
subplot 121
plot(kRange,wcss,'o-k','MarkerFaceColor','auto')
box off
xlabel 'Number of clusters';
ylabel 'WCSS'

subplot 122
plot(kRange,meanSilho,'o-k','MarkerFaceColor','auto')
hold on
plot([optimalK optimalK],[min(meanSilho) max(meanSilho)],'--r')
plot(optimalK, max(meanSilho),'or','MarkerFaceColor','r')
axis tight; hold off; box off
xlabel 'Number of clusters';
ylabel 'Silhouette'

%% COMPARE TO THE GROUND TRUTH

cultivarTags = unique(wine.id);
nCultivar = length(cultivarTags);
clusterCultivar = clusterId(:,optIdx);

fprintf('\nNumber of cultivars: %i; Clustered cultivars: %i\n',...
    nCultivar,optimalK)

missClass = 0;
clustLabels = nan(1,nCultivar);

for iCult = 1:nCultivar
    cultSamples = wine.id == cultivarTags(iCult);
    clustSamples = clusterCultivar(cultSamples);
    clustLabel = mode(clustSamples);
    clustLabels(iCult) = clustLabel;
    correctPercg = 100 * sum(clustSamples == clustLabel) / sum(cultSamples);
    fprintf('\nAccuracy cultivar %i: %.2f%%\n',...
        cultivarTags(iCult),correctPercg)
    missClass = missClass + sum(clustSamples ~= clustLabel);
end
genAccuracy = 100 * (nSamples - missClass)/nSamples;
fprintf('\nGeneral Accuracy: %.2f%%\n',genAccuracy)

fprintf('\nCluster labels asignados: %s\n', mat2str(clustLabels));
if length(unique(clustLabels)) < nCultivar
    warning('Dos o más cultivares comparten el mismo cluster mayoritario')
end

organizedClusters = nan(size(wine.id));
for iCult = 1:3
    organizedClusters(clusterCultivar == clustLabels(iCult)) = iCult;
end
contingencyTable = crosstab(wine.id, organizedClusters);

%%
wfig(6);
heatmap(contingencyTable)
xlabel 'PCA + K-means classification'
ylabel 'Cultivar'

%%

wfig(7);
subplot 131
gscatter(canonScores(:,1), canonScores(:,2), wine.id);
box off
xlabel('LD1'); ylabel('LD2');
title 'LDA Full - Model'

subplot 132
gscatter(canonScores(:,1), canonScores(:,2), organizedClusters);
box off
xlabel('LD1'); ylabel('LD2');
title 'K-means clusters — LDA space (overlay)'

subplot 133
gscatter(pcaScore(:,1), pcaScore(:,2), organizedClusters);
box off
xlabel('PC1'); ylabel('PC2');
title 'K-means clusters — PCA space (native)'

