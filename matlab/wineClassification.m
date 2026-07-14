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

figure(2); % Maps each sample into the LDA space. 
theme(gcf,"light")

gscatter(canonScores(:,1), canonScores(:,2), wine.id);
xlabel('LD1'); ylabel('LD2');

%% RANKING COEFFICIENTS 

attributeStd = std(wine.data{:,:});
attributeRank = nan(4,13);

classPairs = [1 2;1 3;2 3];
classPairsLabels = {...
    'Class1 vs Class2',...
    'Class1 vs Class3',...
    'Class2 vs Class3',...
    'Mean weight'};
for ipair = 1:3
    compCoeffs = wineModelFull.Coeffs(classPairs(ipair,1),classPairs(ipair,2)).Linear;
    attributeRank(ipair,:) = compCoeffs.*attributeStd(:);
end
attributeRank(4,:) = mean(attributeRank(1:3,:));

figure(3); 
theme(gcf,"light")
imagesc(attributeRank)
hbar = colorbar;
hbar.Label.String = 'Weight';
set(gca,'Box', 'off', 'TickDir', 'out',...
    'ytick',1:4,...
    'YTickLabel',classPairsLabels,...
    'xtick',1:13,...
    'XTickLabel',wine.attribute.labels)

%% CLASSIFYING NEW SAMPLES

% In the case we have a new wine sample, it should be in the same order and
% scale as the data used to train the model. Then we can classify the new
% data and 

newWine = [13.5, 1.8, 2.3, 15, 100, 2.5, 2.8, 0.3, 1.5, 5, 1.0, 3.0, 900];  % mismo orden y escala que wine.data
newWineTab = array2table(newWine, "VariableNames", wine.attribute.fieldNames);
[predictedLabel, score] = predict(wineModelFull, newWineTab);

predictedNum = double(predictedLabel);
fprintf('Classification: Cultivar %i with %.2f%% probability\n', ...
    predictedLabel, score(predictedNum) * 100);

% Map the new sample in the LDA space using the eigenvectors
gmean = mean(wine.data{:,:}, 1);
newCanon = (newWine - gmean) * stats.eigenvec;

hold on
plot(newCanon(1), newCanon(2), 'kp', 'MarkerSize', 15, 'MarkerFaceColor', 'y')
hold off
%%

%% PRINCIPAL COMPONENT ANALYSIS
pcaWine = zscore(wine.data{:,2:end},[],1);

[coeff,score,latent,tsquared,explained] = pca(pcaWine);

wfig(4);
theme(gcf,"light")
subplot 121
plot(score(:,1),score(:,2),'.k','MarkerSize',5)
grid on
xlabel 'PCA1'; ylabel 'PCA2'; 

subplot 122
explVariance = cumsum(explained);
plot(explVariance,'-ok','MarkerFaceColor','auto')
box off; grid on; ylim([0 100])
ylabel 'Explained Variance'
xlabel 'PCA components'