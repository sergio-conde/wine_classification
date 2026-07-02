%% READ WINE DATA

% Include some breaf description here

clear; clc
wine = wineConfig;

%% DESCRIPTIVE STATISTICS

% Attribute's distribution
wfig(1);
for iAtt = 1:length(wine.attribute.labels)
    subplot(4,4,iAtt)
    histogram(wine.data{:,iAtt},15)
    box off; 
    xlabel(wine.attribute.labels{iAtt})
    ylabel('Wine Count')
end

%% ARE THE ATTRIBUTES STATISTICALLY INDEPENDENT?


%% PRINCIPAL COMPONENT ANALYSIS
pcaWine = zscore(wine{:,2:end},[],1);
% pcaWine = wine{:,2:end};

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