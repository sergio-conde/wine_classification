%% READ WINE DATA

% Include some breaf description here

clear; clc
wine = wineConfig;

%% DESCRIPTIVE STATISTICS

nAtributtes = width(wine.data);
descriptParams = {'Mean','STD','Median','IQR','Minimum','Maximum'};
descriptTable = array2table(nan(nAtributtes,length(descriptParams)),...
    "RowNames", wine.attribute.labels, "VariableNames", descriptParams);
for iAttribute = 1:nAtributtes
    descriptTable{iAttribute,:} = [
        mean(wine.data{:,iAttribute}),...
        std(wine.data{:,iAttribute}),...
        median(wine.data{:,iAttribute}),...
        iqr(wine.data{:,iAttribute}),...
        min(wine.data{:,iAttribute}),...
        max(wine.data{:,iAttribute})];
end

%% ATTRIBUTE'S VALUE HISTOGRAM 

wfig(1);
for iAtt = 1:length(wine.attribute.labels)
    subplot(4,4,iAtt)
    histogram(wine.data{:,iAtt},20)
    box off; 
    xlabel(wine.attribute.labels{iAtt})
    ylabel('Sample Count')
end

%% ARE THE ATTRIBUTES STATISTICALLY INDEPENDENT?

[attributPearson, pPearson] = corr(wine.data{:,:},'Type','Pearson');
[attributSpearman, pSpearman] = corr(wine.data{:,:},'Type','Spearman');

corrMatrix = triu(attributPearson,1) + tril(attributSpearman,-1);
corrDiff = attributPearson - attributSpearman;

%%
wfig(2)
subplot 121
imagesc(corrMatrix)
axis square; colorbar
subplot 122
imagesc(corrDiff)
axis square; colorbar


