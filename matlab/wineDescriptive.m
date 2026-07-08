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
corrDiff = abs(attributPearson - attributSpearman);

pCorrected = 0.05 / nchoosek(nAtributtes,2);

alphaData = ones(nAtributtes);
alphaData(pPearson > pCorrected | pSpearman > pCorrected) = 0.1;
alphaData(logical(eye(nAtributtes))) = 0;   

%%
wfig(2); clf

subplot 121
h1 = imagesc(corrMatrix);
h1.AlphaData = alphaData;
h1.AlphaDataMapping = 'none';
hold on
plot([0.5 13.5],[0.5 13.5],'--k')
hold off
cb1 = colorbar; cb1.Label.String = 'Correlation';
clim([-1 1])
axis square; set(gca,'Color','w');
box off; xlabel 'Attribute'; ylabel 'Attribute'

subplot 122
h2 = imagesc(corrDiff);
h2.AlphaData = alphaData;
h2.AlphaDataMapping = 'none';
hold on
plot([0.5 13.5],[0.5 13.5],'--k')
hold off
cb2 = colorbar; cb2.Label.String = '\Delta Correlation';
axis square; set(gca,'Color','w');
box off; xlabel 'Attribute'; ylabel 'Attribute'

% format figure function
% white, ticks out
