function predicted = predictPhenolicCluster(data, refAttribute, refValue)
% PREDICTPHENOLICCLUSTER  Predict the other phenolic-cluster attributes
% from one of them, using a simple linear regression fit on the given
% dataset (reflects the natural co-variation of these compounds, not the
% LDA classifier).
%
%   predicted = predictPhenolicCluster(wine.data, 'flavanoids', 3.5)
%   predicted is a struct with one field per other cluster attribute.
%
%   Inputs:
%     data         - table with one column per attribute (e.g. wine.data)
%     refAttribute - name of the attribute being varied (e.g. 'flavanoids')
%     refValue     - value of refAttribute to predict the others from
%
%   Output:
%     predicted    - struct with one field per remaining phenolic-cluster
%                    attribute, holding its predicted value at refValue

clusterFields = {'phenols','flavanoids','odRatio','proanthocyanins'};
otherFields = setdiff(clusterFields, refAttribute, 'stable');

refValues = data{:,refAttribute};

predicted = struct();
for iField = 1:length(otherFields)
    thisField = otherFields{iField};
    fitCoeffs = polyfit(refValues, data{:,thisField}, 1);
    predicted.(thisField) = polyval(fitCoeffs, refValue);
end
end
