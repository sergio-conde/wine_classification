%% CHAPTER 3 - THE HEIST
% What would a typical cultivar-2 wine's chemistry need to look like for a
% linear discriminant model (LDA) to read it as a cultivar 1?
%   3a  The obvious dial: move the phenolic block along its joint
%       direction and follow the wine on the LDA map.
%   3b  The minimal recipe: the smallest change that makes the model call
%       the wine cultivar 1 with probability >= pTarget, measured two ways:
%       L2 (smallest overall move) and L1 (fewest molecules changed).
%   3c  Reality check: the same question with values kept inside the
%       observed ranges, the phenolic block moving as one unit, and the
%       cost measured with the Mahalanobis distance.
% It also reports which molecules drive each axis of the map.
%
% Requires the Statistics and Machine Learning Toolbox and the
% Optimization Toolbox.

clear; clc

%% DATA
% UCI Wine recognition data: 178 wines from three cultivars. Column 1 is
% the cultivar (1-3); columns 2-14 are the 13 chemical measurements.

dataFile = fullfile(fileparts(mfilename('fullpath')), '..', 'example_data', 'wine.data');
raw = readmatrix(dataFile, 'FileType', 'text');
cultivar = raw(:,1);
X = raw(:,2:14);

names = {'Alcohol','Malic acid','Ash','Alcalinity of ash','Magnesium', ...
    'Total phenols','Flavanoids','Nonflavanoid phenols','Proanthocyanins', ...
    'Color intensity','Hue','OD280/OD315','Proline'};

% The phenolic block: four strongly correlated molecules that rise and
% fall together (total phenols, flavanoids, proanthocyanins, OD280/OD315).
phenolicIdx = [6 7 9 12];

% The heist: make a cultivar-2 wine read as cultivar 1 with at least 90%
% probability.
origin = 2;
target = 1;
pTarget = 0.9;

[nWines, nMolecules] = size(X);
classes = unique(cultivar)';
nClasses = numel(classes);

% Everything is computed in standardized units (z-scores), so that a change
% of 1 means "one standard deviation" for every molecule.
xMean = mean(X);
xStd = std(X);
Z = (X - xMean) ./ xStd;
zMin = min(Z);
zMax = max(Z);

%% LDA MODEL
% Each cultivar k gets a linear score s_k(z) = W(k,:)*z' + b(k), built
% from the class means, the pooled within-class covariance and the class
% priors. The probability of each cultivar is the softmax of the scores.

mdl = fitcdiscr(Z, cultivar, 'DiscrimType', 'linear', 'Prior', 'empirical');
classMeans = mdl.Mu;
Sigma = mdl.Sigma;
prior = mdl.Prior(:);

W = classMeans / Sigma;
b = -0.5 * sum(W .* classMeans, 2) + log(prior);

% The hand-built scores must give the same probabilities as MATLAB's own
% predict.
[~, postFit] = predict(mdl, Z);
fprintf('Max |posterior(by hand) - predict|: %.2e\n', max(abs(posterior(Z, W, b) - postFit), [], 'all'));

% Leave-one-out accuracy: each wine is classified by a model that never
% saw it.
looMdl = fitcdiscr(Z, cultivar, 'DiscrimType', 'linear', 'Prior', 'empirical', 'Leaveout', 'on');
fprintf('Leave-one-out accuracy: %.2f%%\n', 100 * (1 - kfoldLoss(looMdl)));

%% LD1-LD2 MAP
% The two discriminant axes solve the generalized eigenproblem
% B v = lambda Sigma v (B: between-class scatter, Sigma: pooled
% within-class covariance), scaled so that the within-class spread is 1 in
% every direction. With three cultivars LDA has exactly two discriminants,
% so this 2D map holds everything the classifier uses: it is the model,
% not a lossy projection of it.

nPerClass = accumarray(cultivar, 1);
Sb = classMeans' * diag(nPerClass) * classMeans;
[V, D] = eig(Sb, Sigma, 'chol');
[eigVals, order] = sort(diag(D), 'descend');
V = V(:, order(1:2));
V = V ./ sqrt(sum(V .* (Sigma * V), 1));

% Eigenvector signs are arbitrary; fix them so cultivar 1 sits to the
% right of cultivar 3 on LD1 and cultivar 2 sits on top on LD2.
centroidsLD = classMeans * V;
if centroidsLD(1,1) < centroidsLD(3,1), V(:,1) = -V(:,1); end
if centroidsLD(2,2) < mean(centroidsLD([1 3],2)), V(:,2) = -V(:,2); end
centroidsLD = classMeans * V;
wineLD = Z * V;

fprintf('LD1 / LD2 share of between-class variance: %.1f%% / %.1f%%\n', ...
    100 * eigVals(1:2) / sum(eigVals(1:2)));

% In this plane, nearest centroid (adjusted by the log prior) gives the
% same probabilities as the full 13-dimensional model.
fprintf('Max |posterior(LD plane) - posterior(13D)|: %.2e\n', ...
    max(abs(posteriorLD(wineLD, centroidsLD, prior) - postFit), [], 'all'));

% Euclidean distance between centroids on the map equals the Mahalanobis
% distance between cultivar means.
pairs = nchoosek(1:nClasses, 2);
fprintf('Mahalanobis distance between cultivar means:');
for iPair = 1:size(pairs,1)
    fprintf('  %d-%d = %.2f', pairs(iPair,:), norm(diff(centroidsLD(pairs(iPair,:),:))));
end
fprintf('\n');

%% WHICH MOLECULES DRIVE LD1 AND LD2
% Two complementary views of each axis:
%   standardized coefficients - the unique contribution of each molecule
%       (weights scaled by the pooled within-class standard deviation);
%   structure correlations - how strongly each molecule moves with the
%       axis across all 178 wines, i.e. what the axis "is made of".

stdCoeffs = V .* sqrt(diag(Sigma));
structCorr = corr(Z, wineLD);

fprintf('\n%-22s %9s %9s   %9s %9s\n', 'Molecule', 'coef LD1', 'coef LD2', 'corr LD1', 'corr LD2');
for iMol = 1:nMolecules
    fprintf('%-22s %9.2f %9.2f   %9.2f %9.2f\n', names{iMol}, stdCoeffs(iMol,:), structCorr(iMol,:));
end

%% PHENOLIC DIAL
% The joint direction of the phenolic block is the first principal
% component of its 4 standardized molecules, signed so that turning the
% dial up raises all four. A wine's dial position is its score on that
% direction.

[eigVecs, eigValsPhen] = eig(corr(X(:,phenolicIdx)));
[~, top] = max(diag(eigValsPhen));
phenolicDir = eigVecs(:, top);
phenolicDir = phenolicDir * sign(sum(phenolicDir));
dialDir = zeros(1, nMolecules);
dialDir(phenolicIdx) = phenolicDir;

fprintf('\nPhenolic direction (PC1, %.0f%% of block variance):\n', 100 * max(diag(eigValsPhen)) / 4);
for iMol = 1:numel(phenolicIdx)
    fprintf('   %-22s %.3f\n', names{phenolicIdx(iMol)}, phenolicDir(iMol));
end
dialScores = Z(:,phenolicIdx) * phenolicDir;

%% STARTING WINES
% The heist starts from the median cultivar-2 wine (median of each
% molecule). To show the results don't hinge on that synthetic wine, it is
% repeated from the real cultivar-2 wine closest to the median
% (Mahalanobis distance).

zMedian = (median(X(cultivar == origin,:)) - xMean) ./ xStd;
originRows = find(cultivar == origin);
dOrigin = mahal2(Z(originRows,:), zMedian, Sigma);
[~, nearest] = min(dOrigin);
realRow = originRows(nearest);

starts = struct('label', {'Cultivar-2 median', sprintf('Real wine #%d', realRow)}, ...
    'z', {zMedian, Z(realRow,:)});

% Plausibility test: if a wine's squared Mahalanobis distance to every
% cultivar mean exceeds the 99th percentile of a chi-square with 13 degrees
% of freedom, no wine in the dataset looks like it.
chi2Cut = chi2inv(0.99, nMolecules);
for iStart = 1:numel(starts)
    z0 = starts(iStart).z;
    fprintf('\n================ START: %s ================\n', starts(iStart).label);
    fprintf('Posterior at start: P1=%.3f  P2=%.3f  P3=%.3f\n', posterior(z0, W, b));
    printPlausibility(z0, classMeans, Sigma, chi2Cut);

    %% 3a - turn the phenolic dial across its observed range
    % Only the four phenolic molecules move; the other nine stay at the
    % starting wine's values.
    dialStart = z0(phenolicIdx) * phenolicDir;
    dial = linspace(min(dialScores), max(dialScores), 201)';
    pathZ = z0 + (dial - dialStart) * dialDir;
    pathPost = posterior(pathZ, W, b);
    pathLD = pathZ * V;

    fprintf('\n3a - phenolic dial from %.2f to %.2f (start at %.2f)\n', dial(1), dial(end), dialStart);
    for iEnd = [1 numel(dial)]
        rawEnd = pathZ(iEnd,phenolicIdx) .* xStd(phenolicIdx) + xMean(phenolicIdx);
        fprintf('   dial %+.2f: P1=%.3f P2=%.3f P3=%.3f  LD=(%.2f, %.2f)  phenolics=[%s]\n', ...
            dial(iEnd), pathPost(iEnd,:), pathLD(iEnd,:), sprintf('%.2f ', rawEnd));
    end
    [maxP1, iMax] = max(pathPost(:,target));
    fprintf('   max P(cultivar %d) along the dial: %.3f at dial %+.2f\n', target, maxP1, dial(iMax));
    for k = classes
        reportCrossing(dial, pathPost(:,k), k, [0.5 pTarget]);
    end

    %% 3b / 3c - recipes
    % Three answers to "what is the smallest change that makes the model
    % call this wine the target cultivar with P >= pTarget?":
    %   L2 (naive)  - smallest overall move, all molecules free;
    %   L1 (sparse) - fewest molecules changed;
    %   Realistic   - values stay within the observed ranges, the phenolic
    %                 block moves as one unit, and the cost is the
    %                 Mahalanobis distance (moves that go against the
    %                 natural correlations between molecules cost more).
    recipes = struct('name', {'L2 (naive)', 'L1 (sparse)', 'Realistic'}, 'kind', {'L2', 'L1', 'real'});
    for iRec = 1:numel(recipes)
        [delta, margin, ok] = solveRecipe(recipes(iRec).kind, z0, W, b, target, pTarget, ...
            Sigma, zMin, zMax, phenolicIdx, phenolicDir);
        recipes(iRec).delta = delta;
        recipes(iRec).ok = ok;
        if ~ok
            fprintf('\n%s: infeasible\n', recipes(iRec).name);
            continue
        end
        zEnd = z0 + delta;
        post = posterior(zEnd, W, b);
        recipes(iRec).zEnd = zEnd;
        fprintf('\n%s  (margin %.3f)  P1=%.3f P2=%.3f P3=%.3f\n', recipes(iRec).name, margin, post);
        fprintf('   L2 size %.2f SD | Mahalanobis cost %.2f | molecules changed: %d\n', ...
            norm(delta), sqrt(delta / Sigma * delta'), sum(abs(delta) > 1e-6));
        printPlausibility(zEnd, classMeans, Sigma, chi2Cut);
        fprintf('   %-22s %9s %9s %9s %7s\n', 'Molecule', 'start', 'change', 'end', 'range?');
        for iMol = find(abs(delta) > 1e-6)
            x0 = z0(iMol) * xStd(iMol) + xMean(iMol);
            x1 = zEnd(iMol) * xStd(iMol) + xMean(iMol);
            rangeFlag = 'ok';
            if zEnd(iMol) < zMin(iMol) - 1e-9 || zEnd(iMol) > zMax(iMol) + 1e-9
                rangeFlag = 'OUT';
            end
            fprintf('   %-22s %9.3f %+9.3f %9.3f %7s\n', names{iMol}, x0, x1 - x0, x1, rangeFlag);
        end
    end
    starts(iStart).pathLD = pathLD;
    starts(iStart).pathPost = pathPost;
    starts(iStart).dial = dial;
    starts(iStart).dialStart = dialStart;
    starts(iStart).recipes = recipes;
end

%% FIGURES
% All figures use the median starting wine. Decision regions are drawn by
% classifying a fine grid of points on the LD plane.

s = starts(1);
gridLD1 = linspace(min(wineLD(:,1)) - 1.5, max(wineLD(:,1)) + 1.5, 400);
gridLD2 = linspace(min(wineLD(:,2)) - 1.5, max(wineLD(:,2)) + 1.5, 400);
[G1, G2] = meshgrid(gridLD1, gridLD2);
[~, region] = max(posteriorLD([G1(:) G2(:)], centroidsLD, prior), [], 2);

figure('Name', '3a - phenolic dial on the LDA map');
drawMap(gridLD1, gridLD2, reshape(region, size(G1)), wineLD, cultivar);
plot(s.pathLD(:,1), s.pathLD(:,2), 'k-', 'LineWidth', 2)
startLD = s.pathLD(abs(s.dial - s.dialStart) == min(abs(s.dial - s.dialStart)),:);
plot(startLD(1), startLD(2), 'kp', 'MarkerSize', 16, 'MarkerFaceColor', 'w')
plot(s.pathLD(1,1), s.pathLD(1,2), 'kv', 'MarkerFaceColor', 'k')
plot(s.pathLD(end,1), s.pathLD(end,2), 'k^', 'MarkerFaceColor', 'k')
title('3a: phenolic dial (\nabla low, \Delta high), start = star')

figure('Name', '3a - posterior along the dial');
plot(s.dial, s.pathPost, 'LineWidth', 2); hold on
xline(s.dialStart, 'k--', 'start'); yline(pTarget, ':'); yline(0.5, ':')
hold off; box off; ylim([0 1])
xlabel('Phenolic dial (block score, SD units)'); ylabel('Posterior')
legend('P(1)', 'P(2)', 'P(3)', 'Location', 'best')
title('3a: P(cultivar) along the phenolic dial')

figure('Name', '3b/3c - recipes on the LDA map');
drawMap(gridLD1, gridLD2, reshape(region, size(G1)), wineLD, cultivar);
z0LD = s.z * V;
plot(z0LD(1), z0LD(2), 'kp', 'MarkerSize', 16, 'MarkerFaceColor', 'w')
markers = {'ko', 'ks', 'kd'};
for iRec = find([s.recipes.ok])
    endLD = s.recipes(iRec).zEnd * V;
    plot([z0LD(1) endLD(1)], [z0LD(2) endLD(2)], 'k-')
    plot(endLD(1), endLD(2), markers{iRec}, 'MarkerSize', 10, 'MarkerFaceColor', 'w', ...
        'DisplayName', s.recipes(iRec).name)
end
title('3b/3c: recipes (circle L2, square L1, diamond realistic)')

figure('Name', '3b/3c - recipe changes per molecule');
okRec = find([s.recipes.ok]);
bar(reshape([s.recipes(okRec).delta], nMolecules, []))
set(gca, 'XTick', 1:nMolecules, 'XTickLabel', names, 'XTickLabelRotation', 45, 'Box', 'off')
ylabel('Change (SD units)'); legend({s.recipes(okRec).name}, 'Location', 'best')
title('3b/3c: change per molecule, cultivar-2 median \rightarrow 1')

figure('Name', 'c - what LD1 and LD2 are made of');
barh(structCorr)
set(gca, 'YTick', 1:nMolecules, 'YTickLabel', names, 'YDir', 'reverse', 'Box', 'off')
xlabel('Correlation with the axis'); legend('LD1', 'LD2', 'Location', 'best')
title('Structure correlations')

%% LOCAL FUNCTIONS

function p = posterior(z, W, b)
% Cultivar probabilities for each row of z: softmax of the linear scores
% (shifted by the row maximum for numerical stability).
s = z * W' + b';
s = s - max(s, [], 2);
p = exp(s) ./ sum(exp(s), 2);
end

function p = posteriorLD(l, centroidsLD, prior)
% The same probabilities computed on the LD plane: squared distance to
% each cultivar centroid, adjusted by the log prior.
s = -0.5 * ((l(:,1) - centroidsLD(:,1)').^2 + (l(:,2) - centroidsLD(:,2)').^2) + log(prior)';
s = s - max(s, [], 2);
p = exp(s) ./ sum(exp(s), 2);
end

function d2 = mahal2(z, mu, Sigma)
% Squared Mahalanobis distance from each row of z to mu.
dz = z - mu;
d2 = sum((dz / Sigma) .* dz, 2);
end

function printPlausibility(z, classMeans, Sigma, chi2Cut)
% Compares the distance to the nearest cultivar mean with the chi-square
% cutoff.
d2 = arrayfun(@(k) mahal2(z, classMeans(k,:), Sigma), 1:size(classMeans,1));
[d2Min, kMin] = min(d2);
verdict = "inside";
if d2Min > chi2Cut, verdict = "NO wine in the dataset looks like this"; end
fprintf('   Plausibility: d2 to nearest mean (cultivar %d) = %.1f vs chi2 99%% = %.1f -> %s\n', ...
    kMin, d2Min, chi2Cut, verdict);
end

function reportCrossing(dial, p, k, levels)
% Prints where along the dial the probability of cultivar k crosses each
% of the given levels.
for level = levels
    above = p >= level;
    if ~any(above)
        fprintf('   P(%d) never reaches %.1f\n', k, level);
    elseif all(above)
        fprintf('   P(%d) >= %.1f along the whole dial\n', k, level);
    else
        flips = find(diff(above) ~= 0);
        fprintf('   P(%d) crosses %.1f at dial %s\n', k, level, sprintf('%+.2f ', dial(flips)));
    end
end
end

function [delta, margin, ok] = solveRecipe(kind, z0, W, b, target, pTarget, Sigma, zMin, zMax, phenIdx, phenDir)
% Smallest change delta (standardized units) that brings P(target) to
% pTarget. Requiring the target's score to beat every other score by a
% margin is a set of linear constraints, so each recipe is an exact QP or
% LP. With three cultivars, P(target) >= pTarget is not itself linear, so
% the margin is found by bisection between log(p/(1-p)) and log(2p/(1-p)),
% which bracket P(target) = pTarget.
lo = log(pTarget / (1 - pTarget));
hi = log(2 * pTarget / (1 - pTarget));
[delta, ok] = solveForMargin(kind, hi, z0, W, b, target, Sigma, zMin, zMax, phenIdx, phenDir);
margin = hi;
if ~ok, return, end
for iter = 1:60
    mid = (lo + hi) / 2;
    [dMid, okMid] = solveForMargin(kind, mid, z0, W, b, target, Sigma, zMin, zMax, phenIdx, phenDir);
    pMid = posterior(z0 + dMid, W, b);
    if okMid && pMid(target) >= pTarget
        hi = mid; delta = dMid; margin = mid;
    else
        lo = mid;
    end
end
end

function [delta, ok] = solveForMargin(kind, margin, z0, W, b, target, Sigma, zMin, zMax, phenIdx, phenDir)
% One recipe for a fixed margin:
%   'L2'   minimize ||delta||^2             (quadratic program)
%   'L1'   minimize sum(|delta|)            (linear program; delta is split
%          into positive and negative parts)
%   'real' minimize the Mahalanobis cost, within the observed ranges and
%          with the phenolic block moving as one unit (quadratic program)
nMol = numel(z0);
others = setdiff(1:size(W,1), target);
dW = W(target,:) - W(others,:);
A = -dW;
c = dW * z0' + b(target) - b(others) - margin;
opts = optimoptions('quadprog', 'Display', 'off');
switch kind
    case 'L2'
        [delta, ~, flag] = quadprog(2 * eye(nMol), zeros(nMol,1), A, c, [], [], [], [], [], opts);
    case 'L1'
        lpOpts = optimoptions('linprog', 'Display', 'off');
        [u, ~, flag] = linprog(ones(2*nMol,1), [A -A], c, [], [], zeros(2*nMol,1), [], lpOpts);
        if flag > 0, delta = u(1:nMol) - u(nMol+1:end); end
    case 'real'
        % delta = E * u: each non-phenolic molecule moves freely, while the
        % four phenolic molecules share a single step along phenDir.
        freeIdx = setdiff(1:nMol, phenIdx);
        E = zeros(nMol, numel(freeIdx) + 1);
        E(freeIdx, 1:numel(freeIdx)) = eye(numel(freeIdx));
        E(phenIdx, end) = phenDir;
        H = 2 * (E' / Sigma * E);
        H = (H + H') / 2;
        Abox = [E; -E];
        cBox = [zMax' - z0'; z0' - zMin'];
        [u, ~, flag] = quadprog(H, zeros(size(E,2),1), [A * E; Abox], [c; cBox], [], [], [], [], [], opts);
        if flag > 0, delta = E * u; end
end
ok = flag > 0;
if ok
    delta = delta(:)';
else
    delta = zeros(1, nMol);
end
end

function drawMap(g1, g2, region, wineLD, cultivar)
% LD1-LD2 map: coloured decision regions with the 178 wines on top.
imagesc(g1, g2, region, 'AlphaData', 0.25); hold on
set(gca, 'YDir', 'normal')
colormap(gca, lines(3))
gscatter(wineLD(:,1), wineLD(:,2), cultivar, lines(3), '.', 10)
box off; xlabel('LD1'); ylabel('LD2')
end
