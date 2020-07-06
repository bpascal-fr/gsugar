%% kmeans_moulinette.m
clear all;
close all;

% generate synthetic data with 3 classes
N=100; MU=[1 2 4]; SS=[0.2 0.2 0.5]/2; 
X=[randn(1,N)*sqrt(SS(1))+MU(1),randn(1,N)*sqrt(SS(2))+MU(2),randn(1,N)*sqrt(SS(3))+MU(3)];

% true class labels
IND=[ones(1,N),2*ones(1,N),3*ones(1,N)];

% classify
K=3; % # classes
B = perms(1:K); % possible class label permutations

[idx,ctrs] = kmeans(X(:),K,'Replicates',5); % k-means

% determine misclassification score for all permutations
for kcb = 1:size(B,1);
    idk{kcb} = zeros(size(idx));b = B(kcb,:);
    for k=1:K
        idk{kcb}(idx==b(k)) = k;
    end
    score(kcb) = mean((idk{kcb}==IND(:)));
end
% Find the best permutation
kbest = find(score==max(score));
id = reshape(idk{kbest},size(X));

misclassif_rate = 1-mean(IND==id);

figure(1);
plot(1:length(X),X, 'ko'); title('data'); grid on;

figure(2);
plot(1:length(X),IND,'bo-',1:length(X),id,'r.-');grid on;xlabel('class label');legend('ground truth','k-means','Location','SouthEast');
title(['misclassification rate: ',num2str(misclassif_rate)]);
