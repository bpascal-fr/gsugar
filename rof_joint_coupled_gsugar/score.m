% Compute the segmentation score in term of proportion of correctly
% classified pixels, up to permutation of labels

function perf = score(seg,MASK)
    
    
    % inputs - seg: map of estimated labels
    %        - MASK: map of true labels
    %
    % ouput  - score: proportion of well-classified pixels
    %
    % Implementation B. Pascal, ENS Lyon
    % May 2020
    
    % Number of labels
    lab = unique(MASK); nlab = length(lab);
    
    % Initialization of scores associated with every permutation of labels
    score = zeros(1,factorial(nlab));
    
    % Permutations
    B = perms(1:nlab);
    
    % Initialization of permuted true maps
    maskB = cell(1,factorial(nlab));
    
    % Compute scores for each permutation
    for nB = 1:factorial(nlab)
        maskB{nB} = zeros(size(MASK));
        for b = 1:nlab
            maskB{nB}(MASK == b) = B(nB,b);
        end
        score(nB) = mean(maskB{nB}(:) == seg(:));
    end
    
    % Select best score
    perf = max(score);
    
    
end