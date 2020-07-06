function C_delta = opC(delta, cov, JJ)
    
    % Multiply multi-scale map delta by covariance matrix including inter-scale and spatial
    % correlations
    %
    % inputs: - delta: multi-scale map composed of |JJ| frames
    %         - cov: inter-scale and spatial correlations between octaves i
    %         and j
    %       
    % ouputs: - Cdelta: multiplication of delta by the covariance matrix
    %         cov
    %
    % Implementation B. Pascal, ENS Lyon
    % May 2020
    
    
    % Multiply C and delta scale by scale
    for ii = JJ
        tmp_S_delta = 0;
        for jj = JJ
            tmp_conv = conv2(delta{jj},cov{ii,jj},'same');
            tmp_S_delta = tmp_S_delta + tmp_conv;
        end
        C_delta{ii} = tmp_S_delta;
    end
    
    
end