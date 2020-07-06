function C_delta = opC_1D(delta, cov, JJ)
    
    % Multiply multi-scale signal delta by covariance matrix including
    % inter-scale and time correlations
    %
    % inputs: - delta: multi-scale signal composed of |JJ| frames
    %         - cov: inter-scale and time correlations between octaves i
    %         and j
    %       
    % ouputs: - Cdelta: multiplication of delta by the covariance matrix
    %         cov
    %
    % Implementation B. Pascal, ENS Lyon
    % May 2020
    
    % Multiply C and delta scale by scale
    C_delta = cell(1,JJ(end));
    for ii = JJ
        tmp_S_delta = 0;
        for jj = JJ
            tmp_conv = conv(delta{jj},cov{ii,jj},'same');
            tmp_S_delta = tmp_S_delta + tmp_conv;
        end
        C_delta{ii} = tmp_S_delta;
    end
    
    
end