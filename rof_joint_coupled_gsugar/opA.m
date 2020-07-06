function Lj = opA(v,h,JJ)
    
    % Linear behavior through octaves j with intercept v and slope h
    % inputs  - v: intercept
    %         - h: slope
    %         - JJ: range of octaves
    %
    % outputs - Lj: intercept + octave * slope at each scale
    %
    % Implementation B. Pascal, ENS Lyon
    % May 2020
    
    Lj = cell(1,JJ(end));
    for jj = JJ
        Lj{jj} = v + jj*h;
    end
    
end