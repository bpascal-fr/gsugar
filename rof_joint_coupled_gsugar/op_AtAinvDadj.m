function [ov, oh] = op_AtAinvDadj(th,tv,gh,gv,use)
    
    % Duality relation to increment the primal iterate from the dual one
    % (ov,oh) = (A'*A)^(-1)D'([th, tv], [gh,gv])
    % inputs  - [th, tv]: dual variable to local power v
    %         - [gh, gv]: dual variable to local regularity h
    %         - use: structure containing coefficients of (A'*A)^(-1)
    %
    % outputs - v, h: primal variables
    %
    % Implementation B. Pascal, ENS Lyon
    % April 2021

    ty = opL_adj(th,tv);
    tz = opL_adj(gh,gv);
    
    ov = (use.S2*ty - use.S1*tz)/use.det;
    oh = (use.S0*tz - use.S1*ty)/use.det;

end