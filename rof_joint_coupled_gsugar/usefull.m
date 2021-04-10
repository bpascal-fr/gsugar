function use = usefull(JJ,L)
    
    % Compute the Lipschitz constant and strong convexity modulus of
    % data-fidelity term
    %     (h,v) --> 1/2 sum_j || Lj - jh - v ||^2
    % (i.e. least square on log-leaders) and
    % sums of log-leaders needed to evaluate its convex conjugate
    %
    % inputs  - JJ: range of octaves
    %         - L: log-leaders
    %
    % ouputs  - use.norm: square root of Lipschitz constant of least squares data-fidelity term
    %         - use.norminv: square root of Lipschitz constant of dual least squares data-fidelity term
    %         - use.mu: strong convexity modulus of data-fidelity term
    %         - use.SLj: sum of the log-leaders over the octaves
    %         - use.SjLj: sum of the log-leaders multiplied by octave number over the octaves
    %
    % Implementation B. Pascal, ENS Lyon
    % May 2020
    
    
    % input : L{ii} contain the log-leaders at scale ii
    [N1,N2] = size(L{JJ(1)});
    
    % S0, S1, S2 coefficients of the matrix J
    use.S0=sum(JJ.^0);
    use.S1=sum(JJ);
    use.S2=sum(JJ.^2);
    J = [[use.S0 use.S1];[use.S1 use.S2]];
    
    % Determinant of the matrix J
    use.det = use.S0*use.S2-use.S1^2;
    
    % Inverse of the matrix J
    Jinv = [[use.S2 -use.S1];[-use.S1 use.S0]]/use.det;
    
    % Sum of the log-leaders over the octaves
    use.SLj = zeros(N1,N2);
    for ii=JJ
        use.SLj = use.SLj + L{ii};
    end
    
    % Sum of the log-leaders times j over the octaves
    use.SjLj = zeros(N1,N2);
    for ii=JJ
        use.SjLj = use.SjLj + ii*L{ii};
    end
    
    
    eigen = eig(J);
    use.mu = min(eigen);
    use.norm = sqrt(max(eigen));
    
    eigeninv = eig(Jinv);
    use.norminv = sqrt(max(eigeninv));
end