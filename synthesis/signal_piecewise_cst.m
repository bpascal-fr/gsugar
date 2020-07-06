function [signal_n,signal] = signal_piecewise_cst(N,sig,l,Ep,Nval)
    
    % Generate noisy piecewise constant signals
    % The noise is correlated along distance l
    % The hops are placed randomly.
    % The number of hops is random but likely to be around Ep.
    % On each tray the value of data is randomly selected among 10
    % regularly spaced values.
    %
    % Inputs:   N      length of signals
    %           sig    standard deviation of the noise (default 1e-1) 
    %           l      correlation length of the noise
    %           Ep     expected number of hops (default 10)
    %           Nval   expected number of differents values taken by the
    %                  signal (default 10) 
    %
    % Outputs:  data   noisy signal
    %           truth  underlying ground truth
    %
    % Implemented by B. PASCAL, ENS de Lyon
    % April 2020
    
    if nargin < 5
        Nval = 10;
        if nargin < 4
            Ep = 10;
            if nargin < 3
                sig = 1e-1;
            end
        end
    end
    
    
    p = Ep/N;
    indic_hop = rand(1,N) < p;
    N_hop = sum(indic_hop);
    trays = cumsum(indic_hop);
    
    signal = zeros(1,N);
    for nh = 1:N_hop
        rnd_val = floor(Nval*rand(1,1));
        rnd_kron = kron(rnd_val,ones(1,sum(trays == nh)));
        signal(trays == nh) = rnd_kron;
    end
    
    
        signal = (signal - min(signal))/(max(signal) - min(signal));
        
        noise    = conv(randn(1,N),ones(1,l)/sqrt(l),'same'); 
        signal_n = signal + sig*noise;
    
end
