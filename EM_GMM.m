function  [wight,mean,var]=EM_GMM(data,k,initial,maxIter)

if nargin <= 1,
    disp('EM_GMM must have 4 inputs!/n')
end

%%%% 初始化 %%%%
wight  = initial.weight;
mean   = initial.mean;
var    = initial.var;

Ln = Likelihood(data,k,wight,mean,var);
Lo = 2*Ln;
threshold = 0.3;%收敛停止的阈值
%%%% EM %%%%

for ithIther = 1:maxIter
    
    if (abs((Ln-Lo)/Lo)>threshold)
        
        expection = Expectation(data,k,wight,mean,var);     % E-step
        [wight,mean,var] = Maximization(data,k,expection);  % M-step
        Lo = Ln;
        Ln = Likelihood(data,k,wight,mean,var);

    end
    disp(sprintf('Number of iterations: %d',ithIther-1));
end

% L = Ln;
    Plot_GM(data,k,wight,mean,var);




function E = Expectation(X,k,W,M,V)

[n,~] = size(X);
E = zeros(n,k);
for j = 1:k
    E(:,j) = W(j).*normpdf( X, M(:,j)', V(:,:,j));% 每个点属于k个高斯分布的概率
end
total = repmat(sum(E,2),1,j);
E = E./total;%每行除以行和 一行一个样本


function [W,M,V] = Maximization(X,k,E)

[n,d] = size(X);
W = sum(E);
M = X'*E./repmat(W,d,1);
for i=1:k,
    dXM = X - repmat(M(:,i)',n,1);
    Wsp = spdiags(E(:,i),0,n,n);
    V(:,:,i) = dXM'*Wsp*dXM/W(i);
end
W = W/n;


function L = Likelihood(X,k,W,M,V)

[n,d] = size(X);
U = mean(X)';
S = cov(double(X));
L = 0;
for i=1:k,
    iV = inv(V(:,:,i));
    L = L + W(i)*(-0.5*n*log(det(2*pi*V(:,:,i))) ...
        -0.5*(n-1)*(trace(iV*S)+(U-M(:,i))'*iV*(U-M(:,i))));
end


function Plot_GM(X,k,W,M,V)
[n,d] = size(X);
if d>2,
    disp('Can only plot 1 dimensional applications!/n');
    return
end
S = zeros(d,k);
R1 = zeros(d,k);
R2 = zeros(d,k);
for i=1:k,  % Determine plot range as 4 x standard deviations
    S(:,i) = sqrt(diag(V(:,:,i)));
    R1(:,i) = M(:,i)-4*S(:,i);
    R2(:,i) = M(:,i)+4*S(:,i);
end
Rmin = min(min(R1));
Rmax = max(max(R2));
R = [Rmin:0.001*(Rmax-Rmin):Rmax];
clf, hold on
    Q = zeros(size(R));
    for i=1:k,
        P = W(i)*normpdf(R,M(:,i),sqrt(V(:,:,i)));
        Q = Q + P;
        plot(R,P,'r-'); grid on
    end
    plot(R,Q,'k-');
    xlabel('X');
    ylabel('Probability density');
title('Gaussian Mixture estimated by EM');



