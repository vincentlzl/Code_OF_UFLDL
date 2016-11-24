function [cost,grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
                                             lambda, sparsityParam, beta, data)

% visibleSize: the number of input units (probably 64) 
% hiddenSize: the number of hidden units (probably 25) 
% lambda: weight decay parameter
% sparsityParam: The desired average activation for the hidden units (denoted in the lecture
%                           notes by the greek alphabet rho, which looks like a lower-case "p").
% beta: weight of sparsity penalty term
% data: Our 64x10000 matrix containing the training data.  So, data(:,i) is the i-th training example. 
  
% The input theta is a vector (because minFunc expects the parameters to be a vector). 
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
% follows the notation convention of the lecture notes. 

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

% Cost and gradient variables (your code needs to compute these values). 
% Here, we initialize them to zeros. 
cost = 0;
W1grad = zeros(size(W1)); 
W2grad = zeros(size(W2));
b1grad = zeros(size(b1)); 
b2grad = zeros(size(b2));

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost/optimization objective J_sparse(W,b) for the Sparse Autoencoder,
%                and the corresponding gradients W1grad, W2grad, b1grad, b2grad.
%
% W1grad, W2grad, b1grad and b2grad should be computed using backpropagation.
% Note that W1grad has the same dimensions as W1, b1grad has the same dimensions
% as b1, etc.  Your code should set W1grad to be the partial derivative of J_sparse(W,b) with
% respect to W1.  I.e., W1grad(i,j) should be the partial derivative of J_sparse(W,b) 
% with respect to the input parameter W1(i,j).  Thus, W1grad should be equal to the term 
% [(1/m) \Delta W^{(1)} + \lambda W^{(1)}] in the last block of pseudo-code in Section 2.2 
% of the lecture notes (and similarly for W2grad, b1grad, b2grad).
% 
% Stated differently, if we were using batch gradient descent to optimize the parameters,
% the gradient descent update to W1 would be W1 := W1 - alpha * W1grad, and similarly for W2, b1, b2. 
% 
data_size=size(data);

z2=zeros(hiddenSize,data_size(2));%25x10000
a2=zeros(hiddenSize,data_size(2));%25x10000

z3=zeros(data_size);%64x10000
a3=zeros(data_size);%64x10000

delta3=zeros(data_size);%64x10000 delta3和数据的维数相同
delta2=zeros(hiddenSize,data_size(2));%25x10000

roh=zeros(hiddenSize,data_size(2));%25x10000
roh_para=ones(hiddenSize,1);

z2=W1*data+repmat(b1,1,data_size(2));%之前未考虑到b1应该是和z2维数一样的矩阵，需要进行列拷贝！！
a2=sigmoid(z2);

z3=W2*a2+repmat(b2,1,data_size(2));
a3=sigmoid(z3);

ave_square=sum(mean((data-a3).^2,2))./2;%对于mean到底除以了行数还是列数要搞清楚，同时"，2"的位置要放对
weight_decay=lambda*0.5*(sum(sum(W1.^2))+sum(sum(W2.^2)));

roh=mean(a2,2);
roh_para=sparsityParam*roh_para;
sparsity=beta.*sum(roh_para.*log(roh_para./roh)+(1-roh_para).*log((1-roh_para)./(1-roh)));

cost=ave_square+weight_decay+sparsity;

% for index2=1:data_size(2)
%     delta3(:,index1)=-(data(:,index1)-a3(:,index1)).*(a3(:,index1).*(ones(visibleSize,1)-a3(:,index1)));
%     delta2(:,index1)=(W2'*delta3+beta*(-sparsityParam./roh+(1-sparsityParam)./(ones(hiddenSize,1)-roh))).*(a2(:,index1).*(ones(hiddenSize,1)-a2(:,index2)));
% end

delta3=(a3-data).*(a3.*(1-a3));
roh_rpt=repmat(roh,1,data_size(2));
roh_pararpt=repmat(sparsityParam,hiddenSize,data_size(2));
delta2=(W2'*delta3+beta*(-roh_pararpt./roh_rpt+(1-roh_pararpt)./(1-roh_rpt))).*(a2.*(1-a2));%roh同样需要repmat

W2grad=(delta3*a2')./data_size(2)+lambda.*W2;
b2grad=mean(delta3,2);
    
W1grad=(delta2*data')./data_size(2)+lambda.*W1;
b1grad=mean(delta2,2);


%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end

%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 

function sigm = sigmoid(x)
  
    sigm = 1 ./ (1 + exp(-x));
end

