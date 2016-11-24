function [ cost, grad ] = stackedAECost(theta, inputSize, hiddenSize, ...
                                              numClasses, netconfig, ...
                                              lambda, data, labels)
                                         
% stackedAECost: Takes a trained softmaxTheta and a training data set with labels,
% and returns cost and gradient using a stacked autoencoder model. Used for
% finetuning.
                                         
% theta: trained weights from the autoencoder
% visibleSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% netconfig:   the network configuration of the stack
% lambda:      the weight regularization penalty
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 
% labels: A vector containing labels, where labels(i) is the label for the
% i-th training example


%% Unroll softmaxTheta parameter

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);

% Extract out the "stack"
stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);

% You will need to compute the following gradients
softmaxThetaGrad = zeros(size(softmaxTheta));
stackgrad = cell(size(stack));
for d = 1:numel(stack)
    stackgrad{d}.w = zeros(size(stack{d}.w));
    stackgrad{d}.b = zeros(size(stack{d}.b));
end

cost = 0; % You need to compute this

% You might find these variables useful
M = size(data, 2);
groundTruth = full(sparse(labels, 1:M, 1));


%% --------------------------- YOUR CODE HERE -----------------------------
%  Instructions: Compute the cost function and gradient vector for 
%                the stacked autoencoder.
%
%                You are given a stack variable which is a cell-array of
%                the weights and biases for every layer. In particular, you
%                can refer to the weights of Layer d, using stack{d}.w and
%                the biases using stack{d}.b . To get the total number of
%                layers, you can use numel(stack).
%
%                The last layer of the network is connected to the softmax
%                classification layer, softmaxTheta.
%
%                You should compute the gradients for the softmaxTheta,
%                storing that in softmaxThetaGrad. Similarly, you should
%                compute the gradients for each layer in the stack, storing
%                the gradients in stackgrad{d}.w and stackgrad{d}.b
%                Note that the size of the matrices in stackgrad should
%                match exactly that of the size of the matrices in stack.
%
n = numel(stack);
m = size(data,2);
hiddenData = cell(n+1,1);
hiddenData{1}.a = data;

for i = 2:n+1
    hiddenData{i}.z = bsxfun(@plus, stack{i-1}.w * hiddenData{i-1}.a, stack{i-1}.b);
    hiddenData{i}.a = sigmoid(hiddenData{i}.z);
end

softin = softmaxTheta * hiddenData{n+1}.a;
softin = bsxfun(@minus, softin, max(softin));
temp = exp(softin);
pb = bsxfun(@rdivide, temp, sum(temp));

cost1 = -sum(sum(groundTruth .* log(pb))) / m;
cost2 = 0.5 * lambda * sum(sum(softmaxTheta .^2));

cost = cost1 + cost2;
softmaxThetaGrad = -(groundTruth - pb) * hiddenData{n+1}.a'/size(data,2) + lambda * softmaxTheta;

delta = cell(n+1,1);
delta{n+1} = -(softmaxTheta' * (groundTruth - pb)) .* hiddenData{n+1}.a .* (1 - hiddenData{n+1}.a);

for i = n:-1:2
    delta{i} = stack{i}.w' * delta{i+1} .* hiddenData{i}.a .* (1 - hiddenData{i}.a);
end

for i = n:-1:1
    stackgrad{i}.w = delta{i+1} * hiddenData{i}.a' / m;
    stackgrad{i}.b = sum(delta{i+1}, 2) / m;
end

% -------------------------------------------------------------------------

%% Roll gradient vector
grad = [softmaxThetaGrad(:) ; stack2params(stackgrad)];

end


% You might find this useful
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end
