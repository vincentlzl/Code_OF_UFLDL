function [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, data, labels)

% numClasses - the number of classes 
% inputSize - the size N of the input vector
% lambda - weight decay parameter
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
% labels - an M x 1 matrix containing the labels corresponding for the input data
%

% Unroll the parameters from theta
theta = reshape(theta, numClasses, inputSize);

numCases = size(data, 2);

groundTruth = full(sparse(labels, 1:numCases, 1));
cost = 0;

thetagrad = zeros(numClasses, inputSize);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost and gradient for softmax regression.
%                You need to compute thetagrad and cost.
%                The groundTruth matrix might come in handy.
data_new=exp(theta*data);
data_newsum=sum(data_new);
data_newsum=repmat(data_newsum,numClasses,1);
ptheta=data_new./data_newsum;
% cost1=0;
% for i=1:size(data,2)
%     cost1=cost1-sum(log(ptheta(:,i)).*groundTruth(:,i));
% end
% cost1=cost1/size(data,2);
cost1=-sum(sum(log(ptheta).*groundTruth))/size(data,2);
weight_decay=sum(sum(theta.^2))*0.5*lambda;
cost=cost1+weight_decay;

grad1=groundTruth-ptheta;
grad2=zeros(inputSize,size(data,2),numClasses);


for i=1:numClasses
    temp=repmat(grad1(i,:),inputSize,1);
    thetagrad(i,:)=-mean(temp.*data,2)';
end

% 下面是矢量化的实现，用时较长
% for i=1:numClasses
%     grad2(:,:,i)=repmat(grad1(i,:),inputSize,1);
% end
% data_rep=repmat(data,1,1,numClasses);
% test=data_rep.*grad2;
% thetagrad=permute(-mean(data_rep.*grad2,2),[3,1,2]);

thetagrad=thetagrad+lambda*theta;


% ------------------------------------------------------------------
% Unroll the gradient matrices into a vector for minFunc
grad = [thetagrad(:)];
end

