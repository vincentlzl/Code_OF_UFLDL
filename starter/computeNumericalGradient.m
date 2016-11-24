function numgrad = computeNumericalGradient(J, theta)
% numgrad = computeNumericalGradient(J, theta)
% theta: a vector of parameters
% J: a function that outputs a real-number. Calling y = J(theta) will return the
% function value at theta. 
  
% Initialize numgrad with zeros
numgrad = zeros(size(theta));

%% ---------- YOUR CODE HERE --------------------------------------
% Instructions: 
% Implement numerical gradient checking, and return the result in numgrad.  
% (See Section 2.3 of the lecture notes.)
% You should write code so that numgrad(i) is (the numerical approximation to) the 
% partial derivative of J with respect to the i-th input argument, evaluated at theta.  
% I.e., numgrad(i) should be the (approximately) the partial derivative of J with 
% respect to theta(i).
%                
% Hint: You will probably want to compute the elements of numgrad one at a time. 
% a1=zeros(size(theta));
% a2=zeros(size(theta));
% epsilon=10^-4;
% for i=1:size(theta)
%     a1(i)=epsilon;
%     a2(i)=-epsilon;
%     a1=theta+a1;
%     a2=theta+a2;
%     numgrad(i)=(J(a1)-J(a2))/epsilon*0.5;
%     a1(i)=0;
%     a2(i)=0;
% end

EPSILON=0.0001;
for i=1:size(theta)
    theta_plus=theta;
    theta_minu=theta;
    theta_plus(i)=theta_plus(i)+EPSILON;
    theta_minu(i)=theta_minu(i)-EPSILON;
    numgrad(i)=(J(theta_plus)-J(theta_minu))/(2*EPSILON);
end

%% ---------------------------------------------------------------
end
