% =========================================================
%  Non-linear regression using rbf functions
%  Phi : feature   transformation of given input x
%==========================================================
    
clc;
load Task1.mat                           
no_of_training_samples = 100;
t_train      = t(1: no_of_training_samples);
no_of_polynomial = 13;
                 
%  COMPUTE BASIS FUNCTION:  phi(x^n)  -> 1,x,x^2,x^3,x^4,x^5 ....x^12          
phi_X = [x.^0,x,x.^1,x.^2,x.^3,x.^4,x.^5,x.^6,x.^7,x.^8,x.^9 ...
           ,x.^10,x.^11,x.^12];
phi_X_train      = phi_X(1:no_of_training_samples,:);    % partitition  the data in the into
phi_X_validation = phi_X(no_of_training_samples+1:1000,:); % training and validation set

Wmle        = zeros(no_of_polynomial,1);
MSE_train   = zeros(no_of_polynomial,1);
MSE_val     = zeros(no_of_polynomial,1);

% traing rbf neural net  % compute Wmle (training  the linear model) , number of columns of phi increases in each loop for a
for i= 1: no_of_polynomial
     % Compute Maximum Likelihood estimate coeffients for n dimensional polynomial 
     Wmle  =  pinv(phi_X_train(:,1:i)'  * phi_X_train(:,1:i) ) * phi_X_train(:,1:i)' * t(1:no_of_training_samples);
     ymle  =  phi_X(:,1: i) *   Wmle ; 
     
     % compute the errors and SSE
     Error       =   t - ymle;    
     E1           =  Error(1:no_of_training_samples)  ;
     E2           =  Error(no_of_training_samples+1:1000)  ;
     SSE_train    =  (norm(E1))^2;
     SSE_val      =  (norm(E2))^2;
   
     % compute the MSE for validation data
     MSE_train(i) = SSE_train/100;
     MSE_val(i)   = SSE_val/900;
     
     %Plot Y(x|mle) for the current polnomial 
     figure(i);
     allPlot =  plot(x(1:100), ymle( 1:100 ),'bd'  );
     xlabel( ['x^{}' num2str(i)]); % label x axis
     ylabel( 'Ymle'); % label y axis
     title(['plot for polynomial' num2str(i)]); % % Title of the current plot
     legend(allPlot, ['Y predicted based on weight obtainted in the polynomial' num2str(i)]); 
     saveas(gcf, ['MPI_Task2a_aadeaina2014' num2str(i)], 'jpg'); % catpure plot
end
clf;
        
% TRACK MINIMUM MSEtrain and MSE validation
   [minMSEtr, minPositiontr]= min(MSE_train);
   [minMSEvl, minPositionvl]= min(MSE_val);
   disp(['ME train : ' num2str(minMSEtr)]);
   disp(['MSE val: '  num2str(minMSEtr)]);
