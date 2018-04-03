%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Implementation of Bayesian Linear 
%  Regression Algorithm  
%  data input /description  :                 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


clc; clear; close all;
load( 'task1.mat');
AnimationFile = 'BLR'; % Filename to save the video to.
imFold = '.\Images\';
diary([AnimationFile '.txt']);
[Sample, Dimension] = size(x);

% Number of w parameters.
NumberOfWeightParameters = 2;
x = [ones(Sample,1), x];     % because of w1 term in tn= w1+w2x+ en
W_true = [1 1];              % True weight
PriorMeanOfWeight = [0 0];   % "Wo"
VarianceOfPriorWeight = 0.4; % small "s"   user defined
MeanOfError = 0; %    
VarianceOfError = 0.25;      % Sigma_squared

PosteriorMeanOfWeight = PriorMeanOfWeight; 
W_mle = (x'*x)^-1*x'*t; 
PosteriorCovarianceMatrix = (VarianceOfPriorWeight)* ...
eye(NumberOfWeightParameters); % (Cw|t (1)  at N=0 i.e before data is seen)

%%%%%%%%%%%%%%%%%%%%%%%%
% PLOT SETUP
%%%%%%%%%%%%%%%%%%%%%%%%

% Generate a grid  that will be used to draw the contour plot later. 
[XGrid, YGrid] = meshgrid(0.9: 0.005: 1.1, 0.9: 0.005: 1.1);
[gx, gy] = size(XGrid); 
  grPts = [XGrid(:), YGrid(:)]; % all possible data samples 
  mkSize = 5; %  
  lineWidth = 1.5;
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Animation  parameters  setup           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
NumberOfFrames = 400;`
A_vector       = floor(linspace(1,Sample, NumberOfFrames));
N              = 0;
M              = 0; 

Animation = [1 A_vector(floor(0.3*NumberOfFrames)), A_vector(floor(0.5*...
    NumberOfFrames)), A_vector(floor(0.99*NumberOfFrames))];
      

sStr = num2str(VarianceOfPriorWeight);
dotPos = sStr == '.';
sStr(dotPos) = [];

writerObj = VideoWriter('MP1_Task1_f.avi');
open(writerObj);
axis tight
set(gca,'nextplot','replacechildren');
set(gcf,'Renderer','zbuffer');

for i = 1 : Sample 
       newx = x(i,:);
       newt = t(i,:);
       % computation of contour plot parameters %
               % G(N+1)  %
       G_Of_N_Plus_One = eye(NumberOfWeightParameters) -((PosteriorCovarianceMatrix *...
          x(i,:)'*x(i,:)) ./(VarianceOfError + x(i,:)*PosteriorCovarianceMatrix ...
           *x(i,:)'));
   
                % C(N+1)  %
        New_PosteriorCovarianceMatrix = G_Of_N_Plus_One* PosteriorCovarianceMatrix ;
    
                % W_map(N+1)  %
        New_PosteriorMeanOfWeight = G_Of_N_Plus_One * PosteriorMeanOfWeight' + ((t(i,:)*  ...
        PosteriorCovarianceMatrix*x(i,:)') ./(VarianceOfError + x(i,:)* ...
        PosteriorCovarianceMatrix*x(i,:)'));
                                              
        N = N + 1; 
        
       
        % Compute Probability of W given data t using multivariate  normal distribution function%
        Probability_W_given_t = mvnpdf(grPts, New_PosteriorMeanOfWeight', New_PosteriorCovarianceMatrix);
        % reshape the probability into the grid dimensions%    
        Probability_Grid = reshape(Probability_W_given_t, gx, gy);
        
        % plot a contour of the pdf  based on the probability grid%
        contourf(XGrid, YGrid, Probability_Grid);
        hold on;

        Hold_True = plot(W_true(1), W_true(2), 'o', 'Markersize', mkSize + 3,...
                    'LineWidth', lineWidth, 'MarkerEdgeColor','k',...
                    'MarkerFaceColor', 'y'); % Plot the true w. 
        Hold_MLE = plot(W_mle(1), W_mle(2), 'm+', 'Markersize', mkSize + 1,...
                    'LineWidth', lineWidth -0.2);  % Plot the wMLE.        
        Hold_MAP = plot(New_PosteriorMeanOfWeight(1), New_PosteriorMeanOfWeight(2), 'yx', 'Markersize', mkSize + 4,...
                    'LineWidth', lineWidth, 'MarkerEdgeColor','g'); % Plot the wMAP.
        
        % Storing the handles in one matrix. 
        C_Handle = [Hold_True; Hold_MLE; Hold_MAP];
        
        %Configure Legend Strings.
        stTr = ['W_true = ' num2str(W_true)];
        stMLE = ['W_mle = ' num2str(W_mle')];
        stMAP = ['W_map = ' num2str(New_PosteriorMeanOfWeight')];
        Nnum = ['N = ' num2str(i)];
        
        % Put in a legend.
        legend(C_Handle, stTr, stMLE, stMAP);
       
        title(['N = ' num2str(i)]);
        axis square;
        
        
        % Capture the frame and store
         Animation_Capture = getframe;
         writeVideo(writerObj,Animation_Capture);
        
        % Save the screen capture every 100 frames
           b=0;
        if (b == mod ( i, 100 ))
            M = M + 1;
      
            saveas(gcf, ['MPI_Task1F_aadeaina2014' num2str(M)], 'jpg');
            
        end
        
         % Clear all but the last figure
        if(i ~= 1000)
            clf;
        end
   
       % Cw|t(N) and W_map(N) Update%
       PosteriorCovarianceMatrix = New_PosteriorCovarianceMatrix ;
       PosteriorMeanOfWeight     = New_PosteriorMeanOfWeight';
end

% convert captured frames to AVI format
%movie2avi(Animation_Capture, 'MP1_TasN1f' ,'compression', 'None', 'quality' , 100);


close(writerObj);
diary off
