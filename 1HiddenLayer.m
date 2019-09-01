
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%             One hidden Layes            %%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  loading the data from data set that called LoadMNIST 
[xTrain, targetTrain, xValid, targetValid, xTest, targetTest] = LoadMNIST(1);

epok = 30;
miniBatchSize = 12;                % we will not word with whole of data at the same time
numberOfminiBatchIteration = 500;  
NumberOfNeuronInHiddenLayer = 100;             
%{
learningrate between [0,1]. High rate make the model to converger fast but might miss 
the global minima and too low requires higher iteration and running time.
%}
learningsRate = 0.3;
%{ 
centering the data and consequenlty making the mean-value to zero partly due to a 
better balance in the data and partly to decrease the collinearity effect.  
%}
meanValueOfxTrain = mean(xTrain, 2);
meanValueOfxValid = mean(xValid, 2);
meanValueOfxTest  = mean(xTest, 2);
centered_xTrain = (xTrain - meanValueOfxTrain);
centered_xValid = (xValid - meanValueOfxValid);
centered_xTest  = (xTest - meanValueOfxTest);

% for saving the error in each epok
classificationErrorsInValidationData = zeros(epok, 1);
classificationErrorsInTrainData = zeros(epok, 1);
classificationErrorsInTestData = zeros(epok, 1);

Weights_jk = randn(NumberOfNeuronInHiddenLayer , 784) .* (1/sqrt(784)); % 100x784    
Weights_i = randn(10 , NumberOfNeuronInHiddenLayer) .* (1/sqrt(NumberOfNeuronInHiddenLayer)); % 10*100  
ThetaInput = zeros(NumberOfNeuronInHiddenLayer, 1); 
ThetaOutput = zeros(10, 1); 
for p = 1:epok
    p   
    for t = 1:numberOfminiBatchIteration
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%     Forwards propagation     %%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        % picking randlomly minibatch amount of the data for training
        randomIndices = unidrnd(length(centered_xTrain(1,:)), miniBatchSize, 1);
        inputMiniBatch = centered_xTrain(:,randomIndices);
        targetMiniBatch  = targetTrain(:,randomIndices);
        
        exp_bj = exp( -Weights_jk*inputMiniBatch + ThetaInput); 
        Vj = 1 ./ (1 + exp_bj); 
       
        exp_b =  exp( -Weights_i*Vj + ThetaOutput); 
        output = 1 ./ (1 + exp_b); 
        
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%      Backwards propagation    %%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        GprimOfB  = (exp_b  ./ (1 + exp_b).^2);
        GprimOfBj = (exp_bj ./ (1 + exp_bj).^2);
        
        
        DeltaMatrix_output= (targetMiniBatch - output) .* GprimOfB; 
        DeltaMatrix_input = (DeltaMatrix_output' * Weights_i)' .* GprimOfBj; 
  
     
        DeltaWi = zeros(10, NumberOfNeuronInHiddenLayer);
        DeltaWjk = zeros(NumberOfNeuronInHiddenLayer, 784);
        
        DeltaWi = DeltaWi + DeltaMatrix_output * Vj'; 
        DeltaWjk = DeltaWjk + DeltaMatrix_input * inputMiniBatch';
    
        
        % %%%%%%%%%%%%%%%%% Updates the wights by deltaweights %%%%%%%%%%%%%%%%%%%%
        
        Weights_i  =   Weights_i  +  learningsRate  *  DeltaWi;
        Weights_jk  =   Weights_jk   + learningsRate *  DeltaWjk;
        
        % %%%%%%%%%%%%%%%%% updating Theta by deltaTheta %%%%%%%%%%%%%%%%%%%%
        DeltaTheta_output = - learningsRate * sum(DeltaMatrix_output, 2);
        DeltaTheta_input = - learningsRate * sum(DeltaMatrix_input, 2);
        ThetaOutput = ThetaOutput + DeltaTheta_output;          
        ThetaInput = ThetaInput + DeltaTheta_input;              
        %%%%%%%%%%%%%%%%%%% done with Back propagation %%%%%%%%%%%%%%%%%
    end
    
    
    
    
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % %%%%%%%%%%%%%%%%%%      Classification Eroor      %%%%%%%%%%%%%%%%%
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
% Using the parameters above to get outputs from Train-, Validation- and Test-data
% only forwards process will be applied here

    ExpBetaTrain = exp(- Weights_jk*centered_xTrain + ThetaInput);
    ExpBetaValid = exp(- Weights_jk*centered_xValid + ThetaInput); 
    ExpBetaTest = exp(- Weights_jk*centered_xTest + ThetaInput);
    
    VjTrain = 1 ./ (1 + ExpBetaTrain); 
    VjValid = 1 ./ (1 + ExpBetaValid); 
    VjTest = 1 ./ (1 + ExpBetaTest); 
       
    ExpbetaTrain2 =  exp(- Weights_i*VjTrain + ThetaOutput);  
    ExpbetaValid2 =  exp(- Weights_i*VjValid + ThetaOutput);
    ExpbetaTest2  =  exp(- Weights_i*VjTest + ThetaOutput);
    
    output_train = 1 ./ (1 + ExpbetaTrain2); 
    output_valid = 1 ./ (1 + ExpbetaValid2); 
    output_test  = 1 ./ (1 + ExpbetaTest2);
   
    
    
    for i = 1:size(centered_xValid, 2) 
        indexOfMax = find(output_valid(:, i)==max(output_valid(:, i))); % return index of the max. 
        output_valid(:, i) = 0;                                         % set the others to zero
        output_valid(indexOfMax, i) = 1;                                % and 1 in index of max
        
        indexOfMax = find(output_test(:, i)==max(output_test(:, i)));
        output_test(:, i) = 0;
        output_test(indexOfMax, i) = 1;
    end
    for i = 1:size(centered_xTrain, 2)
        indexOfMax = find(output_train(:, i)==max(output_train(:, i)));
        output_train(:, i) = 0;
        output_train(indexOfMax, i) = 1;
    end
    
    % compare the output of the model with respective data with corresponding target
    sumOfMissclassifiedTrain = 0;
    sumOfMissclassifiedValid = 0;
    sumOfMissclassifiedTest = 0;
    for i = 1:size(centered_xValid, 2)
        sumOfMissclassifiedValid = sumOfMissclassifiedValid + sum(abs(output_valid(:, i) - targetValid(:, i))); 
        sumOfMissclassifiedTest = sumOfMissclassifiedTest + sum(abs(output_test(:, i) - targetTest(:, i)));
    end
    for i = 1:size(centered_xTrain, 2)
        sumOfMissclassifiedTrain = sumOfMissclassifiedTrain + sum(abs(output_train(:, i) - targetTrain(:, i)));
    end
    
    
    % #### making the score of missclassified to the ratio between [0,1]
    classificationErrorsInTrainData(p) = sumOfMissclassifiedTrain / (2 * size(centered_xTrain, 2));
    classificationErrorsInValidationData(p) = sumOfMissclassifiedValid / (2 * size(centered_xValid, 2));
    classificationErrorsInTestData(p) = sumOfMissclassifiedTest / (2 * size(centered_xTest, 2));
end

% plotting the ratio of missclassification in log-format in respective data
subplot(1,3,1);
plot(log(classificationErrorsInTrainData));
xlabel('epok')
ylabel('missclasifications rate in log-format')
title('Training Data')
subplot(1,3,2);
plot(log(classificationErrorsInValidationData));
title('Validation Data')
xlabel('epok')
ylabel('missclasifications rate in log-format')
subplot(1,3,3);
plot(log(classificationErrorsInTestData));
title('Test Data')
xlabel('epok')
ylabel('missclasifications rate in log-format')

