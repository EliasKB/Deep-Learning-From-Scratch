%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%             No hidden Layes             %%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[xTrain, targetTrain, xValid, targetValid, xTest, targetTest] = LoadMNIST(1);

epok = 30;
learningsRate =  0.3;
numberOfminiBatchIteration = 500;
miniBatchSize = 12

Matrix = cell((epok+1), 2);
meanValueOfxTrain = mean(xTrain, 2);
meanValueOFxValid = mean(xValid, 2);
meanValueOfxTest  = mean(xTest, 2);

centered_xTrain = (xTrain - meanValueOfxTrain);
centered_xValid = (xValid - meanValueOFxValid);
centered_xTest  = (xTest - meanValueOfxTest);

classificationErrorsInTrainData = zeros(epok, 1);
classificationErrorsInValidationData = zeros(epok, 1);
classificationErrorsInTestData = zeros(epok, 1);


Weights_jk = randn(10, length(xTrain(:, 1)))*(1/sqrt(7840)); % 10x784
Theta = zeros(10, 1);
O = zeros(10, miniBatchSize);
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

        exp_bj = exp( -Weights_jk * inputMiniBatch + Theta); 
        output = 1 ./ (1 + exp_bj); 
        
        
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%      Backwards propagation    %%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        GprimOfB = (exp_bj ./ (1 + exp_bj).^2); 
        DeltaMatrix= (targetMiniBatch - output) .* GprimOfB; 

        
        DeltaWjk = zeros(10, 784);
        DeltaWjk = DeltaWjk + DeltaMatrix * inputMiniBatch';
        
        
        %%%%%%%%%%%%%%%%%% Updates the wights by deltaweights %%%%%%%%%%%%%%%%%%%%
        Weights_jk  =   Weights_jk   + learningsRate *  DeltaWjk;
        
        
        %%%%%%%%%%%%%%%%%% updating Theta by deltaTheta %%%%%%%%%%%%%%%%%%%%
        DeltaTheta = - learningsRate * sum(DeltaMatrix, 2);
        Theta = Theta + DeltaTheta;              
        %%%%%%%%%%%%%%%%%%% done with Back propagation %%%%%%%%%%%%%%%%%
    end
    
    
    
    
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % %%%%%%%%%%%%%%%%%%      Classification Eroor      %%%%%%%%%%%%%%%%%
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
% Using the parameters above to get outputs from Train-, Validation- and Test-data
% only forwards process will be applied here

    ExpBetaTrain = exp(- Weights_jk*centered_xTrain + Theta);
    ExpBetaValid = exp(- Weights_jk*centered_xValid + Theta); 
    ExpBetaTest = exp(- Weights_jk*centered_xTest + Theta);   
   
    output_train = 1 ./ (1 + ExpBetaTrain); 
    output_valid = 1 ./ (1 + ExpBetaValid); 
    output_test  = 1 ./ (1 + ExpBetaTest);
   
    % spara classification error
    for i = 1:size(centered_xValid, 2) 
        indexOfMax = find(output_valid(:, i)==max(output_valid(:, i))); 
        output_valid(:, i) = 0;                                      
        output_valid(indexOfMax, i) = 1;                                
        
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



