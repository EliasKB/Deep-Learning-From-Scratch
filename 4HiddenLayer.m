
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%    Four hidden Layes 50 neurons     %%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[xTrain, targetTrain, xValid, targetValid, xTest, targetTest] = LoadMNIST(2);
epok = 50;
MiniBatchSize = 10;
numberOfminiBatchIteration = 5000;
numberOfNeuronInFirstHiddenLayer = 30;
numberOfNeuronInSecondHiddenLayer = 31;
numberOfNeuronInThirdHiddenLayer = 32;
numberOfNeuronInForthHiddenLayer = 33;

learningsRate =  0.003;
LearningSpeed = zeros(epok, 5);

resultsMatrix = cell((epok), 10);
meanValueOfxTrain = mean(xTrain, 2);
meanValueOfxValid = mean(xValid, 2);
meanValueOFxTest  = mean(xTest, 2);
centered_xTrain = (xTrain - meanValueOfxTrain);
centered_xValid = (xValid - meanValueOfxValid);
centered_xTest  = (xTest - meanValueOFxTest);

% for saving the error in each epok
classificationErrorsInValidationData = zeros(epok, 1);
classificationErrorsInTrainData = zeros(epok, 1);
classificationErrorsInTestData = zeros(epok, 1);

Weights1 = randn(numberOfNeuronInFirstHiddenLayer , 784) .* (1/sqrt(784)); 
Weights2 = randn(numberOfNeuronInSecondHiddenLayer , numberOfNeuronInFirstHiddenLayer) .* (1/sqrt(numberOfNeuronInFirstHiddenLayer)); % 30x30
Weights3 = randn(numberOfNeuronInThirdHiddenLayer , numberOfNeuronInSecondHiddenLayer) .* (1/sqrt(numberOfNeuronInSecondHiddenLayer)); % 30x30
Weights4 = randn(numberOfNeuronInForthHiddenLayer , numberOfNeuronInThirdHiddenLayer) .* (1/sqrt(numberOfNeuronInThirdHiddenLayer)); % 30x30
Weights5 = randn(10 , numberOfNeuronInForthHiddenLayer) .* (1/sqrt(numberOfNeuronInForthHiddenLayer)); 
Theta1 = zeros(numberOfNeuronInFirstHiddenLayer, 1);
Theta2 = zeros(numberOfNeuronInSecondHiddenLayer, 1);
Theta3 = zeros(numberOfNeuronInThirdHiddenLayer, 1);
Theta4 = zeros(numberOfNeuronInForthHiddenLayer, 1);
Theta5 = zeros(10, 1);
for p = 1:epok
    p
    for t = 1:numberOfminiBatchIteration
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%     Forwards propagation     %%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        randomIndices = unidrnd(length(centered_xTrain(1,:)), MiniBatchSize, 1);
        inputMiniBatch = centered_xTrain(:,randomIndices);
        targetMiniBatch  = targetTrain(:,randomIndices);
        
        exp_b1 = exp( -Weights1*inputMiniBatch + Theta1); 
        V1 = 1 ./ (1 + exp_b1); 
        exp_b2 = exp(- Weights2 * V1 + Theta2); 
        V2 = 1 ./ (1 + exp_b2); 
        exp_b3 = exp( - Weights3 * V2 + Theta3);
        V3 = 1 ./ (1 + exp_b3);
        exp_b4 = exp( - Weights4 * V3 + Theta4);
        V4 = 1 ./ (1 + exp_b4);
        exp_b5 =  exp( - Weights5 * V4 + Theta5);  
        output = 1 ./ (1 + exp_b5);   
        
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%      Backwards propagation    %%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        GprimOfb5  = (exp_b5  ./ (1 + exp_b5).^2); 
        GprimOfb4  = (exp_b4  ./ (1 + exp_b4).^2);
        GprimOfb3  = (exp_b3  ./ (1 + exp_b3).^2);
        GprimOfb2  = (exp_b2  ./ (1 + exp_b2).^2); 
        GprimOfb1  = (exp_b1  ./ (1 + exp_b1).^2); 
        

        Delta5Matrix = (targetMiniBatch - output) .* GprimOfb5;
        Delta4Matrix = (Delta5Matrix' * Weights5)' .* GprimOfb4;
        Delta3Matrix = (Delta4Matrix' * Weights4)' .* GprimOfb3;
        Delta2Matrix = (Delta3Matrix' * Weights3)' .* GprimOfb2;
        Delta1Matrix = (Delta2Matrix' * Weights2)' .* GprimOfb1;
       
        
        DeltaW5 = zeros(10, numberOfNeuronInForthHiddenLayer);
        DeltaW4 = zeros(numberOfNeuronInForthHiddenLayer, numberOfNeuronInThirdHiddenLayer);
        DeltaW3 = zeros(numberOfNeuronInThirdHiddenLayer, numberOfNeuronInSecondHiddenLayer);
        DeltaW2 = zeros(numberOfNeuronInSecondHiddenLayer, numberOfNeuronInFirstHiddenLayer);
        DeltaW1 = zeros(numberOfNeuronInFirstHiddenLayer, 784);
        
       
        DeltaW5 = DeltaW5 + Delta5Matrix * V4';
        DeltaW4 = DeltaW4 + Delta4Matrix * V3'; 
        DeltaW3 = DeltaW3 + Delta3Matrix * V2'; 
        DeltaW2 = DeltaW2 + Delta2Matrix * V1'; 
        DeltaW1 = DeltaW1 + Delta1Matrix * inputMiniBatch'; 
        
        
        % %%%%%%%%%%%%%%%%% Updates the wights by deltaweights %%%%%%%%%%%%%%%%%%%%
        
        Weights5  =   Weights5  +  learningsRate  *  DeltaW5;
        Weights4  =   Weights4  +  learningsRate  *  DeltaW4;
        Weights3  =   Weights3  +  learningsRate  *  DeltaW3;
        Weights2  =   Weights2   + learningsRate *  DeltaW2;
        Weights1  =   Weights1   + learningsRate *  DeltaW1;
        
        % updating Theta by deltaTheta
        DeltaTheta5 = - learningsRate * sum(Delta5Matrix, 2);
        DeltaTheta4 = - learningsRate * sum(Delta4Matrix, 2);
        DeltaTheta3 = - learningsRate * sum(Delta3Matrix, 2); 
        DeltaTheta2 = - learningsRate * sum(Delta2Matrix, 2); 
        DeltaTheta1 = - learningsRate * sum(Delta1Matrix, 2); 
        
        Theta5 = Theta5 + DeltaTheta5;    
        Theta4 = Theta4 + DeltaTheta4;    
        Theta3 = Theta3 + DeltaTheta3;    
        Theta2 = Theta2 + DeltaTheta2;    
        Theta1 = Theta1 + DeltaTheta1;    
        %%%%%%%%%%%%%%%%%%% done with Back propagation %%%%%%%%%%%%%%%%%
    end
    
    
    LearningSpeed(p, 1) = norm(sum(Delta1Matrix, 2));
    LearningSpeed(p, 2) = norm(sum(Delta2Matrix, 2));
    LearningSpeed(p, 3) = norm(sum(Delta3Matrix, 2));
    LearningSpeed(p, 4) = norm(sum(Delta4Matrix, 2));
    LearningSpeed(p, 5) = norm(sum(Delta5Matrix, 2));

    resultsMatrix{(p), 1} = Weights1; 
    resultsMatrix{(p), 2} = Weights2;
    resultsMatrix{(p), 3} = Weights3;
    resultsMatrix{(p), 4} = Weights4;
    resultsMatrix{(p), 5} = Weights5;

    resultsMatrix{(p), 6} = Theta1;
    resultsMatrix{(p), 7} = Theta2;
    resultsMatrix{(p), 8} = Theta3;
    resultsMatrix{(p), 9} = Theta4;
    resultsMatrix{(p),10} = Theta5;

    
    
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % %%%%%%%%%%%%%%%%%%      Classification Eroor      %%%%%%%%%%%%%%%%%
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
% skicka in alla dina patterns i train, validation och test sen gå framåt, få ut outputs för alla
     ExpBetaTrain1 = exp(- Weights1*centered_xTrain + Theta1); % (100*784 * 784*50000)  exp(-b1)
     ExpBetaValid1 = exp(- Weights1*centered_xValid + Theta1); % (100*784 * 784*50000)  exp(-b1)
     ExpBetaTest1  = exp(- Weights1*centered_xTest + Theta1); % (100*784 * 784*50000)  exp(-b1)

     V1Train = 1 ./ (1 + ExpBetaTrain1); % V1 = 100*50000 or g(-b1)
     V1Valid = 1 ./ (1 + ExpBetaValid1); % V1 = 100*50000 or g(-b1)
     V1Test  = 1 ./ (1 + ExpBetaTest1); % V1 = 100*50000 or g(-b1)

     ExpBetaTrain2 = exp(- Weights2*V1Train + Theta2); % (100*100 * 100*50000)  exp(-b1)
     ExpBetaValid2 = exp(- Weights2*V1Valid + Theta2); % (100*100 * 100*50000)  exp(-b1)
     ExpBetaTest2  = exp(- Weights2*V1Test + Theta2); % (100*100 * 100*50000)  exp(-b1)

     V2Train = 1 ./ (1 + ExpBetaTrain2); % V1 = 100*50000 or g(-b1)
     V2Valid = 1 ./ (1 + ExpBetaValid2); % V1 = 100*50000 or g(-b1)
     V2Test  = 1 ./ (1 + ExpBetaTest2); % V1 = 100*50000 or g(-b1)

     ExpBetaTrain3 = exp(- Weights3*V2Train + Theta3); 
     ExpBetaValid3 = exp(- Weights3*V2Valid + Theta3); 
     ExpBetaTest3  = exp(- Weights3*V2Test + Theta3); 
               
     V3Train = 1 ./ (1 + ExpBetaTrain3);
     V3Valid = 1 ./ (1 + ExpBetaValid3);
     V3Test  = 1 ./ (1 + ExpBetaTest3);
             
     ExpBetaTrain4 = exp(- Weights4*V3Train + Theta4); 
     ExpBetaValid4 = exp(- Weights4*V3Valid + Theta4); 
     ExpBetaTest4  = exp(- Weights4*V3Test + Theta4); 
     
     V4Train = 1 ./ (1 + ExpBetaTrain4);
     V4Valid = 1 ./ (1 + ExpBetaValid4);
     V4Test  = 1 ./ (1 + ExpBetaTest4);
     
     ExpBetaTrain5 = exp(- Weights5*V4Train + Theta5); 
     ExpBetaValid5 = exp(- Weights5*V4Valid + Theta5);
     ExpBetaTest5  = exp(- Weights5*V4Test + Theta5);
               
     output_train = 1 ./ (1 + ExpBetaTrain5); 
     output_valid = 1 ./ (1 + ExpBetaValid5); 
     output_test  = 1 ./ (1 + ExpBetaTest5); 
   
     
     
     
     
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
    
    
    % compare the output of the model with respective data with corresponding target
    sumOfMissclassifiedTrain = 0;
    sumOfMissclassifiedValid = 0;
    sumOfMissclassifiedTest = 0;
    sumOfEnergiFunctionTrain = 0;
    sumOfEnergiFunctionValid = 0;
    sumOfEnergiFunctionTest = 0;
    
    for i = 1:size(centered_xTrain, 2)
        sumOfEnergiFunctionTrain = sumOfEnergiFunctionTrain + sum((output_train(:, i) - targetTrain(:, i)).^2);
        sumOfMissclassifiedTrain   = sumOfMissclassifiedTrain + sum(abs(output_train(:, i) - targetTrain(:, i)));
    end
    
    for i = 1:size(centered_xValid, 2)
        sumOfMissclassifiedValid  = sumOfMissclassifiedValid + sum(abs(output_valid(:, i) - targetValid(:, i)));
        sumOfMissclassifiedTest   = sumOfMissclassifiedTest + sum(abs(output_test(:, i) - targetTest(:, i)));
        sumOfEnergiFunctionValid = sumOfEnergiFunctionValid + sum((output_train(:, i) - targetTrain(:, i)).^2);
        sumOfEnergiFunctionTest  = sumOfEnergiFunctionTest + sum((output_test(:, i) - targetTest(:, i)).^2);
    end

    
    % #### making the score of missclassified to the ratio between [0,1]
    classificationErrorsInTrainData(p) = sumOfMissclassifiedTrain / (2 * size(centered_xTrain, 2));
    classificationErrorsInValidationData(p) = sumOfMissclassifiedValid / (2 * size(centered_xValid, 2));
    classificationErrorsInTestData(p) = sumOfMissclassifiedTest / (2 * size(centered_xTest, 2));
    EnergiFunction(p) = sumOfEnergiFunctionTrain / 2;
    
end

figure(1)
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

figure(2) 
plot(EnergiFunction); 
title('Energy function of training data as a function of epoch')

figure(3) 
plot(1:epok, log(LearningSpeed), 'LineWidth', 2); 
title('Learning Speed for different layers a function in log-format') 
legend("Layer1", "Layer2", "Layer3", "Layer4","Layer5", 'Location', 'southeast')

figure(4) 
plot(1:epok, (LearningSpeed), 'LineWidth', 2); 
title('Learning Speed for different layers a function of epoch') 
legend("Layer1", "Layer2", "Layer3", "Layer4","Layer5", 'Location', 'southeast')
    