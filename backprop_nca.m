
maxepoch=1;

load mnistvh_nca
load mnisthp_nca
load mnisthp2_nca
load mnistpo_nca

makebatches_nca;
[numcases numdims numbatches]=size(batchdata); %5000; 784; 12
N=numcases; 

%%%% PREINITIALIZE WEIGHTS OF THE AUTOENCODER
%%%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% vis-hid에 biases append하는 것으로 보임
w1=[vishid; hidrecbiases];
w2=[hidpen; penrecbiases];
w3=[hidpen2; penrecbiases2];
w4=[hidtop; toprecbiases];

%%%%%%%%%% END OF PREINITIALIZATIO OF WEIGHTS  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%% 이건 각 레이어의 노드 사이즈로 보임
l1=size(w1,1)-1;
l2=size(w2,1)-1;
l3=size(w3,1)-1;
l4=size(w4,1)-1;
l5=size(w4,2);

test_err=[];
train_err=[];



%에포크 시작
for epoch = 1:maxepoch

    %%%%%%%%%%%%%%%%%%%% COMPUTE TRAINING RECONSTRUCTION ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [numcases numdims numbatches]=size(batchdata); %%4000 * 784 * 600
    N=numcases;
     
    for batch = 1:numbatches
        data = [batchdata(:,:,batch)]; %%% batch index에 해당하는 배치 데이터 셋을 모셔옴.
        targets = [batchtargets(:,:,batch)];
        
        %%%%%%%%%%%%%%% PERFORM CONJUGATE GRADIENT WITH 3 LINESEARCHES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        max_iter=3;
        data = data(:,1:l1);

        VV = [w1(:)' w2(:)' w3(:)' w4(:)']';
        Dim = [l1; l2; l3; l4; l5];

        [X, fX] = minimize(VV,'CG_MNIST_NCA',max_iter,Dim,data, targets);

        w1 = reshape(X(1:(l1+1)*l2),l1+1,l2);
        xxx = (l1+1)*l2;
        w2 = reshape(X(xxx+1:xxx+(l2+1)*l3),l2+1,l3);
        xxx = xxx+(l2+1)*l3;
        w3 = reshape(X(xxx+1:xxx+(l3+1)*l4),l3+1,l4);
        xxx = xxx+(l3+1)*l4;
        w4 = reshape(X(xxx+1:xxx+(l4+1)*l5),l4+1,l5);

        %%%%%%%%%%%%%%% END OF CONJUGATE GRADIENT WITH 3 LINESEARCHES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %%%%%%%%%%%%%%%%%%%% COMPUTE TEST MISCLASSIFICATION ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
            %%%%%%%%%%%%%%%%%%%% 트레이닝 셋 매핑하기 %%%%%%%%%%%%%%%%%%%% 
            
            f_x_array = [];
            target_array = [];
            make_database;
            
            %%%%%%%%%%%%%%%%%%%% 각 테스트 데이터 배치에 대해 %%%%%%%%%%%%%%%%%%%% 
            counter=0;
            total = 0;
            [testnumcases testnumdims testnumbatches]=size(testbatchdata);
            tN=testnumcases;

            for test_batch = 1:testnumbatches
                
                total = total + testnumcases;
                
                data = [testbatchdata(:,:,test_batch)];
                target = [testbatchtargets(:,:,test_batch)];
                data = [data ones(tN,1)];
                
                w1probs    = 1./(1 + exp(-data*w1)); w1probs = [w1probs  ones(tN,1)];
                w2probs    = 1./(1 + exp(-w1probs*w2)); w2probs = [w2probs ones(tN,1)];
                w3probs    = 1./(1 + exp(-w2probs*w3)); w3probs = [w3probs  ones(tN,1)];
                testbatch_to_low = 1./(1 + exp(-w3probs*w4));
                
                D = pdist2(testbatch_to_low, f_x_array);
                [M,I] = min(D, [], 2);
                
                for i = 1:testnumcases
                    if( find ( 1== target_array(I(i), :)) == find( 1== target(i,:)) )
                        counter = counter +1;
                    end
                end

            end
            
        error_rate = counter / total * 100;
        
        fprintf('epoch: %d / %d, batch: %d / %d, value %2.2f\r', ...
            epoch, maxepoch, batch, numbatches, error_rate);
        
        %%%%%%%%%%%%%%%%%%%% COMPUTE TEST MISCLASSIFICATION ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    end
    
 save mnist_weights_nca w1 w2 w3 w4

end

