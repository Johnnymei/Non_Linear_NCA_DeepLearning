maxepoch=1;

load mnistvh_nca
load mnisthp_nca
load mnisthp2_nca
load mnistpo_nca

makebatches_nca;c
[numcases numdims numbatches]=size(batchdata);
N=numcases; 

%%%%%% NCA를 위한 웨이트값 설정 %%%%%% 
w1=[vishid; hidrecbiases];
w2=[hidpen; penrecbiases];
w3=[hidpen2; penrecbiases2];
w4=[hidtop; toprecbiases];
%%%%%% NCA를 위한 웨이트값 설정 %%%%%% 

%%%%%% NCA2를 위한 웨이트값 설정 %%%%%% 
w1_nca2=[vishid; hidrecbiases];
w2_nca2=[hidpen; penrecbiases];
w3_nca2=[hidpen2; penrecbiases2];
w4_nca2=[hidtop; toprecbiases];
%%%%%% NCA2를 위한 웨이트값 설정 %%%%%% 


%%%%%%%%%% 이건 각 레이어 노드 사이즈 %%%%%%%%%%
l1=size(w1,1)-1;
l2=size(w2,1)-1;
l3=size(w3,1)-1;
l4=size(w4,1)-1;
l5=size(w4,2);
%%%%%%%%%% 이건 각 레이어 노드 사이즈 %%%%%%%%%%


%%%%%%%%%% 에포크 시작 %%%%%%%%%% 
for epoch = 1:maxepoch

    [numcases numdims numbatches]=size(batchdata);
    N=numcases;
     
    for batch = 1:numbatches
        data = [batchdata(:,:,batch)]; %%% batch index에 해당하는 배치 데이터 셋을 모셔옴.
        targets = [batchtargets(:,:,batch)];
        
        %%%%%%%%%%%%%%% PERFORM CONJUGATE GRADIENT WITH 3 LINESEARCHES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        max_iter=3;
        data = data(:,1:l1);

        %%%%%%%%%%%%%%% NCA에 대한 최적화 %%%%%%%%%%%%%%% 
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
        
            %%%%%%%%%%%%%%%%%%%%%%%% COMPUTE TEST MISCLASSIFICATION ERROR %%%%%%%%%%%%%%%%%%%%%%%% 
            %%%%%%%%%%%%%%%%%%%% 트레이닝 셋 매핑하기 %%%%%%%%%%%%%%%%%%%% 
            f_x_array = [];
            target_array = [];
            make_database_nca;
            
            %%%%%%%%%%%%%%%%%%%% 각 테스트 데이터 배치에 대해 %%%%%%%%%%%%%%%%%%%% 
            counter=0;
            total = 0;
            [testnumcases testnumdims testnumbatches]=size(testbatchdata);
            tN=testnumcases;

            for test_batch = 1:testnumbatches
                
                total = total + testnumcases;
                
                test_data = [testbatchdata(:,:,test_batch)];
                target = [testbatchtargets(:,:,test_batch)];
                test_data = [test_data ones(tN,1)];
                
                w1probs    = 1./(1 + exp(-test_data*w1)); w1probs = [w1probs  ones(tN,1)];
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

            fprintf('NCA, epoch: %d / %d, batch: %d / %d, value %2.5f\r', ...
                epoch, maxepoch, batch, numbatches, error_rate);
            
            %%%%%%%%%%%%%%%%%%%% 테스트 데이터 정확도 계산 끝 %%%%%%%%%%%%%%%%%%%% 
      
        %%%%%%%%%%%%%%% NCA에 대한 최적화 끝%%%%%%%%%%%%%%% 

        %%%%%%%%%%%%%%% NCA2에 대한 최적화 %%%%%%%%%%%%%%% 
        VV = [w1_nca2(:)' w2_nca2(:)' w3_nca2(:)' w4_nca2(:)']';
        Dim = [l1; l2; l3; l4; l5];
        [X, fX] = minimize(VV,'CG_MNIST_NCA2',max_iter,Dim,data, targets,0.99);
        
        w1_nca2 = reshape(X(1:(l1+1)*l2),l1+1,l2);
        xxx = (l1+1)*l2;
        w2_nca2 = reshape(X(xxx+1:xxx+(l2+1)*l3),l2+1,l3);
        xxx = xxx+(l2+1)*l3;
        w3_nca2 = reshape(X(xxx+1:xxx+(l3+1)*l4),l3+1,l4);
        xxx = xxx+(l3+1)*l4;
        w4_nca2 = reshape(X(xxx+1:xxx+(l4+1)*l5),l4+1,l5);
        %%%%%%%%%%%%%%% NCA2에 대한 최적화 %%%%%%%%%%%%%%% 

        
        %%%%%%%%%%%%%%%%%%%%%%%% COMPUTE TEST MISCLASSIFICATION ERROR %%%%%%%%%%%%%%%%%%%%%%%% 
            %%%%%%%%%%%%%%%%%%%% 트레이닝 셋 매핑하기 %%%%%%%%%%%%%%%%%%%% 
            f_x_array = [];
            target_array = [];
            make_database_nca2;
            
            %%%%%%%%%%%%%%%%%%%% 각 테스트 데이터 배치에 대해 %%%%%%%%%%%%%%%%%%%% 
            counter=0;
            total = 0;
            [testnumcases testnumdims testnumbatches]=size(testbatchdata);
            tN=testnumcases;

            for test_batch = 1:testnumbatches
                
                total = total + testnumcases;
                
                test_data = [testbatchdata(:,:,test_batch)];
                target = [testbatchtargets(:,:,test_batch)];
                test_data = [test_data ones(tN,1)];
                
                w1probs    = 1./(1 + exp(-test_data*w1_nca2)); w1probs = [w1probs  ones(tN,1)];
                w2probs    = 1./(1 + exp(-w1probs*w2_nca2)); w2probs = [w2probs ones(tN,1)];
                w3probs    = 1./(1 + exp(-w2probs*w3_nca2)); w3probs = [w3probs  ones(tN,1)];
                testbatch_to_low = 1./(1 + exp(-w3probs*w4_nca2));
                
                D = pdist2(testbatch_to_low, f_x_array);
                [M,I] = min(D, [], 2);
                
                for i = 1:testnumcases
                    if( find ( 1== target_array(I(i), :)) == find( 1== target(i,:)) )
                        counter = counter +1;
                    end
                end

            end
            
            error_rate = counter / total * 100;

            fprintf('NCA2, epoch: %d / %d, batch: %d / %d, value %2.5f\r', ...
                epoch, maxepoch, batch, numbatches, error_rate);
            
            %%%%%%%%%%%%%%%%%%%% 테스트 데이터 정확도 계산 끝 %%%%%%%%%%%%%%%%%%%% 
        
        %%%%%%%%%%%%%%%%%%%% COMPUTE TEST MISCLASSIFICATION ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    end
    
 save mnist_weights_nca w1_nca2 w2_nca2 w3_nca2 w4_nca2

end

