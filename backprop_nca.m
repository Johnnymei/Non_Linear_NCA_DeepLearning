
maxepoch=5;
fprintf(1,'\nFine-tuning deep autoencoder by minimizing cross entropy error. \n');
fprintf(1,'50 batches of 5000 cases each. \n');

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
    err=0; 
    [numcases numdims numbatches]=size(batchdata); %%4000 * 784 * 600
    N=numcases;
     
    for batch = 1:numbatches
        data = [batchdata(:,:,batch)]; %%% batch index에 해당하는 배치 데이터 셋을 모셔옴.
        targets = [batchtargets(:,:,batch)];
        data = [data ones(N,1)];       %%% 마지막 데이터는 1로 bias를 위한 값 append. : 100 x 785
        w1probs = 1./(1 + exp(-data*w1)); w1probs = [w1probs  ones(N,1)]; %// 100 x 785 -> 100 x 100 -> 100 x 101
        w2probs = 1./(1 + exp(-w1probs*w2)); w2probs = [w2probs ones(N,1)];
        w3probs = 1./(1 + exp(-w2probs*w3)); w3probs = [w3probs  ones(N,1)];
        w4probs = w3probs*w4; w4probs = [w4probs  ones(N,1)];
    end
    
    %train_err(epoch)=err/numbatches;
    % 한번 돌면 train_err[epoch]에 미니배치 리컨스트럭션 데이터 에러의 평균을 더함
 
    %%%%%%%%%%%%%%% PERFORM CONJUGATE GRADIENT WITH 3 LINESEARCHES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    max_iter=3;
    VV = [w1(:)' w2(:)' w3(:)' w4(:)']';
    Dim = [l1; l2; l3; l4; l5];
    data = data(:,1:l1);

    [X, fX] = minimize(VV,'CG_MNIST_NCA',max_iter,Dim,data, targets);
    
    w1 = reshape(X(1:(l1+1)*l2),l1+1,l2);
    xxx = (l1+1)*l2;
    w2 = reshape(X(xxx+1:xxx+(l2+1)*l3),l2+1,l3);
    xxx = xxx+(l2+1)*l3;
    w3 = reshape(X(xxx+1:xxx+(l3+1)*l4),l3+1,l4);
    xxx = xxx+(l3+1)*l4;
    w4 = reshape(X(xxx+1:xxx+(l4+1)*l5),l4+1,l5);
       
%%%%%%%%%%%%%%% END OF CONJUGATE GRADIENT WITH 3 LINESEARCHES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 save mnist_weights_nca w1 w2 w3 w4

end

