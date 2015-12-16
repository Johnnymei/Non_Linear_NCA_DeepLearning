% Version 1.000
%
% Code provided by Ruslan Salakhutdinov and Geoff Hinton
%
% Permission is granted for anyone to copy, use, modify, or distribute this
% program and accompanying programs and documents for any purpose, provided
% this copyright notice is retained and prominently displayed, along with
% a note saying that the original programs are available from our
% web page.
% The programs and documents are distributed without any warranty, express or
% implied.  As the programs were written for research purposes only, they have
% not been tested to the degree that would be advisable in any important
% application.  All use of these programs is entirely at the user's own risk.

% This program fine-tunes an autoencoder with backpropagation.
% Weights of the autoencoder are going to be saved in mnist_weights.mat
% and trainig and test reconstruction errors in mnist_error.mat
% You can also set maxepoch, default value is 200 as in our paper.  

maxepoch=200;
fprintf(1,'\nFine-tuning deep autoencoder by minimizing cross entropy error. \n');
fprintf(1,'60 batches of 1000 cases each. \n');

load mnistvh
load mnisthp
load mnisthp2
load mnistpo 

makebatches;
[numcases numdims numbatches]=size(batchdata); %100; 784; 600
N=numcases; 

%%%% PREINITIALIZE WEIGHTS OF THE AUTOENCODER
%%%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% vis-hid에 biases append하는 것으로 보임
w1=[vishid; hidrecbiases];
w2=[hidpen; penrecbiases];
w3=[hidpen2; penrecbiases2];
w4=[hidtop; toprecbiases];
w5=[hidtop'; topgenbiases]; 
w6=[hidpen2'; hidgenbiases2]; 
w7=[hidpen'; hidgenbiases]; 
w8=[vishid'; visbiases];

%%%%%%%%%% END OF PREINITIALIZATIO OF WEIGHTS  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%% 이건 각 레이어의 노드 사이즈로 보임
l1=size(w1,1)-1;
l2=size(w2,1)-1;
l3=size(w3,1)-1;
l4=size(w4,1)-1;
l5=size(w5,1)-1;
l6=size(w6,1)-1;
l7=size(w7,1)-1;
l8=size(w8,1)-1;
l9=l1; 
test_err=[];
train_err=[];

%에포크 시작
for epoch = 1:maxepoch

%%%%%%%%%%%%%%%%%%%% COMPUTE TRAINING RECONSTRUCTION ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
err=0; 
[numcases numdims numbatches]=size(batchdata); %%100 * 784 * 600
N=numcases;
 for batch = 1:numbatches
  data = [batchdata(:,:,batch)]; %%% batch index에 해당하는 배치 데이터 셋을 모셔옴.
  data = [data ones(N,1)];       %%% 마지막 데이터는 1로 bias를 위한 값 append. : 100 x 785
  w1probs = 1./(1 + exp(-data*w1)); w1probs = [w1probs  ones(N,1)]; %// 100 x 785 -> 100 x 100 -> 100 x 101
  w2probs = 1./(1 + exp(-w1probs*w2)); w2probs = [w2probs ones(N,1)];
  w3probs = 1./(1 + exp(-w2probs*w3)); w3probs = [w3probs  ones(N,1)];
  w4probs = w3probs*w4; w4probs = [w4probs  ones(N,1)];
  w5probs = 1./(1 + exp(-w4probs*w5)); w5probs = [w5probs  ones(N,1)];
  w6probs = 1./(1 + exp(-w5probs*w6)); w6probs = [w6probs  ones(N,1)];
  w7probs = 1./(1 + exp(-w6probs*w7)); w7probs = [w7probs  ones(N,1)];
  dataout = 1./(1 + exp(-w7probs*w8)); %데이터 아웃: 역으로 만들어냄
  err= err +  1/N*sum(sum( (data(:,1:end-1)-dataout).^2 )); 
  end
 train_err(epoch)=err/numbatches;
% 한번 돌면 train_err[epoch]에 미니배치 리컨스트럭션 데이터 에러의 평균을 더함
 
%%%%%%%%%%%%%% END OF COMPUTING TRAINING RECONSTRUCTION ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%% DISPLAY FIGURE TOP ROW REAL DATA BOTTOM ROW RECONSTRUCTIONS %%%%%%%%%%%%%%%%%%%%%%%%%
%fprintf(1,'Displaying in figure 1: Top row - real data, Bottom row -- reconstructions \n');
%output=[];
% for ii=1:15
%  output = [output data(ii,1:end-1)' dataout(ii,:)'];
% end
%   if epoch==1 
%   close all 
%   figure('Position',[100,600,1000,200]);
%   else 
%   figure(1)
%   end 
%   mnistdisp(output);
%   drawnow;

%%%%%%%%%%%%%%%%%%%% COMPUTE TEST RECONSTRUCTION ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[testnumcases testnumdims testnumbatches]=size(testbatchdata);
N=testnumcases;
err=0;
for batch = 1:testnumbatches
  data = [testbatchdata(:,:,batch)];
  data = [data ones(N,1)];
  w1probs = 1./(1 + exp(-data*w1)); w1probs = [w1probs  ones(N,1)];
  w2probs = 1./(1 + exp(-w1probs*w2)); w2probs = [w2probs ones(N,1)];
  w3probs = 1./(1 + exp(-w2probs*w3)); w3probs = [w3probs  ones(N,1)];
  w4probs = w3probs*w4; w4probs = [w4probs  ones(N,1)];
  w5probs = 1./(1 + exp(-w4probs*w5)); w5probs = [w5probs  ones(N,1)];
  w6probs = 1./(1 + exp(-w5probs*w6)); w6probs = [w6probs  ones(N,1)];
  w7probs = 1./(1 + exp(-w6probs*w7)); w7probs = [w7probs  ones(N,1)];
  dataout = 1./(1 + exp(-w7probs*w8));
  err = err +  1/N*sum(sum( (data(:,1:end-1)-dataout).^2 ));
  end
 test_err(epoch)=err/testnumbatches;
 fprintf(1,'Before epoch %d Train squared error: %6.3f Test squared error: %6.3f \t \t \n',epoch,train_err(epoch),test_err(epoch));

%%%%%%%%%%%%%% END OF COMPUTING TEST RECONSTRUCTION ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 tt=0;
 for batch = 1:numbatches/10
 fprintf(1,'epoch %d batch %d\r',epoch,batch);

%%%%%%%%%%% COMBINE 10 MINIBATCHES INTO 1 LARGER MINIBATCH %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 tt=tt+1; 
 data=[];
 for kk=1:10
  data=[data 
        batchdata(:,:,(tt-1)*10+kk)]; 
 end 

%%%%%%%%%%%%%%% PERFORM CONJUGATE GRADIENT WITH 3 LINESEARCHES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  max_iter=3;
  VV = [w1(:)' w2(:)' w3(:)' w4(:)' w5(:)' w6(:)' w7(:)' w8(:)']';
  Dim = [l1; l2; l3; l4; l5; l6; l7; l8; l9];

  [X, fX] = minimize(VV,'CG_MNIST',max_iter,Dim,data);

  w1 = reshape(X(1:(l1+1)*l2),l1+1,l2);
  xxx = (l1+1)*l2;
  w2 = reshape(X(xxx+1:xxx+(l2+1)*l3),l2+1,l3);
  xxx = xxx+(l2+1)*l3;
  w3 = reshape(X(xxx+1:xxx+(l3+1)*l4),l3+1,l4);
  xxx = xxx+(l3+1)*l4;
  w4 = reshape(X(xxx+1:xxx+(l4+1)*l5),l4+1,l5);
  xxx = xxx+(l4+1)*l5;
  w5 = reshape(X(xxx+1:xxx+(l5+1)*l6),l5+1,l6);
  xxx = xxx+(l5+1)*l6;
  w6 = reshape(X(xxx+1:xxx+(l6+1)*l7),l6+1,l7);
  xxx = xxx+(l6+1)*l7;
  w7 = reshape(X(xxx+1:xxx+(l7+1)*l8),l7+1,l8);
  xxx = xxx+(l7+1)*l8;
  w8 = reshape(X(xxx+1:xxx+(l8+1)*l9),l8+1,l9);

%%%%%%%%%%%%%%% END OF CONJUGATE GRADIENT WITH 3 LINESEARCHES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 end

 save mnist_weights w1 w2 w3 w4 w5 w6 w7 w8 
 save mnist_error test_err train_err;

end



