digitdata=[]; 
targets=[]; 
load digit0; digitdata = [digitdata; D]; targets = [targets; repmat([1 0 0 0 0 0 0 0 0 0], size(D,1), 1)];  
load digit1; digitdata = [digitdata; D]; targets = [targets; repmat([0 1 0 0 0 0 0 0 0 0], size(D,1), 1)];
load digit2; digitdata = [digitdata; D]; targets = [targets; repmat([0 0 1 0 0 0 0 0 0 0], size(D,1), 1)]; 
load digit3; digitdata = [digitdata; D]; targets = [targets; repmat([0 0 0 1 0 0 0 0 0 0], size(D,1), 1)];
load digit4; digitdata = [digitdata; D]; targets = [targets; repmat([0 0 0 0 1 0 0 0 0 0], size(D,1), 1)]; 
load digit5; digitdata = [digitdata; D]; targets = [targets; repmat([0 0 0 0 0 1 0 0 0 0], size(D,1), 1)];
load digit6; digitdata = [digitdata; D]; targets = [targets; repmat([0 0 0 0 0 0 1 0 0 0], size(D,1), 1)];
load digit7; digitdata = [digitdata; D]; targets = [targets; repmat([0 0 0 0 0 0 0 1 0 0], size(D,1), 1)];
load digit8; digitdata = [digitdata; D]; targets = [targets; repmat([0 0 0 0 0 0 0 0 1 0], size(D,1), 1)];
load digit9; digitdata = [digitdata; D]; targets = [targets; repmat([0 0 0 0 0 0 0 0 0 1], size(D,1), 1)];
digitdata = digitdata/255;

totnum=size(digitdata,1);
fprintf(1, 'Size of the training dataset= %5d \n', totnum);

rand('state',0); %so we know the permutation of the training data
randomorder=randperm(totnum);

numbatches=totnum/100;
numdims  =  size(digitdata,2);
batchsize = 100; %minibatch size가 6천 (TODO: 맞는지?)
batchdata = zeros(batchsize, numdims, numbatches);
batchtargets = zeros(batchsize, 10, numbatches);
%batchdata[5000 784 12]: 600*100 x 784를 분리한 것, 그걸 랜덤하게 아래에서 채우는 중
for b=1:numbatches
  batchdata(:,:,b) = digitdata(randomorder(1+(b-1)*batchsize:b*batchsize), :);
  batchtargets(:,:,b) = targets(randomorder(1+(b-1)*batchsize:b*batchsize), :);
end;

%batchdata(:,:,b) 는 b번째 미니배치 데이터 집합을 뜻함 
clear digitdata targets;

digitdata_test=[];
targets_test=[];
load test0; digitdata_test = [digitdata_test; D]; targets_test = [targets_test; repmat([1 0 0 0 0 0 0 0 0 0], size(D,1), 1)]; 
load test1; digitdata_test = [digitdata_test; D]; targets_test = [targets_test; repmat([0 1 0 0 0 0 0 0 0 0], size(D,1), 1)]; 
load test2; digitdata_test = [digitdata_test; D]; targets_test = [targets_test; repmat([0 0 1 0 0 0 0 0 0 0], size(D,1), 1)];
load test3; digitdata_test = [digitdata_test; D]; targets_test = [targets_test; repmat([0 0 0 1 0 0 0 0 0 0], size(D,1), 1)];
load test4; digitdata_test = [digitdata_test; D]; targets_test = [targets_test; repmat([0 0 0 0 1 0 0 0 0 0], size(D,1), 1)];
load test5; digitdata_test = [digitdata_test; D]; targets_test = [targets_test; repmat([0 0 0 0 0 1 0 0 0 0], size(D,1), 1)];
load test6; digitdata_test = [digitdata_test; D]; targets_test = [targets_test; repmat([0 0 0 0 0 0 1 0 0 0], size(D,1), 1)];
load test7; digitdata_test = [digitdata_test; D]; targets_test = [targets_test; repmat([0 0 0 0 0 0 0 1 0 0], size(D,1), 1)];
load test8; digitdata_test = [digitdata_test; D]; targets_test = [targets_test; repmat([0 0 0 0 0 0 0 0 1 0], size(D,1), 1)];
load test9; digitdata_test = [digitdata_test; D]; targets_test = [targets_test; repmat([0 0 0 0 0 0 0 0 0 1], size(D,1), 1)];
digitdata_test = digitdata_test/255;

totnum=size(digitdata_test,1);
fprintf(1, 'Size of the test dataset= %5d \n', totnum);

rand('state',0); %so we know the permutation of the training data
randomorder=randperm(totnum);

numbatches=totnum/100;
numdims  =  size(digitdata_test,2);
batchsize = 100;
testbatchdata = zeros(batchsize, numdims, numbatches);
testbatchtargets = zeros(batchsize, 10, numbatches);

for b=1:numbatches
  testbatchdata(:,:,b) = digitdata_test(randomorder(1+(b-1)*batchsize:b*batchsize), :);
  testbatchtargets(:,:,b) = targets_test(randomorder(1+(b-1)*batchsize:b*batchsize), :);
end;
%clear digitdata_test targets_test;


%%% Reset random seeds 
rand('state',sum(100*clock)); 
randn('state',sum(100*clock)); 



