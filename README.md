#Learning a Nonlinear Embedding by preserving Class Neighbourhood Structure

##1. Slide
<iframe src='https://onedrive.live.com/embed?cid=7230FE4126F9D3CC&resid=7230FE4126F9D3CC%212098&authkey=APYihjB4VYTEWDo&em=2&wdAr=1.7777777777777777' width='962px' height='565px' frameborder='0'>포함된 <a target='_blank' href='http://office.com'>Microsoft Office</a> 프레젠테이션, 제공: <a target='_blank' href='http://office.com/webapps'>Office Online</a></iframe>

##2. Source Code
- base로 사용한 코드: [Code provided by Ruslan Salakhutdinov and Geoff Hinton ](http://www.cs.toronto.edu/~hinton/MatlabForSciencePaper.html)
```
Permission is granted for anyone to copy, use, modify, or distribute this program and accompanying programs and documents for any purpose, provided this copyright notice is retained and prominently displayed, along with a note saying that the original programs are available from our web page. The programs and documents are distributed without any warranty, express or implied. As the programs were written for research purposes only, they have not been tested to the degree that would be advisable in any important application. All use of these programs is entirely at the user's own risk.
```
 
- NLNCA(Nonlinear NCA)를 위해 base 코드를 수정: [GitHub Link](https://github.com/ws-choi/Lon_Linear_NCA_DeepLearning)
	- [Download zip](https://github.com/ws-choi/Lon_Linear_NCA_DeepLearning/archive/master.zip)
	- Git Clone: https://github.com/ws-choi/Lon_Linear_NCA_DeepLearning.git
	- 실행 순서
		- [download MNIST](http://yann.lecun.com/exdb/mnist/)
		- download source code above
		- exceute mnist_nlnca.m
		- ++test mnist_nlnca.m++ (not implemented yet)


##3. Source Code에 대한 설명

| File | 역할 |
|--------:|--------|
|[mnist_nlnca.m](https://github.com/ws-choi/Lon_Linear_NCA_DeepLearning/blob/master/mnist_nlnca.m)| Epoch 수 및 DNN Structure 정의. minst를 위한 nlnca를 training 시키기 위해 Converter, nca_pretraining, backpro_nca를 순서대로 호출함|
|[converter.m](https://github.com/ws-choi/Lon_Linear_NCA_DeepLearning/blob/master/converter.m)| idx1-ubyte, idx3-ubyte 형식의 mnist 데이터셋을 텍스트 형식으로 Converting함|
|[nca_pretraining.m](https://github.com/ws-choi/Lon_Linear_NCA_DeepLearning/blob/master/nca_pretraining.m)| nlnca를 위한 DNN을 Pretraining하여 각 레이어의 weight들을 mnistvh_nca, mnisthp_nca, mnisthp2_nca, nistpo_nca라는 이름의 파일에 저장함|
|[backprop_nca](https://github.com/ws-choi/Lon_Linear_NCA_DeepLearning/blob/master/backprop_nca.m)|nlnca를 위한 finetuning 작업을 수행함. mnistvh_nca, mnisthp_nca, mnisthp2_nca, nistpo_nca를 읽어들여서 DNN을 로딩함. makebatchs_nca를 사용해서 미니배치 데이터 셋을 만듦. epoch가 maxepoch에 도달할때까지 backpropagation하며, 이때 minimize.m 를 호출하여 웨이트 수정값을 구함. |
|[minimize.m](https://github.com/ws-choi/Lon_Linear_NCA_DeepLearning/blob/master/minimize.m)|function [X, fX, i] = minimize(X, f, length, varargin)을 정의함. 이 함수는 line search가 length 이하로 수행되는 Conjugate-Gradient Method를 이용해서 파라미터가 VV일때 f라는 목적함수를 최소화 시키는 방향으로 파라미터를 조정함. 이후 조정된 파라미터인 X를 반환함. 본 구현 버전에서 f는 'CG_MNIST_NCA'라는 함수임. Conjugate-Gradient-for-MNIST_NLNCA의 약자임.|
|[CG_MNIST_NCA.m](https://github.com/ws-choi/Lon_Linear_NCA_DeepLearning/blob/master/CG_MNIST_NCA.m)|ONCA값 및 ONCA의 derivative를 반환하는 CG_MNIST_NCA라는 함수를 정의함|

### 주요 부분인 CG_MNIST_NCA.m
- ONCA값 및 ONCA의 derivative를 반환하는 CG_MNIST_NCA라는 함수를 정의함
- O_NCA를 최대화하는 것이 목적이지만, 제공된 minimize.m은 목적함수 최소화코드이기 때문에 -1 * O_NCA를 최소화하게끔 구현했음
	- 즉, 엄밀히 말하면 이 함수는 -1 * ONCA 및 -1 * ONCA의 derivative를 반환함

```MATLAB
% Original Copyright:
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

% ws_choi@korea.ac.kr at KOREA UNIV.

function [f, df] = CG_MNIST_NCA(VV,Dim,XX,TT);

l1 = Dim(1);
l2 = Dim(2);
l3 = Dim(3);
l4= Dim(4);
l5= Dim(5);
N = size(XX,1);

% Do decomversion.
w1 = reshape(VV(1:(l1+1)*l2),l1+1,l2);
xxx = (l1+1)*l2;
w2 = reshape(VV(xxx+1:xxx+(l2+1)*l3),l2+1,l3);
xxx = xxx+(l2+1)*l3;
w3 = reshape(VV(xxx+1:xxx+(l3+1)*l4),l3+1,l4);
xxx = xxx+(l3+1)*l4;
w4 = reshape(VV(xxx+1:xxx+(l4+1)*l5),l4+1,l5);


XX = [XX ones(N,1)];
%N=25;
%XX = XX(1:N, :); %TODO
%TT = TT(1:N, :); %TODO: remove;
w1probs    = 1./(1 + exp(-XX*w1)); w1probs = [w1probs  ones(N,1)];
w2probs    = 1./(1 + exp(-w1probs*w2)); w2probs = [w2probs ones(N,1)];
w3probs    = 1./(1 + exp(-w2probs*w3)); w3probs = [w3probs  ones(N,1)];
f_x_W = 1./(1 + exp(-w3probs*w4)); %TODO

dab     = zeros(l5, N, N);
dab_2   = zeros(N, N);
eab     = zeros(N, N);
pab     = zeros(N, N);


% precomputation 

for i=1:N
    for j=i+1:N
        dab(:, i,j) = f_x_W(i,:) -f_x_W(j,:);%
        dab(:, j,i) = dab(:, i,j);
        dab_2(i,j)  = dab(:, i,j)' * dab(:, i,j);
        dab_2(j,i)  = dab_2(i,j);
        eab(i,j)    = exp (-dab_2(i,j));
        eab(j,i)    = eab(i,j);
    end
end 

psum = sum(eab')';

for i=1:N
    for j=i+1:N
        pab(i,j) = eab(i,j)/psum(i);
        pab(j,i) = eab(j,i)/psum(j);
    end
end 


pab_dab = zeros(l5, N, N);

for i=1:N
    for j=1:N
        pab_dab(:, i,j) = pab(i,j)*dab(:, i,j);
    end
end

pab_sig_pazdaz = zeros(N, 30);

for a=1:N
    
    %i = a, for each a
    
    %%%%% SUM(pab[SUMpazdaz]) %%%%%%%%% %%%%% SUM(pab[SUMpazdaz]) %%%%%%%%%
    a_pab_sum = 0;
    
    for b=1:N
        if TT(a,:) * TT(b,:)' > 0
            a_pab_sum = a_pab_sum + pab(a,b);
        end
    end
    
    pazdaz = 0;
    for z=1:N
        if a == z
            continue
        else
            pazdaz = pazdaz + pab_dab(:,a,z);
        end
    end
    
    a_pab_pazdaz = a_pab_sum .* pazdaz;
    a_pab_pazdaz = 2 .* a_pab_pazdaz;
    a_pab_pazdaz = a_pab_pazdaz';   
    %%%%% SUM(pab[SUMpazdaz]) %%%%%%%%% %%%%% SUM(pab[SUMpazdaz]) %%%%%%%%%
    
        
    %%%%% SUM[SUMplq]pladla]  %%%%%%%%% %%%%% SUM[SUMplq]pladla]  %%%%%%%%%
    plqpladla = zeros(1,l5);
    
    for L=1:N
        if a == L
            continue
        else
            
            sum_pLq = 0;
            for q=1:N
                if TT(L,:) * TT(q,:)' > 0
                    sum_pLq = sum_pLq + pab(L,q);
                end
            end
            plqpladla = plqpladla + (sum_pLq * pab(L,a) * dab(:,L,a))';
        end
            
    end
    
    plqpladla = -2 .* plqpladla;
    %%%%% SUM[SUMplq]pladla]  %%%%%%%%% %%%%% SUM[SUMplq]pladla]  %%%%%%%%%
    
        
    
    pab_sig_pazdaz(a,:) = a_pab_pazdaz + plqpladla;

end 

%%%%% calculate O_NCA  %%%%%%%%%
f=0;
for a=1:N   
    for b=1:N
        if TT(a,:) * TT(b,:)' > 0
            f = f - pab(a,b);
        end
    end
end

fprintf(1,'%f\n',f);

IO = pab_sig_pazdaz;

Ix4= -1 .* IO; 

dw4 = w3probs'*Ix4;

Ix3 = (Ix4*w4').*w3probs.*(1-w3probs); 
Ix3 = Ix3(:,1:end-1);
dw3 =  w2probs'*Ix3;

Ix2 = (Ix3*w3').*w2probs.*(1-w2probs); 
Ix2 = Ix2(:,1:end-1);
dw2 =  w1probs'*Ix2;

Ix1 = (Ix2*w2').*w1probs.*(1-w1probs); 
Ix1 = Ix1(:,1:end-1);
dw1 =  XX'*Ix1;

df = [dw1(:)' dw2(:)' dw3(:)' dw4(:)' ]'; 



```
##4. Experiment
![NLNCA.png](img/NLNCA.png)

- 실험이 끝난 것은 아니나(수일이상 학습시켜야 할 것으로 보임) 의도되로 학습되고 있음
	- 매 backpropagation 실행마다 목적함수인 O_NCA의 값을 출력해보니 ONCA의 값이 일관되게 증가하는 추세임
		- 단, minibatch의 수는 논문에서는 5000값을 사용했으나
		- 메모리 소모가 지나치게 커서 디버깅이 곤란하여 현재는 500으로 설정했음

    - 그래프의 y축은 -1 * O_NCA임. O_NCA를 최대화해야하지만 
    	- 제공된 minimize.m은 목적함수 최소화코드이기 때문에 -1 * O_NCA를 최소화하게끔 구현했음
    	- O_NCA의 값은 증가하고 있는 추세임

- minibatch size = 500 이라면
	- 목적함수인 ONCA가 500이어야 가장 이상적이다.
		- 이유: a 번째 데이터가 잘 분류될 확률은 1인 것이 가장 이상적이다. 


$$

\sum_{b:c^a = c^b} p_{ab} \leq 1

$$

- 그런데, 

$$

O_{NCA}= \sum_{n=1}^{500}{\sum_{b:c^a = c^b} p_{ab}} \leq 500

$$

- 따라서 $$$O_{NCA}$$$ 는 500인 것이 이상적.
	- finetuning 초기에는 $$$O_{NCA}$$$가 206에 불과했으나 CG_MNIST_NCA가 500번 호출된 시점에서는 230으로 올랐음 (이틀째 수행중인 시점)

## 5. 결론 및 제언

- 실험
	- 본 논문에서 제시한 derivative로 fine-tuning해본 결과, 학습이 의도대로 되고 있음.
	- 학습이 완료되면 평가실험을 해볼 계획임
- 개선 가능성
	- 슬라이드(#1)에서는 informal한 방식으로 목적함수의 성질을 다음과 같이 기술했었음
		1. a와 b의 클래스가 같다면 embedding 공간에서 가까워야 한다.
		2. a와 b의 클래스가 다르다면 embedding 공간에서 멀리 떨어지게 해야한다.
    - 이러한 방식으로 목적 함수를 기술하는 방법은 LSH[2,3]나 DSH[4]에서도 찾아볼 수 있음[#7 참고슬라이드]
    	- 납득갈만한 목적함수 설계 철학임!
		- LSH의 기본 원리는 1과 2를 모두 만족하는 함수 집합을 만드는 것임
		- DSH의 기본 원리는 1과 2를 만족시키는 함수 집합을 기계학습으로 만들어 냄. [#7 참고슬라이드]
    - 그러나 본 논문의 objective function에서는
    	1. 은 반영하고 있으나
    	2. 는 반영이 되지 않고 있음
    - 만일 1과 2를 동시에 고려한 O_NCA 목적함수를 개발한다면 어느정도의 성능 향상을 꾀할 수 있을 것으로 보임
    	- 그러나 아직 이와 비슷한 취지의 논문이 발표되었는지 Survey해보지는 않았음
    	- 계산량은 더욱 많아질 것으로 보임.


##6. 참조문헌
[1] Salakhutdinov, Ruslan, and Geoffrey E. Hinton. "Learning a nonlinear embedding by preserving class neighbourhood structure." International Conference on Artificial Intelligence and Statistics. 2007.
[2] Gionis, Aristides, Piotr Indyk, and Rajeev Motwani. "Similarity search in high dimensions via hashing." VLDB. Vol. 99. 1999.
[3] Datar, Mayur, et al. "Locality-sensitive hashing scheme based on p-stable distributions." Proceedings of the twentieth annual symposium on Computational geometry. ACM, 2004.
[4] Gao, Jinyang, et al. "DSH: Data sensitive hashing for high-dimensional k-nnsearch." Proceedings of the 2014 ACM SIGMOD international conference on Management of data. ACM, 2014.

##7. 참고슬라이드
Woosung Choi at el. - "DSH 발표자료", [Dataknow Lab. Seminar](http://wiki.dataknow.net/lab%20seminar), 2014
<iframe src='https://onedrive.live.com/embed?cid=7230FE4126F9D3CC&resid=7230FE4126F9D3CC%211081&authkey=AArqhkAOVJcDAKI&em=2&wdAr=1.7777777777777777' width='962px' height='565px' frameborder='0'>포함된 <a target='_blank' href='http://office.com'>Microsoft Office</a> 프레젠테이션, 제공: <a target='_blank' href='http://office.com/webapps'>Office Online</a></iframe>
