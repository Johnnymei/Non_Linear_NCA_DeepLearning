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


