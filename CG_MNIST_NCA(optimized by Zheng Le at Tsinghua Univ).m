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

function [f, df] = CG_MNIST_NCA_simple(VV,Dim,XX,TT)

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
w1probs    = 1./(1 + exp(-XX*w1)); w1probs = [w1probs  ones(N,1)];
w2probs    = 1./(1 + exp(-w1probs*w2)); w2probs = [w2probs ones(N,1)];
w3probs    = 1./(1 + exp(-w2probs*w3)); w3probs = [w3probs  ones(N,1)];
f_x_W = 1./(1 + exp(-w3probs*w4)); %TODO


dab_2 = squareform(pdist(f_x_W));
dab_2 = dab_2.^2;
eab = exp(-dab_2);
eab = eab-diag(diag(eab));

% how to optimize?
psum = sum(eab, 2)';
pab = zeros(N, N);
for i=1:N
    for j=i+1:N
        pab(i,j) = eab(i,j)/psum(i);
        pab(j,i) = eab(j,i)/psum(j);
    end
end

pabdab = repmat(sum(TT*TT'.*pab,2),1,ll).*f_x_W-TT*TT'.*pab*f_x_W;
a_pab_pazdaz = repmat((sum(TT*TT'.*pab,2)-diag(pab)),1,ll).*(repmat(sum(pab, 2),1,ll).*f_x_W-pab*f_x_W);
pladla = TT*TT'.*pab'*f_x_W-repmat(diag(TT*TT'*pab),1,ll).*f_x_W;
plqpladla = (repmat(diag(TT*TT'*pab')-diag(pab), 1, N).*pab)'*f_x_W-repmat(pab'*(diag(TT*TT'*pab')-diag(pab)), 1, ll).*f_x_W;
pab_sig_pazdaz = -2*pabdab + 2*a_pab_pazdaz + 2*pladla - 2*plqpladla;

f = -sum(sum(TT*TT'.*pab));

fprintf(1,'%f\n',f);

IO = pab_sig_pazdaz;

Ix4= -1.*IO; 

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


