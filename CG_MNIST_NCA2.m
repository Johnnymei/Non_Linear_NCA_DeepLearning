function [ f, df ] = CG_MNIST_NCA2(VV,Dim,XX,TT, alpha)

[nca_f, nca_df] = CG_MNIST_NCA(VV,Dim,XX,TT);
[rnca_f, rnca_df] = CG_MNIST_reverse_NCA(VV,Dim,XX,TT);

f= alpha * nca_f + (1-alpha) * rnca_f;
df = alpha * nca_df + (1-alpha) * rnca_df;

end

