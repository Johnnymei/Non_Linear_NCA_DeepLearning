for training_batch = 1:numbatches
    make_data = [batchdata(:,:,training_batch)];
    make_data = [make_data ones(N,1)];
    w1probs    = 1./(1 + exp(-make_data*w1)); w1probs = [w1probs  ones(N,1)];
    w2probs    = 1./(1 + exp(-w1probs*w2)); w2probs = [w2probs ones(N,1)];
    w3probs    = 1./(1 + exp(-w2probs*w3)); w3probs = [w3probs  ones(N,1)];
    f_x_W = 1./(1 + exp(-w3probs*w4)); %TODO
    f_x_array = [f_x_array; f_x_W];
    target_array = [target_array; batchtargets(:,:,training_batch)];
end