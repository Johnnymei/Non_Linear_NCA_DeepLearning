## Converter

line-by-line code tracing

1. Make sure you downloaded MNIST materials.

  ```MATLAB
  fprintf(1,'You first need to download files:\n train-images-idx3-ubyte.gz\n train-labels-idx1-ubyte.gz\n t10k-images-idx3-ubyte.gz\n t10k-labels-idx1-ubyte.gz\n from http://yann.lecun.com/exdb/mnist/\n and gunzip them \n');
  ```

2. Open file that contains digital images/labels for test.

  ```MATLAB
  f = fopen('t10k-images-idx3-ubyte.idx3-ubyte','r');
  [a,count] = fread(f,4,'int32');

  g = fopen('t10k-labels-idx1-ubyte.idx1-ubyte','r');
  [l,count] = fread(g,2,'int32');
  ```

3. set n = 1000

  ```MATLAB
  fprintf(1,'Starting to convert Test MNIST images (prints 10 dots) \n');
  n = 1000;
  ```

4. Create test files by chunking 1000 test tuple. Df(d+1) is a file descriptor related to 'testd.ascii'

  ```MATLAB
  Df = cell(1,10);
  for d=0:9,
    Df{d+1} = fopen(['test' num2str(d) '.ascii'],'w');
  end;
  ```

5. Make test files for each label (test0.ascii will contain test data with label 0 after this phase)

  * Read 1000 test data tuples
    * rawimages <- read 1000 images( (28*28) * 1000=n ) from from f
    * rawimages <- reshape it into 2-dimensional matrix (784 * 1000)
    * rawlabels <- read 1000 labels

  * Write each test data tuple into the file to which it's label is related
    * Read jth column(i.e. jth image entry)
    * Write it into the file to which DF{rawlables(j)+1} refers

  ```MATLAB
  for i=1:10,
    fprintf('.');
    rawimages = fread(f,28*28*n,'uchar');
    rawlabels = fread(g,n,'uchar');
    rawimages = reshape(rawimages,28*28,n);

    for j=1:n,
      fprintf(Df{rawlabels(j)+1},'%3d ',rawimages(:,j));
      fprintf(Df{rawlabels(j)+1},'\n');
    end;
  end;
  ```

6. Close each testd.ascii file for each d=0:9, and save it as .mat file by loading and rewriting it

  ```MATLAB
  fprintf(1,'\n');
  for d=0:9,
    fclose(Df{d+1});
    D = load(['test' num2str(d) '.ascii'],'-ascii');
    fprintf('%5d Digits of class %d\n',size(D,1),d);
    save(['test' num2str(d) '.mat'],'D','-mat');
  end;
  ```

7. In the same way as above, create training files

  ```MATLAB
  % Work with trainig files second  
  f = fopen('train-images-idx3-ubyte.idx3-ubyte','r');
  [a,count] = fread(f,4,'int32');

  g = fopen('train-labels-idx1-ubyte.idx1-ubyte','r');
  [l,count] = fread(g,2,'int32');

  fprintf(1,'Starting to convert Training MNIST images (prints 60 dots)\n');
  n = 1000;

  Df = cell(1,10);
  for d=0:9,
    Df{d+1} = fopen(['digit' num2str(d) '.ascii'],'w');
  end;

  for i=1:60,
    fprintf('.');
    rawimages = fread(f,28*28*n,'uchar');
    rawlabels = fread(g,n,'uchar');
    rawimages = reshape(rawimages,28*28,n);

    for j=1:n,
      fprintf(Df{rawlabels(j)+1},'%3d ',rawimages(:,j));
      fprintf(Df{rawlabels(j)+1},'\n');
    end;
  end;

  fprintf(1,'\n');
  for d=0:9,
    fclose(Df{d+1});
    D = load(['digit' num2str(d) '.ascii'],'-ascii');
    fprintf('%5d Digits of class %d\n',size(D,1),d);
    save(['digit' num2str(d) '.mat'],'D','-mat');
  end;
  ```

8. Remove .ascii files

  ```MATLAB
  dos('rm *.ascii');
  ```
