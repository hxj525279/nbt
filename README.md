# Nyström Basis Transfer
Matlab Source Code for the ICLR 2020 Paper submission "Domain Adaptation via Low-Rank Basis Approximation". 

Folders are self-explaining. 

## Demo and Reproducing:
For a demo and reproducing of performance/time results start
_demo.m_

## Main file:
_pd_cl_nbt.m_ (Submission Algorithm)

## Secondardy Files:
_pd_ny_svd.m_<br/>
_libsvm (folder)_<br/>
_augmentation.m_
 
## Tutorial:
```matlab 
[model,K,m,n] = pd_cl_nbt(Xs',Xt',Ys,options);
[label, acc,scores] = svmpredict(full(Yt), [(1:n)', K(m+1:end, 1:m)], model);
fprintf('\n PD_NBT %.2f%% \n', acc(1));
```
Assume traning data Xs with size d x m and test data Xt with size d x n. Label vector Ys and Yt accordingly. 
pd_cl_nbt accecpts the data and an options struct. With this struct the user can specify:
```
NBT Specific:
options.landmarks ~ Number of Landmarks (int)
SVM Specific: 
options.gamma ~ Gamma of SVM (int)
options.smvc ~ C Parameter of SVM (int)
options.ker ~ Kernel Type "linear|rbf|lab" (string)
```
The functions outputs a libsvm model and a kernel over training and test data modified. The training data is modified by NBT algorithm. <br/>
## Reproducing Plots:
Figure 1: Sensitivity of landmark-parameter: _landmarkperformance_plot.m_<br/>
Figure 2: Process of NBT: _plot_process.m_

## Additional Results: 
Additional results with another domain adaptation sampling strategy can be obtained with demo_gong.m 


## Abstract of Submission:
Domain adaptation focuses on the reuse of supervised learning models in a new context. Prominent applications can be found in robotics, image processing or web mining. In these areas, learning scenarios change by nature, but often remain related and motivate the reuse of existing supervised models.<br/>
While the majority of symmetric and asymmetric domain adaptation algorithms utilize all available source and target domain data, we show that domain adaptation requires only a substantial smaller subset from both domains. This makes it more suitable for real-world scenarios where target domain data is rare and provides a sparse solution. The presented approach finds a target subspace representation for source and target data to address domain differences by orthogonal basis transfer. We employ Nyström techniques and show the reliability of this approximation without a particular landmark matrix by applying post-transfer normalization.<br/>
It is evaluated on typical domain adaptation tasks with standard benchmark data.
