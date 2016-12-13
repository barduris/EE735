# DEEP LEARNING PRACTICE #2.
# Uncomment and execute a line you want to execute.

# 1. Initial baseline.
CUDA_VISIBLE_DEVICES=0 th main.lua -task wsol -poolType max -learnRate 1e-2,1e-2 -batchSize 64 -epochSize 10 -numEpoch 1;

# 2. Average pooling
#CUDA_VISIBLE_DEVICES=0 th main.lua -task wsol -poolType average -learnRate 1e-2,1e-2 -batchSize 64 -epochSize 10 -numEpoch 1; # -batchSize 64 -epochSize 10 -numEpoch 10;; # -eval map -keepAspect 1 -batchSize 64 -epochSize 25 -numEpoch 50;

# 3. Siamese single
#CUDA_VISIBLE_DEVICES=0 th main.lua -task wsol -net siamese -learnRate 1e-2,1e-2  -batchSize 64 -epochSize 10 -numEpoch 1; #; # -keepAspect 1 -batchSize 64 -epochSize 25 -numEpoch 50;

# 3. Siamese mulit
#CUDA_VISIBLE_DEVICES=0 th main.lua -task wsol -net siamese -multiScale 1 -learnRate 1e-2,1e-2 -batchSize 64 -epochSize 10 -numEpoch 1; #; #-keepAspect 1 -batchSize 64 -epochSize 25 -numEpoch 50;




# 2. Learning from scratch
#CUDA_VISIBLE_DEVICES=0 th main.lua -task mlcls -net alexNetScratch -learnRate 1e-3,1e-3 -epochSize 20 -keepAspect 1 -batchSize 64 -numEpoch 20;

# 3. Entropy-based loss
#CUDA_VISIBLE_DEVICES=0 th main.lua -task mlcls -loss entropy -learnRate 1e-5,1e-3 -epochSize 20 -keepAspect 1 -batchSize 64 -numEpoch 20;

# 4. Mean average precision
#-eval map
#CUDA_VISIBLE_DEVICES=0 th main.lua -task mlcls -eval map -learnRate 1e-5,1e-3 -epochSize 20 -keepAspect 1 -batchSize 64 -numEpoch 20;


# 2. Overfitting.
#CUDA_VISIBLE_DEVICES=0 th main.lua -task slcls -weightDecay 0;
#CUDA_VISIBLE_DEVICES=0 th main.lua -task slcls -dropout 0;
#CUDA_VISIBLE_DEVICES=0 th main.lua -task slcls -augment 0;
#CUDA_VISIBLE_DEVICES=0 th main.lua -task slcls -net cifarNetLarge;

# 3. Loss function.
#CUDA_VISIBLE_DEVICES=0 th main.lua -task slcls -loss hinge;
#CUDA_VISIBLE_DEVICES=0 th main.lua -task slcls -loss l2 -learnRate 1e-4,1e-4;

# 4. Convergence speed.
#CUDA_VISIBLE_DEVICES=0 th main.lua -task slcls -learnRate 1e-3,1e-3;
#CUDA_VISIBLE_DEVICES=0 th main.lua -task slcls -net cifarNetBatchNorm -learnRate 1e-1,1e-1;
