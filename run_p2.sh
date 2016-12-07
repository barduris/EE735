# DEEP LEARNING PRACTICE #2.
# Uncomment and execute a line you want to execute.

# 1. Initial baseline.
CUDA_VISIBLE_DEVICES=0 th main.lua -task mlcls -learnRate 1e-5,1e-3 -epochSize 20 -keepAspect 1 -batchSize 64 -numEpoch 20;

# 2. Learning from scratch
#CUDA_VISIBLE_DEVICES=0 th main.lua -task mlcls -net alexNetScratch -learnRate 1e-2,1e-2 -epochSize 20 -keepAspect 1 -batchSize 64 -numEpoch 20;

# 3. Entropy-based loss
#CUDA_VISIBLE_DEVICES=0 th main.lua -task mlcls -loss entropy -learnRate 1e-5,1e-2 -epochSize 20 -keepAspect 1 -batchSize 64 -numEpoch 20;

# 4. Mean average precision



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
