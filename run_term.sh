# DEEP LEARNING PRACTICE #2.
# Uncomment and execute a line you want to execute.

# 1. Initial baseline.
CUDA_VISIBLE_DEVICES=0 th main.lua -task hyperface -net lnet; # -batchSize 128 -epochSize 10 -numEpoch 10; # -poolType max -learnRate 1e-2,1e-2 -batchSize 64 -epochSize 10 -numEpoch 1;

# 2. Average pooling
#CUDA_VISIBLE_DEVICES=0 th main.lua -task wsol -poolType average; # -learnRate 1e-2,1e-2 -batchSize 10 -epochSize 1 -numEpoch 1; # -batchSize 64 -epochSize 10 -numEpoch 10;; # -eval map -keepAspect 1 -batchSize 64 -epochSize 25 -numEpoch 50;

# 3. Siamese single
#CUDA_VISIBLE_DEVICES=0 th main.lua -task wsol -net siamese; # -learnRate 1e-2,1e-2  -batchSize 10 -epochSize 1 -numEpoch 1; #; # -keepAspect 1 -batchSize 64 -epochSize 25 -numEpoch 50;

# 3. Siamese multi
#CUDA_VISIBLE_DEVICES=0 th main.lua -task wsol -net siamese -multiScale 1; # -learnRate 1e-2,1e-2 -batchSize 48 -epochSize 10 -numEpoch 1; #; #-keepAspect 1 -batchSize 64 -epochSize 25 -numEpoch 50;


