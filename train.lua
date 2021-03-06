----------------------------------------------------
---- Large-scale deep learning framework -----------
---- This script trains a given network. -----------
---- This script is independent from any tasks. ----
---- Author: Donggeun Yoo, KAIST. ------------------
------------ dgyoo@rcv.kaist.ac.kr -----------------
----------------------------------------------------
require 'optim'
local train = {  }
train.inputs = torch.CudaTensor(  )
train.labels = torch.CudaTensor(  )
train.netTimer = torch.Timer(  )
train.dataTimer = torch.Timer(  )
function train.setOption( opt )
	assert( opt.batchSize > 0 )
	assert( opt.epochSize > 0 )
	assert( opt.pathModel:match( '(.+).t7$' ):len(  ) > 0 )
	assert( opt.pathOptim:match( '(.+).t7$' ):len(  ) > 0 )
	assert( opt.pathTrainLog:match( '(.+).log$' ):len(  ) > 0 )
	train.batchSize = opt.batchSize
	train.epochSize = opt.epochSize
	train.pathModel = opt.pathModel
	train.pathOptim = opt.pathOptim
	train.pathTrainLog = opt.pathTrainLog
	if opt.net == 'siamese' then
		train.inputs = {
			torch.CudaTensor(),
			torch.CudaTensor(),
			torch.CudaTensor()
		}
	end
end
function train.setModel( modelSet )
	assert( #modelSet.params == #modelSet.grads )
	assert( #modelSet.params == #modelSet.optims )
	for g = 1, #modelSet.params do 
		assert( modelSet.params[ g ]:numel(  ) == modelSet.grads[ g ]:numel(  ) ) 
	end
	train.model = modelSet.model
	train.criterion = modelSet.criterion
	train.params = modelSet.params
	train.grads = modelSet.grads
	train.optims = modelSet.optims
end
function train.setDonkey( donkeys )
	train.donkeys = donkeys
end
function train.setFunction( getBatch, evalBatch )
	train.getBatch = getBatch
	train.evalBatch = evalBatch
end
function train.train( epoch )
	-- Initialization.
	local trainLogger = io.open( train.pathTrainLog, 'a' )
	local epochTimer = torch.Timer(  )
	local getBatch = train.getBatch
	local trainBatch = train.trainBatch
	train.epoch = epoch
	train.evalEpoch = 0
	train.lossEpoch = 0
	train.batchNumber = 0
	-- Do the job.
	train.print( string.format( 'Train epoch %d.', epoch ) )
	cutorch.synchronize(  )
	train.model:training(  )
	for b = 1, train.epochSize do
		train.donkeys:addjob(
			function(  ) 
				return getBatch(  )
			end, -- Job callback.
			function( x, y ) 
				trainBatch( x, y ) 
			end -- End callback.
		)
	end
	train.donkeys:synchronize(  )
	cutorch.synchronize(  )
	train.evalEpoch = train.evalEpoch / train.epochSize
	train.lossEpoch = train.lossEpoch / train.epochSize
	train.print( string.format( 'Epoch %d, time %.2fs, avg loss %.4f, eval %.4f', 
		epoch, epochTimer:time(  ).real, train.lossEpoch, train.evalEpoch ) )
	trainLogger:write( string.format( '%03d %.4f %.4f\n', epoch, train.lossEpoch, train.evalEpoch ) )
	trainLogger:close(  )
	-- Save model.
	--train.print( 'Save model.' )
	--train.model:clearState()
	--saveDataParallel( train.pathModel:format( epoch ), train.model )
	--torch.save( train.pathOptim:format( epoch ), train.optims )
	train.print( 'Done.' )
	collectgarbage(  )
end
function train.trainBatch( inputsCpu, labelsCpu )
	--print(#inputsCpu)
	--print(type(inputsCpu) == 'table')
	-- Initialization.
	local dataTime = train.dataTimer:time(  ).real
	train.netTimer:reset(  )

	if (type(inputsCpu) == 'table') then
		for i = 1, #inputsCpu do
			train.inputs[i]:resize( inputsCpu[i]:size(  ) ):copy( inputsCpu[i] )
			--train.labels[i]:resize( labelsCpu[i]:size(  ) ):copy( labelsCpu[i] )
		end
	else
		train.inputs:resize( inputsCpu:size(  ) ):copy( inputsCpu )
		--train.labels:resize( labelsCpu:size(  ) ):copy( labelsCpu )
	end
	train.labels:resize( labelsCpu:size(  ) ):copy( labelsCpu )
	train.model:zeroGradParameters(  )
	cutorch.synchronize(  )
	---------------------
	-- FILL IN THE BLANK.
	-- See https://github.com/torch/nn/blob/master/doc/module.md
	-- 1. Feed-forward.
	-- 2. Compute loss and accumulate that to train.lossEpoch.
	-- 3. Backpropagation to get gradients.
	-- 4. Update the model with the gradients by using SGD.
	--    See https://github.com/torch/optim/blob/master/doc/algos.md
	--    Also, make it possible to apply different learning rates for different network modules.
	--    See the function task:groupParams() which will help your understanding.
	-- 5. Compute evaluation metric (e.g. top-1) and accumulate that to train.evalEpoch.
	--    You must call train.evalBatch().

	-- 1. Feed-forward
    local output = train.model:forward(train.inputs)
    
    -- 2. Estimate loss
	local err = train.criterion:forward(output, train.labels)
	train.lossEpoch = train.lossEpoch + err

	-- 3. Estimate gradients
	local outputGradients = train.criterion:backward(output, train.labels)
	train.model:backward(train.inputs, outputGradients)

	-- 4. SGD
	local layerParameters = {}
	local layerGradientParameters = {}

	for i = 1, train.model:size() do
		local layer = train.model:get(i)
		layerParameters[i], layerGradientParameters[i] = layer:getParameters()
	end
	for i = 1, #layerParameters do

		if layerParameters[i]:nDimension() > 0 then
			local feval = function(x)
				return _, layerGradientParameters[i]
			end

			optim.sgd(feval, layerParameters[i], train.optims[i])

		end
	end

	-- 5. Evaluate epoch
	local eval = train.evalBatch(output, train.labels)
	train.evalEpoch = train.evalEpoch + eval

	-- END BLANK.
	-------------
	if train.model.needsSync then train.model:syncParameters(  ) end
	cutorch.synchronize(  )
	train.batchNumber = train.batchNumber + 1
	local netTime = train.netTimer:time(  ).real
	local totalTime = dataTime + netTime
	local speed = train.batchSize / totalTime
	-- Print information
	train.print( string.format( 'Epoch %d, %d/%d, %dim/s (data %.2fs, net %.2fs), err %.2f, eval %.2f', 
		train.epoch, train.batchNumber, train.epochSize, speed, dataTime, netTime, err, eval ) )
	train.dataTimer:reset(  )
	collectgarbage(  )
end
function train.print( str )
	print( 'TRAIN) ' .. str )
end
return train
