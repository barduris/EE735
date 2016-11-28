-------------------------------------------------------------------------------
---- Large-scale deep learning framework --------------------------------------
---- This script handles task-specific jobs out of train.lua and val.lua. -----
---- You only need to define this script for a specific task. -----------------
---- Author: Donggeun Yoo, KAIST. ---------------------------------------------
------------ dgyoo@rcv.kaist.ac.kr --------------------------------------------
-------------------------------------------------------------------------------
local ffi = require 'ffi'
local task = torch.class( 'TaskManager' )
-----------------------------------------------------------------------------
---- Task-independent functions ---------------------------------------------
---- These functions are used in common regardless of any specific task. ----
-----------------------------------------------------------------------------
function task:__init(  )
	self.opt = {  }
	self.dbtr = {  }
	self.dbval = {  }
	self.inputStat = {  }
	self.numBatchTrain = 0
	self.numBatchVal = 0
end
function task:setOption( arg )
	self.opt = self:parseOption( arg )
	assert( self.opt.numGpu )
	assert( self.opt.backend )
	assert( self.opt.numDonkey )
	assert( self.opt.data )
	assert( self.opt.imageSize )
	assert( self.opt.cropSize )
	assert( self.opt.net )
	assert( self.opt.dropout )
	assert( self.opt.loss )
	assert( self.opt.numEpoch )
	assert( self.opt.epochSize )
	assert( self.opt.batchSize )
	assert( self.opt.learnRate )
	assert( self.opt.momentum )
	assert( self.opt.weightDecay )
	assert( self.opt.startFrom )
	assert( self.opt.dirRoot )
	assert( self.opt.pathDbTrain )
	assert( self.opt.pathDbVal )
	assert( self.opt.pathImStat )
	assert( self.opt.dirModel )
	assert( self.opt.pathModel )
	assert( self.opt.pathOptim )
	assert( self.opt.pathTrainLog )
	assert( self.opt.pathValLog )
	paths.mkdir( self.opt.dirRoot )
	paths.mkdir( self.opt.dirModel )
end
function task:getOption(  )
	return self.opt
end
function task:setDb(  )
	paths.dofile( string.format( '../db/%s.lua', self.opt.data ) )
	if paths.filep( self.opt.pathDbTrain ) then
		self:print( 'Load train db.' )
		self.dbtr = torch.load( self.opt.pathDbTrain )
		self:print( 'Done.' )
	else
		self:print( 'Create train db.' )
		self.dbtr = self:createDbTrain(  )
		torch.save( self.opt.pathDbTrain, self.dbtr )
		self:print( 'Done.' )
	end
	if paths.filep( self.opt.pathDbVal ) then
		self:print( 'Load val db.' )
		self.dbval = torch.load( self.opt.pathDbVal )
		self:print( 'Done.' )
	else
		self:print( 'Create val db.' )
		self.dbval = self:createDbVal(  )
		torch.save( self.opt.pathDbVal, self.dbval )
		self:print( 'Done.' )
	end
	self.numBatchTrain, self.numBatchVal = self:setNumBatch(  )
	assert( self.numBatchTrain > 0 )
	assert( self.numBatchVal > 0 )
end
function task:getNumBatch(  )
	return self.numBatchTrain, self.numBatchVal
end
function task:setInputStat(  )
	if paths.filep( self.opt.pathImStat ) then
		self:print( 'Load input data statistics.' )
		self.inputStat = torch.load( self.opt.pathImStat )
		self:print( 'Done.' )
	else
		self:print( 'Estimate input data statistics.' )
		self.inputStat = self:estimateInputStat(  )
		torch.save( self.opt.pathImStat, self.inputStat )
		self:print( 'Done.' )
	end
end
function task:getFunctionTrain(  )
	return
		function(  ) return self:getBatchTrain(  ) end,
		function( x, y ) return self:evalBatch( x, y ) end
end
function task:getFunctionVal(  )
	return
		function( i ) return self:getBatchVal( i ) end,
		function( x, y ) return self:evalBatch( x, y ) end
end
function task:getModel(  )
	local numEpoch = self.opt.numEpoch
	local pathModel = self.opt.pathModel
	local pathOptim = self.opt.pathOptim
	local numGpu = self.opt.numGpu
	local startFrom = self.opt.startFrom
	local backend = self.opt.backend
	local startEpoch = 1
	for e = 1, numEpoch do
		local modelPath = pathModel:format( e )
		local optimPath = pathOptim:format( e )
		if not paths.filep( modelPath ) then startEpoch = e break end 
	end
	local model, params, grads, optims
	if startEpoch == 1 and startFrom:len(  ) == 0 then
		self:print( 'Create model.' )
		model = self:defineModel(  )
		if backend == 'cudnn' then
			require 'cudnn'
			cudnn.convert( model, cudnn )
		end
		params, grads, optims = self:groupParams( model )
	elseif startEpoch == 1 and startFrom:len(  ) > 0 then
		self:print( 'Load user-defined model.' .. startFrom )
		model = loadDataParallel( startFrom, numGpu, backend )
		params, grads, optims = self:groupParams( model )
	elseif startEpoch > 1 then
		self:print( string.format( 'Load model from epoch %d.', startEpoch - 1 ) )
		model = loadDataParallel( pathModel:format( startEpoch - 1 ), numGpu, backend )
		params, grads, _ = self:groupParams( model )
		optims = torch.load( pathOptim:format( startEpoch - 1 ) )
	end
	self:print( 'Done.' )
	local criterion = self:defineCriterion(  )
	self:print( 'Model looks' )
	print( model )
	print(criterion)
	self:print( 'Convert model to cuda.' )
	model = model:cuda(  )
	criterion:cuda(  )
	self:print( 'Done.' )
	cutorch.setDevice( 1 )
	local modelSet = {  }
	modelSet.model = model
	modelSet.criterion = criterion
	modelSet.params = params
	modelSet.grads = grads
	modelSet.optims = optims
	-- Verification
	assert( #self.opt.learnRate == #modelSet.params )
	assert( #self.opt.learnRate == #modelSet.grads )
	assert( #self.opt.learnRate == #modelSet.optims )
	for g = 1, #modelSet.params do
		assert( modelSet.params[ g ]:numel(  ) == modelSet.grads[ g ]:numel(  ) )
	end
	return modelSet, startEpoch
end
function task:print( str )
	print( 'TASK MANAGER) ' .. str )
end
------------------------------------------------------------
---- Task-dependent functions ------------------------------
---- These functions should be specified for your task. ----
------------------------------------------------------------
function task:parseOption( arg )
	local cmd = torch.CmdLine(  )
	cmd:option( '-task', arg[ 2 ] )
	-- System.
	cmd:option( '-numGpu', 1, 'Number of GPUs.' )
	cmd:option( '-backend', 'cudnn', 'cudnn or nn.' )
	cmd:option( '-numDonkey', 4, 'Number of donkeys for data loading.' )
	-- Data.
	cmd:option( '-data', 'CIFAR10', 'Name of dataset defined in "./db/"' )
	cmd:option( '-imageSize', 36, 'Short side of initial resize.' )
	cmd:option( '-cropSize', 32, 'Size of random square crop.' )
	cmd:option( '-keepAspect', 0, '1 for keep, 0 for no.' )
	cmd:option( '-normalizeStd', 0, '1 for normalize piexel std to 1, 0 for no.' )
	cmd:option( '-augment', 1, '1 for data augmentation, 0 for no.' )
	cmd:option( '-caffeInput', 0, '1 for caffe input, 0 for no.' )
	-- Model.
	cmd:option( '-net', 'cifarNet', 'Network like cifarNet, cifarNetLarge, cifarNetBatchNorm.' )
	cmd:option( '-dropout', 0.5, 'Dropout ratio.' )
	cmd:option( '-loss', 'logSoftMax', 'Loss like logSoftMax, hinge, L2.' )
	-- Train.
	cmd:option( '-numEpoch', 200, 'Number of total epochs to run.' )
	cmd:option( '-epochSize', 195, 'Number of batches per epoch.' )
	cmd:option( '-batchSize', 256, 'Mini-batch size.' )
	cmd:option( '-learnRate', '1e-2,1e-2', 'Supports multi-lr for multi-module like lr1,lr2,lr3,...' )
	cmd:option( '-momentum', 0.9, 'Momentum.' )
	cmd:option( '-weightDecay', 1e-4, 'Weight decay.' )
	cmd:option( '-startFrom', '', 'Path to the initial model. Use it for LR decay is recommended.' )
	local opt = cmd:parse( arg or {  } )
	-- Set dst paths.
	local dirRoot = paths.concat( gpath.dataout, opt.data )
	local pathDbTrain = paths.concat( dirRoot, 'dbTrain.t7' )
	local pathDbVal = paths.concat( dirRoot, 'dbVal.t7' )
	local pathImStat = paths.concat( dirRoot, 'inputStat.t7' )
	if opt.caffeInput == 1 then pathImStat = pathImStat:match( '(.+).t7$' ) .. 'Caffe.t7' end
	local ignore = { numGpu=true, backend=true, numDonkey=true, data=true, numEpoch=true, startFrom=true }
	local dirModel = paths.concat( dirRoot, cmd:string( opt.task, opt, ignore ) )
	if opt.startFrom ~= '' then
		local baseDir, epoch = opt.startFrom:match( '(.+)/model_(%d+).t7' )
		dirModel = paths.concat( baseDir, cmd:string( 'model_' .. epoch, opt, ignore ) )
	end
	opt.dirRoot = dirRoot
	opt.pathDbTrain = pathDbTrain
	opt.pathDbVal = pathDbVal
	opt.pathImStat = pathImStat
	opt.dirModel = dirModel
	opt.pathModel = paths.concat( opt.dirModel, 'model_%03d.t7' )
	opt.pathOptim = paths.concat( opt.dirModel, 'optimState_%03d.t7' )
	opt.pathTrainLog = paths.concat( opt.dirModel, 'train.log' )
	opt.pathValLog = paths.concat( opt.dirModel, 'val.log' )
	-- Value processing.
	opt.normalizeStd = opt.normalizeStd > 0
	opt.keepAspect = opt.keepAspect > 0
	opt.caffeInput = opt.caffeInput > 0
	opt.augment = opt.augment > 0
	opt.learnRate = opt.learnRate:split( ',' )
	for k,v in pairs( opt.learnRate ) do opt.learnRate[ k ] = tonumber( v ) end
	-- Verification.
	assert( opt.numGpu >= 0 )
	assert( opt.backend:len(  ) > 0 )
	assert( opt.numDonkey >= 0 )
	assert( opt.data:len(  ) > 0 )
	assert( opt.cropSize > 0 )
	assert( opt.imageSize >= opt.cropSize )
	assert( opt.net:len(  ) > 0 )
	assert( opt.dropout <= 1 and opt.dropout >= 0 )
	assert( opt.loss:len(  ) > 0 )
	assert( opt.numEpoch > 0 )
	assert( opt.epochSize > 0 )
	assert( opt.batchSize > 0 )
	assert( #opt.learnRate > 0 )
	assert( opt.momentum >= 0 )
	assert( opt.weightDecay >= 0 )
	return opt
end
function task:createDbTrain(  )
	local dbtr = {  }
	dbtr.iid2path,	dbtr.iid2cid, dbtr.cid2name = createDb( 'train' )
	local numImage = dbtr.iid2path:size( 1 )
	local numClass = dbtr.cid2name:size( 1 )
	self:print( string.format( 'Train: %d images, %d classes.', numImage, numClass ) )
	-- Verification.
	assert( dbtr.iid2path:size( 1 ) == dbtr.iid2cid:numel(  ) )
	assert( dbtr.cid2name:size( 1 ) == dbtr.iid2cid:max(  ) )
	return dbtr
end
function task:createDbVal(  )
	local dbval = {  }
	dbval.iid2path, dbval.iid2cid, dbval.cid2name = createDb( 'val' )
	local numImage = dbval.iid2path:size( 1 )
	local numClass = dbval.cid2name:size( 1 )
	self:print( string.format( 'Val: %d images, %d classes.', numImage, numClass ) )
	-- Verification.
	assert( dbval.iid2path:size( 1 ) == dbval.iid2cid:numel(  ) )
	assert( dbval.cid2name:size( 1 ) == dbval.iid2cid:max(  ) )
	return dbval
end
function task:setNumBatch(  )
	local batchSize = self.opt.batchSize
	---------------------
	-- FILL IN THE BLANK.
	-- Determine number of train/val batches per epoch.

	local numBatchTrain = self.opt.epochSize --math.floor( self.dbtr.iid2cid:numel(  ) / batchSize )
	local numBatchVal = math.floor( self.dbval.iid2cid:numel(  ) / batchSize )

	-- END BLANK.
	-------------
	return numBatchTrain, numBatchVal
end
function task:estimateInputStat(  )
	local numIm = 10000
	local batchSize = self.opt.batchSize
	local meanEstimate = torch.Tensor( 3 ):fill( 0 )
	---------------------
	-- FILL IN THE BLANK.
	-- Estimate RGB-mean vector from numIm training images.
	-- You can use self:getBatchTrain().
	it = math.floor(numIm / batchSize)
	for i = 1, it do
		local batch = self:getBatchTrain()
		local rgbMean = torch.squeeze(torch.mean(torch.mean(torch.mean(batch, 1), 3), 4))
		meanEstimate = meanEstimate + rgbMean
	end
	--print(meanEstimate)
	meanEstimate:div(it)
	--print(meanEstimate)

	-- END BLANK.
	-------------
	return { mean = meanEstimate, std = 0 }
end
function task:defineModel(  )
	-- Set params.
	local netName = self.opt.net
	local numClass = self.dbtr.cid2name:size( 1 )
	local dropout = self.opt.dropout
	local model
	if netName == 'cifarNet' then
		---------------------
		-- FILL IN THE BLANK.
		-- Define cifarNet as follows.
		-- See https://github.com/torch/nn/tree/master/doc
		-- 1. Create a feature network.
		--    Conv1) 32 filters of 5*5 size with a stride 1 and a pad 2.
		--           Add ReLU activation function.
		--           Add max pooling with a 3*3 size window with a stride 2 and a pad 1.
		--    Conv2) 32 filters of 5*5 size with stride 1 and padding 2.
		--           Add ReLU activation function.
		--           Add average pooling with a 3*3 size window with a stride 2 and a pad 1.
		--    Conv3) 64 filters of 5*5 size with a stride 1 and a pad 2.
		--           Add ReLU activation function.
		--           Add average pooling with a 3*3 size window with a stride 2 and a pad 1.
		--    Conv4) 64 filters of 4*4 size with a stride 1 and a pad 0.
		--           Add ReLU activation function.
		--           Add dropout.
		-- 2. Create a classifier network.
		--    FC5) Fully connect input nodes with numClass hidden nodes.
		--         Add log-softmax normalizatioin.
		-- 3. Create full model by concatenating the feature net and the classification net.
		--    The reason we separately build two networks and combine them is to 
		--    apply different learning rate to different network modules for the next dl-practice.
		--    See the function task:groupParams() which will help your understanding.
		
		local outSize = self.opt.cropSize
		local feature = nn.Sequential()
		-- Conv 1
		feature:add(nn.SpatialConvolutionMM(3, 32, 5, 5, 1, 1, 2, 2))
		outSize = math.floor((outSize + 2*2 - 5) / 1 + 1)
		feature:add(nn.ReLU())
		feature:add(nn.SpatialMaxPooling(3, 3, 2, 2, 1, 1))
		outSize = math.floor((outSize + 2*1 - 3) / 2 + 1)
		-- Conv 2
		feature:add(nn.SpatialConvolutionMM(32, 32, 5, 5, 1, 1, 2, 2))
		outSize = math.floor((outSize + 2*2 - 5) / 1 + 1)
		feature:add(nn.ReLU())
		feature:add(nn.SpatialAveragePooling(3, 3, 2, 2, 1, 1))
		outSize = math.floor((outSize + 2*1 - 3) / 2 + 1)
		-- Conv 3
		feature:add(nn.SpatialConvolutionMM(32, 64, 5, 5, 1, 1, 2, 2))
		outSize = math.floor((outSize + 2*2 - 5) / 1 + 1)
		feature:add(nn.ReLU())
		feature:add(nn.SpatialAveragePooling(3, 3, 2, 2, 1, 1))
		outSize = math.floor((outSize + 2*1 - 3) / 2 + 1)
		-- Conv 4
		feature:add(nn.SpatialConvolutionMM(64, 64, 4, 4, 1, 1, 0, 0))
		outSize = math.floor((outSize + 2*0 - 4) / 1 + 1)
		feature:add(nn.ReLU())
		feature:add(nn.SpatialDropout(dropout))
		
		-- Size of feature output
		outSize = 64*outSize
		print("Size of input to FC network: " .. outSize)

		-- FC 5
		local classifier = nn.Sequential()
		classifier:add(nn.Reshape(outSize))
		classifier:add(nn.Linear(outSize, numClass))
		classifier:add(nn.LogSoftMax())

		-- Concatenation
		model = nn.Sequential()
		model:add(feature)
		model:add(classifier)

		-- END BLANK.
		-------------
	elseif netName == 'cifarNetLarge' then
		---------------------
		-- FILL IN THE BLANK.
		-- Define cifarNetLarge,
		-- where the number of filters in each conv-layer of cifarNet is increased two times.
		-- See https://github.com/torch/nn/tree/master/doc
		
		-- END BLANK.
		-------------
	elseif netName == 'cifarNetBatchNorm' then
		---------------------
		-- FILL IN THE BLANK.
		-- Define cifarNetBatchNorm,
		-- where a batch normalization layer is added at each conv-layer of cifarNet.
		-- See https://github.com/torch/nn/tree/master/doc
		
		-- END BLANK.
		-------------
	end
	model:cuda(  )
	-- Check options.
	assert( model )
	assert( not self.opt.caffeInput )
	assert( not self.opt.normalizeStd )
	return model
end
function task:defineCriterion(  )
	local lossName = self.opt.loss
	local loss
	if lossName == 'logSoftMax' then
		---------------------
		-- FILL IN THE BLANK.
		-- Choose a built-in log-softmax loss function in torch.
		-- See https://github.com/torch/nn/blob/master/doc/criterion.md
		loss = nn.ClassNLLCriterion()
		-- END BLANK.
		-------------
	elseif lossName == 'hinge' then
		---------------------
		-- FILL IN THE BLANK.
		-- Choose a built-in hinge loss function in torch.
		-- See https://github.com/torch/nn/blob/master/doc/criterion.md
	
		-- END BLANK.
		-------------
	elseif lossName == 'l2' then
		---------------------
		-- FILL IN THE BLANK.
		-- Choose a built-in l2 loss function in torch.
		-- See https://github.com/torch/nn/blob/master/doc/criterion.md
		
		-- END BLANK.
		-------------
	end
	-- Check options.
	assert( loss )
	return loss
end
function task:groupParams( model )
	local params, grads, optims = {  }, {  }, {  }
	params[ 1 ], grads[ 1 ] = model.modules[ 1 ]:getParameters(  ) -- Feature.
	params[ 2 ], grads[ 2 ] = model.modules[ 2 ]:getParameters(  ) -- Classifier.
	optims[ 1 ] = { -- Feature.
		learningRate = self.opt.learnRate[ 1 ],
		learningRateDecay = 0.0,
		momentum = self.opt.momentum,
		dampening = 0.0,
		weightDecay = self.opt.weightDecay 
	}
	optims[ 2 ] = { -- Classifier.
		learningRate = self.opt.learnRate[ 2 ],
		learningRateDecay = 0.0,
		momentum = self.opt.momentum,
		dampening = 0.0,
		weightDecay = self.opt.weightDecay 
	}
	return params, grads, optims
end
function task:getBatchTrain(  )
	local batchSize = self.opt.batchSize
	local cropSize = self.opt.cropSize
	local augment = self.opt.augment
	local numImage = self.dbtr.iid2path:size( 1 )
	local lossName = self.opt.loss
	---------------------
	-- FILL IN THE BLANK.
	-- 1. Randomly sample batchSize training images from self.dbtr.
	--    You must call self:processImageTrain() to load images.
	--    You can decode a file path by ffi.string( torch.data( self.dbtr.iid2path[ iid ] ) ).
	--    The image batch must have a size of batchSize*3*cropSize*cropSize.
	-- 2. Make a label batch.
	--    The shape of the label batch depends on the type of loss.
	--    See https://github.com/torch/nn/blob/master/doc/criterion.md
	
	local indeces = torch.randperm(numImage)
	indeces = indeces[{{1, batchSize}}]

	local input = torch.Tensor(batchSize, 3, cropSize, cropSize)
	local label = torch.Tensor(batchSize) --zeros(batchSize, numClass)
	local path
	local rw
	local rh
	local rf

	for i = 1, batchSize do
		path = ffi.string( torch.data( self.dbtr.iid2path[ indeces[i] ] ) )
		rw = torch.uniform()
		rh = torch.uniform()
		rf = torch.uniform()
		input[i] = self:processImageTrain(path, rw, rh, rf)
		label[i] = self.dbtr.iid2cid[ indeces[i] ]
	end

	-- for some others maybe
	-- classId = self.dbtr.iid2cid[ indeces[i] ]
	--	label[i][classId] = 1


	-- END BLANK.
	-------------
	return input, label
end
function task:getBatchVal( iidStart )
	local batchSize = self.opt.batchSize
	local cropSize = self.opt.cropSize
	local numImage = self.dbval.iid2path:size( 1 )
	local lossName = self.opt.loss
	---------------------
	-- FILL IN THE BLANK.
	-- 1. Starting from iidStart, get consecutive batchSize validation images from self.dbval.
	--    You must call self:processImageVal() to load images.
	--    You can decode a file path by ffi.string( torch.data( self.dbval.iid2path[ iid ] ) ).
	--    The image batch must have a size of batchSize*3*cropSize*cropSize.
	-- 2. Make a label batch.
	--    The shape of the label batch depends on the type of loss.
	--    See https://github.com/torch/nn/blob/master/doc/criterion.md

	batchSize = math.min(batchSize, numImage - iidStart)

	local input = torch.Tensor(batchSize, 3, cropSize, cropSize)
	local path
	for i = 1, batchSize do
		path = ffi.string( torch.data( self.dbval.iid2path[iidStart + i - 1] ))
		input[i] = self:processImageVal(path)
	end

	-- Need to reshape for other criterion
	local label = self.dbVal.iid2cid[{{iidStart, iidStart+batchSize}}]
	
	-- END BLANK.
	-------------
	return input, label
end
function task:evalBatch( outs, labels )
	local batchSize = self.opt.batchSize
	local lossName = self.opt.loss
	assert( batchSize == outs:size( 1 ) )
	---------------------
	-- FILL IN THE BLANK.
	-- Compare the network output and label to find top-1 accuracy.
	-- This also depends on the type of loss.
	
	local _, outLabels = torch.max(outs, 2)
	local top1 = 0.0
	for i = 1, batchSize do
		if (outLabels[i] == labels[i]) then
			top1 = top1 + 1
		end
	end
	top1 = top1 / batchSize

	-- END BLANK.
	-------------
	return top1 * 100
end
require 'image'
function task:processImageTrain( path, rw, rh, rf )
	collectgarbage(  )
	local cropSize = self.opt.cropSize
	---------------------
	-- FILL IN THE BLANK.
	-- 1. Load an image. You must call self:loadImage()
	-- 2. Do random crop.
	--    You must use rw, rh, which are random values of a range [0,1]
	-- 3. Do random horizontal-flip.
	--    You must use rf which is a random values of a range [0,1]
	-- 4. Normalize the augmented image by the RGB mean.
	--    You must call self:normalizeImage()

	local im = self:loadImage( path )
	--notdone()
	local x = math.floor(rw * ( im:size(3) - cropSize ))
	local y = math.floor(rh * ( im:size(2) - cropSize ))
	im = image.crop(im, x, y, x + cropSize, y + cropSize )
	if rf > 0.5 then
		im = image.hflip(im)
	end
	out = self:normalizeImage(im)
	-- END BLANK.
	-------------
	return out
end
function task:processImageVal( path )
	collectgarbage(  )
	local cropSize = self.opt.cropSize
	---------------------
	-- FILL IN THE BLANK.
	-- 1. Load an image. You must call self:loadImage()
	-- 2. Do central crop.
	-- 4. Normalize the image by the RGB mean.
	--    You must call self:normalizeImage()

	local im = self:loadImage( path )
	local x = math.floor( im:size(3) - cropSize / 2.0 )
	local y = math.floor( im:size(2) - cropSize / 2.0 )
	im = image.crop(im, x, y, x + cropSize, y + cropSize )
	local out = self:normalizeImage(im)
	-- END BLANK.
	-------------
	return out
end
function task:loadImage( path )
	local im = image.load( path, 3, 'float' )
	im = self:resizeImage( im )
	if self.opt.caffeInput then
		im = im * 255
		im = im:index( 1, torch.LongTensor{ 3, 2, 1 } )
	end
	return im
end
function task:resizeImage( im )
	local imageSize = self.opt.imageSize
	local keepAspect = self.opt.keepAspect
	---------------------
	-- FILL IN THE BLANK.
	-- If keepAspect is false, resize the image to imageSize*imageSize.
	-- If keepAspect is true, keep the image aspect ratio,
	-- and resize the image so that the short side is imageSize.
	
	if keepAspect then
		local width = im:size(3)
		local height = im:size(2)
		if width < height then
			local ratio = imageSize / width
			im = scale( im, imageSize, math.floor(ratio * height) )
		else
			local ratio = imageSize / height
			im = scale( im, math.floor(ratio * width), imageSize )
		end
	else
		im = image.scale( im, imageSize, imageSize )
	end
	-- END BLANK.
	-------------
	return im
end
function task:normalizeImage( im )
	local rgbMean = self.inputStat.mean
	---------------------
	-- FILL IN THE BLANK.
	-- Subtract the RGB mean vector from image.

	if self.inputStat.mean then
		for i = 1, 3 do 
			im[i]:add(-rgbMean[i])
		end
	end
	
	-- END BLANK.
	-------------
	return im
end
