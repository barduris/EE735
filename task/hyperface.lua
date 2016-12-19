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
	assert( self.opt.eval )
	assert( self.opt.poolType )
	assert( self.opt.numEpoch )
	assert( self.opt.epochSize )
	assert( self.opt.batchSize )
	assert( self.opt.learnRate )
	assert( self.opt.multiScale )
	assert( self.opt.heatmaps )
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
	cmd:option( '-data', 'AFW', 'Name of dataset defined in "./db/"' )
	cmd:option( '-imageSize', 256, 'Short side of initial resize.' )
	cmd:option( '-cropSize', 224, 'Size of random square crop.' )
	cmd:option( '-keepAspect', 1, '1 for keep, 0 for no.' )
	cmd:option( '-normalizeStd', 0, '1 for normalize piexel std to 1, 0 for no.' )
	cmd:option( '-augment', 1, '1 for data augmentation, 0 for no.' )
	cmd:option( '-caffeInput', 0, '1 for caffe input, 0 for no.' )
	-- Model.
	cmd:option( '-net', 'hyperface', 'Network like cifarNet, cifarNetLarge, cifarNetBatchNorm.' ) --------------------------------------------
	cmd:option( '-dropout', 0.5, 'Dropout ratio.' )
	cmd:option( '-loss', 'L2', 'Loss like logSoftMax, hinge, L2.' )
	cmd:option( '-eval', 'map', 'Evaluation metric.' )
	cmd:option( '-poolType', 'max', 'Type of global pooling layers.' )
	cmd:option( '-multiScale', 0, 'Test type.' )
	cmd:option( '-heatmaps', 1, 'Generate heatmaps.' )
	-- Train.
	cmd:option( '-numEpoch', 50, 'Number of total epochs to run.' )
	cmd:option( '-epochSize', 30, 'Number of batches per epoch.' )
	cmd:option( '-batchSize', 48, 'Mini-batch size.' )
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
	assert( opt.eval:len(  ) > 0 )
	assert( opt.poolType:len(  ) > 0 )
	assert( opt.multiScale >= 0 )
	assert( opt.heatmaps >= 0 )
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
	--print(dbtr.iid2path:size())
	--print(dbtr.iid2cid:size())
	--print(dbtr.cid2name:size())
	--print(dbtr.iid2cid:max())
	assert( dbtr.iid2path:size( 1 ) == dbtr.iid2cid:size( 1 ) )
	--assert( dbtr.cid2name:size( 1 ) == dbtr.iid2cid:max(  ) )
	return dbtr
end
function task:createDbVal(  )
	local dbval = {  }
	dbval.iid2path, dbval.iid2cid, dbval.cid2name = createDb( 'val' )
	local numImage = dbval.iid2path:size( 1 )
	local numClass = dbval.cid2name:size( 1 )
	self:print( string.format( 'Val: %d images, %d classes.', numImage, numClass ) )
	-- Verification.
	assert( dbval.iid2path:size( 1 ) == dbval.iid2cid:size( 1 ) )
	-- assert( dbval.cid2name:size( 1 ) == dbval.iid2cid:max(  ) )
	return dbval
end
function task:setNumBatch(  )
	local batchSize = self.opt.batchSize
	---------------------
	-- FILL IN THE BLANK.
	-- Determine number of train/val batches per epoch.

	local numBatchTrain = self.opt.epochSize
	local numBatchVal = math.floor( self.dbval.iid2cid:size( 1 ) / batchSize )

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
	meanEstimate:div(it)

	-- END BLANK.
	-------------
	return { mean = meanEstimate, std = 0 }
end
function task:defineModel(  )
	-- Set params.
	local netName = self.opt.net
	local numClass = self.dbtr.cid2name:size( 1 )
	local dropout = self.opt.dropout
	local poolType = self.opt.poolType
	local model
	
	if netName == 'hyperface' then-- or netName == 'alexNetScratch' then
		require 'loadcaffe'
		local alexnet = loadcaffe.load(gpath.net.alex_caffe_proto, gpath.net.alex_caffe_model, 'cudnn')
		print(alexnet)
		

        for i = 16, 24 do
            alexnet:remove()
        end

        local net1 = nn.Sequential()
        local net2 = nn.Sequential()
        local net3 = nn.Sequential()

        for i = 1, 4 do
            net1:add(alexnet:get(i))
        end
       
        for i = 5, 10 do
            net2:add(alexnet:get(i))
        end
        
        for i = 11, 15 do
            net3:add(alexnet:get(i))
        end

        local conv1a = nn.SpatialConvolution(96, 256, 4, 4, 4, 4)
        local conv3a = nn.SpatialConvolution(384, 256, 2, 2, 2, 2)
        
        local subsubnet = nn.ConcatTable()--nn.Concat(2)
        subsubnet:add(net3)
        subsubnet:add(conv3a)

        local submod = nn.Sequential()
        submod:add(net2)
        submod:add(subsubnet)
        submod:add(nn.JoinTable(1, 3))

        local subnet = nn.ConcatTable() --Concat(2)
        subnet:add(submod)
        subnet:add(conv1a)

        local feature = nn.Sequential()
        
        feature:add(net1)
        feature:add(subnet)
        feature:add(nn.JoinTable(1, 3))
        feature:add(nn.SpatialConvolution(768, 192, 1, 1))
        feature:cuda()



        local classifier = nn.Sequential()
        classifier:add(nn.View(-1))


        --[[
        classifier:add(nn.SpatialConvolution(256, 4096, 6, 6))
        classifier:add(nn.ReLU())
        classifier:add(nn.Dropout(0.5))
        classifier:add(nn.SpatialConvolution(4096, 4096, 1, 1))
        classifier:add(nn.ReLU())
        classifier:add(nn.Dropout(0.5))
        classifier:add(nn.SpatialConvolution(4096, numClass, 1, 1))
        --classifier:add(nn.ReLU())

        if poolType == 'max' then
        	classifier:add(nn.Max(3,3))
        	classifier:add(nn.Max(2,2))
    	elseif poolType == 'average' then
    		classifier:add(nn.Mean(3,3))
        	classifier:add(nn.Mean(2,2))
        end

        --]]
        model = nn.Sequential()

		--model:add(alexnet)
        model:add(feature)
		model:add(classifier)
        print(model)
        local inp = torch.Tensor(1, 3, 227, 227):cuda()
        local outp = model:forward(inp)
        print(outp:size())
        assert(false)

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
	if lossName == 'multiHinge' then
		---------------------
		-- FILL IN THE BLANK.
		-- Choose a built-in hinge loss function in torch.
		-- See https://github.com/torch/nn/blob/master/doc/criterion.md

		loss = nn.MultiLabelMarginCriterion()

		-- END BLANK.
		-------------
	elseif lossName == 'entropy' then
		---------------------
		-- FILL IN THE BLANK.
		-- Choose a built-in log-softmax loss function in torch.
		-- See https://github.com/torch/nn/blob/master/doc/criterion.md

		loss = nn.MultiLabelSoftMarginCriterion() --nn.ClassNLLCriterion()
		
		-- END BLANK.
		-------------
	elseif lossName == 'l2' then
		---------------------
		-- FILL IN THE BLANK.
		-- Choose a built-in l2 loss function in torch.
		-- See https://github.com/torch/nn/blob/master/doc/criterion.md
		
		loss = nn.MSECriterion()
		loss.sizeAverage = false

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
	local netName = self.opt.net
	---------------------
	-- FILL IN THE BLANK.
	-- 1. Randomly sample batchSize training images from self.dbtr.
	--    You must call self:processImageTrain() to load images.
	--    You can decode a file path by ffi.string( torch.data( self.dbtr.iid2path[ iid ] ) ).
	--    The image batch must have a size of batchSize*3*cropSize*cropSize.
	-- 2. Make a label batch.
	--    The shape of the label batch depends on the type of loss.
	--    See https://github.com/torch/nn/blob/master/doc/criterion.md
	
	local numClass = self.dbtr.cid2name:size( 1 )

	local indeces = torch.randperm(numImage)
	indeces = indeces[{{1, batchSize}}]

	local input = torch.Tensor(batchSize, 3, cropSize, cropSize)

	local path
	local rw = 0.5
	local rh = 0.5
	local rf = 0

	local label = torch.Tensor(batchSize, numClass)
	if lossName == 'entropy' then
		label:fill(0)
	end
	--print(numClass)
	--if lossName == 'l2' then
	--	label = torch.Tensor(batchSize, numClass):fill(-1)
	--else
	--	label = torch.Tensor(batchSize)
	--end
	for i = 1, batchSize do
		path = ffi.string( torch.data( self.dbtr.iid2path[ indeces[i] ] ) )
		if augment then
			rw = torch.uniform()
			rh = torch.uniform()
			rf = torch.uniform()
		end
		input[i] = self:processImageTrain(path, rw, rh, rf)
		local cid = self.dbtr.iid2cid[ indeces[i] ]
		if lossName == 'entropy' then
			for icid = 1,cid:size(1) do
				if cid[icid] == 0 then break end
				label[i][cid[icid]] = 1
			end
		else
			label[i] = cid
		end
		--local cid = self.dbtr.iid2cid[ indeces[i] ]
		--if lossName == 'l2' then

		--	label[i][cid] = 1
		--else
		--	label[i] = cid
		--end
	end

	--if netName == 'siamese' then
	--	input = {input, input, input}
	--end
	local inputs
	if netName == 'siamese' then
		local sizes = {224, 336, 448}
		--print(#input)

		local im2 = torch.Tensor(input:size(1), input:size(2), sizes[2], sizes[2])
		local im3 = torch.Tensor(input:size(1), input:size(2), sizes[3], sizes[3])
		for i = 1, input:size(1) do
			im2[i] = image.scale(input[i], sizes[2])
			im3[i] = image.scale(input[i], sizes[3])
		end

		inputs = {input, im2, im3}
	else
		inputs = input
	end
	--print("Size of train batch")
	--print(#input)

	-- END BLANK.
	-------------
	return inputs, label
end
function task:getBatchVal( iidStart )
	local batchSize = self.opt.batchSize
	local cropSize = self.opt.cropSize
	local numImage = self.dbval.iid2path:size( 1 )
	local lossName = self.opt.loss
	local netName = self.opt.net
	local multiScale = self.opt.multiScale
	---------------------
	-- FILL IN THE BLANK.
	-- 1. Starting from iidStart, get consecutive batchSize validation images from self.dbval.
	--    You must call self:processImageVal() to load images.
	--    You can decode a file path by ffi.string( torch.data( self.dbval.iid2path[ iid ] ) ).
	--    The image batch must have a size of batchSize*3*cropSize*cropSize.
	-- 2. Make a label batch.
	--    The shape of the label batch depends on the type of loss.
	--    See https://github.com/torch/nn/blob/master/doc/criterion.md

	local numClass = self.dbval.cid2name:size( 1 )

	local input = torch.Tensor(batchSize, 3, cropSize, cropSize)
	local label = torch.Tensor(batchSize, numClass)
	if lossName == 'entropy' then
		label:fill(0)
	end
	local path
	for i = 1, batchSize do
		path = ffi.string( torch.data( self.dbval.iid2path[iidStart + i - 1] ))
		input[i] = self:processImageVal(path)
		--label[i] = self.dbval.iid2cid[iidStart + i - 1]
		local cid = self.dbval.iid2cid[ iidStart + i - 1 ]
		if lossName == 'entropy' then
			for icid = 1,cid:size(1) do
				if cid[icid] == 0 then break end
				label[i][cid[icid]] = 1
			end
		else
			label[i] = cid
		end
	end

	--print(iidStart)

	--local label = torch.Tensor(batchSize, numClass)
	-- Need to reshape for other criterion
	--[[
	if lossName == 'l2' then
		label = torch.Tensor(batchSize, numClass):fill(-1)
		label[i] = self.dbtr.iid2cid[ indeces[i] ]
		local cid
		for i = 1, batchSize do
			cid = self.dbval.iid2cid[iidStart + i - 1]
			label[i][cid] = 1
		end
	else
		label = self.dbval.iid2cid[{{iidStart, iidStart+batchSize-1}}]
	end
	--]]
	local inputs
	if netName == 'siamese' then
		if multiScale > 0 then
			local sizes = {224, 336, 448}
			--print(#input)

			local im2 = torch.Tensor(input:size(1), input:size(2), sizes[2], sizes[2])
			local im3 = torch.Tensor(input:size(1), input:size(2), sizes[3], sizes[3])
			for i = 1, input:size(1) do
				im2[i] = image.scale(input[i], sizes[2])
				im3[i] = image.scale(input[i], sizes[3])
			end

			inputs = {input, im2, im3}
		else
			inputs = {input, input:clone(), input:clone()}
		end
	else
		inputs = input
	end
	--print("Size of val batch")
	--print(#input)

	-- END BLANK.
	-------------
	return inputs, label
end

function task:getBatchHeatmap(  )
	local batchSize = self.opt.batchSize
	local cropSize = self.opt.cropSize
	local numImage = self.dbval.iid2path:size( 1 )
	local lossName = self.opt.loss
	local netName = self.opt.net
	local multiScale = self.opt.multiScale
	local dbDir = gpath.db.voc07
	
	---------------------
	-- FILL IN THE BLANK.
	-- 1. Starting from iidStart, get consecutive batchSize validation images from self.dbval.
	--    You must call self:processImageVal() to load images.
	--    You can decode a file path by ffi.string( torch.data( self.dbval.iid2path[ iid ] ) ).
	--    The image batch must have a size of batchSize*3*cropSize*cropSize.
	-- 2. Make a label batch.
	--    The shape of the label batch depends on the type of loss.
	--    See https://github.com/torch/nn/blob/master/doc/criterion.md



	--cid2name = {'aeroplane', 'bicycle', 'bird',
	-- 	'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
	-- 	'cow', 'diningtable', 'dog', 'horse', 'motorbike',
	-- 	'person', 'pottedplant', 'sheep', 'sofa', 'train',
	 -- 	'tvmonitor'}


	local batchSize = 1
	local numClass = 20--self.dbtr.cid2name:size( 1 )

	local input = torch.Tensor(batchSize, 3, cropSize, cropSize)
	local label = torch.zeros(batchSize, numClass)
	if lossName == 'entropy' then
		label[{1, 1}] = 1
		--label[{2, 2}] = 1
		--label[{2, 15}] = 1
	else
		label[{1, 1}] = 1
		--label[{2, 1}] = 2
		--label[{2, 2}] = 15
	end

	imgs = {'000647'}--'004555.jpg', '000030.jpg'}--{'000228.jpg', '000812.jpg'}
	local path
	for i = 1, batchSize do
		path = dbDir .. 'JPEGImages/' .. imgs[i] .. '.jpg'--dbDir .. 'HeatmapTest/' .. imgs[i]
		input[i] = self:processImageVal(path)
	end

	local inputs
	if netName == 'siamese' and multiScale then
		local sizes = {224, 336, 448}
		--print(#input)

		local im2 = torch.Tensor(input:size(1), input:size(2), sizes[2], sizes[2])
		local im3 = torch.Tensor(input:size(1), input:size(2), sizes[3], sizes[3])
		for i = 1, input:size(1) do
			im2[i] = image.scale(input[i], sizes[2])
			im3[i] = image.scale(input[i], sizes[3])
		end

		inputs = {input, im2, im3}
	elseif netName == 'siamese' then
		local iSize = 448
		local aux = torch.Tensor(input:size(1), input:size(2), iSize, iSize)--image.scale(input, 448)
		--local im2 = torch.Tensor(input:size(1), input:size(2), sizes[2], sizes[2])
		--local im3 = torch.Tensor(input:size(1), input:size(2), sizes[3], sizes[3])
		for i = 1, input:size(1) do
			aux[i] = image.scale(input[i], iSize)
			--im3[i] = image.scale(input[i], sizes[3])
		end
		inputs = {aux, aux, aux}
	else
		local iSize = 448
		inputs = torch.Tensor(input:size(1), input:size(2), iSize, iSize)--image.scale(input, 448)
		for i = 1, input:size(1) do
			inputs[i] = image.scale(input[i], iSize)
		end
		--inputs = image.scale(input, 448)
	end

	--print(#inputs)
	-- END BLANK.
	-------------
	return inputs, label, batchSize
end

function task:evalBatch( outs, labels )
	local batchSize = self.opt.batchSize
	local lossName = self.opt.loss
	local eval = self.opt.eval
	assert( batchSize == outs:size( 1 ) )
	---------------------
	-- FILL IN THE BLANK.
	-- Compare the network output and label to find top-1 accuracy.
	-- This also depends on the type of loss.
	
	local numClass = self.dbtr.cid2name:size( 1 )
	if eval == 'map' then


		local TP = torch.zeros(numClass) -- True positives
		--local P = torch.zeros(numClass) -- 	Condition Positive: True positives + False Negatives

        local FP = torch.zeros(numClass)
        local FN = torch.zeros(numClass)

        local AP = torch.zeros(numClass)--batchSize)
        local Rlast = torch.zeros(numClass)

        local PP = torch.zeros(batchSize, numClass)
        local RR = torch.zeros(batchSize, numClass)

        --print(outs:size())

        
        if lossName == 'multiHinge' then
            local aux = torch.zeros(batchSize, numClass)

            for i = 1, batchSize do
                for j = 1, numClass do
                    local label = labels[i][j]
                    if label == 0 then break end
                    aux[i][label] = 1
                end
            end
            labels = aux
        end

        local TP = torch.zeros(numClass) -- True positives
        local FP = torch.zeros(numClass)
        local FN = torch.zeros(numClass)

		for i = 1, batchSize do
            

			--if lossName == 'entropy' then
				for j = 1, numClass do
					if labels[i][j] == 1 then
						--P[j] = P[j] + 1
						if outs[i][j] > 0 then
							TP[j] = TP[j] + 1
						else
                            FN[j] = FN[j] + 1
                        end
                    elseif labels[i][j] == 0 then
                        --P[j] = P[j] + 1
                        --if outs[i][j] > 0 then
                        --    TP[j] = TP[j] + 1
                        --top1 = top1 + 1
                        --break
                        --end
                        if outs[i][j] > 0 then
                            FP[j] = FP[j] + 1
                        --else
                        --    FN[j] = FN[j] + 1
                        end
                    end

				--end
			--[[else
                for j = 1, numClass do
                    local label = labels[i][j]
                    --if label == 0 then break end
                    --P[label] = P[label] + 1
                    print(label)
                    if label == 1 then
                        if outs[i][label] > 0 then
                            TP[label] = TP[label] + 1
                        else
                            FN[label] = FN[label] + 1
                        end
                    elseif outs[i][label] > 0 then
                        FP[label] = FP[label] + 1
                    end
                    --if labels[i][j] == 1 then --(outLabels[i] == labels[i][j]) then
                        --top1 = top1 + 1
                        --break
                        --truth[j] = truth[j] + 1
                    --end
                end --]]
            end

            local P = torch.cdiv(TP, TP + FP)
            local R = torch.cdiv(TP, TP + FN)
            --print(TP)
            --print(FP)
            --print(FN)
            --print(P)
            --print(R)
            --print(R - Rlast)
            --local tmp = P / batchSize
            local nan_mask_P = P:ne(P)
            P[nan_mask_P] = 0
            local nan_mask_R = R:ne(R)
            R[nan_mask_R] = 0

            --AP[i] = P --AP + P --* (R - Rlast) / i --/ i--batchSize --P[i] = torch.mean(P) -- torch.cmul(P, R - Rlast)
            --Rlast = R

            PP[i] = P
            RR[i] = R

            --local nan_mask = AP:ne(AP)
            --AP[nan_mask] = 0

        end

        --AP = AP / batchSize

        --AP = torch.zeros(numClass)

        for iC = 1, numClass do
            local P = PP[{{}, iC}]
            local R = RR[{{}, iC}]

            local r, idx = torch.sort(R)
            local p = PP[{{i}}]
            local r_last = 0

            for i = 1, #r do
                AP[idx[i]] = AP[idx[i]] + p[i] * (r[i] - r_last)
                r_last = r[i]
            end
        end


        --[[
        local N = 0
        local sum = 0
        for i = 1, batchSize do
            if AP[i] ~= AP[i] then 
                AP[i] = 0
            end
                sum = sum + PPV[i]
                N = N + 1
            end
        end
        local mAP = sum/N
        --]]

        --local P = torch.cdiv(TP, TP + FP)
        --local R = torch.cdiv(TP, TP + FN)

        --AP[i] = torch.mean(P)

        --local nan_mask = AP:ne(AP)
        --AP[nan_mask] = 0
        mAP = torch.mean(AP)
        

        --[[
				for j = 1, numClass do
					local label = labels[i][j]
					if label == 0 then break end
					P[label] = P[label] + 1
					if outs[i][label] > 0 then
						TP[label] = TP[label] + 1
					end
					--if labels[i][j] == 1 then --(outLabels[i] == labels[i][j]) then
						--top1 = top1 + 1
						--break
						--truth[j] = truth[j] + 1
					--end
				end
			end
		end


		local PPV = torch.cdiv(TP, P)
		-- ignore NaN
		
		local N = 0
		local sum = 0
		for i = 1, PPV:size(1) do
			if PPV[i] == PPV[i] then
				sum = sum + PPV[i]
				N = N + 1
			end
		end
		mAP = sum/N
        --]]
		return mAP--torch.mean(torch.cdiv(classified, truth))


	elseif eval == 'top-1' then

		
		local _, outLabels = torch.max(outs, 2)
		--[[
		local label
		
		if lossName == 'l2' then
			_, label = torch.max(labels, 2)
		else
			label = labels
		end
		--]]
		--if lossName == 'entropy' then
		--	_, labels = torch.max(labels, 2)
		--else

		outLabels = outLabels:squeeze()
		--label = label:squeeze()

		local top1 = 0
		for i = 1, batchSize do
			if lossName == 'entropy' then
				if (labels[i][outLabels[i]] == 1) then
						top1 = top1 + 1
					end
			else
				for j = 1, numClass do
					if (outLabels[i] == labels[i][j]) then
						top1 = top1 + 1
						break
					end
				end
			end
		end
		top1 = top1 / batchSize

		-- END BLANK.
		-------------
		return top1 * 100
	end
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
	local x = math.floor(rw * ( im:size(3) - cropSize )) + 1
	local y = math.floor(rh * ( im:size(2) - cropSize )) + 1
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
	local x = math.floor( (im:size(3) - cropSize) / 2.0 ) + 1
	local y = math.floor( (im:size(2) - cropSize) / 2.0 ) + 1
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
			im = image.scale( im, imageSize, math.floor(ratio * height) )
		else
			local ratio = imageSize / height
			im = image.scale( im, math.floor(ratio * width), imageSize )
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
