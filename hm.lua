----------------------------------------------------
---- Large-scale deep learning framework -----------
---- This script evaluates a given network. --------
---- This script is independent from any tasks. ----
---- Author: Donggeun Yoo, KAIST. ------------------
------------ dgyoo@rcv.kaist.ac.kr -----------------
----------------------------------------------------
local hm = {  }
hm.inputs = torch.CudaTensor(  )
hm.labels = torch.CudaTensor(  )
function hm.setOption( opt, batchSize, pathLog )
	--assert( numBatchVal > 0 )
	--assert( numBatchVal % 1 == 0 )
	--assert( opt.batchSize > 0 )
	assert( pathLog:match( '(.+).log$' ):len(  ) > 0 )
	hm.batchSize = batchSize
	hm.pathLog = pathLog
	--hm.epochSize = numBatchVal
	if opt.net == 'siamese' then
		hm.inputs = {
			torch.CudaTensor(),
			torch.CudaTensor(),
			torch.CudaTensor()
		}
		
	end
	--hm.dirRoot = opt.dirRoot
	hm.opt = opt
end
function hm.setModel( modelSet )
	hm.model = modelSet.model
	hm.criterion = modelSet.criterion
end
function hm.setDonkey( donkeys )
	hm.donkeys = donkeys
end
function hm.setFunction( getBatch, evalBatch )
	hm.getBatch = getBatch
	hm.evalBatch = evalBatch
end
--[[
function val.evaluate( epoch )
	-- Initialization.
	local logger = io.open( val.pathLog, 'a' )
	local epochTimer = torch.Timer(  )
	local getBatch = val.getBatch
	local valBatch = val.evaluateBatch
	val.epoch = epoch
	val.evalEpoch = 0
	val.lossEpoch = 0
	val.batchNumber = 0
	-- Do the job.
	val.print( string.format( 'Validation epoch %d.', epoch ) )
	cutorch.synchronize(  )
	val.model:evaluate(  )
	for b = 1, val.epochSize do
		local s = ( b - 1 ) * val.batchSize + 1
		val.donkeys:addjob(
			function(  )
				return getBatch( s )
			end, -- Job callback.
			function( x, y )
				valBatch( x, y )
			end -- End callback.
		)
	end
	val.donkeys:synchronize(  )
	cutorch.synchronize(  )
	val.evalEpoch = val.evalEpoch / val.epochSize
	val.lossEpoch = val.lossEpoch / val.epochSize
	val.print( string.format( 'Epoch %d, time %.2fs, avg loss %.4f, eval %.4f', 
		epoch, epochTimer:time(  ).real, val.lossEpoch, val.evalEpoch ) )
	valLogger:write( string.format( '%03d %.4f %.4f\n', epoch, val.lossEpoch, val.evalEpoch ) )
	valLogger:close(  )
	collectgarbage(  )
end
--]]

function hm.evaluate( inputsCpu, labelsCpu )
	-- Initialization.
	--local dataTime = val.dataTimer:time(  ).real
	--val.netTimer:reset(  )
	--print(#inputsCpu)
	--print(type(inputsCpu))
	--require 'image'
	if (type(inputsCpu) == 'table') then
		for i = 1, #inputsCpu do
			hm.inputs[i]:resize( inputsCpu[i]:size(  ) ):copy( inputsCpu[i] )
			--train.labels[i]:resize( labelsCpu[i]:size(  ) ):copy( labelsCpu[i] )
		end
	else
		hm.inputs:resize( inputsCpu:size(  ) ):copy( inputsCpu )
		--train.labels:resize( labelsCpu:size(  ) ):copy( labelsCpu )
	end
	--val.inputs:resize( inputsCpu:size(  ) ):copy( inputsCpu )
	hm.labels:resize( labelsCpu:size(  ) ):copy( labelsCpu )
	cutorch.synchronize(  )
	---------------------
	-- FILL IN THE BLANK.
	-- See https://github.com/torch/nn/blob/master/doc/module.md
	-- 1. Feed-forward.
	-- 2. Compute loss and accumulate that to val.lossEpoch.
	-- 3. Compute evaluation metric (e.g. top-1) and accumulate that to train.evalEpoch.
	--    You must call val.evalBatch().
	
	-- 1.
	local output = hm.model:forward(hm.inputs)
	--print(#val.model.modules[2])
	--print(#hm.model.modules[2].modules[1].modules[1].modules[6].output)--.modules[1].modules[6].output)--[6])--.output)
	--print(#hm.model.modules[2].modules[1].modules[2].modules[6].output)--.modules[1].modules[6].output)--[6])--.output)
	--print(#hm.model.modules[2].modules[1].modules[3].modules[6].output)--.modules[1].modules[6].output)--[6])--.output)
	--print(#hm.model.modules[2].modules[1].modules[3].modules[6].output[1])
	
	local prfx
	if hm.opt.net ~= 'siamese' then
		if hm.opt.poolType == 'max' then
			prfx = 'default'
		else
			prfx = 'avg'
		end
	elseif hm.opt.multiScale > 0 then
		--print(hm.opt.multiScale)
		prfx = 'multi'
	else
		prfx = 'single'
	end

	--local netType
	--opt.net == 'siamese'

	local maxval = 0
	local maxidx = 0

	for i = 1, hm.batchSize do
		local heatmaps = {}
		--local maxval = 0
		--local maxidx = 0
		for j = 1, 20 do
			local heatmap
			if hm.opt.net == 'siamese' then
				-- TODO implement this shiiiite
				if hm.opt.multiScale > 0 then
					local tmp3 = hm.model.modules[2].modules[1].modules[3].modules[7].output[{i, {j}}]:double()
					local sz = tmp3:size(3)
					local tmp2 = image.scale(hm.model.modules[2].modules[1].modules[2].modules[7].output[{i, {j}}]:double(), sz)
					local tmp1 = image.scale(hm.model.modules[2].modules[1].modules[1].modules[7].output[{i, {j}}]:double(), sz)
					local htmp = torch.Tensor(3, sz, sz)
					htmp[1] = tmp1
					htmp[2] = tmp2
					htmp[3] = tmp3
					heatmap = torch.mean(htmp, 1)
					--print(heatmap:size())
				else
					heatmap = hm.model.modules[2].modules[1].modules[3].modules[7].output[{i, {j}}]
				end
			else
				heatmap = hm.model.modules[2].modules[7].output[{i, {j}}]
			end
			heatmap = heatmap:double()
			heatmap = heatmap - torch.min(heatmap)
			local mx = torch.max(heatmap)
			if mx > maxval then
				maxval = mx
				maxidx = j
			end
			heatmaps[j] = heatmap
			--heatmap = heatmap:div(torch.max(heatmap)) * 255 + 1
			--print(heatmap)
			--heatmap = image.y2jet(heatmap)--:double())
		end
		for j = 1, 20 do
			heatmap = heatmaps[j]
			heatmap = image.scale(heatmap, 448)
			heatmap[{1,1,1}] = maxval
			heatmap = heatmap:div(maxval) * 255 + 1
			heatmap = image.y2jet(heatmap)--:double())
			image.save(paths.concat( hm.opt.dirRoot, 'figures/' .. prfx .. '/' .. i ..',' .. j .. '.png' ), heatmap)--hm.model.modules[2].modules[1].modules[3].modules[6].output[{i, {j}}])
			--print(heatmap:size())
		end
		if hm.opt.net == 'siamese' then
			image.save(paths.concat( hm.opt.dirRoot, 'figures/' .. prfx .. '/output' .. i .. '.png' ), hm.inputs[3][i])

		else
			image.save(paths.concat( hm.opt.dirRoot, 'figures/' .. prfx .. '/output' .. i .. '.png' ), hm.inputs[i])
		end
	end
	--assert(1 == 2)

	--print(hm.inputs:size())
	--print(output:size())
	-- 2.
	local err = hm.criterion:forward(output, hm.labels)
	--val.lossEpoch = val.lossEpoch + err

	-- 3.
	local eval = hm.evalBatch(output, hm.labels)
	--val.evalEpoch = val.evalEpoch + eval


	

	-- END BLANK.
	-------------
	cutorch.synchronize(  )

	local logger = io.open( hm.pathLog, 'a' )
	logger:write( string.format( '%d %.4f %.4f\n\n', maxidx, err, eval ) )
	for i = 1, hm.batchSize do
		local str = ''
		for j = 1, 20 do
			str = str .. " " .. output[{i, j}]
		end
		logger:write( string.format('%s\n', str))
	end
	logger:close(  )

	--[[
	val.batchNumber = val.batchNumber + 1
	local netTime = val.netTimer:time(  ).real
	local totalTime = dataTime + netTime
	local speed = val.batchSize / totalTime
	-- Print information.
	val.print( string.format( 'Epoch %d, %d/%d, %dim/s (data %.2fs, net %.2fs), err %.2f, eval %.2f', 
		val.epoch, val.batchNumber, val.epochSize, speed, dataTime, netTime, err, eval ) )
	val.dataTimer:reset(  )
	--]]

	collectgarbage(  )
end
function hm.print( str )
	print( 'HM) ' .. str )
end



function hm.evalBatch( outs, labels )
	local batchSize = hm.batchSize
	--local lossName = self.opt.loss
	--local eval = self.opt.eval
	assert( batchSize == outs:size( 1 ) )
	---------------------
	-- FILL IN THE BLANK.
	-- Compare the network output and label to find top-1 accuracy.
	-- This also depends on the type of loss.
	
	local lossName = 'hinge'
	local numClass = 20--self.dbtr.cid2name:size( 1 )

	local _, outLabels = torch.max(outs, 2)
	if outs:size(1) > 1 then
		outLabels = outLabels:squeeze()
	end
	

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

return hm

