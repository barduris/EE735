-------------------------------------------------------------------------------
---- Large-scale deep learning framework --------------------------------------
---- This script extracts the information needed for --------------------------
---- learning from a specific dataset and processes ---------------------------
---- it into a predefined protocal. -------------------------------------------
---- You only need to define this script for a specific dataset. --------------
---- Author: Donggeun Yoo, KAIST. ---------------------------------------------
------------ dgyoo@rcv.kaist.ac.kr --------------------------------------------
-------------------------------------------------------------------------------
require 'paths'
require 'sys'
local ffi = require 'ffi'
local function strTableToTensor( strTable )
	local maxStrLen = 0
	local numStr = #strTable
	for _, path in pairs( strTable ) do
		if maxStrLen < path:len(  ) then maxStrLen = path:len(  ) end
	end
	maxStrLen = maxStrLen + 1
	local charTensor = torch.CharTensor( numStr, maxStrLen ):fill( 0 )
	local pt = charTensor:data(  )
	for _, path in pairs( strTable ) do
		ffi.copy( pt, path )
		pt = pt + maxStrLen
	end
	for i = 1, #strTable do strTable[ i ] = nil end strTable = nil
	collectgarbage(  )
	return charTensor
end
function createDb( setName )
	local matio = require 'matio'
	local percentile = 0.7
	local dbDir = gpath.db.afw
	---------------------
	-- FILL IN THE BLANK.
	-- Create dataset information that satisfies the following format.
	-- If setName is 'train', create training db information,
	-- or create validation db information.
	-- 1. cid2name: A table in which a key is a class id (cid)
	--              and a value is a name of that class.
	--              The class id starts from 1.
	-- 2. iid2path: A table in which a key is an image id (iid)
	--              and a value is a global path of that image.
	--              The image id starts from 1.
	-- 3. iid2cid: A table in which a key is an image id (iid)
	--             and a value is a class id (cid) of that image.
	--              The image id starts from 1.
	
	local cid2name = {}
	local iid2path = {}
	local iid2cid = {}

	matio.use_lua_strings = true
	local anno = matio.load(dbDir .. 'anno_fixed.mat')
	--local names = anno['names']
	--local bbox = anno['bbox']

	--iid2path = anno['names']
	--iid2cid = anno['bbox']
	
	cid2name = {'x1', 'y1', 'x2', 'y2'}

	local split = math.floor(percentile * #anno['names'])
	--local shift = 0
	--if setName == 'val' then shift = split + 1 end
	if setName == 'train' then
		for i = 1, split do
			iid2path[i] = dbDir .. anno['names'][i]
			--iid2cid[i] = anno['bbox'][i][1]
			local bbox = anno['bbox'][i][1]
			iid2cid[i] = { bbox[1][1], bbox[1][2], bbox[2][1], bbox[2][2] }
		end
	else
		for i = split+1, #anno['names'] do
			iid2path[i - split] = dbDir .. anno['names'][i]
			local bbox = anno['bbox'][i][1]
			iid2cid[i - split] = { bbox[1][1], bbox[1][2], bbox[2][1], bbox[2][2] }
		end
	end
	print("Only taking first face")
	--print('Size of ' .. setName .. ' dataset')
	--print(#iid2cid)
	--print(#iid2path)


	-- END BLANK.
	-------------
	assert( #iid2path == #iid2cid )
	-- Convert tables to tensors.
	-- Lua has a fatal drawback that the garbage collection 
	-- is getting very slow when it holds large tables. Therefore, 
	-- this process is quite important when the size of the table grows.
	iid2cid = torch.LongTensor( iid2cid )
	iid2path = strTableToTensor( iid2path )
	cid2name = strTableToTensor( cid2name )
	collectgarbage(  )
	return iid2path, iid2cid, cid2name
end
