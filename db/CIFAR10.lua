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
	local dbDir = gpath.db.cifar10
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

	local file = io.open(dbDir .. 'classnames.txt')
	if file then
		for line in file:lines() do
			cid2name[#cid2name + 1] = line
		end
	end


	if setName == 'train' then
		file = io.open(dbDir .. 'train_impaths.txt')
		for line in file:lines() do
			iid2path[#iid2path + 1] = dbDir .. line
		end
		file = io.open(dbDir .. 'train_imclasses.txt')
		for line in file:lines() do
			iid2cid[#iid2cid + 1] = line
		end
	elseif setName == 'val' then
		file = io.open(dbDir .. 'val_impaths.txt')
		for line in file:lines() do
			iid2path[#iid2path + 1] = dbDir .. line
		end
		file = io.open(dbDir .. 'val_imclasses.txt')
		for line in file:lines() do
			iid2cid[#iid2cid + 1] = line
		end
	end

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
