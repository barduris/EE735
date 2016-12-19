--------------------------------------------------------------------------
---- Large-scale deep learning framework ---------------------------------
---- This script defines the required locations in a global variable. ----
---- Author: Donggeun Yoo, KAIST. ----------------------------------------
------------ dgyoo@rcv.kaist.ac.kr ---------------------------------------
--------------------------------------------------------------------------
gpath = { db = {  }, net = {  } }
gpath.db.cifar10 = 'data/db/CIFAR10/'
gpath.db.voc07 = 'data/db/VOC07/'
gpath.db.afw = 'data/db/AFW/'
gpath.net.alex_caffe_model = 'data/net/bvlc_alexnet.caffemodel'
gpath.net.alex_caffe_proto = 'data/net/bvlc_alexnet_deploy.prototxt'
gpath.dataout = 'data/cache/'
paths.mkdir( gpath.dataout )
