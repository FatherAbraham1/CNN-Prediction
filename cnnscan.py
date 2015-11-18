# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 16:13:12 2015

@author: fujunliu
"""

import scipy.io as sio
import numpy as np
from scipy import signal as sg
from scipy import misc
import matplotlib.pyplot as plt
#from im2col_cython import col2im_cython, im2col_cython
import im2col
from datetime import datetime
#import numbapro.cudalib.cublas as cublas
import pickle

class cnnScan(object):
    # test single image
    # currenly, convolutional layer is slow

    def __init__(self, cnnMat_path, indice_dict_path = None):
        self.model = self.load_matcnn(cnnMat_path)
        patchsz = self.model[0]['mapsize'][0]
        self.padsz = np.floor(patchsz/2.0)
        if not indice_dict_path:
            self.indice_dict = {}
        else:
            self.indice_dict = pickle.load(open(indice_dict_path, "rb"))
            
    def save_indice_dict(self, indice_dict_path):
        pickle.dump(self.indice_dict, open( indice_dict_path, "wb"))
        
    def pad_img(self, img):
        # img H x W * channels
        img = np.pad(img, ((self.padsz, self.padsz),(self.padsz, self.padsz), (0,0)), 'symmetric')
        return img
        
    def fast_max_pooling(self, x, pool_scale):
        # non-overlapping max-pooling and zero pading  
        N, H, W = x.shape
        pool_height = pool_width = pool_scale
        assert H % pool_height == 0
        assert W % pool_height == 0
        x_reshaped = x.reshape(N, H / pool_height, pool_height,
                                 W / pool_width, pool_width)
        out = x_reshaped.max(axis=2).max(axis=3)
        return out
    
    def max_pooling_frag(self, x, offset, pool_scale):
        start_x, start_y = offset[0], offset[1]
        # non-overlapping max-pooling and zero pading  
        N, H, W = x.shape
        H_out = pool_scale*np.floor((H-start_y)/pool_scale)
        W_out = pool_scale*np.floor((W-start_x)/pool_scale)
        x = x[:, start_y:start_y+H_out, start_x:start_x+W_out]
        return self.fast_max_pooling(x, pool_scale)
        
    
    def gather_frags(self, frags, offsets, img_size, patch_size, map_size):
        if len(map_size) > 1: # square maps
            map_size = map_size[0]
        patch_num = img_size - patch_size + 1
        N = frags[0].shape[0]
        frags_reunion = np.zeros((N*map_size*map_size, patch_num[0]*patch_num[1]))
        
        for ifrag in range(len(frags)):
            feat_maps = frags[ifrag]
            #print feat_maps.shape
            offset = offsets[ifrag]
            noffset = offset.shape[0]
            nrows, ncols = feat_maps.shape[1:] - map_size + 1
            for irow in range(nrows):
                for jcol in range(ncols):
                    # get this patch data
                    patch_data = feat_maps[:, irow:irow+map_size, jcol:jcol+map_size].flatten()
                    # found the patch id
                    px, py = jcol, irow
                    for i in range(noffset-1, -1, -1):
                        ox, oy, k = offset[i, :]
                        px = ox + px*k
                        py = oy + py*k
                    
                    patch_id = py*patch_num[1] + px
                    frags_reunion[:, patch_id] = patch_data
                    
        return frags_reunion
    
    def frags2cols(self, frags, offsets, img_size, patch_size, map_size):
        if len(map_size) > 1: # square maps
            map_size = map_size[0]
        patch_num = img_size - patch_size + 1
        frags_cols = []
        patch_id_cols = np.zeros(patch_num[0]*patch_num[1], dtype='int')
        index_start = 0
        #indice_dict = {}
        padding, stride = 0, 1
        for ifrag in range(len(frags)):
            feat_maps = frags[ifrag][None,:,:,:]
            # cache indice_map
            indice_key = '-'.join(['-'.join([str(se) for se in feat_maps.shape]), str(map_size), 
                        str(map_size), str(padding), str(stride)])
                        
            if indice_key in self.indice_dict:
                indice_map = self.indice_dict[indice_key]
            else:
                indice_map = im2col.get_im2col_indices(feat_maps.shape, map_size, map_size, 
                                                       padding, stride)
                self.indice_dict[indice_key] = indice_map
                
            feat_maps_cols = im2col.im2col_indices_cache(feat_maps, indice_map, map_size, map_size, 0, 1)
            
            #feat_maps_cols = im2col.im2col_indices(feat_maps[None,:,:,:], map_size, map_size, 0, 1)
            offset = offsets[ifrag]
            noffset = offset.shape[0]
            nrows, ncols = feat_maps.shape[2:] - map_size + 1
            
            px, py = np.arange(ncols), np.arange(nrows)
            for i in range(noffset-1, -1, -1):
                ox, oy, k = offset[i, :]
                px = ox + px*k
                py = oy + py*k
            
            py_all = np.repeat(py, ncols)
            px_all = np.tile(px, nrows)
            
            frags_cols.append(feat_maps_cols)
            patch_id_cols[index_start:index_start+nrows*ncols] = py_all*patch_num[1] + px_all
            index_start = index_start + nrows*ncols
            
        return frags_cols, patch_id_cols
        
    def gather_frags_fast(self, frags, offsets, img_size, patch_size, map_size):
        if len(map_size) > 1: # square maps
            map_size = map_size[0]
        patch_num = img_size - patch_size + 1
        N = frags[0].shape[0]
        frags_reunion = np.zeros((N*map_size*map_size, patch_num[0]*patch_num[1]))
        patch_id_reunion = np.zeros(patch_num[0]*patch_num[1], dtype='int')
        index_start = 0
        #indice_dict = {}
        padding, stride = 0, 1
        for ifrag in range(len(frags)):
            feat_maps = frags[ifrag][None,:,:,:]
            
            indice_key = '-'.join(['-'.join([str(se) for se in feat_maps.shape]), str(map_size), 
                        str(map_size), str(padding), str(stride)])
                        
            if indice_key in self.indice_dict:
                indice_map = self.indice_dict[indice_key]
            else:
                indice_map = im2col.get_im2col_indices(feat_maps.shape, map_size, map_size, 
                                                       padding, stride)
                self.indice_dict[indice_key] = indice_map
                
            feat_maps_cols = im2col.im2col_indices_cache(feat_maps, indice_map, map_size, map_size, 0, 1)
           
            offset = offsets[ifrag]
            noffset = offset.shape[0]
            nrows, ncols = feat_maps.shape[2:] - map_size + 1
            
            px, py = np.arange(ncols), np.arange(nrows)
            for i in range(noffset-1, -1, -1):
                ox, oy, k = offset[i, :]
                px = ox + px*k
                py = oy + py*k
            
            py_all = np.repeat(py, ncols)
            px_all = np.tile(px, nrows)
            #patch_id = py_all*patch_num[1] + px_all
            
            #frags_reunion[:, patch_id.astype(int)] = feat_maps_cols
            frags_reunion[:, index_start:index_start+nrows*ncols] = feat_maps_cols
            patch_id_reunion[index_start:index_start+nrows*ncols] = py_all*patch_num[1] + px_all
            index_start = index_start + nrows*ncols
            
        return frags_reunion, patch_id_reunion           
            
        
    def max_pooling_frags(self, frags_in, in_offset, k):
         k2 = k*k
         x,y = np.meshgrid(range(k),range(k))
         pooling_offset = np.concatenate((y.reshape(1,k2), x.reshape(1,k2))).T
         ks = k*np.ones(k2).reshape(k2,1)
         pooling_offset = np.concatenate((pooling_offset, ks), axis=1)
         
         frags_out = []
         out_offset = []
         for i in range(len(frags_in)):
             frag_in = frags_in[i]
             for j in range(k2):
                 frags_out.append(self.max_pooling_frag(frag_in, pooling_offset[j,:], k))
                 new_offset = pooling_offset[j,:].reshape(1,3)
                 if not in_offset: # empty
                     out_offset.append(new_offset)
                 else:
                     curr_offset = np.concatenate((in_offset[i], new_offset))
                     out_offset.append(curr_offset)
                 
         return frags_out, out_offset
    
    def get_sub_img(self, img, hDivs, wDivs, ih, iw, divsize, overlap):
        # used for parallel processing
        H, W, Channels = img.shape
        start_h = ih*divsize
        if ih == hDivs-1:
            end_h = H
        else:
            end_h = start_h + divsize + 2*overlap            
        
        start_w = iw*divsize
        if iw == wDivs-1:
            end_w = W
        else:
            end_w = start_w + divsize + 2*overlap
            
        sub_img = img[start_h:end_h, start_w:end_w, :]
        return sub_img
        
    def divide_image(self, img, divsize, overlap):
        H, W, Channels = img.shape
        if H > divsize or W > divsize:
            divs = {(0,0):img}
            return divs
        hDivs, wDivs = np.floor(H/divsize), np.floor(W/divsize)
        divs = {}
        for ih in range(hDivs):
            for iw in range(wDivs):
                divs[(ih, iw)] = self.get_sub_img(img, ih, iw, divsize, overlap)
        return divs
    
    def predictImageOneDivide(self, img, hDivs, wDivs, ih, iw, divsize, overlap):
        sub_img = self.get_sub_img(img, hDivs, wDivs, ih, iw, divsize, overlap)
        edgemap = self.predictImage(sub_img)
        return edgemap
        
    def predictImageWithDivide(self, img, divsize):
        # pad image first
        H0, W0, Channels = img.shape
        edgemap = np.zeros((H0,W0))
        if H0 < divsize or W0 < divsize:
            # divide is not needed
            return self.predictImage(img, padding = True)
        
        img = self.pad_img(img)
        H, W, Channels = img.shape
        
        hDivs, wDivs = int(np.floor(H/divsize)), int(np.floor(W/divsize))
        print 'There are totally %d divisions' %(hDivs*wDivs)
        for ih in range(hDivs):
            for iw in range(wDivs):
                sub_edgemap = self.predictImageOneDivide(img, hDivs, wDivs, ih, iw, divsize, self.padsz)
                #plt.imshow(sub_edgemap, cmap = plt.get_cmap('gray'))
                #plt.show()
                sub_H, sub_W = sub_edgemap.shape
                #print sub_H, sub_W, divsize
                edgemap[ih*divsize:ih*divsize+sub_H, iw*divsize:iw*divsize+sub_W] = sub_edgemap
        return edgemap
    
    def convFast(self, x, k, b, nonlinear_type, stride = 1, pad = 0):
        x = x[None,:,:,:]        
        N, C, H, W = x.shape
        num_filters, _, filter_height, filter_width = k.shape
        
        # Check dimensions
        assert (W + 2 * pad - filter_width) % stride == 0, 'width does not work'
        assert (H + 2 * pad - filter_height) % stride == 0, 'height does not work'
        
        # Create output
        out_height = (H + 2 * pad - filter_height) / stride + 1
        out_width = (W + 2 * pad - filter_width) / stride + 1
        out = np.zeros((N, num_filters, out_height, out_width))
        
        x_cols = im2col.im2col_indices(x, k.shape[2], k.shape[3], pad, stride)
        #print x_cols.shape
        #x_cols = im2col_cython(x, k.shape[2], k.shape[3], pad, stride)
        res = k.reshape((k.shape[0], -1)).dot(x_cols) + b.reshape(-1, 1)
        
        out = res.reshape(k.shape[0], out.shape[2], out.shape[3], x.shape[0])
        out = out.transpose(3, 0, 1, 2)
        
        out = self.nonlinear_unit(np.squeeze(out), nonlinear_type)
        return out
    
    def convSimple(self, frag, k, b, nonlinear_type):
        outputmaps, inputmaps, kernelsize, _ = k.shape
        fraginmaps, fraginH, fraginW = frag.shape
        assert fraginmaps == inputmaps
        fragoutH, fragoutW = fraginH - kernelsize + 1, fraginW - kernelsize + 1
        a = np.zeros((outputmaps, fragoutH, fragoutW))    
        for kout in xrange(outputmaps):
            for kin in xrange(inputmaps):
                a[kout, :, :] += sg.convolve(frag[kin, :, :], k[kout, kin, :, :], 'valid')
            a[kout, :, :] = self.nonlinear_unit(a[kout, :, :] + b[kout], nonlinear_type)
        return a
    
    def predictImageFast(self, img, padding = False):
        frags, offset = [], []
        for i in xrange(len(self.model)):
            layer = self.model[i]
            print 'processing layer %s ...' %layer['type']
            tstart = datetime.now()
            #print 'layer %s' %layer['type']
            if layer['type'] == 'i':
                if padding:
                    img = self.pad_img(img)
                x = np.transpose(img, (2,0,1))
                imgChannels, imgH, imgW = x.shape
                imgsize = np.array([imgH, imgW])
                patchsize = layer['mapsize']
                patchNum = imgsize - patchsize + 1
                frags.append(x)
            elif layer['type'] == 'c': # convolutional layer 
            
                k = layer['k']
                b = layer['b']
                
                out_frags = []
                for frag in frags:
                    out_frags.append(self.convFast(frag, k, b, layer['nonlinear']))
                frags = out_frags
                
            elif layer['type'] == 's': # pooling layer
                frags, offset = self.max_pooling_frags(frags, offset, layer['scale'])
            elif layer['type'] == 'f' or layer['type'] == 'o':
                if self.model[i-1]['type'] != 'f': # column vector
                   gatherstart = datetime.now()
                   rsps_prev, patchid_order = self.frags2cols(frags, offset, imgsize, patchsize, self.model[i-1]['mapsize'])
                   gatherend = datetime.now()
                   print 'it took gather %d seconds' %(gatherend - gatherstart).seconds
                
                rsps = []       
                for rsp in rsps_prev:
                    rsps.append(self.nonlinear_unit(np.dot(layer['W'], rsp) + layer['b'].reshape(layer['W'].shape[0], 1), 
                                                    layer['nonlinear']))
                    
                rsps_prev = rsps
                
            else:
                print 'not supported currently'
            
            tend = datetime.now()
            print 'it took %s layer %d seconds' %(layer['type'], (tend - tstart).seconds)
        # reshape into 2d array
        edgemap = np.zeros(len(patchid_order), dtype='float')
        index = 0
        for rsp in rsps_prev:
            edgemap[index:index+rsp.shape[1]] = rsp[1, :]
            index = index + rsp.shape[1]
        edgemap = edgemap[np.argsort(patchid_order)].reshape(patchNum[0], patchNum[1])
        return edgemap
           
    def predictImage(self, img, padding = False):
        frags, offset = [], []
        for i in xrange(len(self.model)):
            layer = self.model[i]
            print 'processing layer %s ...' %layer['type']
            tstart = datetime.now()
            #print 'layer %s' %layer['type']
            if layer['type'] == 'i':
                if padding:
                    img = self.pad_img(img)
                x = np.transpose(img, (2,0,1))
                imgChannels, imgH, imgW = x.shape
                imgsize = np.array([imgH, imgW])
                patchsize = layer['mapsize']
                patchNum = imgsize - patchsize + 1
                frags.append(x)
            elif layer['type'] == 'c': # convolutional layer 
            
                k = layer['k']
                b = layer['b']
                
                out_frags = []
                for frag in frags:
                    out_frags.append(self.convFast(frag, k, b, layer['nonlinear']))
                frags = out_frags
                
            elif layer['type'] == 's': # pooling layer
                frags, offset = self.max_pooling_frags(frags, offset, layer['scale'])
            elif layer['type'] == 'f' or layer['type'] == 'o':
                if self.model[i-1]['type'] != 'f': # column vector
                   gatherstart = datetime.now()
                   rsp, patchid_order = self.gather_frags_fast(frags, offset, imgsize, patchsize, self.model[i-1]['mapsize'])
                   gatherend = datetime.now()
                   print 'it took gather %d seconds' %(gatherend - gatherstart).seconds
                
                
                rsp = self.nonlinear_unit(np.dot(layer['W'], rsp) + layer['b'].reshape(layer['W'].shape[0], 1), layer['nonlinear'])
                
#                M, N, K = layer['W'].shape[0], rsp.shape[1], layer['W'].shape[1]
#                print layer['W'].shape, rsp.shape
#                wxrsp = np.zeros([M, N], order='F')
#                blas = cublas.Blas()
#                blas.gemm('T', 'T', M, N, K, 1.0, layer['W'], rsp, .0, wxrsp)
#                
#                rsp = self.nonlinear_unit(wxrsp + layer['b'].reshape(layer['W'].shape[0], 1), layer['nonlinear'])
                                
            else:
                print 'not supported currently'
            
            tend = datetime.now()
            print 'it took %s layer %d seconds' %(layer['type'], (tend - tstart).seconds)
        # reshape into 2d array
        print rsp.shape
        rsp = 1.0 - rsp[0, np.argsort(patchid_order)].reshape(patchNum[0], patchNum[1])
        return rsp
                
    def predictPatch(self, x):
        a_prev = x
        for i in xrange(len(self.model)):
            layer = self.model[i]
            #print 'layer %s' %layer['type']
            if layer['type'] == 'i':
                pass
            elif layer['type'] == 'c': # convolutional layer 
                outputmaps = layer['outputmaps']
                inputmaps  = self.model[i-1]['outputmaps']
                mapsize = layer['mapsize']
            
                a = np.zeros((outputmaps, mapsize[0], mapsize[1]))
                k = layer['k']
                b = layer['b']
                
                for kout in xrange(outputmaps):
                    for kin in xrange(inputmaps):
                        a[kout, :, :] += sg.convolve(a_prev[kin, :, :], k[kout, kin, :, :], 'valid')
                    a[kout, :, :] = self.nonlinear_unit(a[kout, :, :] + b[kout], layer['nonlinear'])
                #print a
                a_prev = a
                
            elif layer['type'] == 's': # pooling layer
                a_prev = self.fast_max_pooling(a_prev, layer['scale'])
            elif layer['type'] == 'f' or layer['type'] == 'o':
                if self.model[i-1]['type'] != 'f': # column vector
                   n,h,w = a_prev.shape
                   a_prev = a_prev.flatten()
                
                a_prev = self.nonlinear_unit(np.dot(layer['W'], a_prev) + layer['b'], layer['nonlinear'])
            else:
                print 'not supported currently'   
            
        return a_prev[1]
    
    def nonlinear_unit(self, x, non_linear_type):
        if non_linear_type == 3: # rectifier
            x = np.maximum(x, 0)
        elif non_linear_type == 4: # softmax
            xe = np.exp(x)
            x = np.divide(xe, np.sum(xe, axis=0))
        else:
            print "not used now"
        
        return x
            
    def parse_cnn_layer(self, mlayer):
        dict = {}
        layer_type = mlayer['type'][0,0][0]
        dict['type'] = layer_type
        
        
        if layer_type == 'i':
            # this input layer
            dict['outputmaps'] = mlayer['outputmaps'][0,0][0,0]
            dict['mapsize'] = mlayer['mapsize'][0,0][0]
            #pass
        elif layer_type == 'c':
            # this convolutional layer        
            kernelsize = mlayer['kernelsize'][0,0][0,0]
            dict['kernelsize'] = kernelsize
            dict['nonlinear'] = mlayer['nonlinear'][0,0][0,0]
            dict['neurons'] = mlayer['neurons'][0,0][0,0]
            dict['outputmaps'] = mlayer['outputmaps'][0,0][0,0]
            dict['mapsize'] = mlayer['mapsize'][0,0][0]
            # parse weight k 
            matk = mlayer['k'][0,0]
            n_out, n_in = len(matk[0]), len(matk[0,0][0])
            k = np.zeros((n_out, n_in, kernelsize, kernelsize))
            for i in xrange(n_out):
                ki = matk[0,i]
                for j in xrange(n_in):
                    k[i,j,:,:] = ki[0,j]
            # parse b
            matb = mlayer['b'][0,0]
            b = np.zeros((n_out,1))
            for i in xrange(n_out):
                b[i] = np.squeeze(np.asarray(matb[0,i]))
            
            dict['k'] = k
            dict['b'] = b
        elif layer_type == 's':
            # this is pooling layer
            dict['neurons'] = mlayer['neurons'][0,0][0,0]
            dict['scale'] = mlayer['scale'][0,0][0,0]
            dict['outputmaps'] = mlayer['outputmaps'][0,0][0,0]
            dict['mapsize'] = mlayer['mapsize'][0,0][0]
        elif layer_type == 'f' or layer_type == 'o':
            # this is fully connected layer
            dict['nonlinear'] = mlayer['nonlinear'][0,0][0,0]
            dict['neurons'] = mlayer['neurons'][0,0][0,0]
            # parse W
            dict['W'] = mlayer['W'][0,0]
            # parse b
            dict['b'] = np.squeeze(mlayer['b'][0,0])
        else:
            # layer not supported
            print "layer %s is not supported now." %layer_type
            exit(1)
        return dict
    
    def load_matcnn(self, cnnmat_path):
        
        cnnmat = sio.loadmat(cnnmat_path)
        layers = cnnmat['cnn']['layers'][0,0]
        model = []
        for i in xrange(len(layers[0])):
            #print "parsing layer %d" %i
            model.append(self.parse_cnn_layer(layers[0,i]))
            
        return model

if __name__ == "__main__":
    #load model
    cnnModel = cnnScan('muscle-deepcontour_2_im2col.mat')
    print 'test image scanner'
    #load test image
    img = misc.imread('Bmal1 WT#17-3_1.jpg')
    plt.imshow(img)
    plt.show()
    # run cnn scan
    print 'testing fast scanning ...'
    tstart = datetime.now()
    edgemap =  cnnModel.predictImage(img/255.0, padding = True)
    misc.toimage(edgemap, cmin=0.0, cmax=1.0).save('edgemap.png')
    tend = datetime.now()
    print 'it took fast scanning %d seconds' %(tend - tstart).seconds
    # show result
    plt.imshow(edgemap, cmap = plt.get_cmap('gray'))
    plt.show()    
    
    #print img.shape, edgemap.shape