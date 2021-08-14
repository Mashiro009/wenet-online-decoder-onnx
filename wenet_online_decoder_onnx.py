#!/usr/bin/env python

# authors: Mashiro009

# modifed from https://github.com/jiay7/wenet_onlinedecode

# from __future__ import print_function

import argparse
import copy
import logging
import os
import sys
import time
import numpy as np
import torch
import yaml
import torchaudio.compliance.kaldi as kaldi
import onnxruntime
from tqdm import tqdm
import struct
import webrtcvad


from wenet.utils.common import (IGNORE_ID, add_sos_eos, log_add,
                                remove_duplicates_and_blank, th_accuracy)
# from python_speech_features import logfbank
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format="%(levelname)8s %(filename)s:%(lineno)d %(asctime)s %(message)s ")

def np_topk(x,k,dim=0):
    """
        return numpy array topk index. don't finish

    Args:
        x (numpy.array) : numpy array
        k (int): top k num.
        dim (int): which dimension

    Returns:
        array: topk index

    Examples:

    """
    topk_index = np.argsort(-x,axis=dim)[:k]


class WeNetDecoder:
    def __init__(self,conf_file):
        with open(conf_file, 'r') as fin:
            configs = yaml.load(fin, Loader=yaml.FullLoader)

        # store configs
        self.conf_file = configs
        
        # Define online feature pipeline
        feat_config = configs.get("collate_conf", {})
        # self.sr = feat_config["sample_rate"]
        self.sr = feat_config.get("sample_rate",16000)
        self.feature_type = feat_config["feature_extraction_conf"]["feature_type"]
        self.frame_shift = feat_config["feature_extraction_conf"]["frame_shift"]
        self.frame_length = feat_config["feature_extraction_conf"]["frame_length"]
        self.using_pitch = feat_config["feature_extraction_conf"]["using_pitch"]
        self.mel_bins = feat_config["feature_extraction_conf"]["mel_bins"]

        # VAD module
        self.max_sil_frame = 50

        # Wenet model
        # self.model = self.get_wenetmodel(configs)
        # load_checkpoint(self.model, configs["checkpoint"])
        use_cuda = configs.get("gpu","-1") != "-1" and torch.cuda.is_available()
        if use_cuda:
            os.environ["CUDA_VISIBLE_DEVICES"] = configs.get("gpu","-1")
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        # self.model = self.model.to(self.device)
        # self.model.eval()

        # decode mode
        self.decode_mode = configs["decode_mode"]

        # dict model
        self.dict_model,self.eos = self.get_dict(configs)

        # buffer
        self.decoding_chunk_size = 16
        input_layer_type = configs["encoder_conf"]["input_layer"]
        if input_layer_type == "linear":
            right_context = 0
            subsampling_rate = 1
        elif input_layer_type == "conv2d":
            right_context = 6
            subsampling_rate = 4
        elif input_layer_type == "conv2d6":
            right_context = 14
            subsampling_rate = 6
        elif input_layer_type == "conv2d8":
            right_context = 14
            subsampling_rate = 5
        else:
            raise ValueError("unknown input_layer: " + input_layer_type)
        # self.context = self.model.encoder.embed.right_context + 1 
        self.context = right_context + 1 
        # self.decoding_window = self.model.encoder.embed.subsampling_rate * int(self.decoding_chunk_size - 1) + self.context
        self.decoding_window = subsampling_rate * int(self.decoding_chunk_size - 1) + self.context
        # self.stride = int(self.model.encoder.embed.subsampling_rate*self.decoding_chunk_size)
        self.stride = int(subsampling_rate*self.decoding_chunk_size)
        self.sig_buffer = np.zeros(400) # signal history buffer
        self.bytes_offline_notfinished_buffer = bytes() # offline bytes buffer for last buffer not finished signal
        self.dat_buffer = [] # maybe put feature in    feature buffer
        self.max_dat_buffer_block = 5000 # feature buffer max len
        self.chunk_size = 1600 # signal chunk size

        # onnx model
        onnx_model_dir = configs["onnx_dir"]
        self.encoder_ort_session = onnxruntime.InferenceSession(os.path.join(onnx_model_dir,'encoder_chunk.onnx'))
        self.ctc_ort_session = onnxruntime.InferenceSession(os.path.join(onnx_model_dir,'ctc.onnx'))
        self.decoder_ort_session = onnxruntime.InferenceSession(os.path.join(onnx_model_dir,'decoder.onnx'))

        # cache
        # self.subsampling_cache: Optional[torch.Tensor] = None
        # self.elayers_output_cache: Optional[List[torch.Tensor]] = None
        # self.conformer_cnn_cache: Optional[List[torch.Tensor]] = None
        # self.subsampling_cache = self.to_numpy(torch.rand(1,1,256))
        # self.elayers_output_cache = self.to_numpy(torch.rand(12,1,1,256))
        # self.conformer_cnn_cache = self.to_numpy(torch.rand(12,1,256,15))
        self.subsampling_cache = np.zeros((1,1,256), dtype = np.float32, order = 'C')
        self.elayers_output_cache = np.zeros((12,1,1,256), dtype = np.float32, order = 'C')
        self.conformer_cnn_cache = np.zeros((12,1,256,15), dtype = np.float32, order = 'C')
        # self.offset = 0
        # self.offset = self.to_numpy(torch.tensor(1,dtype=torch.int64))
        self.offset = np.array(1,dtype = np.int64)
        # self.required_cache_size = self.decoding_chunk_size * 50
        self.required_cache_size = -1
        self.cache_size_max = 400
        self.outputs = [] # maybe put encoder output
        self.num_decode = 0

        # set sos eos index
        self.sos = configs["output_dim"] - 1 
        self.eos = configs["output_dim"] - 1


        # offline signal
        self.offline_signal = np.zeros(20)


        #online decode setting
        self.rescoring = configs["rescoring"]
        if self.rescoring:
            self.ctc_weight = float(configs["model_conf"]["ctc_weight"])
            
        if self.decode_mode == "ctc_prefix_beam_search":
            self.beam,self.time_step,self.cur_hyps = self.setting_ctc_prefix_search(configs)
        elif self.decode_mode == "ctc_greedy_search":
            self.time_step,self.cur_result = self.setting_ctc_greedy_search(configs)


    def to_numpy(self,tensor):
        """ change tensor to numpy

        Args:
            tensor (Tensor)
            
        Returns:
            array (numpy.array)
        
        """
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    def setting_ctc_greedy_search(self,configs):
        """ setting_ctc_prefix_search

        Args:
            configs (dict): the dict from config file
            
        Returns:
            time_step (int): not use
            cur_result (List): List means the sequence of the ctc output
        
        """
        time_step = 0
        cur_result = []
        return time_step,cur_result

    def setting_ctc_prefix_search(self,configs):
        """ setting_ctc_prefix_search

        Args:
            configs (dict): the dict from config file
            
        Returns:
            beam (int): beam size
            time_step (int): not use
            cur_hyps (List[ Tuple(Tuple1, Tuple2(float1, float2))]): List means nbest, 
                Tuple1 means 规整字符串
                float1 p_b(L)表示所有以blank结尾且规整后是L的各ctc字符串的概率之和
                float2 p_nb(L)表示所有以非blank结尾且规整后是L的各ctc字符串的概率之和
                details see http://placebokkk.github.io/asr/2020/02/01/asr-ctc-decoder.html
        
        """
        # beam = configs["beam"]
        beam = configs.get("beam",10)
        time_step = 0
        cur_hyps = [(tuple(), (0.0, -float('inf')))]
        return beam,time_step,cur_hyps

    def get_dict(self,configs):
        """ get the vocabulary

        Args:
            configs (dict): the dict from config file
            
        Returns:
            char_dict (dict): vocabulary dict, key=id, value=word
            eos (int): the id of <sos/eos>
        
        """
        char_dict = {}
        with open(configs["dict"], 'r') as fin:
            for line in fin:
                arr = line.strip().split()
                assert len(arr) == 2
                char_dict[int(arr[1])] = arr[0]
        eos = len(char_dict) - 1
        return char_dict,eos

    def _extract_feature(self,waveform):
        """ real extract acoustic features 
            get waveform and return fbank

        Args:
            waveform (numpy.array): (Time,) the signal
            
        Returns:
            feats (numpy.array): (FrameTime, dim=80)
        
        """
        waveform = torch.from_numpy(np.expand_dims(waveform,0))
        mat = kaldi.fbank(
                    waveform,
                    num_mel_bins=self.mel_bins,
                    frame_length=self.frame_length,
                    frame_shift=self.frame_shift,
                    dither=0,
                    energy_floor=0.0,
                    sample_frequency=self.sr
                )
        # mat = logfbank(waveform,nfilt=80)
        mat = mat.detach().numpy()
        return mat

    def extract_feat(self, cur_signal, pre_signal):
        """ extract acoustic features 
            concatenate pre_signal and cur_signal
            extract feats and don't use the end (less than a frame ) of the signal
        Args:
            cur_signal (numpy.array): (Time1,) the signal new
            pre_signal (numpy.array): (Time2,) the signal stored in self.sig_buffer
            
        Returns:
            feats (numpy.array): (FrameTime, dim=80)
            signal (numpy.array): (Time1+Time2, ) the concatenation of pre_signal and cur_signal
        
        """
        if pre_signal is not None:
            signal = np.concatenate([pre_signal, cur_signal])
            rest_points = int((len(signal) - (0.015 * self.sr)) % (0.01 * self.sr))
            run_signal = signal if rest_points == 0 else signal[:-rest_points]
            feats = self._extract_feature(run_signal).astype('float32')
            feats = feats[1:,:]
        else:
            signal = cur_signal
            rest_points = int((len(signal) - (0.015 * self.sr)) % (0.01 * self.sr))
            run_signal = signal[:-rest_points]
            feats = self._extract_feature(run_signal).astype('float32')
  
        return feats, signal

    def buffer_signal(self, signal):
        """ buffer_signal
            called in detect_process \n
            store some signal that is less than a frame in sig_buffer and Add a frame further ahead
        Args:
            signal (numpy.array): (Time,)
            
        Returns:
            Nothing
        """
        rest_points = (len(signal) - (0.015 * self.sr)) % (0.01 * self.sr)
        buf_sig_len = int((0.025 * self.sr) + rest_points) # protect frame
        return signal[-buf_sig_len:]

    def detect_process(self, signal):
        """ detect_process
            called in detect \n
            we extract the feats of signal then put it into buffer and store some signal that is less than a frame in sig_buffer

        Args:
            signal (numpy.array): (Time,)
            
        Returns:
            Nothing
        """
        # extract acoustic features
        feat, _signal_buf = self.extract_feat(signal, self.sig_buffer) #signal和self.sig_buffer拼起来提特征，然后取第二帧及以后
        self.dat_buffer.append(feat) #self.dat_buffer存储了所有特征，以list形式，每个元素是feat矩阵
        if len(self.dat_buffer) > self.max_dat_buffer_block:
            self.dat_buffer.pop(0)
        self.sig_buffer = self.buffer_signal(_signal_buf) #把_signal_buf取最后一帧的采样点

    def reset(self):
        """ reset
            we reset sig_buffer, dat_buffer, \n
                subsampling_cache, elayers_output_cache, conformer_cnn_cache, offset,\n
                outputs(encoder_outputs), \n
                num_decode, self.beam, self.time_step, self.cur_hyps \n
                self.time_step maybe not used

        Args:
            Nothing
            
        Returns:
            Nothing
        """
        # buffer
        self.sig_buffer = np.zeros(400)
        self.dat_buffer = []
        # self.max_dat_buffer_block = 5000

        # cache
        # self.subsampling_cache = None
        # self.elayers_output_cache = None
        # self.conformer_cnn_cache = None
        self.subsampling_cache = np.zeros((1,1,256), dtype = np.float32, order = 'C')
        self.elayers_output_cache = np.zeros((12,1,1,256), dtype = np.float32, order = 'C')
        self.conformer_cnn_cache = np.zeros((12,1,256,15), dtype = np.float32, order = 'C')
        self.outputs = []
        # self.offset = 0
        self.offset = np.array(1,dtype = np.int64)

        self.num_decode = 0

        # self.time_step = 0
        # self.cur_hyps = [(tuple(), (0.0, -float('inf')))]

        #online decode setting
        # 在这里把self.cur_hyps清空
        if self.decode_mode == "ctc_prefix_beam_search":
            self.beam,self.time_step,self.cur_hyps = self.setting_ctc_prefix_search(self.conf_file)
        elif self.decode_mode == "ctc_greedy_search":
            self.time_step,self.cur_result = self.setting_ctc_greedy_search(self.conf_file)


    def dat_buffer_read(self,decoding_window,stride):
        """ read all the feats from buffer and clean the buffer

        Args:
            Nothing
            
        Returns:
            (Bool): if we get feats or not
            feats (numpy.array): (Time, Dim=80) the fbank feats
        """
        if self.dat_buffer == []:
            return False,-1
        feats = np.concatenate(self.dat_buffer)
        if feats.shape[0] < decoding_window:
            return False,-1
        else:
            feats_to_predict = feats[:decoding_window]
            self.dat_buffer = [feats[stride:]]
            return True,feats_to_predict
    
    def dat_buffer_read_all(self):
        """ read all the feats from buffer and clean the buffer

        Args:
            Nothing
            
        Returns:
            (Bool): if we get feats or not
            feats (numpy.array): (Time, Dim=80) the fbank feats
        """
        if self.dat_buffer == []:
            return False,-1
        feats = np.concatenate(self.dat_buffer)
        self.dat_buffer = []
        return True,feats

    # def ctc_greedy_search_atten(self):

    #     while True:
    #         more_data,feats_to_predict = self.dat_buffer_read(self.decoding_window,self.stride)
    #         if not more_data:
    #             break
    #         feats_to_predict = torch.from_numpy(np.expand_dims(feats_to_predict,0)).float()
    #         y, self.subsampling_cache, self.elayers_output_cache,self.conformer_cnn_cache = self.model.encoder.forward_chunk(feats_to_predict, self.offset,
    #                             self.required_cache_size,self.subsampling_cache,self.elayers_output_cache,self.conformer_cnn_cache)
            
    #         subsampling_cache_shape = [item.shape for item in self.subsampling_cache]
    #         elayers_output_cache_shape = [item.shape for item in self.elayers_output_cache]
    #         conformer_cnn_cache_shape = [item.shape for item in self.conformer_cnn_cache]
    #         self.outputs.append(y)
    #         ys = torch.cat(self.outputs, 1)
    #         self.offset += y.size(1)
    #         ctc_probs = self.model.ctc.log_softmax(ys)
    #         topk_prob, topk_index = ctc_probs.topk(1, dim=2)
    #         topk_index = topk_index.view(ys.shape[1],)
    #         hyps = topk_index.tolist()
    #         hyps = remove_duplicates_and_blank(hyps)
    #         content = [self.dict_model[index] for index in hyps]
    #         if ''.join(content) == '':
    #             continue
    #         print(''.join(content))
    #         self.num_decode += 1

    def ctc_greedy_search_purn(self):
        """ decode feats
            we continues run this code until the buffer is empty \n
            one step we get feats(the shape is controled by decode_window) from buffer,
                and use encode,ctc process the feats and get encoder_out and ctc_out \n
            then we endpoint if find endpoint we will rescore \n
            if no endpoint we use ctc_greedy_search to get nbest(beam) result and show \n
            by the way we use keep_cache_size_moderate in this function

        Args:
            Nothing
            
        Returns:
            Nothing
        """

        while True:
            get_data,feats_to_predict = self.dat_buffer_read(self.decoding_window,self.stride)
            if not get_data:
                break
            feats_to_predict = torch.from_numpy(np.expand_dims(feats_to_predict,0)).float()
            # # y, self.subsampling_cache, self.elayers_output_cache,self.conformer_cnn_cache = self.model.encoder.forward_chunk(feats_to_predict, self.offset,
            # #                     self.required_cache_size,self.subsampling_cache,self.elayers_output_cache,self.conformer_cnn_cache)
            
            # # subsampling_cache_shape = [item.shape for item in self.subsampling_cache]
            # # elayers_output_cache_shape = [item.shape for item in self.elayers_output_cache]
            # # conformer_cnn_cache_shape = [item.shape for item in self.conformer_cnn_cache]
            # # encoder_out = y
            # # self.offset += y.size(1)
            # ort_inputs = {self.encoder_ort_session.get_inputs()[0].name: self.to_numpy(feats_to_predict),
            #     self.encoder_ort_session.get_inputs()[1].name: self.offset,
            #     # ort_session.get_inputs()[2].name: to_numpy(required_cache_size),
            #     self.encoder_ort_session.get_inputs()[2].name: self.subsampling_cache,
            #     self.encoder_ort_session.get_inputs()[3].name: self.elayers_output_cache,
            #     #   ort_session.get_inputs()[5].name: to_numpy(conformer_cnn_cache)
            #     }
            # ort_outs = self.encoder_ort_session.run(None, ort_inputs)
            # y = torch.Tensor(ort_outs[0])
            # # self.offset += ort_outs[0].shape[1]
            # self.subsampling_cache = ort_outs[1]
            # self.elayers_output_cache = ort_outs[2]

            # self.offset += y.size(1)
            y = self.get_encoder_output(feats_to_predict=feats_to_predict)
            encoder_out = y

            self.keep_cache_size_moderate()

            # # ctc_probs = self.model.ctc.log_softmax(encoder_out)
            # ort_inputs = {self.ctc_ort_session.get_inputs()[0].name: self.to_numpy(encoder_out),
            #     }
            # ctc_probs = self.ctc_ort_session.run(None, ort_inputs)[0]
            # ctc_probs = torch.Tensor(ctc_probs)
            ctc_probs = self.get_ctc_output(encoder_out=encoder_out)
            topk_prob, topk_index = ctc_probs.topk(1, dim=2)
            topk_index = topk_index.view(encoder_out.shape[1],)
            hyps = topk_index.tolist()
            hyps = remove_duplicates_and_blank(hyps)
            content = [self.dict_model[index] for index in hyps]
            if ''.join(content) == '':
                continue
            self.cur_result = self.cur_result + content
            # print(''.join(self.cur_result))
            logger.debug("Partial: %s",''.join(self.cur_result))
            self.num_decode += 1

    def decoder_rescoring(self,noLog=False):
        """ decoder rescoring
            we rescore the ctc output use teacher force \n
            the nbest sequence don't change, just rescore and find the new best
            we will return partial final result and final result likelihood

        Args:
            noLog (Bool): if true, we don't print log, default False
            
        Returns:
            Final result (string): the result after attention decoder teacher force rescoring. It is the best one
            Final result likelihood (float): the likelihood(score) of the Final result. 
            ( There is some bug, the Returns sometime maybe None )
        """
        if len(self.outputs) == 0:
            return
        encoder_out = torch.cat(self.outputs, 1)
        encoder_out = encoder_out.to(self.device)
        hyps = [(y[0], log_add([y[1][0], y[1][1]])) for y in self.cur_hyps]
        assert len(hyps) == self.beam
        hyps_pad = pad_sequence([
            torch.tensor(hyp[0], device=self.device, dtype=torch.long)
            for hyp in hyps
        ], True, -1) 
        hyps_lens = torch.tensor([len(hyp[0]) for hyp in hyps],
                                 device=self.device,
                                 dtype=torch.long)
        
        # hyps_pad, _ = add_sos_eos(hyps_pad, self.model.sos, self.model.eos, -1)
        hyps_pad, _ = add_sos_eos(hyps_pad, self.sos, self.eos, -1)
        hyps_lens = hyps_lens + 1
        encoder_out = encoder_out.repeat(self.beam, 1, 1)
        encoder_mask = torch.ones(self.beam,1,encoder_out.size(1),dtype=torch.bool,device=self.device)
        # with torch.no_grad():
        #     decoder_out,_ , _ = self.model.decoder(encoder_out, encoder_mask, hyps_pad,hyps_lens)
        # decoder_out = torch.nn.functional.log_softmax(decoder_out, dim=-1)
        decoder_out = self.get_decoder_output(encoder_out,encoder_mask,hyps_pad,hyps_lens)
        # ort_inputs = {self.decoder_ort_session.get_inputs()[0].name: self.to_numpy(encoder_out),
        #         self.decoder_ort_session.get_inputs()[1].name: self.to_numpy(encoder_mask),
        #         self.decoder_ort_session.get_inputs()[2].name: self.to_numpy(hyps_pad),
        #         self.decoder_ort_session.get_inputs()[3].name: self.to_numpy(hyps_lens),
        #         }
        # ort_outs = self.decoder_ort_session.run(None, ort_inputs)
        # decoder_out = torch.Tensor(ort_outs[0])
        # decoder_out = torch.nn.functional.log_softmax(decoder_out, dim=-1)
        decoder_out = decoder_out.detach().cpu().numpy() if decoder_out.requires_grad else decoder_out.cpu().numpy()
        best_score = -float('inf')
        best_index = 0
        for i, hyp in enumerate(hyps):
            score = 0.0
            for j, w in enumerate(hyp[0]):
                score += decoder_out[i][j][w]
            score += decoder_out[i][len(hyp[0])][self.eos]
            score += hyp[1] * self.ctc_weight
            if score > best_score:
                best_score = score
                best_index = i
        likelihood = hyps[best_index][1] if hyps[best_index][1] != None else -float("inf")
        hyps_best = hyps[best_index][0]
        content = [self.dict_model[index] for index in hyps_best]
        # print("Rescoing: "+''.join(content))
        if not noLog:
            logger.info("Rescoing: "+''.join(content))
        return ''.join(content),likelihood

    def endpoint(self):
        """ check if there is endpoint
            we use several output of encoder and use ctc process it \n
            if there is too many blank in the start of ctc_output we will find silence begining \n
            if there is too many blank in the end of ctc_output we will find endpoint then return true \n
            too many means greater than self.max_sil_frame ( the number is after subsampling ) \n
            if we find silence or endpoint, we will drop some encoder_out (equals to self.max_sil_frame) \n
            reference see https://zhuanlan.zhihu.com/p/367223398 (we don't completely reproduct the reference)
        Args:
            Nothing
            
        Returns:
            find_endpoint (Bool): if we find endpoint or not
        """
        output_step = self.max_sil_frame // self.decoding_chunk_size
        encoder_out = torch.cat(self.outputs, 1)
        if encoder_out.shape[1] < self.max_sil_frame:
            return False
        # # ctc_probs = self.model.ctc.log_softmax(encoder_out)
        # ort_inputs = {self.ctc_ort_session.get_inputs()[0].name: self.to_numpy(encoder_out),
        #         }
        # ctc_probs = self.ctc_ort_session.run(None, ort_inputs)[0]
        # ctc_probs = torch.Tensor(ctc_probs)
        ctc_probs = self.get_ctc_output(encoder_out=encoder_out)
        ctc_probs = torch.exp(ctc_probs)[0,:,0]
        is_blank = ctc_probs > 0.5
        if sum(is_blank[:self.max_sil_frame]) == self.max_sil_frame:
            # print("beginging sil....")
            logger.info("beginging sil....")
            self.outputs = self.outputs[output_step:]
        elif sum(is_blank[-1 * self.max_sil_frame:]) == self.max_sil_frame:
            self.outputs = self.outputs[:-output_step]
            # print("endpoint detect!...")
            logger.info("endpoint detect!...")
            return True
        return False

    def ctc_prefix_beam_search_purn_onestep(self):
        """ decode feats one step
            one step we get feats(the shape is controled by decoding_window) from buffer,
                and use encode,ctc process the feats and get encoder_out and ctc_out \n
            then we endpoint if find endpoint we will rescore \n
            if no endpoint we use ctc_prefix_beam_search to get nbest(beam) result and show \n
            by the way we use keep_cache_size_moderate in this function

        Args:
            Nothing
            
        Returns:
            continue_run (Bool): will run this code again or not. 
                (If true, maybe we get real feats from buffer, maybe we don't find endpoint)
            Partial result (string): the result after ctc prefix_beam_search
            Final result (string): the result after attention decoder teacher force rescoring. It is the best one
            Final result likelihood (float): the likelihood(score) of the Final result. 
        """
        get_data,feats_to_predict = self.dat_buffer_read(self.decoding_window,self.stride)
        if not get_data:
            return False, "", "", -float('inf')
        logger.debug("ctc_prefix_beam_search read %s data",str(feats_to_predict.shape))
        feats_to_predict = torch.from_numpy(np.expand_dims(feats_to_predict,0)).float()
        y = self.get_encoder_output(feats_to_predict=feats_to_predict)

        self.keep_cache_size_moderate()
        
        if self.rescoring:
            self.outputs.append(y)
        if self.endpoint():
            if self.rescoring:
                final_result, final_result_likelihood = self.decoder_rescoring()
            self.reset()
            return False, "", final_result, final_result_likelihood

        encoder_out = y
        maxlen = encoder_out.size(1)

        ctc_probs = self.get_ctc_output(encoder_out=encoder_out)
        self.ctc_prefix_beam_search_Algorithm(ctc_probs=ctc_probs)
        # ctc_probs = ctc_probs.squeeze(0)
        # for t in range(0, maxlen):
        #     logp = ctc_probs[t]  # (vocab_size,)
        #     # key: prefix, value (pb, pnb), default value(-inf, -inf)
        #     next_hyps = defaultdict(lambda: (-float('inf'), -float('inf')))
        #     # 2.1 First beam prune: select topk best
        #     top_k_logp, top_k_index = logp.topk(self.beam)  # (beam_size,)
        #     for s in top_k_index:
        #         s = s.item()
        #         ps = logp[s].item()
        #         for prefix, (pb, pnb) in self.cur_hyps:
        #             last = prefix[-1] if len(prefix) > 0 else None
        #             if s == 0:  # blank
        #                 n_pb, n_pnb = next_hyps[prefix]
        #                 n_pb = log_add([n_pb, pb + ps, pnb + ps])
        #                 next_hyps[prefix] = (n_pb, n_pnb)
        #             elif s == last:
        #                 #  Update *ss -> *s;
        #                 n_pb, n_pnb = next_hyps[prefix]
        #                 n_pnb = log_add([n_pnb, pnb + ps])
        #                 next_hyps[prefix] = (n_pb, n_pnb)
        #                 # Update *s-s -> *ss, - is for blank
        #                 n_prefix = prefix + (s, )
        #                 n_pb, n_pnb = next_hyps[n_prefix]
        #                 n_pnb = log_add([n_pnb, pb + ps])
        #                 next_hyps[n_prefix] = (n_pb, n_pnb)
        #             else:
        #                 n_prefix = prefix + (s, )
        #                 n_pb, n_pnb = next_hyps[n_prefix]
        #                 n_pnb = log_add([n_pnb, pb + ps, pnb + ps])
        #                 next_hyps[n_prefix] = (n_pb, n_pnb)

        #     # 2.2 Second beam prune
        #     next_hyps = sorted(next_hyps.items(),
        #                     key=lambda x: log_add(list(x[1])),
        #                     reverse=True)
        #     self.cur_hyps = next_hyps[:self.beam]
        hyps = [(y[0], log_add([y[1][0], y[1][1]])) for y in self.cur_hyps]
        hyps = hyps[0][0]
        content = [self.dict_model[index] for index in hyps]
        #print("content:",content)
        if ''.join(content) == '':
            return True, ''.join(content), "", -float('inf')
        else:
            # print('Partial:'+''.join(content))
            logger.info('Partial:'+''.join(content))
            return True, ''.join(content), "", -float('inf')
        
    def ctc_prefix_beam_search_purn_try(self):
        """ decode feats
            the aim of this function is same like ctc_prefix_beam_search_purn function \n
            we write it in another way and Call ctc_prefix_beam_search_purn_onestep function

        Args:
            Nothing
            
        Returns:
            Nothing
        """
        continue_run = True
        while continue_run:
            continue_run, partial_result, final_result, final_result_likelihood = self.ctc_prefix_beam_search_purn_onestep()


    def ctc_prefix_beam_search_purn(self):
        """ decode feats
            we continues run this code until the buffer is empty \n
            one step we get feats(the shape is controled by decode_window) from buffer,
                and use encode,ctc process the feats and get encoder_out and ctc_out \n
            then we endpoint if find endpoint we will rescore \n
            if no endpoint we use ctc_prefix_beam_search to get nbest(beam) result and show \n
            by the way we use keep_cache_size_moderate in this function

        Args:
            Nothing
            
        Returns:
            Nothing
        """

        while True:
            get_data,feats_to_predict = self.dat_buffer_read(self.decoding_window,self.stride)
            if not get_data:
                break
            feats_to_predict = torch.from_numpy(np.expand_dims(feats_to_predict,0)).float()
            y = self.get_encoder_output(feats_to_predict=feats_to_predict)
            # # feats_to_predict = feats_to_predict.to(self.device)
            # # y, self.subsampling_cache, self.elayers_output_cache,self.conformer_cnn_cache = self.model.encoder.forward_chunk(feats_to_predict, self.offset,
            # #                     self.required_cache_size,self.subsampling_cache,self.elayers_output_cache,self.conformer_cnn_cache)
            # ort_inputs = {self.encoder_ort_session.get_inputs()[0].name: self.to_numpy(feats_to_predict),
            #     self.encoder_ort_session.get_inputs()[1].name: self.offset,
            #     # ort_session.get_inputs()[2].name: to_numpy(required_cache_size),
            #     self.encoder_ort_session.get_inputs()[2].name: self.subsampling_cache,
            #     self.encoder_ort_session.get_inputs()[3].name: self.elayers_output_cache,
            #     self.encoder_ort_session.get_inputs()[4].name: self.conformer_cnn_cache
            #     }
            # ort_outs = self.encoder_ort_session.run(None, ort_inputs)
            # y = torch.Tensor(ort_outs[0])
            # # self.offset += ort_outs[0].shape[1]
            # self.subsampling_cache = ort_outs[1]
            # self.elayers_output_cache = ort_outs[2]
            # self.conformer_cnn_cache = ort_inputs[3]

            self.keep_cache_size_moderate()
            
            # # subsampling_cache_shape = [item.shape for item in self.subsampling_cache]
            # # elayers_output_cache_shape = [item.shape for item in self.elayers_output_cache]
            # # conformer_cnn_cache_shape = [item.shape for item in self.conformer_cnn_cache]
            # self.offset += y.size(1)
            if self.rescoring:
                self.outputs.append(y)
            if self.endpoint():
                if self.rescoring:
                    self.decoder_rescoring()
                self.reset()
                break

            encoder_out = y
            # maxlen = encoder_out.size(1) # not use

            # ctc_probs = self.model.ctc.log_softmax(encoder_out)
            # ort_inputs = {self.ctc_ort_session.get_inputs()[0].name: self.to_numpy(encoder_out),
            #     }
            # ctc_probs = self.ctc_ort_session.run(None, ort_inputs)[0]
            # ctc_probs = torch.Tensor(ctc_probs)
            ctc_probs = self.get_ctc_output(encoder_out=encoder_out)
            self.ctc_prefix_beam_search_Algorithm(ctc_probs=ctc_probs)
            # ctc_probs = ctc_probs.squeeze(0)
            # for t in range(0, maxlen):
            #     logp = ctc_probs[t]  # (vocab_size,)
            #     # key: prefix, value (pb, pnb), default value(-inf, -inf)
            #     next_hyps = defaultdict(lambda: (-float('inf'), -float('inf')))
            #     # 2.1 First beam prune: select topk best
            #     top_k_logp, top_k_index = logp.topk(self.beam)  # (beam_size,)
            #     for s in top_k_index:
            #         s = s.item()
            #         ps = logp[s].item()
            #         for prefix, (pb, pnb) in self.cur_hyps:
            #             last = prefix[-1] if len(prefix) > 0 else None
            #             if s == 0:  # blank
            #                 n_pb, n_pnb = next_hyps[prefix]
            #                 n_pb = log_add([n_pb, pb + ps, pnb + ps])
            #                 next_hyps[prefix] = (n_pb, n_pnb)
            #             elif s == last:
            #                 #  Update *ss -> *s;
            #                 n_pb, n_pnb = next_hyps[prefix]
            #                 n_pnb = log_add([n_pnb, pnb + ps])
            #                 next_hyps[prefix] = (n_pb, n_pnb)
            #                 # Update *s-s -> *ss, - is for blank
            #                 n_prefix = prefix + (s, )
            #                 n_pb, n_pnb = next_hyps[n_prefix]
            #                 n_pnb = log_add([n_pnb, pb + ps])
            #                 next_hyps[n_prefix] = (n_pb, n_pnb)
            #             else:
            #                 n_prefix = prefix + (s, )
            #                 n_pb, n_pnb = next_hyps[n_prefix]
            #                 n_pnb = log_add([n_pnb, pb + ps, pnb + ps])
            #                 next_hyps[n_prefix] = (n_pb, n_pnb)

            #     # 2.2 Second beam prune
            #     next_hyps = sorted(next_hyps.items(),
            #                     key=lambda x: log_add(list(x[1])),
            #                     reverse=True)
            #     self.cur_hyps = next_hyps[:self.beam]
            hyps = [(y[0], log_add([y[1][0], y[1][1]])) for y in self.cur_hyps]
            hyps = hyps[0][0]
            content = [self.dict_model[index] for index in hyps]
            #print("content:",content)
            if ''.join(content) == '':
                continue
            # print('Partial:'+''.join(content))
            logger.info('Partial:'+''.join(content))



    def keep_cache_size_moderate(self):
        """ keep_cache_size_moderate

            because we use onnx model, we can't set require_cache_size > 0,
            so we use another way to control the cache size moderate. \n
            if the time of cache is greater than self.cache_size_max, then cut the cache, 
            then the time of cache will be same as self.cache_size_max \n
                (actually will be one greater than self.cache_size_max because a pad zero dimension)\n
                (因为第一维的时间在model处理时会被扔掉,这一维是冗余且必要的,实际上会pad一下,比 self.cache_size_max 大 1)

        Args:
            Nothing
            
        Returns:
            Nothing
        """
        cache_max_size = self.cache_size_max
        if self.subsampling_cache.shape[1] > cache_max_size:
            # self.subsampling_cache = self.subsampling_cache[:,-cache_max_size:,:]
            # print("\nsubsampling_cache.shape: " + str(self.subsampling_cache.shape))
            logger.debug("\nsubsampling_cache.shape: " + str(self.subsampling_cache.shape))
            self.subsampling_cache = np.concatenate( (np.zeros((1,1,256),dtype=np.float32),self.subsampling_cache[:,-cache_max_size:,:]) , axis=1)
            # print("\nsubsampling_cache.shape: " + str(self.subsampling_cache.shape))
            logger.debug("\nsubsampling_cache.shape: " + str(self.subsampling_cache.shape))
        if self.elayers_output_cache.shape[2] > cache_max_size:
            # print("elayers_output_cache.shape: " + str(self.elayers_output_cache.shape))
            logger.debug("elayers_output_cache.shape: " + str(self.elayers_output_cache.shape))
            self.elayers_output_cache = np.concatenate((np.zeros((12,1,1,256),dtype=np.float32),self.elayers_output_cache[:,:,-cache_max_size:,:]),axis=2)
            # self.elayers_output_cache = np.concatenate((np.zeros(12,1,1,256),self.elayers_output_cache[:,:,-cache_max_size:,:]) 
            # print("elayers_output_cache.shape: " + str(self.elayers_output_cache.shape))
            logger.debug("elayers_output_cache.shape: " + str(self.elayers_output_cache.shape))


    def ctc_prefix_beam_search_purn_all(self):
        """ a tmp code maybe have bug \n
            we get all the feats in buffer and decode them all \n
            we use ctc and attention decoder

        Args:
            Nothing
            
        Returns:
            Nothing
        """

        while True:
            get_data,feats_to_predict = self.dat_buffer_read_all()
            if not get_data:
                break
            feats_to_predict = torch.from_numpy(np.expand_dims(feats_to_predict,0)).float()
            y = self.get_encoder_output(feats_to_predict)
            # # feats_to_predict = feats_to_predict.to(self.device)
            # # y, self.subsampling_cache, self.elayers_output_cache,self.conformer_cnn_cache = self.model.encoder.forward_chunk(feats_to_predict, self.offset,
            # #                     self.required_cache_size,self.subsampling_cache,self.elayers_output_cache,self.conformer_cnn_cache)
            # ort_inputs = {self.encoder_ort_session.get_inputs()[0].name: self.to_numpy(feats_to_predict),
            #     self.encoder_ort_session.get_inputs()[1].name: self.offset,
            #     # ort_session.get_inputs()[2].name: to_numpy(required_cache_size),
            #     self.encoder_ort_session.get_inputs()[2].name: self.subsampling_cache,
            #     self.encoder_ort_session.get_inputs()[3].name: self.elayers_output_cache,
            #     #   ort_session.get_inputs()[5].name: to_numpy(conformer_cnn_cache)
            #     }
            # ort_outs = self.encoder_ort_session.run(None, ort_inputs)
            # y = torch.Tensor(ort_outs[0])
            # # self.offset += ort_outs[0].shape[1]
            # self.subsampling_cache = ort_outs[1]
            # self.elayers_output_cache = ort_outs[2]


            # # subsampling_cache_shape = [item.shape for item in self.subsampling_cache]
            # # elayers_output_cache_shape = [item.shape for item in self.elayers_output_cache]
            # # conformer_cnn_cache_shape = [item.shape for item in self.conformer_cnn_cache]
            # self.offset += y.size(1)

            if self.rescoring:
                self.outputs.append(y)
            if self.endpoint():
                if self.rescoring:
                    self.decoder_rescoring()
                self.reset()
                break

            encoder_out = y
            maxlen = encoder_out.size(1)
            # ctc_probs = self.model.ctc.log_softmax(encoder_out)
            # ort_inputs = {self.ctc_ort_session.get_inputs()[0].name: self.to_numpy(encoder_out),
            #     }
            # ctc_probs = self.ctc_ort_session.run(None, ort_inputs)[0]
            # ctc_probs = torch.Tensor(ctc_probs)
            ctc_probs = self.get_ctc_output(encoder_out=encoder_out)
            self.ctc_prefix_beam_search_Algorithm(ctc_probs=ctc_probs)
            # ctc_probs = ctc_probs.squeeze(0)
            # for t in range(0, maxlen):
            #     logp = ctc_probs[t]  # (vocab_size,)
            #     # key: prefix, value (pb, pnb), default value(-inf, -inf)
            #     next_hyps = defaultdict(lambda: (-float('inf'), -float('inf')))
            #     # 2.1 First beam prune: select topk best
            #     top_k_logp, top_k_index = logp.topk(self.beam)  # (beam_size,)
            #     for s in top_k_index:
            #         s = s.item()
            #         ps = logp[s].item()
            #         for prefix, (pb, pnb) in self.cur_hyps:
            #             last = prefix[-1] if len(prefix) > 0 else None
            #             if s == 0:  # blank
            #                 n_pb, n_pnb = next_hyps[prefix]
            #                 n_pb = log_add([n_pb, pb + ps, pnb + ps])
            #                 next_hyps[prefix] = (n_pb, n_pnb)
            #             elif s == last:
            #                 #  Update *ss -> *s;
            #                 n_pb, n_pnb = next_hyps[prefix]
            #                 n_pnb = log_add([n_pnb, pnb + ps])
            #                 next_hyps[prefix] = (n_pb, n_pnb)
            #                 # Update *s-s -> *ss, - is for blank
            #                 n_prefix = prefix + (s, )
            #                 n_pb, n_pnb = next_hyps[n_prefix]
            #                 n_pnb = log_add([n_pnb, pb + ps])
            #                 next_hyps[n_prefix] = (n_pb, n_pnb)
            #             else:
            #                 n_prefix = prefix + (s, )
            #                 n_pb, n_pnb = next_hyps[n_prefix]
            #                 n_pnb = log_add([n_pnb, pb + ps, pnb + ps])
            #                 next_hyps[n_prefix] = (n_pb, n_pnb)

            #     # 2.2 Second beam prune
            #     next_hyps = sorted(next_hyps.items(),
            #                     key=lambda x: log_add(list(x[1])),
            #                     reverse=True)
            #     self.cur_hyps = next_hyps[:self.beam]
            hyps = [(y[0], log_add([y[1][0], y[1][1]])) for y in self.cur_hyps]
            hyps = hyps[0][0]
            content = [self.dict_model[index] for index in hyps]
            #print("content:",content)
            if ''.join(content) == '':
                continue
            # print('Partial:'+''.join(content))
            logger.info('Partial:'+''.join(content))

    def detect(self, signal):
        """ detect signal into decoder wavebuffer.\n
            signal data is numpy, waveform 数量级在int (order of magnitude is int)
        Args:
            signal (numpy.array):  (L, ), L for time
            
        Returns:
            Nothing
        """
        self.offline_signal = np.concatenate([self.offline_signal, signal])
        chunk_num = int(len(signal) / self.chunk_size) # 1600
        for index in range(chunk_num):
            self.detect_process(signal[index * self.chunk_size : (index + 1) * self.chunk_size])
        # return 0

    def get_encoder_output(self,feats_to_predict):
        """ get onnx encoder output.
            in this function update self.subsampling_cache,self.elayers_output_cache,self.conformer_cnn_cache,self.offset \n
            if self.encoder = transformer encoder, self.conformer_cnn_cache always be zeros
        Args:
            feats_to_predict (torch.Tensor):  (Batch, L, Dim), L for time
            
        Returns:
            encoder_output (torch.Tensor): (Batch, L/subsampling_rate, Dim) The output of the encoder onnx process.
        """
        ort_inputs = {self.encoder_ort_session.get_inputs()[0].name: self.to_numpy(feats_to_predict),
            self.encoder_ort_session.get_inputs()[1].name: self.offset,
            # ort_session.get_inputs()[2].name: to_numpy(required_cache_size),
            self.encoder_ort_session.get_inputs()[2].name: self.subsampling_cache,
            self.encoder_ort_session.get_inputs()[3].name: self.elayers_output_cache,
            self.encoder_ort_session.get_inputs()[4].name: self.conformer_cnn_cache
            }
        ort_outs = self.encoder_ort_session.run(None, ort_inputs)
        self.subsampling_cache = ort_outs[1]
        self.elayers_output_cache = ort_outs[2]
        self.conformer_cnn_cache = ort_outs[3]
        y = torch.Tensor(ort_outs[0])
        self.offset += y.size(1)
        return y

    def get_decoder_output(self,encoder_out,encoder_mask,hyps_pad,hyps_lens) -> (torch.Tensor):
        """ get onnx encoder output.
            in this function update self.subsampling_cache,self.elayers_output_cache,self.conformer_cnn_cache,self.offset \n
            if self.encoder = transformer encoder, self.conformer_cnn_cache always be zeros
        Args:
            encoder_out (torch.Tensor):  (Batch, L, Dim), L for time after subsampling
            encoder_mask (torch.Tensor): (Batch or Beam size, 1, L), come from following expression
                encoder_mask = torch.ones(self.beam,1,encoder_out.size(1),dtype=torch.bool,device=self.device)
            hyps_pad (torch.Tensor): (Batch or Beam size, sentenceMaxLen), ctc_output after beam search nbest result then after padding process and add_sos_eos 
                examples:
                    >>> hyps = [(y[0], log_add([y[1][0], y[1][1]])) for y in self.cur_hyps]
                    >>> assert len(hyps) == self.beam
                    >>> hyps_pad = pad_sequence([
                            torch.tensor(hyp[0], device=self.device, dtype=torch.long)
                            for hyp in hyps
                        ], True, -1) 
                    >>> hyps_pad, _ = add_sos_eos(hyps_pad, self.sos, self.eos, -1)
                    >>> hyps_lens = torch.tensor([len(hyp[0]) for hyp in hyps],
                                 device=self.device,
                                 dtype=torch.long)
            hyps_lens (torch.Tensor): (Batch or Beam size), show every beam hyps_pad actual len

        Returns:
            decoder_out (torch.Tensor): (Batch, L+1, VocabularySize) The output of the decoder_out onnx process after log_softmax.
        """
        ort_inputs = {self.decoder_ort_session.get_inputs()[0].name: self.to_numpy(encoder_out),
                self.decoder_ort_session.get_inputs()[1].name: self.to_numpy(encoder_mask),
                self.decoder_ort_session.get_inputs()[2].name: self.to_numpy(hyps_pad),
                self.decoder_ort_session.get_inputs()[3].name: self.to_numpy(hyps_lens),
                }
        ort_outs = self.decoder_ort_session.run(None, ort_inputs)
        decoder_out = torch.Tensor(ort_outs[0])
        decoder_out = torch.nn.functional.log_softmax(decoder_out, dim=-1)
        return decoder_out

    def get_ctc_output(self,encoder_out):
        """ get onnx ctc output.
            in this function update nothing
        Args:
            encoder_out (torch.Tensor):  (Batch or Beam size, L, Dim), L for time after subsampling
            
        Returns:
            ctc_out (torch.Tensor): (Batch or Beam size, L, VocabularySize) The output of the ctc onnx. must be vocabulary logprobabilities after softmax
        """
        ort_inputs = {self.ctc_ort_session.get_inputs()[0].name: self.to_numpy(encoder_out),
            }
        ctc_probs = self.ctc_ort_session.run(None, ort_inputs)[0]
        ctc_probs = torch.Tensor(ctc_probs)
        return ctc_probs



    def offline_decode(self,audio_path,isbytes=False,useVad=False,noLog=False):
        """ offline_decode audio
            we can recieve bytes data or wav file path \n
            we will return partial result, final result and final result 
            
            if we use vad, we will call self.offline_decode_one_sentence one sentence by one sentence
                and if we read a audio like Streaming long part by long part (Maybe not recommended) \n
                we will Cut off a short sentence then we Keep the first half of the sentence and put them together when you get the second half \n
                anyway if we want to decode for Streaming we better use a streaming model (maybe the model trained by train_unified_transformer.conf )

        Args:
            audio_path (string or bytes): path string or wav bytes
            isbytes (Bool): set audio_path is wav bytes or not, default False
            useVad (Bool): set we use vad or not, default False
                The Vad tools we use is webrtcvad. See details in https://github.com/wiseman/py-webrtcvad
            noLog (Bool): if true, we don't print log
            
        Returns:
            Partial result (string): the result after ctc prefix_beam_search
            Final result (string): the result after attention decoder teacher force rescoring. It is the best one
            Final result likelihood (float): the likelihood(score) of the Final result. 
        """
        partial_result_list = []
        final_result_list = []
        final_result_likelihood = -float("inf")
        from utils.vad_ext import frame_generator,read_wave,raw_to_float,vad_collector
        if not isbytes:
            audio, sample_rate = read_wave(audio_path)
        else:
            audio = audio_path
            sample_rate = 16000
        # print("decoder recieve {} audio bytes ".format(len(audio)))
        if not noLog:
            logger.info("decoder recieve {} audio bytes ".format(len(audio)))
        audio = self.bytes_offline_notfinished_buffer + audio
        if not noLog:
            if self.bytes_offline_notfinished_buffer != b'':
                logger.info("after concatenate {} audio bytes ".format(len(audio)))
        if useVad:
            vad = webrtcvad.Vad(0)
            # frames = frame_generator(30, audio, 16000)
            frames = frame_generator(20, audio, 16000)
            frames = list(frames)
            # segments = vad_collector(sample_rate, 30, 200, vad, frames, 0.8)
            segments = vad_collector(sample_rate, 20, 200, vad, frames,0.6)
            signal = np.zeros(1)
            seg_num = 0
            if not noLog:
                logger.info("use vad has {} segments".format(len(segments)))
            for index,seg in enumerate(segments):
                seg_num += 1

                now_audio_bytes = seg[0]
                signal = np.concatenate([signal,raw_to_float(now_audio_bytes)])
                seg_finished = seg[3]
                if index == len(segments) - 1:
                    if seg_finished:
                        partial_result, final_result, final_result_likelihood = self.offline_decode_one_sentence(signal)
                        partial_result_list.append(partial_result)
                        final_result_list.append(final_result)
                        signal = np.zeros(1)
                        self.bytes_offline_notfinished_buffer = bytes()
                    else:
                        self.bytes_offline_notfinished_buffer = now_audio_bytes
                    continue
                partial_result, final_result, final_result_likelihood = self.offline_decode_one_sentence(signal)
                partial_result_list.append(partial_result)
                final_result_list.append(final_result)
                # if (index+1) % 3 == 0:
                
                signal = np.zeros(1)
                
            # print("use vad has {} segments".format(seg_num))
            if not noLog:
                # logger.info("use vad has {} segments".format(seg_num))
                if not seg_finished:
                    logger.info("the last segment not finished")
        else:
            signal = raw_to_float(audio)
            partial_result, final_result, final_result_likelihood = self.offline_decode_one_sentence(signal)
            partial_result_list.append(partial_result)
            final_result_list.append(final_result)
        
        if not noLog:
            logger.info("final partial_result: {}".format(" ".join(partial_result_list)))
            logger.info("final final_result: {}".format(" ".join(final_result_list)))
        return " ".join(partial_result_list),  " ".join(final_result_list), final_result_likelihood
        # feats = self._extract_feature(signal)
        # feats_to_predict = torch.from_numpy(np.expand_dims(feats,0)).float()
        
        # y = self.get_encoder_output(feats_to_predict)
        
        # if self.rescoring:
        #     self.outputs.append(y)
        # # if self.endpoint():
        # #     if self.rescoring:
        # #         self.decoder_recoring()
        # #     self.reset()

        # encoder_out = y
        # maxlen = encoder_out.size(1)
        # # ctc_probs = self.model.ctc.log_softmax(encoder_out)
        # ctc_probs = self.get_ctc_output(encoder_out)
        # self.ctc_prefix_beam_search_Algorithm(ctc_probs=ctc_probs)
        # # ctc_probs = ctc_probs.squeeze(0)
        # # for t in range(0, maxlen):
        # #     logp = ctc_probs[t]  # (vocab_size,)
        # #     # key: prefix, value (pb, pnb), default value(-inf, -inf)
        # #     next_hyps = defaultdict(lambda: (-float('inf'), -float('inf')))
        # #     # 2.1 First beam prune: select topk best
        # #     top_k_logp, top_k_index = logp.topk(self.beam)  # (beam_size,)
        # #     for s in top_k_index:
        # #         s = s.item()
        # #         ps = logp[s].item()
        # #         for prefix, (pb, pnb) in self.cur_hyps:
        # #             last = prefix[-1] if len(prefix) > 0 else None
        # #             if s == 0:  # blank
        # #                 n_pb, n_pnb = next_hyps[prefix]
        # #                 n_pb = log_add([n_pb, pb + ps, pnb + ps])
        # #                 next_hyps[prefix] = (n_pb, n_pnb)
        # #             elif s == last:
        # #                 #  Update *ss -> *s;
        # #                 n_pb, n_pnb = next_hyps[prefix]
        # #                 n_pnb = log_add([n_pnb, pnb + ps])
        # #                 next_hyps[prefix] = (n_pb, n_pnb)
        # #                 # Update *s-s -> *ss, - is for blank
        # #                 n_prefix = prefix + (s, )
        # #                 n_pb, n_pnb = next_hyps[n_prefix]
        # #                 n_pnb = log_add([n_pnb, pb + ps])
        # #                 next_hyps[n_prefix] = (n_pb, n_pnb)
        # #             else:
        # #                 n_prefix = prefix + (s, )
        # #                 n_pb, n_pnb = next_hyps[n_prefix]
        # #                 n_pnb = log_add([n_pnb, pb + ps, pnb + ps])
        # #                 next_hyps[n_prefix] = (n_pb, n_pnb)

        # #     # 2.2 Second beam prune
        # #     next_hyps = sorted(next_hyps.items(),
        # #                     key=lambda x: log_add(list(x[1])),
        # #                     reverse=True)
        # #     self.cur_hyps = next_hyps[:self.beam]
        # hyps = [(y[0], log_add([y[1][0], y[1][1]])) for y in self.cur_hyps]
        # hyps = hyps[0][0]
        # content = [self.dict_model[index] for index in hyps]
        # #print("content:",content)
        # # if ''.join(content) == '':
        # #     continue
        # # print('Partial:'+''.join(content))
        # if not noLog:
        #     logger.info('Partial:'+''.join(content))
        # if self.rescoring:
        #     final_result, final_result_likelihood = self.decoder_rescoring(noLog=noLog)
        # self.reset()
        # return ''.join(content) , final_result, final_result_likelihood

    def offline_decode_one_sentence(self,signal,noLog=False):
        """ offline_decode audio
            we can recieve one short sentence signal \n
            the signal is one segment after vad
            we will return partial result, final result and final result likelihood

            TODO 现在这个函数不能记住以前的信息，有时候是合理的，有时候是不合理

        Args:
            audio_path (string or bytes): path string or wav bytes
            noLog (Bool): if true, we don't print log
            
        Returns:
            Partial result (string): the result after ctc prefix_beam_search
            Final result (string): the result after attention decoder teacher force rescoring. It is the best one
            Final result likelihood (float): the likelihood(score) of the Final result. 
        """
        feats = self._extract_feature(signal)
        feats_to_predict = torch.from_numpy(np.expand_dims(feats,0)).float()
        
        y = self.get_encoder_output(feats_to_predict)

        self.keep_cache_size_moderate()
        
        if self.rescoring:
            self.outputs.append(y)
        # if self.endpoint():
        #     if self.rescoring:
        #         self.decoder_recoring()
        #     self.reset()

        encoder_out = y
        maxlen = encoder_out.size(1)
        # ctc_probs = self.model.ctc.log_softmax(encoder_out)
        ctc_probs = self.get_ctc_output(encoder_out)
        self.ctc_prefix_beam_search_Algorithm(ctc_probs=ctc_probs)
        hyps = [(y[0], log_add([y[1][0], y[1][1]])) for y in self.cur_hyps]
        hyps = hyps[0][0]
        content = [self.dict_model[index] for index in hyps]
        #print("content:",content)
        # if ''.join(content) == '':
        #     continue
        # print('Partial:'+''.join(content))
        if not noLog:
            logger.info('Partial:'+''.join(content))
        if self.rescoring:
            final_result, final_result_likelihood = self.decoder_rescoring(noLog=noLog)
        self.reset()
        partial_result = ''.join(content) if ''.join(content) != '' else ''
        if partial_result == '':
            final_result = ''
        return ''.join(content) , final_result, final_result_likelihood

    def ctc_prefix_beam_search_Algorithm(self,ctc_probs):
        """ ctc_prefix_beam_search_Algorithm
            details see http://placebokkk.github.io/asr/2020/02/01/asr-ctc-decoder.html and wenet/transformer/asr_model.py _ctc_prefix_beam_search function
        Args:
            ctc_probs (Tensor): (Batch, Time, VocabularySize)
            
        Returns:
            Nothing
            but we change the self.cur_hyps,
            we store the new nbest result in self.cur_hyps
        """
        ctc_probs = ctc_probs.squeeze(0)
        maxlen = ctc_probs.size(0)
        for t in range(0, maxlen):
            logp = ctc_probs[t]  # (vocab_size,)
            # key: prefix, value (pb, pnb), default value(-inf, -inf)
            next_hyps = defaultdict(lambda: (-float('inf'), -float('inf')))
            # 2.1 First beam prune: select topk best
            top_k_logp, top_k_index = logp.topk(self.beam)  # (beam_size,)
            for s in top_k_index:
                s = s.item()
                ps = logp[s].item()
                for prefix, (pb, pnb) in self.cur_hyps:
                    last = prefix[-1] if len(prefix) > 0 else None
                    if s == 0:  # blank
                        n_pb, n_pnb = next_hyps[prefix]
                        n_pb = log_add([n_pb, pb + ps, pnb + ps])
                        next_hyps[prefix] = (n_pb, n_pnb)
                    elif s == last:
                        #  Update *ss -> *s;
                        n_pb, n_pnb = next_hyps[prefix]
                        n_pnb = log_add([n_pnb, pnb + ps])
                        next_hyps[prefix] = (n_pb, n_pnb)
                        # Update *s-s -> *ss, - is for blank
                        n_prefix = prefix + (s, )
                        n_pb, n_pnb = next_hyps[n_prefix]
                        n_pnb = log_add([n_pnb, pb + ps])
                        next_hyps[n_prefix] = (n_pb, n_pnb)
                    else:
                        n_prefix = prefix + (s, )
                        n_pb, n_pnb = next_hyps[n_prefix]
                        n_pnb = log_add([n_pnb, pb + ps, pnb + ps])
                        next_hyps[n_prefix] = (n_pb, n_pnb)

            # 2.2 Second beam prune
            next_hyps = sorted(next_hyps.items(),
                            key=lambda x: log_add(list(x[1])),
                            reverse=True)
            self.cur_hyps = next_hyps[:self.beam]


    def offline_decode_wavscp(self,wavscp_path,result_file_path,rtf_file_path,useVad=False):
        """ offline_decode wav.scp
            we recieve a wav.scp and decode them all
            write result_file and rtf_file

        Args:
            wavscp_path (string): wav.scp path string 
            result_file_path (string): output text in this path
            rtf_file_path (string): output real time factor in this path
            useVad (Bool): set we use vad or not, default False
                The Vad tools we use is webrtcvad. See details in https://github.com/wiseman/py-webrtcvad
            
        Returns:
            Nothing
        """

        utt2wav = { line.strip().split(' ')[0]:line.strip().split(' ')[1] for line in open(wavscp_path,'r') }
        with open(result_file_path, 'w') as fout, open(rtf_file_path, 'w') as rtfout:
            for index, key in enumerate(utt2wav.keys()):
                pre_time = time.time()
                _,final_result,_ = self.offline_decode(utt2wav[key],isbytes=False,useVad=useVad,noLog=True)
                end_time = time.time()
                process_time = end_time - pre_time
                logger.info('{} {}'.format(key, final_result))
                logger.info('{} precess {} seconds'.format(key, str(process_time)))
                fout.write('{} {}\n'.format(key, final_result))
                rtfout.write('{} {}\n'.format(key, str(process_time)))
            

    def resetAll(self):
        """ resetAll
            we reset self.bytes_offline_notfinished_buffer and call self.reset \n
            reset self.bytes_offline_notfinished_buffer means we recieve all the bytes from a long audio file

        Args:
            Nothing
            
        Returns:
            Nothing
        """
        self.bytes_offline_notfinished_buffer = bytes()
        self.reset()
        pass

    # def 


if __name__ == '__main__':

    # weNetDecoder = WeNetDecoder('exp/transformer/model.yaml')
    # weNetDecoder = WeNetDecoder('exp/unified_transformer/model.yaml')
    # weNetDecoder = WeNetDecoder('exp/unified_conformer/onnx_model/model_onnx.yaml')
    # weNetDecoder = WeNetDecoder('exp/conformer/onnx_model/model_onnx.yaml')
    weNetDecoder = WeNetDecoder('20210204_conformer_exp/onnx_model/model_onnx.yaml')
    # weNetDecoder = WeNetDecoder('model_onnx_template.yaml')
    # weNetDecoder.offline_decode('./long_sil.wav',useVad=False)
    # weNetDecoder.resetAll()
    # weNetDecoder.offline_decode('./long_sil.wav',useVad=True)
    # # weNetDecoder.resetAll()

    # weNetDecoder.offline_decode('./1-car.wav',useVad=True)
    # weNetDecoder.resetAll()

    # with open('./1-car.wav','rb') as audiostream:
    #     audiostream.read(44)
    #     for dataflow in tqdm(iter(lambda:audiostream.read(32000 * 20),"")):
    #         if len(dataflow) == 0:
    #             break
    #         weNetDecoder.offline_decode(dataflow,isbytes=True,useVad=True)
    #         time.sleep(0.1)
    # weNetDecoder.resetAll()


    # weNetDecoder.offline_decode('./long_sil.wav',useVad=False)
    
    weNetDecoder.offline_decode('./BAC009S0764W0121.wav',useVad=True)
    weNetDecoder.resetAll()
    
    weNetDecoder = WeNetDecoder('20210204_unified_transformer_exp/onnx_model/model_onnx.yaml')

    with open('./BAC009S0764W0121.wav','rb') as audiostream:
    # with open('./long_sil.wav','rb') as audiostream:
        audiostream.read(44)
        for dataflow in tqdm(iter(lambda:audiostream.read(8000),"")):
            if len(dataflow) == 0:
                break
            sig = struct.unpack("%ih" % (len(dataflow) / 2), dataflow)
            data = np.array([float(val) for val in sig])
            weNetDecoder.detect(data)
            # weNetDecoder.ctc_prefix_beam_search_purn()
            weNetDecoder.ctc_prefix_beam_search_purn_try()
            #SR_model.ctc_greedy_search_purn()
            time.sleep(0.1)
    # weNetDecoder.ctc_prefix_beam_search_purn_all()
    weNetDecoder.decoder_rescoring()
    # pass