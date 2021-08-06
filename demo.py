import pyaudio
import numpy as np
import time 
import array
import wave
import struct
import logging
# from wenet_online_decoder import WeNetDecoder
from wenet_online_decoder_onnx import WeNetDecoder
import os
from tqdm import tqdm
from scipy.io import wavfile


logging.basicConfig()
logger = logging.getLogger("ASRmodel")
logger.setLevel(logging.INFO)
TOP_DIR = os.path.dirname(os.path.abspath(__file__))


class waveBuffer:
    def __init__(self):
        self.receive_eos = False
        self.data_stream = np.ones(1,)
    def read(self, buffer_size):
        if self.data_stream.shape[0] == 0:
            return False, self.data_stream
        buffer_size = self.data_stream.shape[0] if self.data_stream.shape[0] <= buffer_size else buffer_size
        data_buffer = self.data_stream[0:buffer_size]
        self.data_stream = self.data_stream[buffer_size:self.data_stream.shape[0] - buffer_size]
        return True, data_buffer

    def push(self, dataflow):
        length = len(dataflow)
        for i in range(0, length):
            try:
                sig = struct.unpack("%ih" % (len(dataflow) / 2), dataflow)
                data = np.array([float(val) for val in sig])
                #data = array.array('h', dataflow)
            except ValueError:
                continue
            if i > 0:
                break
        extend = np.array(data)
        #print("extend data:",extend)
        org_data = np.array(self.data_stream)
        self.data_stream = np.array(np.r_[org_data, extend])
    def size(self):
        return self.data_stream.shape[0]


class ASRDetector(object):
    def __init__(self):
        self.num_channels = 1
        self.sample_rate  = 16000
        self.bits_per_sample = 16
        self.frames_per_buffer = 1600
        self.ring_buffer = waveBuffer()
        self.audio = pyaudio.PyAudio()

    def start(self):
        self._running = True
        def audio_callback(in_data, frame_count, time_info, status):
            self.ring_buffer.push(in_data)
            play_data = chr(0) * len(in_data)
            return play_data, pyaudio.paContinue
        self.stream_in = self.audio.open(
            input=True, output=False,
            format=self.audio.get_format_from_width(
                self.bits_per_sample / 8),
            channels=self.num_channels,
            rate=self.sample_rate,
            frames_per_buffer=self.frames_per_buffer,
            stream_callback=audio_callback)
        # ASR_model = WeNetDecoder("./exp/unified_conformer/train.yaml")
        ASR_model = WeNetDecoder("model_onnx_template.yaml")
        logger.debug("detecting...")
        while self._running is True:
            is_speech,data = self.ring_buffer.read(1600)
            if not is_speech:
                time.sleep(0.1)
                continue
            ASR_model.detect(data)
            ASR_model.ctc_prefix_beam_search_purn_try()
        logger.debug("finished")

    def terminate(self):
        """
        Terminate audio stream. Users can call start() again to detect.
        :return: None
        """
        self.stream_in.stop_stream()
        self.stream_in.close()
        self.audio.terminate()
        self._running = False

def online_decode():
    audio_buffer = ASRDetector()
    audio_buffer.start()

def offline_decode():
    # ASR_model = WeNetDecoder("./exp/unified_conformer/train.yaml")
    # ASR_model = WeNetDecoder("exp/unified_transformer/model.yaml")
    ASR_model = WeNetDecoder("model_onnx_template.yaml")
    with open('./BAC009S0764W0121.wav','rb') as audiostream:
        audiostream.read(44) # 把头信息读掉
        for dataflow in tqdm(iter(lambda:audiostream.read(8000),"")):
            if len(dataflow) == 0:
                break
            sig = struct.unpack("%ih" % (len(dataflow) / 2), dataflow)
            data = np.array([float(val) for val in sig])
            ASR_model.detect(data)
            ASR_model.ctc_prefix_beam_search_purn_try()
            #SR_model.ctc_greedy_search_purn()
            time.sleep(0.1)
    ASR_model.decoder_rescoring()

if __name__=="__main__":
    
    # samplerate, fdata = wavfile.read('./BAC009S0764W0121.wav')
    # tdata = np.array([])
    # with open('./BAC009S0764W0121.wav','rb') as audiostream:
    #     audiostream.read(44)
    #     for dataflow in tqdm(iter(lambda:audiostream.read(8000),"")):
    #         if len(dataflow) == 0:
    #             break
    #         sig = struct.unpack("%ih" % (len(dataflow) / 2), dataflow)
    #         data = np.array([float(val) for val in sig])
    #         tdata = np.hstack((tdata,data))
    offline_decode()
    #online_decode()
