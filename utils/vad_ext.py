import collections
import contextlib
import sys
import wave
import struct
import webrtcvad
import numpy as np
def raw_to_float(raw_signal):
    sig = struct.unpack("%ih" % (len(raw_signal) / 2), raw_signal)
    # sig = np.array([float(val) / pow(2, 15) for val in sig])
    sig = np.array([float(val) for val in sig])
    return sig



def read_wave(path):
    
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        if num_channels == 1:
            #sample_width = wf.getsampwidth()
            #assert sample_width == 2
            sample_rate = wf.getframerate()
            #assert sample_rate in (8000, 16000, 32000)
            pcm_data = wf.readframes(wf.getnframes())
            #print(pcm_data)
        else:
            params = wf.getparams()
            sample_rate = wf.getframerate()
            nchannels, sampwidth, framerate, nframes = params[:4]
            #print(nchannels, sampwidth, framerate, nframes)  # 2 2 44100 11625348
            # 读取波形数据
            str_data = wf.readframes(nframes)
            # 将波形数据转换为数组
            wave_data = np.fromstring(str_data, dtype=np.int16)
            wave_data.shape = -1, num_channels
            wave_data = wave_data.T
            wave_data_1 = wave_data[0]  # 声道1
            pcm_data = wave_data_1.tostring()
        return pcm_data, sample_rate


def write_wave(path, audio, sample_rate):
    with contextlib.closing(wave.open(path, 'wb')) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)
class Frame(object):
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


def frame_generator(frame_duration_ms, audio, sample_rate):
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n
def vad_collector(sample_rate, frame_duration_ms,
                  padding_duration_ms, vad, frames, padding_max_length):
    """Filters out non-voiced audio frames.
    Given a webrtcvad.Vad and a source of audio frames, yields only
    the voiced audio.
    Uses a padded, sliding window algorithm over the audio frames.
    When more than 90% of the frames in the window are voiced (as
    reported by the VAD), the collector triggers and begins yielding
    audio frames. Then the collector waits until 90% of the frames in
    the window are unvoiced to detrigger.
    The window is padded at the front and back to provide a small
    amount of silence or the beginnings/endings of speech around the
    voiced frames.
    Arguments:
    sample_rate - The audio sample rate, in Hz.
    frame_duration_ms - The frame duration in milliseconds.
    padding_duration_ms - The amount to pad the window, in milliseconds.
    vad - An instance of webrtcvad.Vad.
    frames - a source of audio frames (sequence or generator).
    padding_max_length - ratio of num_voiced in ring_buffer.maxlen
    Returns: A generator that yields PCM audio data.
    """
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    triggered = False
    voiced_frames = []
    segments = [] # 不再使用异步方案
    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)
        #sys.stdout.write(
        #    '1' if vad.is_speech(frame.bytes, sample_rate) else '0')
        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            # num_voiced = len([f for f in ring_buffer
            #                   if vad.is_speech(f.bytes, sample_rate)]) # 这样的写法是不是错误的？21.8.13
            # vad要读很多遍相同的内容 在没有trigger之前 因为ring_buffer的每个值都要在这一步过一遍vad
            if num_voiced > padding_max_length * ring_buffer.maxlen:
                #sys.stdout.write('+(%s)' % (ring_buffer[0].timestamp,))
                triggered = True
                start = ring_buffer[0][0].timestamp
                voiced_frames.extend(ring_buffer)
                ring_buffer.clear()
        else:
            # voiced_frames.append(frame)
            voiced_frames.append((frame, is_speech))
            # ring_buffer.append(frame)
            # num_unvoiced = len([f for f in ring_buffer
            #                     if not vad.is_speech(f.bytes, sample_rate)])
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            # num_unvoiced = 0
            # for f, speech in ring_buffer:
            #     if not speech:
            #         num_unvoiced += 1
            #     else:
            #         num_unvoiced = 0
            if num_unvoiced > padding_max_length * ring_buffer.maxlen:
                #sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
                triggered = False
                end = ring_buffer[-1][0].timestamp + ring_buffer[-1][0].duration
                segments.append((b''.join([f[0].bytes for f in voiced_frames]),start,end,True))
                # yield b''.join([f.bytes for f in voiced_frames])
                ring_buffer.clear()
                voiced_frames = []
    #if triggered:
    #    sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
    #sys.stdout.write('\n')
    # if voiced_frames:
    #     yield b''.join([f.bytes for f in voiced_frames])
    if triggered:
        end = frame.timestamp + frame.duration
        segments.append((b''.join([f[0].bytes for f in voiced_frames]),start, end,False))
    # if len(frames)!=0:
    #     sys.stderr.write('Speech_percent: {:4d} {:.2f}\n'.format(len(frames), count / len(frames)))
    return segments


def main(args):
    if len(args) != 2:
        sys.stderr.write(
            'Usage: example.py <aggressiveness> <path to wav file>\n')
        sys.exit(1)
    audio, sample_rate = read_wave(args[1])
    vad = webrtcvad.Vad(int(args[0]))
    frames = frame_generator(30, audio, sample_rate)
    frames = list(frames)
    segments = vad_collector(sample_rate, 25, 200, vad, frames)
    for i, segment in enumerate(segments):
        path = 'chunk-%002d.wav' % (i,)
        print(' Writing %s' % (path,))
        write_wave(path, segment, sample_rate)


if __name__ == '__main__':
    main(sys.argv[1:])




