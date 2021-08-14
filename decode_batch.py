# decode a wav.scp list

"""
# we can use this file by shell commend following
test_dir=exp/conformer/test_result
python3 decode_batch.py \
            --config exp/conformer/onnx_model/model_onnx.yaml \
            --test_data data/test/wav.scp \
            --result_file $test_dir/text \
            --rtf_file $test_dir/rtf

# and then compute the wer by wenet code

python3 tools/compute-wer.py --char=1 --v=1 \
			data/test/text $test_dir/text > $test_dir/wer

"""

import argparse

from wenet_online_decoder_onnx import WeNetDecoder

parser = argparse.ArgumentParser(description='recognize with your model')
parser.add_argument('--config', required=True, help='config file')
parser.add_argument('--test_data', required=True, help='test data file')
parser.add_argument('--result_file', required=True, help='asr result file')
# 添加了存放rtf的文件
parser.add_argument('--rtf_file', required=True, help='asr rtf file')
args = parser.parse_args()


weNetDecoder = WeNetDecoder(args.config)
weNetDecoder.offline_decode_wavscp(args.test_data,
    args.result_file,
    args.rtf_file)