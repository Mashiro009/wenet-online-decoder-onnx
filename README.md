# wenet online onnx decoder

## 准备onnx模型

本系统使用[wenet-onnx](https://github.com/Mashiro009/wenet-onnx)导出的onnx模型
* encoder_chunk.onnx
* decoder.onnx
* ctc.onnnx

## 环境配置

torch、CUDA版本以及其他python包安装，参考wenet官方文档：https://github.com/mobvoi/wenet

## 文件配置

根据model_onnx_template.yaml中的描述补全lang_char.txt关键文件和onnx_model文件夹，默认模型为16k采样率数据训练所得。

## 系统运行

在正确的环境下直接执行python wenet_online_decoder_onnx.py，执行offline的识别 和 读取wav文件仿流式识别。

## 参考

本代码参考[wenet_onlinedecode](https://github.com/jiay7/wenet_onlinedecode)进行修改