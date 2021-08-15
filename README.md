# wenet online onnx decoder

## 准备onnx模型

本系统使用[wenet-onnx](https://github.com/Mashiro009/wenet-onnx)导出的onnx模型
* encoder_chunk.onnx
* decoder.onnx
* ctc.onnnx

已准备了两个模型分别在20210204_conformer_exp和20210204_unified_transformer_exp中，可以直接进行测试，详细内容见各自的readme文件,

因无法上传100MB以上的文件到github上,请前去[CSDN链接](https://download.csdn.net/download/MashiroRin/21094226)进行下载

或百度云(删除链接中 中文后 访问)：
链接：https://pan盘.baidu百度.com/s/139mycaFN3JHNoY0xCHLjxw 
提取码：k3gk

或在issues中留下邮箱,发送该zip文件

## 环境配置

torch、CUDA版本以及其他python包安装，参考wenet官方文档：https://github.com/mobvoi/wenet

## 文件配置

根据model_onnx_template.yaml中的描述补全lang_char.txt关键文件和onnx_model文件夹，默认模型为16k采样率数据训练所得。

## 系统运行

在正确的环境下直接执行python wenet_online_decoder_onnx.py，执行offline的识别 和 读取wav文件仿流式识别。

## 参考

本代码参考[wenet_onlinedecode](https://github.com/jiay7/wenet_onlinedecode)进行修改

## 21.8.14更新

* 准备了两个模型,分别为离线conformer和在线transformer
* 修改了对长音频的离线解码方式,使用vad来一句一句进行解码（代码不一定完善