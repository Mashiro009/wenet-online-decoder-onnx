# wenet onnx model

## 介绍

根据 https://github.com/wenet-e2e/wenet/blob/main/examples/aishell/s0/README.md

当前原始torch模型从 http://mobvoi-speech-public.ufile.ucloud.cn/public/wenet/aishell/20210204_conformer_exp.tar.gz 下载而来(为节省空间已删除final.pt)

通过 [wenet-onnx](https://github.com/Mashiro009/wenet-onnx)导出onnx模型 放在onnx_model文件夹内

该模型为离线conformer

来自aishell1的例子

因无法上传100MB以上的文件到github上,请前去[CSDN链接](https://download.csdn.net/download/MashiroRin/21094226)进行下载

或百度云(删除中文后下载)：
链接：https://pan盘.baidu百度.com/s/139mycaFN3JHNoY0xCHLjxw 
提取码：k3gk


## 测试

可以直接运行python wenet_online_decoder_onnx.py进行测试