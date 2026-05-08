# 在有 cmakeLists 的基础上 

cmake -B build 建立build 
cmkae --build build 指定编译

然后cd build ， ./编译后的文件



# ONNX_runtime 的作用

是的，ONNX 主要是推理核心；
但工程部署里，预处理和后处理通常还是要在 C++ 里自己补齐。
只是你这版模型已经把“归一化”内置进 ONNX 了，所以少做了一步。


IMU原始数据 -> 预处理 -> ONNX -> 后处理 -> 报警结果




ONNX 主要负责“模型前向计算”，也就是：

输入一组 6 轴数据 -> 经过网络层计算 -> 输出 change_logit

但完整部署通常分成 3 段：

1:
预处理
把原始 IMU 数据整理成模型能吃的格式。
比如：
取哪 6 个通道
通道顺序怎么排
数据类型转成 float
是否做窗口拼接
是否做归一化/去噪/滤波

2:
ONNX 推理
这一段由 onnxruntime 执行。
也就是：
把输入 tensor 喂进去
得到输出 tensor

3:
后处理
把模型输出变成业务判断结果。
比如：
logit -> sigmoid -> probability
probability >= 0.5 判为剧烈变化
连续几帧都超过阈值才报警
加防抖、平滑、状态机
