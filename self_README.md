------------------------------------------------------------------------------

dataloader = DataLoader(
    dataset,           # 数据集
    batch_size=32,     # 🔥 批次大小（你问的关键）
    shuffle=True,      # 是否打乱数据（训练时True，测试时False）
    num_workers=4,     # 多线程加载数据（加快速度）
    drop_last=False,   # 是否丢弃最后一个不完整的batch
    pin_memory=True,   # 加速GPU传输
)


DataLoader 函数


---------------------------------------------------------------------------------------------

TensorDataset 是 PyTorch 中用于将多个张量包装成数据集的工具函数，让多个张量可以按索引一起取出。 

张量第一维大小必须相同


    TensorDataset = 把多个张量"粘"在一起

    最简单：不需要写自定义 Dataset 类

    用途：快速将张量数据包装成 PyTorch Dataset

    配合：通常和 DataLoader 一起使用

    限制：所有张量第一维长度必须相同

简单记忆：TensorDataset 就是给 DataLoader 准备数据的，当你的数据已经是张量格式时，用它最方便！


-----------------------------------------------------

# Data 就像一个"数据容器"或"数据类"
# TensorDataset 就是把这个容器里填上多个张量
# 并且保证每个张量的"样本数"相同


形象类比
python

# 想象一个表格（Excel）
# TensorDataset 就是把几列数据"粘"成一整个表格

列1 = [特征1, 特征2, 特征3, ...]  # 张量1
列2 = [标签1, 标签2, 标签3, ...]   # 张量2
列3 = [权重1, 权重2, 权重3, ...]   # 张量3

# TensorDataset = 把这3列并排放在一起
数据集 = TensorDataset(列1, 列2, 列3)

# 现在你可以按"行"访问
第1行 = 数据集[0]  # (特征1, 标签1, 权重1)
第2行 = 数据集[1]  # (特征2, 标签2, 权重2)


--------------------------------------------------------------------------

# ################# 代码验证
from torch.utils.data import TensorDataset
import torch

# 你的理解："将数据类里面填充别的类型也就是别的张量"
# 就像创建一个包含多个字段的数据结构

# 方式1：使用普通类（传统方式）
class MyDataClass:
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

my_data = MyDataClass(
    features=torch.randn(100, 10),
    labels=torch.randint(0, 2, (100,))
)
# 但这样不能直接按索引访问 (my_data[0])，需要自己实现

# 方式2：使用 TensorDataset（PyTorch 方式）
dataset = TensorDataset(
    torch.randn(100, 10),  # 特征张量
    torch.randint(0, 2, (100,))  # 标签张量
)
# ✅ 可以直接按索引访问 dataset[0]
# ✅ 自动保证"个数相等"（都是100个样本）




MyDataClass 是“存数据的普通类”
TensorDataset 是“PyTorch 规定格式的数据集类”
TensorDataset 的重点不是“能装张量”，而是“能按样本索引访问，并和 DataLoader 配合



总结为：

    Data 是概念上的数据结构/数据类 - 用来组织数据

    TensorDataset 是具体实现 - 把多个张量"粘"在一起

    填充不同类型的张量 - 特征、标签、权重等都可以

    个数必须相等 - 第一维长度（样本数）必须一致

一句话：TensorDataset 就像个"数据容器"，让你能把多个张量按"行"对齐存放，然后方便地按索引访问每一"行"的所有数据。


# 用哪个python  
python 用哪个，不看你“有没有想用虚拟环境”，只看 which python 指向哪里。

which python
which pip
python -m pip list



ros 用的是系统环境里面的 python 不能用虚拟环境的python  前面带(base)也是虚拟环境


# python 学习
__xx__ 的意义：Python 的"魔法方法"

__xx__ 是 Python 中的特殊方法（也称为魔术方法或双下方法），它们定义了类的对象如何与 Python 内置操作交互。
核心意义

让自定义类的对象像 Python 内置类型一样工作

__xx__ 就像类的"契约"或"接口"，告诉 Python：

    __len__：我知道怎么计算长度，你可以用 len() 问我

    __getitem__：我知道怎么按索引取值，你可以用 [] 问我

    __init__：我知道怎么初始化，创建对象时自动调用我

    __add__：我知道怎么相加，你可以用 + 运算符


__xx__ 的意义：

    标准化：让不同类用统一的方式与 Python 交互

    语法糖：让代码更优雅（len(obj) 比 obj.get_length() 更自然）

    集成：让自定义类无缝融入 Python 生态系统

    契约：告诉 Python 解释器"我支持这个功能"


# randint: 随机整数（指定范围）
integers = torch.randint(0, 10, (1000,))
print(f"randint: {integers[:5]}")  # tensor([3, 7, 1, 9, 4])

# randn: 标准正态分布（浮点数，均值0，方差1）
normal = torch.randn(1000)
print(f"randn: {normal[:5]}")  # tensor([0.52, -1.23, 0.87, -0.45, 1.34])

# rand: 均匀分布（浮点数，[0,1)）
uniform = torch.rand(1000)
print(f"rand: {uniform[:5]}")  # tensor([0.123, 0.456, 0.789, 0.234, 0.567])


tensor(0) = “值是 0 的张量”

tensor(0)：0 维张量，标量
tensor([0])：1 维张量，长度为 1
你可以这样理解：

tensor(0) 像一个单独的数字
tensor([0]) 像一个只装了一个元素的数组




# 张量就是描述 ，向量的维数

向量就是 1 维张量

# 1维张量时，len() = 元素个数
x = torch.tensor([1, 2, 3])  # 形状 [3]
print(len(x))        # 3
print(x.numel())     # 3（相等）

# 多维张量， len() = 张量的行数

# len() 永远返回张量的第一维（最外层）的大小
# 无论张量是几维




# 1维: len([a, b, c])          → 3
# 2维: len([[a,b], [c,d]])     → 2
# 3维: len([[[...]], [[...]]]) → 2
# 4维: len(批次, 通道, 高, 宽) → 批次大小



shape 的表达方式
python

import torch

x = torch.randn(2, 3, 4)

# 方式1：直接打印
print(x.shape)        # torch.Size([2, 3, 4])

# 方式2：使用 size() 方法
print(x.size())       # torch.Size([2, 3, 4])

# 方式3：获取特定维度大小
print(x.shape[0])     # 2 (第一维大小)
print(x.shape[1])     # 3 (第二维大小)
print(x.shape[2])     # 4 (第三维大小)

# 方式4：使用 size() 获取特定维度
print(x.size(0))      # 2
print(x.size(1))      # 3
print(x.size(2))      # 4

不同维度张量的 shape
张量描述	创建代码	shape	含义
标量	torch.tensor(5)	torch.Size([])	0维，一个数字
向量	torch.tensor([1,2,3])	torch.Size([3])	1维，3个元素
矩阵	torch.randn(2,3)	torch.Size([2,3])	2行3列
3维	torch.randn(2,3,4)	torch.Size([2,3,4])	2个3x4矩阵
4维	torch.randn(32,3,224,224)	torch.Size([32,3,224,224])	32张RGB 224x224图


----------------------------------------------------------------------------------------------------------

    Data 是概念上的数据结构/数据类 - 用来组织数据

    TensorDataset 是具体实现 - 把多个张量"粘"在一起

    填充不同类型的张量 - 特征、标签、权重等都可以

    个数必须相等 - 第一维长度（样本数）必须一致

一句话：TensorDataset 就像个"数据容器"，让你能把多个张量按"行"对齐存放，然后方便地按索引访问每一"行"的所有数据。

就是不同张量的 len( )，要一致

重要内容:

# 1维张量时，len() = 元素个数
# 记忆方式：len() 看最外层的方括号里有多少个元素

"张量是存储和组织数据的容器" ， 

张量的维度顺序是约定俗成的，不是数学强制规定的
你可以按任何顺序排列维度，只要保持一致就行。

关键是在你的代码/项目中保持一致性，并遵循所用框架的惯例！


# randint: 随机整数（指定范围）
integers = torch.randint(0, 10, (1000,))
print(f"randint: {integers[:5]}")  # tensor([3, 7, 1, 9, 4])

# randn: 标准正态分布（浮点数，均值0，方差1）
normal = torch.randn(1000)
print(f"randn: {normal[:5]}")  # tensor([0.52, -1.23, 0.87, -0.45, 1.34])

# rand: 均匀分布（浮点数，[0,1)）
uniform = torch.rand(1000)
print(f"rand: {uniform[:5]}")  # tensor([0.123, 0.456, 0.789, 0.234, 0.567])

# python 学习
__xx__ 是 Python 中的特殊方法（也称为魔术方法或双下方法），它们定义了类的对象如何与 Python 内置操作交互。
核心意义

让自定义类的对象像 Python 内置类型一样工作,自然


# 关键的一点， 1维向量 和 2维张量
特性	(100,)	(1, 100)
维度	1维	2维

torch.Size([100])  torch.Size([1,100])
形状	[100]	[1, 100]
视觉	一条线	一个只有一行的表格
索引方式	[i]	[0, i]
括号层级	一层 []	两层 [[]]
批量操作	不能直接作为batch	可以看作batch_size=1


(100,) = 一条线，100个点

(1, 100) = 只有一行的一列的表格

后者多一层括号，需要多一个索引

在神经网络中，需要注意（很多API对形状敏感）

# ############################ 很关键 ， 2维度张量可以当作一个batch size
在数学中一样在编程中不一样

数学中，向量就是向量
(100,) 和 (1, 100) 都表示 100 维向量
没有"维度"这个词的歧义

与matlab 中不一样
MATLAB 里没有单独的数据类型叫“向量”，向量就是特殊的矩阵。虽然形状（维度）不同会让运算规则天差地别，但它们在本质上就是同一个东西。

python 的 torch 中， 向量和矩阵不是一个东西



# 如何部署强化学习的模型

部署时通常只用 actor，critic 和 optimizer 一般不需要上线。

.pt 文件是训练检查点，

# 部署流程
model_299.pt 不是直接部署终点，它更像训练存档

你需要先找到 actor 的网络定义，部署时通常只导出并上线 actor

如果你想“更方便跨平台部署”，更推荐：

先把 actor 导出成 ONNX
再用 ONNX Runtime 部署


state_dict 
是 PyTorch 中的一个核心概念，简单来说，它就是 Python 字典对象，用来存储模型中所有可学习参数（权重和偏置）的键值对。


# 训练的流程 

用 Isaac Lab 的 rsl_rl/train.py 训练后，默认会生成 checkpoint 权重文件，比如 model_0.pt、model_50.pt、model_299.pt。这些 .pt 是训练保存的模型状态，不是专门给部署用的 ONNX。

之后你运行 rsl_rl/play.py 并加载某个 checkpoint 时，脚本里会顺手把推理用 policy 导出成：

exported/policy.pt

exported/policy.onnx

也就是说，常见流程就是：

train.py 产出训练 checkpoint：model_xxx.pt

play.py 加载 checkpoint
play.py 自动导出 policy.onnx 和 policy.pt

你现在这个 MyRobot_flat 就正是这样：

训练权重在 model_299.pt

导出 ONNX 在 policy.onnx

补一句更准确的说法：

不是“play 运行完才一定有 onnx”，而是这个 play.py 脚本内部写了导出逻辑，所以你一跑它，它就会帮你导出。



#  文件	          什么时候出现	                       里面是什么	              主要用途
model_299.pt	训练 train.py 时定期保存	训练 checkpoint，包含策略参数，以及恢复训练所需状态	继续训练、恢复训练、评估时加载原始训练结果
policy.pt	跑 play.py 时自动导出	TorchScript 推理模型，只保留推理需要的 policy	       在 PyTorch / LibTorch 环境里部署推理
policy.onnx	跑 play.py 时自动导出	ONNX 推理模型，只保留推理需要的 policy	        跨框架部署，给 ONNX Runtime、TensorRT 等用



model_xxx.pt 是训练 checkpoint。它通常不只是 actor 网络权重，还会带上训练过程相关的状态，比如当前模型参数、优化器状态、可能还有归一化器、训练轮次等。这类文件的重点是“我以后还能继续接着训，或者完整恢复当时的训练上下文”。

policy.pt 是从 checkpoint 里抽出来并转换过的 TorchScript 推理模型。它的重点不是恢复训练，而是“拿来直接前向推理”。在你这个 Isaac Lab 导出逻辑里，policy.pt 本质上只保留 policy 需要的部分，主要就是 actor，以及可能用到的 observation normalizer；不会保留 optimizer 这类训练状态。

你可以这样记：

model_xxx.pt：面向训练
用途是 resume、继续训练、重新加载 runner。
policy.pt：面向部署
用途是 obs -> action，直接推理。
再说得更直白一点：

你想“接着训”，用 model_299.pt
你想“给机器人跑动作输出”，用 policy.pt 或 policy.onnx
补一个很关键的点：
虽然它们都叫 .pt，但格式和用途并不一样。

model_xxx.pt 往往是 torch.save(...) 存出来的 checkpoint 字典
policy.pt 是 TorchScript，属于 torch.jit.script(...).save(...) 导出的可部署模型
所以这两个文件不能简单互换。policy.pt 不能拿去无缝恢复训练，model_xxx.pt 也不是最适合直接部署的推理文件。


# 网络参数相关的

model_xxx.pt 也带 actor 的网络参数，通常也间接对应那套网络结构；
policy.pt 则是把“网络结构 + 权重”一起固化成了一个可直接推理的 TorchScript 模型。

区别不在于“一个带结构、一个不带结构”，而在于“结构是怎么保存的”。

model_xxx.pt
更像 checkpoint 数据包，通常主要保存参数和训练状态。
它本身一般不是一个独立可执行的模型定义文件，加载它时还需要代码里先有对应的网络类定义，然后再把参数塞进去。
policy.pt
是已经导出的 TorchScript 模型。
它把前向计算图和权重都打包好了，所以更接近“拿来就能推理”。
你可以把它理解成：

model_xxx.pt：像“零件包 + 存档信息”
还需要 Isaac Lab / rsl_rl 里的网络代码把它重新组装起来
policy.pt：像“已经装好的整机”
可以直接拿来跑推理
所以如果一句话总结：

不是 model_xxx.pt 不带 actor，
而是 model_xxx.pt 不会像 policy.pt 那样，把 actor 以可直接部署执行的形式完整封装好。



# 如何推理

果你是在 Isaac Lab / rsl_rl 这套流程里，用 runner 去加载 model_xxx.pt，那通常是正常的，不会报错。比如 play.py 里加载的就是这种 checkpoint。

但如果你把 model_xxx.pt 当成 policy.pt 那样，直接拿去做 TorchScript 推理加载，例如这种思路：

torch.jit.load("model_299.pt")
那大概率会报错。因为 model_xxx.pt 不是 TorchScript 导出模型，它是训练 checkpoint。

对应地：

model_xxx.pt 的常见加载方式：
先创建网络 / runner，再 load checkpoint

policy.pt 的常见加载方式：
直接 torch.jit.load(...)


policy.onnx 的常见加载方式：
用 ONNX Runtime / TensorRT 等加载
所以关键不是“能不能 load”，而是“用哪种方式 load”。

你可以这样记：

想继续训练、在 Isaac Lab 里回放：model_xxx.pt
想直接部署推理：policy.pt 或 policy.onnx
如果你愿意，我下一条可以直接给你写三个最小例子：

model_xxx.pt 正确加载方式
policy.pt 正确加载方式
policy.onnx 正确加载方式


# -----------------------------------------------------------------------
1. 明确部署输入输出

你先要确认这几个问题：

输入是什么：观测 obs
输入 shape 是多少：比如 [1, obs_dim]


输出是什么：动作 action
输出 shape 是多少：比如 [1, act_dim]


推理前是否要做 obs_normalizer
推理后是否要做动作裁剪或缩放
这一步非常重要。很多“模型能跑但机器人不动”的问题，不是模型坏了，而是漏了训练时的归一化和后处理。

2. 本地先用 ONNX Runtime 跑通

最小推理代码通常长这样：

import onnxruntime as ort
import numpy as np

onnx_path = "/home/cy/onnx_study/my/model/actor.onnx"

session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

obs = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)   # 按你的 obs_dim 改
action = session.run(None, {"obs": obs})[0]

print("action:", action)
print("shape:", action.shape)
注意这里的 "obs" 必须和 ONNX 模型的输入名一致。
如果不确定，可以先打印：

print(session.get_inputs()[0].name, session.get_inputs()[0].shape)
print(session.get_outputs()[0].name, session.get_outputs()[0].shape)


3. 把预处理和后处理补完整

如果训练时有 obs_normalizer，部署时也要保留同样逻辑。通常流程是：

obs
-> 归一化
-> actor.onnx 推理
-> 动作裁剪/缩放
-> 发给机器人/仿真器
也就是说部署不只是“跑一下 ONNX”，而是要把整条推理链接起来。

4. 按目标场景选部署方式

如果你是这几种目标，可以这样选：

本机 Python 程序调用：ONNX Runtime
C++ 程序调用：ONNX Runtime C++
Jetson / 边缘设备：ONNX Runtime 或 TensorRT
Web 或跨平台服务：包一层 FastAPI
机械臂/小车实时控制：推理模块单独封装成一个控制节点
最简单实用的一版，通常是 Python + ONNX Runtime + FastAPI。

一个最小 API 例子：

from fastapi import FastAPI
import onnxruntime as ort
import numpy as np

app = FastAPI()
session = ort.InferenceSession("actor.onnx", providers=["CPUExecutionProvider"])

@app.post("/infer")
def infer(obs: list[float]):
    x = np.array([obs], dtype=np.float32)
    action = session.run(None, {"obs": x})[0]
    return {"action": action[0].tolist()}
启动：

uvicorn app:app --host 0.0.0.0 --port 8000
调用方只需要发观测，服务返回动作。

你现在最推荐的做法

先别碰 .pt，直接：

用 actor.onnx + onnxruntime 写一个本地推理脚本
确认输入输出 shape
补上 obs_normalizer 和动作后处理
再封装成 API 或控制节点
你当前真正要确认的核心只有两个：

ONNX 输入名和 shape
训练时是否用了 observation normalization
如果你愿意，我下一步可以直接帮你写一个可运行的 deploy_onnx.py，你只要把 actor.onnx 路径和一条观测向量填进去就能测。



# 训练的时候是怎么定义模型的
在强化学习里，训练的输入输出不是像监督学习那样“人工标好标签”，而是这样定义的：

输入：环境给智能体的 observation，也就是当前状态信息
输出：策略网络给出的 action，也就是下一步动作
你现在这个 Isaac Lab 的 actor 就是：

输入 obs：36 维观测
输出 action：8 维动作
可以把它理解成：

obs_t  ->  policy(actor)  ->  action_t
然后环境再根据这个动作往前走一步：

action_t -> environment -> reward_t, obs_{t+1}, done
所以强化学习训练里真正流动的是这一串：

智能体拿到当前观测 obs_t
actor 输出动作 action_t
环境执行动作
返回奖励 reward_t 和下一时刻观测 obs_{t+1}
算法根据这些数据更新网络参数
和监督学习最大的区别是：

监督学习：
输入是 x，输出标签是人工给的 y
强化学习：
输入是 obs，输出是 action
但“正确答案动作”不是提前给你的，而是靠奖励 reward 反过来优化出来
如果放到你这个模型上，通常是这样定义：

输入 obs
可能包含机器人当前的状态，比如：

关节角度
关节速度
机身姿态
目标位置
上一时刻动作
接触信息
这些拼起来正好 36 维。

输出 action
通常是 8 维控制量，比如：

8 个关节的目标位置增量
8 个关节的目标力矩
8 个电机控制指令
具体含义要看你的任务配置。

还有一个常见点：

actor 输出动作
critic 输出价值 value
也就是说在训练时常常有两个网络：

Actor 输入 obs，输出 action
Critic 输入 obs，输出一个标量价值
所以：

部署时通常只用 actor
训练时 actor 和 critic 都要用
最短一句话总结：

强化学习里，训练时的“输入”是环境观测 obs，“输出”是策略产生的动作 action；它不像分类任务那样有固定标签，而是通过奖励信号来学会输出更好的动作。



#  my_robot 任务配置怎么继承下来，以及观测张量怎么一路被拼成 

这条链路分两层看最清楚：my_robot 任务配置怎么继承下来，以及观测张量怎么一路被拼成 policy 输入。

配置继承

你现在用的 my_robot 环境入口注册在 init.py (line 11)。如果是平地任务，入口类是 flat_env_cfg.py (line 12) 里的 MyRobotFlatEnvCfg。

它的继承链是：

MyRobotFlatEnvCfg

flat_env_cfg.py (line 12)
MyRobotRoughEnvCfg

rough_env_cfg.py (line 16)
LocomotionVelocityRoughEnvCfg

velocity_env_cfg.py (line 264)
ManagerBasedRLEnvCfg

在 isaaclab.envs 里，被 LocomotionVelocityRoughEnvCfg 继承

这几层分别做的事大概是：


LocomotionVelocityRoughEnvCfg：定义通用 locomotion 任务，包括 scene、actions、observations、rewards、events。

MyRobotRoughEnvCfg：把通用 locomotion 的 robot 换成你的 MY_ROBOT_CFG，并覆写部分 reward / event / termination。

MyRobotFlatEnvCfg：再从 rough 版本改成平地，关键是把 height_scan 去掉。

ManagerBasedRLEnvCfg：最底层 RL 环境配置基类。


观测继承和拼接

观测不是在 my_robot 里重新定义的，my_robot 只是继承并局部改掉。

链路是：

velocity_env_cfg.py (line 115) 定义 ObservationsCfg
其中 velocity_env_cfg.py (line 119) 的 PolicyCfg(ObsGroup) 定义各个 ObsTerm
在 velocity_env_cfg.py (line 141) 里 self.concatenate_terms = True
MyRobotFlatEnvCfg.__post_init__() 在 flat_env_cfg.py (line 25) 和 flat_env_cfg.py (line 26) 把 height_scan 关掉
运行时由 observation_manager.py (line 344) 的 compute_group() 逐项算 observation
最后在 observation_manager.py (line 433) 用 torch.cat(...) 拼成一个 tensor




# 看 policyCfg 里面的配置 ， 知道输入的张量是多少维度的

对，就是这段。这里定义的是“候选观测项”，不是每个环境都会把它们全用上。

你之所以看到 36，是因为 my_robot 的 flat 配置把最后那个 height_scan 关掉了，在 flat_env_cfg.py (line 25) 和 flat_env_cfg.py (line 26)：

python



self.scene.height_scanner = None
self.observations.policy.height_scan = None



所以 flat 环境实际参与拼接的只有前 7 项：


base_lin_vel: 3

base_ang_vel: 3

projected_gravity: 3

velocity_commands: 3

joint_pos: 8

joint_vel: 8

actions: 8


加起来就是：

python



3 + 3 + 3 + 3 + 8 + 8 + 8 = 36



如果是 rough 环境，height_scan 还在，维度会更大，不是 36。

补一句每项为什么是这些维度：


base_lin_vel / base_ang_vel / projected_gravity 都是 XYZ，所以各 3 维

velocity_commands 对这个任务实际是 3 维命令

你的机器人有 8 个关节，所以 joint_pos、joint_vel、actions 都是 8 维


如果你愿意，我可以直接把它展开成：


obs[0] 到 obs[35] 每一维分别是什么字段。



# 
个关节名来自 URDF，顺序通常就是模型里的 joint 顺序；你这个 URDF 里依次是 quadruped_8dof.urdf (line 210)


关节 8 维顺序：

front_left_1_joint
front_left_2_joint
front_right_1_joint
front_right_2_joint
rear_left_1_joint
rear_left_2_joint
rear_right_1_joint
rear_right_2_joint

obs[0]   = base_lin_vel.x
obs[1]   = base_lin_vel.y
obs[2]   = base_lin_vel.z

obs[3]   = base_ang_vel.x
obs[4]   = base_ang_vel.y
obs[5]   = base_ang_vel.z

obs[6]   = projected_gravity.x
obs[7]   = projected_gravity.y
obs[8]   = projected_gravity.z

obs[9]   = velocity_commands[0]
obs[10]  = velocity_commands[1]
obs[11]  = velocity_commands[2]

obs[12]  = joint_pos(front_left_1_joint)
obs[13]  = joint_pos(front_left_2_joint)
obs[14]  = joint_pos(front_right_1_joint)
obs[15]  = joint_pos(front_right_2_joint)
obs[16]  = joint_pos(rear_left_1_joint)
obs[17]  = joint_pos(rear_left_2_joint)
obs[18]  = joint_pos(rear_right_1_joint)
obs[19]  = joint_pos(rear_right_2_joint)

obs[20]  = joint_vel(front_left_1_joint)
obs[21]  = joint_vel(front_left_2_joint)
obs[22]  = joint_vel(front_right_1_joint)
obs[23]  = joint_vel(front_right_2_joint)
obs[24]  = joint_vel(rear_left_1_joint)
obs[25]  = joint_vel(rear_left_2_joint)
obs[26]  = joint_vel(rear_right_1_joint)
obs[27]  = joint_vel(rear_right_2_joint)

obs[28]  = last_action(front_left_1_joint)
obs[29]  = last_action(front_left_2_joint)
obs[30]  = last_action(front_right_1_joint)
obs[31]  = last_action(front_right_2_joint)
obs[32]  = last_action(rear_left_1_joint)
obs[33]  = last_action(rear_left_2_joint)
obs[34]  = last_action(rear_right_1_joint)
obs[35]  = last_action(rear_right_2_joint)

joint_pos 用的是 mdp.joint_pos_rel，所以它不是绝对关节角，而是“当前关节角 - default joint pos”，定义在 observations.py (line 212)

joint_vel 用的是 mdp.joint_vel_rel，这里默认速度一般是 0，所以通常就是当前关节速度，定义在 observations.py (line 257)

actions 用的是 mdp.last_action，就是上一步 policy 输出的 8 维动作，定义在 observations.py (line 657)



# onnx 模型创建
1. 核心对象的创建

    if (!ctx) ctx = std::make_unique<ModelContext>(); (准备容器)

    Ort::SessionOptions session_options; (即便不加优化项，也必须传一个空的配置对象)

    ctx->session = std::make_unique<Ort::Session>(...); (将 .onnx 文件载入内存，这是最核心的一步)

2. 获取输入输出元数据 (Metadata)
模型像一个黑盒，你必须知道它有几个入口、几个出口、叫什么名字、需要什么形状的数据。

    GetInputCount() / GetOutputCount() (获取节点数量)

    GetInputNameAllocated() / GetOutputNameAllocated() (获取节点名称)

    GetInputTypeInfo()....GetShape() (获取维度，比如 [1, 3, 224, 224])

3. 名称指针的格式转换

    ctx->input_names_raw 和 output_names_raw 的组装。

        原因： ONNX Runtime 的底层是 C API，它的 Run 函数只认 const char* const* 这种老式的 C 指针数组，不认现代的 std::string。这一步的转换是 API 强制要求的。

4. 内存位置声明

    Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU)

        原因： ORT 需要知道你的数据是放在 CPU 内存还是 GPU 显存里，否则它不知道去哪里取数据。




# 对，普通神经网络“本身”不一定需要多帧。

要不要多帧，不取决于它是不是神经网络，而取决于：

当前这一帧信息，够不够你做判断。

如果单帧就够，比如：

一张图片分类猫狗
根据当前温度预测是否过热
根据当前状态做一个静态判断
那单帧输入就行，不需要多帧。

但如果任务里“时间变化”很重要，单帧看不出来，就常常需要多帧或历史信息。比如：

判断物体是朝左还是朝右运动
强化学习里判断速度、加速度、趋势
语音识别
视频动作识别
举个最简单的例子：

只看一张小球图片，你只能知道“小球在这里”；
看两三帧，你才能知道“小球是在往左走还是往右走”。

所以：

普通前馈网络可以只吃单帧
如果任务需要时序信息，就要人为把多帧拼进去，或者改用 RNN / LSTM / Transformer 这类能处理序列的网络
在强化学习里经常拼多帧，不是因为“神经网络必须多帧”，而是因为单帧 obs 往往不够描述环境的动态状态。



frame stack  和 RNN

机器人 RL 更喜欢 frame stack，不是因为 RNN 不强，而是因为大多数控制任务里，短时历史就够了，而 frame stack 更简单、更稳、更好训。  

机器人很多时候只需要“短时历史”
    比如控制里最重要的是：
    速度趋势
    关节最近怎么动
    动作刚刚下发了什么
    传感器最近几帧怎么变化

frame stack 更容易训练稳定
    RNN 在 RL 里训练通常更麻烦，因为：
    序列展开更复杂
    梯度传播更难
    更容易不稳定
    采样和训练代码都更绕



# 一句话总结

frame_stack 在这份部署代码里是通过 inference.yaml 设置的；
但它通常不是部署阶段独立设计出来的，而是训练时（比如 Isaac Lab 里）就已经定义好的策略输入结构，部署时必须和训练保持一致。




# 输入 obs 的拼接

1：维度
2：frame_stack 


# obs 获得 

1： 存入
2： 拼接