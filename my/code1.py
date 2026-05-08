from torch.utils.data import TensorDataset
import torch

# 你的理解："将数据类里面填充别的类型也就是别的张量"
# 就像创建一个包含多个字段的数据结构


#这会报错，因为 MyDataClass 没有实现 __len__，所以 Python 不知道它的长度。它只是一个普通对象，不会自动像 dataset 一样支持 len() 或索引。
#如果你想让它更像数据集，至少要加上 __len__，最好再加 __getitem__：


# 方式1：使用普通类（传统方式）
class MyDataClass:
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
    
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
# ✅ 自动保证"个数相等"


print("样本1", len(dataset))
print("样本2", len(my_data))
print("dataset[0]:", dataset[0])
print("my_data[0]:", my_data[0])

# 这两个是一样的
print("dataset.tensors[0].shape", dataset.tensors[0].shape)
print("dataset.tensors[1].shape", dataset.tensors[1].shape)

print("dataset.tensors[0].size", dataset.tensors[0].size())
print("dataset.tensors[1].size", dataset.tensors[1].size())


x = torch.tensor([1,2,3])
print("x.size", x.size())
# dataset[0] 是一个样本，类型是 tuple
# dataset[0][0] 是这个样本的特征张量
# dataset[0][1] 是这个样本的标签张量