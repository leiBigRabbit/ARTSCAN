import math

# 定义实部和虚部
i = 1
j = 2

# 计算 e^(i+j)
result = 1 + math.exp(-(i*i + j*j) / 72)

print(result)
