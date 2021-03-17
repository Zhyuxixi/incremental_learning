# LWF
pytorch implementation of "Large Scale Incremental Learning" from 
paper：https://link.springer.com/chapter/10.1007/978-3-319-46493-0_37
code：https://github.com/lizhitwo/LearningWithoutForgetting

# Dataset
Download Cifar100 dataset from https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz

Put meta, train, test into ./cifar100

# Train
```
python main.py
```

# Result

|    |  20  |  40  |  60  |  80  |  100  |
| ---- | ---- | ---- | ---- | ---- | ---- |
|  Implementation  | 0.815 | 0.655 | 0.488 | 0.406 | 0.320|




