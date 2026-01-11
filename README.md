# Trendalyze

## 结构

- mocker/mocker.py: 生成虚拟设备信息，根据剧本生成有时序的虚拟用户设备信息
- models/habits.py: 构建一个LSTM&Transformer encoder模型，该模型根据历史数据，对未来的设备事件进行预测。

## 结果

![results](docs/predict_results.png)

![result_text](docs/train_predict_results.png)

![model_arch](docs/model_arch.png)