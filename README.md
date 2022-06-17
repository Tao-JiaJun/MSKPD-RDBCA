# MSKPD-RDBCA

We will update the environment settings and the details of how to use the pre-trained model as soon as possible.

## Pre-trained models

You may download all models reported in the paper from Google Drive or Baidu Cloud.

We have trained our model with backbone of ResNet-18, ResNet-50 and MobileNet V2.

- Google (https://drive.google.com/drive/folders/1wmziQIaOQz1Pr_OndeWygZc3K81w7s-6?usp=sharing)

- Baidu (https://pan.baidu.com/s/12CDz25jR4X_6CYqcSd13yw the access code is：**mskp**)

## How to use

```bash
git clone https://github.com/Tao-JiaJun/MSKPD-RDBCA.git
cd mskpd-rdbca/
```

Please read the comment in eval_voc.py, and use the correct model and correct config files.
For instance, if you want to evaluate the model with ResNet-18 at resolution of 384x384, please change the **cfg file** first.

```python
# line 66
model = Detector(device, input_size=voc['input_size'], num_cls=20, strides = voc['strides'], scales=voc['scales'], cfg=RES18_RDB_384)
```

Second, select the model correspongding to the config file.

```python
# line 70
checkpoint = torch.load('./weights/RES18_RDB_384.pth',map_location=device)
```

Third, run eval_voc.py

```python
python eval_voc.py
```

Last, get the mean average precision.

``python
python evel/get_map.py
```
