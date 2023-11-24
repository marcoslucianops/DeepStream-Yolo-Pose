# YOLO-NAS-Pose usage

**NOTE**: The yaml file is not required.

* [Convert model](#convert-model)
* [Edit the config_infer_primary_yolonas_pose file](#edit-the-config_infer_primary_yolonas_pose-file)

##

### Convert model

#### 1. Download the YOLO-NAS repo and install the requirements

```
git clone https://github.com/Deci-AI/super-gradients.git
cd super-gradients
pip3 install -r requirements.txt
python3 setup.py install
pip3 install onnx onnxsim onnxruntime
```

**NOTE**: It is recommended to use Python virtualenv.

#### 2. Copy conversor

Copy the `export_yolonas_pose.py` file from `DeepStream-Yolo-Pose/utils` directory to the `super-gradients` folder.

#### 3. Download the model

Download the `pth` file from [YOLO-NAS-Pose](https://sghub.deci.ai/) releases (example for YOLO-NAS-Pose S)

```
wget https://sghub.deci.ai/models/yolo_nas_pose_s_coco_pose.pth
```

**NOTE**: You can use your custom model.

#### 4. Convert model

Generate the ONNX model file (example for YOLO-NAS-Pose S)

```
python3 export_yolonas_pose.py -m yolo_nas_pose_s -w yolo_nas_pose_s_coco_pose.pth --dynamic
```

**NOTE**: Model names

```
-m yolo_nas_pose_s
```

or

```
-m yolo_nas_pose_m
```

or

```
-m yolo_nas_pose_l
```

**NOTE**: To change the inference size (defaut: 640)

```
-s SIZE
--size SIZE
-s HEIGHT WIDTH
--size HEIGHT WIDTH
```

Example for 1280

```
-s 1280
```

or

```
-s 1280 1280
```

**NOTE**: To simplify the ONNX model (DeepStream >= 6.0)

```
--simplify
```

**NOTE**: To use dynamic batch-size (DeepStream >= 6.1)

```
--dynamic
```

**NOTE**: To use static batch-size (example for batch-size = 4)

```
--batch 4
```

#### 5. Copy generated files

Copy the generated ONNX model file to the `DeepStream-Yolo-Pose` folder.

##

### Edit the config_infer_primary_yolonas_pose file

Edit the `config_infer_primary_yolonas_pose.txt` file according to your model (example for YOLO-NAS-Pose S)

```
[property]
...
onnx-file=yolo_nas_pose_s_coco_pose.onnx
...
parse-bbox-func-name=NvDsInferParseYoloPoseE
...
```
