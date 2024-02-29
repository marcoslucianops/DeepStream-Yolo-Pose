# YOLOv8-Pose usage

**NOTE**: The yaml file is not required.

* [Convert model](#convert-model)
* [Edit the config_infer_primary_yoloV8_pose file](#edit-the-config_infer_primary_yolov8_pose-file)

##

### Convert model

#### 1. Install Ultralytics
To install the Ultralytics package along with all its requirements, make sure you are in a Python environment with version 3.8 or higher and PyTorch version 1.8 or higher. You can use pip to install the necessary packages.

```bash
pip3 install ultralytics
pip3 install onnx onnxsim onnxruntime
```

**NOTE**: **1**.It is recommended to use Python virtualenv.  **2**.Make sure to replace `pip3` with `pip` if you're using a Python environment where `pip` is the default package manager.

#### 2. Copy conversor

Copy the `export_yoloV8_pose.py` file from `DeepStream-Yolo-Pose/utils` directory to the `ultralytics` folder.

#### 3. Download the model

Download the `pt` file from [YOLOv8](https://github.com/ultralytics/assets/releases/) releases (example for YOLOv8s-Pose)

```
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-pose.pt
```

**NOTE**: You can use your custom model.

#### 4. Convert model

Generate the ONNX model file (example for YOLOv8s-Pose)

```
python3 export_yoloV8_pose.py -w yolov8s-pose.pt --dynamic
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

### Edit the config_infer_primary_yoloV8_pose file

Edit the `config_infer_primary_yoloV8_pose.txt` file according to your model (example for YOLOv8s-Pose)

```
[property]
...
onnx-file=yolov8s-pose.onnx
...
parse-bbox-func-name=NvDsInferParseYoloPose
...
```
