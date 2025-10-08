# YOLOv7-Pose usage

**NOTE**: The yaml file is not required.

* [Convert model](#convert-model)
* [Compile the lib](#compile-the-lib)
* [Edit the config_infer_primary_yoloV7_pose file](#edit-the-config_infer_primary_yolov7_pose-file)

##

### Convert model

#### 1. Download the YOLOv7 repo and install the requirements

```
git clone https://github.com/WongKinYiu/yolov7.git
cd yolov7
pip3 install -r requirements.txt
pip3 install onnx onnxslim onnxruntime
```

**NOTE**: It is recommended to use Python virtualenv.

#### 2. Copy conversor

Copy the `export_yoloV7_pose.py` file from `DeepStream-Yolo-Pose/utils` directory to the `yolov7` folder.

#### 3. Download the model

Download the `pt` file from [YOLOv7](https://github.com/WongKinYiu/yolov7/releases/) releases (example for YOLOv7-w6-Pose)

```
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-pose.pt
```

**NOTE**: You can use your custom model.

#### 4. Reparameterize your model (for custom models)

Custom YOLOv7 models cannot be directly converted to engine file. Therefore, you will have to reparameterize your model using the code [here](https://github.com/WongKinYiu/yolov7/blob/main/tools/reparameterization.ipynb). Make sure to convert your custom checkpoints in YOLOv7 repository, and then save your reparmeterized checkpoints for conversion in the next step.

#### 5. Convert model

Generate the ONNX model file (example for YOLOv7-w6-Pose)

```
python3 export_yoloV7_pose.py -w yolov7-w6-pose.pt --dynamic --p6
```

**NOTE**: To convert a P6 model

```
--p6
```

**NOTE**: To change the inference size (defaut: 640 / 1280 for `--p6` models)

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

#### 6. Copy generated files

Copy the generated ONNX model file and labels.txt file (if generated) to the `DeepStream-Yolo-Pose` folder.

##

### Compile the lib

1. Open the `DeepStream-Yolo-Pose` folder and compile the lib

2. Set the `CUDA_VER` according to your DeepStream version

```
export CUDA_VER=XY.Z
```

* x86 platform

  ```
  DeepStream 8.0 = 12.8
  DeepStream 7.1 = 12.6
  DeepStream 7.0 / 6.4 = 12.2
  DeepStream 6.3 = 12.1
  DeepStream 6.2 = 11.8
  DeepStream 6.1.1 = 11.7
  DeepStream 6.1 = 11.6
  DeepStream 6.0.1 / 6.0 = 11.4
  ```

* Jetson platform

  ```
  DeepStream 8.0 = 13.0
  DeepStream 7.1 = 12.6
  DeepStream 7.0 / 6.4 = 12.2
  DeepStream 6.3 / 6.2 / 6.1.1 / 6.1 = 11.4
  DeepStream 6.0.1 / 6.0 = 10.2
  ```

3. Make the lib

```
make -C nvdsinfer_custom_impl_Yolo_pose clean && make -C nvdsinfer_custom_impl_Yolo_pose
```

##

### Edit the config_infer_primary_yoloV7_pose file

Edit the `config_infer_primary_yoloV7_pose.txt` file according to your model (example for YOLOv7-w6-Pose)

```
[property]
...
onnx-file=yolov7-w6-pose.onnx
...
num-detected-classes=1
...
parse-bbox-func-name=NvDsInferParseYoloPose
...
```
