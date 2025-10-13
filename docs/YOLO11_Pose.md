# YOLO11-Pose usage

**NOTE**: The yaml file is not required.

* [Convert model](#convert-model)
* [Compile the lib](#compile-the-lib)
* [Edit the config_infer_primary_yolo11_pose file](#edit-the-config_infer_primary_yolo11_pose-file)

##

### Convert model

#### 1. Download the YOLO11 repo and install the requirements

```
git clone https://github.com/ultralytics/ultralytics.git
cd ultralytics
pip3 install -e .
pip3 install onnx onnxslim onnxruntime
```

**NOTE**: It is recommended to use Python virtualenv.

#### 2. Copy conversor

Copy the `export_yolo11_pose.py` file from `DeepStream-Yolo-Pose/utils` directory to the `ultralytics` folder.

#### 3. Download the model

Download the `pt` file from [YOLO11](https://github.com/ultralytics/assets/releases/) releases (example for YOLO11s-Pose)

```
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-pose.pt
```

**NOTE**: You can use your custom model.

#### 4. Convert model

Generate the ONNX model file (example for YOLO11s-Pose)

```
python3 export_yolo11_pose.py -w yolo11s-pose.pt --dynamic
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

**NOTE**: To simplify the ONNX model

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

### Edit the config_infer_primary_yolo11_pose file

Edit the `config_infer_primary_yolo11_pose.txt` file according to your model (example for YOLO11s-Pose)

```
[property]
...
onnx-file=yolo11s-pose.onnx
...
num-detected-classes=80
...
parse-bbox-func-name=NvDsInferParseYoloPose
...
```

**NOTE**: The **DeepStream-Yolo-Pose** requires

```
[property]
...
maintain-aspect-ratio=1
symmetric-padding=1
...
```
