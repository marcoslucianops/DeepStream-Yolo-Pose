# DeepStream-Yolo-Pose

NVIDIA DeepStream SDK application for YOLO-Pose models

--------------------------------------------------------------------------------------------------
### YOLO objetct detection models and other infos: https://github.com/marcoslucianops/DeepStream-Yolo
--------------------------------------------------------------------------------------------------

### Supported models

* [YOLOv8](https://github.com/ultralytics/ultralytics)

##

### Basic usage

#### 1. Download the DeepStream-Yolo-Pose repo

```
git clone https://github.com/marcoslucianops/DeepStream-Yolo-Pose.git
cd DeepStream-Yolo-Pose
```

#### 2. Download the YOLOv8 repo and install the requirements

```
git clone https://github.com/ultralytics/ultralytics.git
cd ultralytics
pip3 install -r requirements.txt
python3 setup.py install
pip3 install onnx onnxsim onnxruntime
```

**NOTE**: It is recommended to use Python virtualenv.

#### 3. Copy conversor

Copy the `export_yoloV8_pose.py` file from `DeepStream-Yolo-Pose/utils` directory to the `ultralytics` folder.

#### 4. Download the model

Download the `pt` file from [YOLOv8](https://github.com/ultralytics/assets/releases/) releases (example for YOLOv8s-Pose)

```
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-pose.pt
```

**NOTE**: You can use your custom model.

#### 5. Convert model

Generate the ONNX model file (example for YOLOv8s)

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

**NOTE**: To use implicit batch-size (example for batch-size = 4)

```
--batch 4
```

**NOTE**: If you are using the DeepStream 5.1, remove the `--dynamic` arg and use opset 12 or lower. The default opset is 16.

```
--opset 12
```

#### 6. Copy generated files

Copy the generated ONNX model file to the `DeepStream-Yolo-Pose` folder.

#### 7. Compile the libs

Export the CUDA_VER env according to your DeepStream version and platform:

* DeepStream 6.3 on x86 platform

  ```
  export CUDA_VER=12.1
  ```

* DeepStream 6.2 on x86 platform

  ```
  export CUDA_VER=11.8
  ```

* DeepStream 6.1.1 on x86 platform

  ```
  export CUDA_VER=11.7
  ```

* DeepStream 6.1 on x86 platform

  ```
  export CUDA_VER=11.6
  ```

* DeepStream 6.0.1 / 6.0 on x86 platform

  ```
  export CUDA_VER=11.4
  ```

* DeepStream 5.1 on x86 platform

  ```
  export CUDA_VER=11.1
  ```

* DeepStream 6.3 / 6.2 / 6.1.1 / 6.1 on Jetson platform

  ```
  export CUDA_VER=11.4
  ```

* DeepStream 6.0.1 / 6.0 / 5.1 on Jetson platform

  ```
  export CUDA_VER=10.2
  ```

Compile the libs

```
make -C nvdsinfer_custom_impl_Yolo_pose
make
```

#### 8. Run

```
./deepstream -s file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4 -c config_infer_primary_yoloV8_pose.txt
```

**NOTE**: The TensorRT engine file may take a very long time to generate (sometimes more than 10 minutes).

**NOTE**: To change the source

```
-s file:// or rtsp:// or http://
--source file:// or rtsp:// or http://
```

**NOTE**: To change the config infer file (example for config_infer.txt file)

```
-c config_infer.txt
--config-infer config_infer.txt
```

**NOTE**: To change the nvstreammux batch-size (example for 2; default: 1)

```
-b 2
--streammux-batch-size 2
```

**NOTE**: To change the nvstreammux width (example for 1280; default: 1920)

```
-w 1280
--streammux-width 1280
```

**NOTE**: To change the nvstreammux height (example for 720; default: 1080)

```
-h 720
--streammux-height 720
```

**NOTE**: To change the GPU id (example for 1; default: 0)

```
-g 1
--gpu-id 1
```

**NOTE**: To change the FPS measurement interval (example for 10; default: 5)

```
-f 10
--fps-interval 10
```

##

### Edit the config_infer_primary_yoloV8_pose file

Edit the `config_infer_primary_yoloV8_pose.txt` file according to your model (example for YOLOv8s-Pose)

```
[property]
...
onnx-file=yolov8s-pose.onnx
...
```

**NOTE**: The **YOLOv8-Pose** resizes the input with center padding. To get better accuracy, use

```
...
maintain-aspect-ratio=1
symmetric-padding=1
...
```

**NOTE**: By default, the dynamic batch-size is set. To use implicit batch-size, uncomment the line

```
...
force-implicit-batch-dim=1
...
```

##

### NMS configuration

For now, the `nms-iou-threshold` is fixed to `0.45`.

**NOTE**: Make sure to set `cluster-mode=4` in the config_infer file.

##

### Detection threshold configuration

```
[class-attrs-all]
pre-cluster-threshold=0.25
topk=300
```

##

My projects: https://www.youtube.com/MarcosLucianoTV
