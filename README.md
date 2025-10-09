# DeepStream-Yolo-Pose

NVIDIA DeepStream SDK 8.0 / 7.1 / 7.0 / 6.4 / 6.3 / 6.2 / 6.1.1 / 6.1 / 6.0.1 / 6.0 application for YOLO-Pose models

--------------------------------------------------------------------------------------------------
### YOLO object detection models and other infos: https://github.com/marcoslucianops/DeepStream-Yolo
--------------------------------------------------------------------------------------------------
### Important: Please export the ONNX model with the new export file, generate the TensorRT engine again with the updated files, and use the new config_infer_primary file according to your model
--------------------------------------------------------------------------------------------------

### Getting started

* [Supported models](#supported-models)
* [Instructions](#basic-usage)
* [YOLOv7-Pose usage](docs/YOLOv7_Pose.md)
* [YOLOv8-Pose usage](docs/YOLOv8_Pose.md)
* [YOLO11-Pose usage](docs/YOLO11_Pose.md)
* [YOLO-NAS-Pose usage](docs/YOLONAS_Pose.md)
* [NMS configuration](#nms-configuration)
* [Detection threshold configuration](#detection-threshold-configuration)

##

### Supported models

* [YOLO-NAS-Pose](https://github.com/Deci-AI/super-gradients/blob/master/YOLONAS-POSE.md)
* [YOLO11-Pose](https://github.com/ultralytics/ultralytics)
* [YOLOv8-Pose](https://github.com/ultralytics/ultralytics)
* [YOLOv7-Pose](https://github.com/WongKinYiu/yolov7)

##

### Instructions

#### 1. Download the DeepStream-Yolo-Pose repo

```
git clone https://github.com/marcoslucianops/DeepStream-Yolo-Pose.git
cd DeepStream-Yolo-Pose
```

#### 3. Compile the libs

3.1. Set the `CUDA_VER` according to your DeepStream version

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

3.2. Make the libs

```
make -C nvdsinfer_custom_impl_Yolo_pose clean && make -C nvdsinfer_custom_impl_Yolo_pose
make clean && make
```

**NOTE**: To use the Python code, you need to install the DeepStream Python bindings.

Reference: https://github.com/NVIDIA-AI-IOT/deepstream_python_apps


* x86 platform: 

  ```
  pip3 install https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/releases/download/v1.2.2/pyds-1.2.2-cp312-cp312-linux_x86_64.whl
  ```

* Jetson platform:

  ```
  pip3 install https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/releases/download/v1.2.2/pyds-1.2.2-cp312-cp312-linux_aarch64.whl
  ```

**NOTE**: It is recommended to use Python virtualenv.

**NOTE**: The steps above only work on **DeepStream 8.0**. For previous versions, please check the files on the `NVIDIA-AI-IOT/deepstream_python_apps` repo.

#### 3. Run

* C code

  ```
  ./deepstream -s file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4 -c config_infer_primary_yoloV8_pose.txt
  ```

* Python code

  ```
  python3 deepstream.py -s file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4 -c config_infer_primary_yoloV8_pose.txt
  ```

**NOTE**: The TensorRT engine file may take a very long time to generate (sometimes more than 10 minutes).

**NOTE**: To change the source

```
-s file:// or rtsp:// or http://
--source file:// or rtsp:// or http://
```

**NOTE**: To change the infer config file (example for config_infer.txt file)

```
-c config_infer.txt
--infer-config config_infer.txt
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
-e 720
--streammux-height 720
```

**NOTE**: To change the GPU id (example for 1; default: 0)

```
-g 1
--gpu-id 1
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
```

##

My projects: https://www.youtube.com/MarcosLucianoTV
