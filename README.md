# DeepStream-Yolo-Pose

NVIDIA DeepStream SDK application for YOLO-Pose models

--------------------------------------------------------------------------------------------------
### YOLO objetct detection models and other infos: https://github.com/marcoslucianops/DeepStream-Yolo
--------------------------------------------------------------------------------------------------
### Important: I've changed the output logic to prevent the TensorRT to use the wrong output order. Please export the ONNX model with the new export file, generate the TensorRT engine again with the updated files, and use the new config_infer_primary file according to your model
--------------------------------------------------------------------------------------------------

### Getting started

* [Supported models](#supported-models)
* [Instructions](#basic-usage)
* [YOLOv7-Pose usage](docs/YOLOv7_Pose.md)
* [YOLOv8-Pose usage](docs/YOLOv8_Pose.md)
* [YOLO-NAS-Pose usage](docs/YOLONAS_Pose.md)
* [NMS configuration](#nms-configuration)
* [Detection threshold configuration](#detection-threshold-configuration)

##

### Supported models

* [YOLO-NAS-Pose](https://github.com/Deci-AI/super-gradients/blob/master/YOLONAS-POSE.md)
* [YOLOv8-Pose](https://github.com/ultralytics/ultralytics)
* [YOLOv7-Pose](https://github.com/WongKinYiu/yolov7)

##

### Instructions

#### 1. Download the DeepStream-Yolo-Pose repo

```
git clone https://github.com/marcoslucianops/DeepStream-Yolo-Pose.git
cd DeepStream-Yolo-Pose
```

#### 2. Compile the libs

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

* DeepStream 6.3 / 6.2 / 6.1.1 / 6.1 on Jetson platform

  ```
  export CUDA_VER=11.4
  ```

* DeepStream 6.0.1 / 6.0 on Jetson platform

  ```
  export CUDA_VER=10.2
  ```

Compile the libs

```
make -C nvdsinfer_custom_impl_Yolo_pose
make
```

**NOTE**: To use the Python code, you need to install the DeepStream Python bindings.

Reference: https://github.com/NVIDIA-AI-IOT/deepstream_python_apps


* x86 platform: 

  ```
  wget https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/releases/download/v1.1.8/pyds-1.1.8-py3-none-linux_x86_64.whl
  pip3 install pyds-1.1.8-py3-none-linux_x86_64.whl
  ```

* Jetson platform:

  ```
  wget https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/releases/download/v1.1.8/pyds-1.1.8-py3-none-linux_aarch64.whl
  pip3 install pyds-1.1.8-py3-none-linux_aarch64.whl
  ```

**NOTE**: It is recommended to use Python virtualenv.

**NOTE**: The steps above only work on **DeepStream 6.3**. For previous versions, please check the files on the `NVIDIA-AI-IOT/deepstream_python_apps` repo.

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
-e 720
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
