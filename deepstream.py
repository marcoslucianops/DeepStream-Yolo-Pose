import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

import os
import sys
import time
import argparse
import platform
from threading import Lock
from ctypes import sizeof, c_float

sys.path.append("/opt/nvidia/deepstream/deepstream/lib")
import pyds

MAX_ELEMENTS_IN_DISPLAY_META = 16

SOURCE = ""
INFER_CONFIG = ""
STREAMMUX_BATCH_SIZE = 1
STREAMMUX_WIDTH = 1920
STREAMMUX_HEIGHT = 1080
GPU_ID = 0

PERF_MEASUREMENT_INTERVAL_SEC = 5
JETSON = False

skeleton = [
    [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11],
    [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]
]

perf_struct = {}


class GETFPS:
    def __init__(self, stream_id):
        self.stream_id = stream_id
        self.start_time = time.time()
        self.is_first = True
        self.frame_count = 0
        self.total_fps_time = 0
        self.total_frame_count = 0
        self.fps_lock = Lock()

    def update_fps(self):
        with self.fps_lock:
            if self.is_first:
                self.start_time = time.time()
                self.is_first = False
                self.frame_count = 0
                self.total_fps_time = 0
                self.total_frame_count = 0
            else:
                self.frame_count = self.frame_count + 1

    def get_fps(self):
        with self.fps_lock:
            end_time = time.time()
            current_time = end_time - self.start_time
            self.total_fps_time = self.total_fps_time + current_time
            self.total_frame_count = self.total_frame_count + self.frame_count
            current_fps = float(self.frame_count) / current_time
            avg_fps = float(self.total_frame_count) / self.total_fps_time
            self.start_time = end_time
            self.frame_count = 0
        return current_fps, avg_fps

    def perf_print_callback(self):
        if not self.is_first:
            current_fps, avg_fps = self.get_fps()
            sys.stdout.write(f"DEBUG - Stream {self.stream_id + 1} - FPS: {current_fps:.2f} ({avg_fps:.2f})\n")
        return True


def set_custom_bbox(obj_meta):
    border_width = 6
    font_size = 18
    x_offset = int(min(STREAMMUX_WIDTH - 1, max(0, obj_meta.rect_params.left - border_width * 0.5)))
    y_offset = int(min(STREAMMUX_HEIGHT - 1, max(0, obj_meta.rect_params.top - font_size * 2 + border_width * 0.5 + 1)))

    obj_meta.rect_params.border_width = border_width
    obj_meta.rect_params.border_color.red = 0.0
    obj_meta.rect_params.border_color.green = 0.0
    obj_meta.rect_params.border_color.blue = 1.0
    obj_meta.rect_params.border_color.alpha = 1.0
    obj_meta.text_params.font_params.font_name = "Ubuntu"
    obj_meta.text_params.font_params.font_size = font_size
    obj_meta.text_params.x_offset = x_offset
    obj_meta.text_params.y_offset = y_offset
    obj_meta.text_params.font_params.font_color.red = 1.0
    obj_meta.text_params.font_params.font_color.green = 1.0
    obj_meta.text_params.font_params.font_color.blue = 1.0
    obj_meta.text_params.font_params.font_color.alpha = 1.0
    obj_meta.text_params.set_bg_clr = 1
    obj_meta.text_params.text_bg_clr.red = 0.0
    obj_meta.text_params.text_bg_clr.green = 0.0
    obj_meta.text_params.text_bg_clr.blue = 1.0
    obj_meta.text_params.text_bg_clr.alpha = 1.0


def parse_pose_from_meta(batch_meta, frame_meta, obj_meta):
    display_meta = None

    num_joints = int(obj_meta.mask_params.size / (sizeof(c_float) * 3))

    gain = min(obj_meta.mask_params.width / STREAMMUX_WIDTH, obj_meta.mask_params.height / STREAMMUX_HEIGHT)

    pad_x = (obj_meta.mask_params.width - STREAMMUX_WIDTH * gain) * 0.5
    pad_y = (obj_meta.mask_params.height - STREAMMUX_HEIGHT * gain) * 0.5

    for i in range(num_joints):
        data = obj_meta.mask_params.get_mask_array()

        xc = int((data[i * 3 + 0] - pad_x) / gain)
        yc = int((data[i * 3 + 1] - pad_y) / gain)
        confidence = data[i * 3 + 2]

        if confidence < 0.5:
            continue

        if display_meta is None or display_meta.num_circles == MAX_ELEMENTS_IN_DISPLAY_META:
            display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
            pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

        circle_params = display_meta.circle_params[display_meta.num_circles]
        circle_params.xc = xc
        circle_params.yc = yc
        circle_params.radius = 6
        circle_params.circle_color.red = 1.0
        circle_params.circle_color.green = 1.0
        circle_params.circle_color.blue = 1.0
        circle_params.circle_color.alpha = 1.0
        circle_params.has_bg_color = 1
        circle_params.bg_color.red = 0.0
        circle_params.bg_color.green = 0.0
        circle_params.bg_color.blue = 1.0
        circle_params.bg_color.alpha = 1.0
        display_meta.num_circles += 1

    for i in range(num_joints + 2):
        data = obj_meta.mask_params.get_mask_array()

        x1 = int((data[(skeleton[i][0] - 1) * 3 + 0] - pad_x) / gain)
        y1 = int((data[(skeleton[i][0] - 1) * 3 + 1] - pad_y) / gain)
        confidence1 = data[(skeleton[i][0] - 1) * 3 + 2]
        x2 = int((data[(skeleton[i][1] - 1) * 3 + 0] - pad_x) / gain)
        y2 = int((data[(skeleton[i][1] - 1) * 3 + 1] - pad_y) / gain)
        confidence2 = data[(skeleton[i][1] - 1) * 3 + 2]

        if confidence1 < 0.5 or confidence2 < 0.5:
            continue

        if display_meta is None or display_meta.num_lines == MAX_ELEMENTS_IN_DISPLAY_META:
            display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
            pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

        line_params = display_meta.line_params[display_meta.num_lines]
        line_params.x1 = x1
        line_params.y1 = y1
        line_params.x2 = x2
        line_params.y2 = y2
        line_params.line_width = 6
        line_params.line_color.red = 0.0
        line_params.line_color.green = 0.0
        line_params.line_color.blue = 1.0
        line_params.line_color.alpha = 1.0
        display_meta.num_lines += 1


def nvosd_sink_pad_buffer_probe(pad, info, user_data):
    buf = info.get_buffer()
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))

    l_frame = batch_meta.frame_meta_list
    while l_frame:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        l_obj = frame_meta.obj_meta_list
        while l_obj:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            parse_pose_from_meta(batch_meta, frame_meta, obj_meta)
            set_custom_bbox(obj_meta)

            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        perf_struct[frame_meta.source_id].update_fps()

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


def uridecodebin_child_added_callback(child_proxy, Object, name, user_data):
    if name.find("decodebin") != -1:
        Object.connect("child-added", uridecodebin_child_added_callback, user_data)
    elif name.find("nvv4l2decoder") != -1:
        Object.set_property("drop-frame-interval", 0)
        Object.set_property("num-extra-surfaces", 1)
        Object.set_property("qos", 0)
        if JETSON:
            Object.set_property("enable-max-performance", 1)
        else:
            Object.set_property("cudadec-memtype", 0)
            Object.set_property("gpu-id", GPU_ID)


def uridecodebin_pad_added_callback(decodebin, pad, user_data):
    nvstreammux_sink_pad = user_data

    caps = pad.get_current_caps()
    if not caps:
        caps = pad.query_caps()

    structure = caps.get_structure(0)
    name = structure.get_name()
    features = caps.get_features(0)

    if name.find("video") != -1:
        if features.contains("memory:NVMM"):
            if pad.link(nvstreammux_sink_pad) != Gst.PadLinkReturn.OK:
                sys.stderr.write("ERROR - Failed to link source to nvstreammux sink pad\n")
        else:
            sys.stderr.write("ERROR - decodebin did not pick NVIDIA decoder plugin\n")


def create_uridecodebin(stream_id, uri, nvstreammux):
    bin_name = f"source-bin-{stream_id:04d}"

    uridecodebin = Gst.ElementFactory.make("uridecodebin", bin_name)

    if "rtsp://" in uri:
        pyds.configure_source_for_ntp_sync(uridecodebin)

    uridecodebin.set_property("uri", uri)

    pad_name = f"sink_{stream_id}"

    nvstreammux_sink_pad = nvstreammux.get_request_pad(pad_name)
    if not nvstreammux_sink_pad:
        sys.stderr.write(f"ERROR - Failed to get nvstreammux {pad_name} pad\n")
        return None

    uridecodebin.connect("pad-added", uridecodebin_pad_added_callback, nvstreammux_sink_pad)
    uridecodebin.connect("child-added", uridecodebin_child_added_callback, None)

    perf_struct[stream_id] = GETFPS(stream_id)
    GLib.timeout_add(PERF_MEASUREMENT_INTERVAL_SEC * 1000, perf_struct[stream_id].perf_print_callback)

    return uridecodebin


def bus_call(bus, message, user_data):
    loop = user_data
    t = message.type
    if t == Gst.MessageType.EOS:
        sys.stdout.write("DEBUG - EOS\n")
        loop.quit()
    elif t == Gst.MessageType.WARNING:
        error, debug = message.parse_warning()
        sys.stderr.write(f"WARNING - {error.message} - {debug}\n")
    elif t == Gst.MessageType.ERROR:
        error, debug = message.parse_error()
        sys.stderr.write(f"ERROR - {error.message} - {debug}\n")
        loop.quit()
    return True


def is_aarch64():
    return platform.uname()[4] == "aarch64"


def main():
    Gst.init(None)

    loop = GLib.MainLoop()

    pipeline = Gst.Pipeline()
    if not pipeline:
        sys.stderr.write("ERROR - Failed to create pipeline\n")
        return -1

    nvstreammux = Gst.ElementFactory.make("nvstreammux", "nvstreammux")
    if not nvstreammux or not pipeline.add(nvstreammux):
        sys.stderr.write("ERROR - Failed to create nvstreammux\n")
        return -1

    uridecodebin = create_uridecodebin(0, SOURCE, nvstreammux)
    if not uridecodebin or not pipeline.add(uridecodebin):
        sys.stderr.write("ERROR - Failed to create uridecodebin\n")
        return -1

    nvinfer = Gst.ElementFactory.make("nvinfer", "nvinfer")
    if not nvinfer or not pipeline.add(nvinfer):
        sys.stderr.write("ERROR - Failed to create nvinfer\n")
        return -1

    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "nvvideoconvert")
    if not nvvidconv or not pipeline.add(nvvidconv):
        sys.stderr.write("ERROR - Failed to create nvvideoconvert\n")
        return -1

    capsfilter = Gst.ElementFactory.make("capsfilter", "capsfilter")
    if not capsfilter or not pipeline.add(capsfilter):
        sys.stderr.write("ERROR - Failed to create capsfilter\n")
        return -1

    nvosd = Gst.ElementFactory.make("nvdsosd", "nvdsosd")
    if not nvosd or not pipeline.add(nvosd):
        sys.stderr.write("ERROR - Failed to create nvdsosd\n")
        return -1

    nvsink = None
    if JETSON:
        nvsink = Gst.ElementFactory.make("nv3dsink", "nv3dsink")
        if not nvsink or not pipeline.add(nvsink):
            sys.stderr.write("ERROR - Failed to create nv3dsink\n")
            return -1
    else:
        nvsink = Gst.ElementFactory.make("nveglglessink", "nveglglessink")
        if not nvsink or not pipeline.add(nvsink):
            sys.stderr.write("ERROR - Failed to create nveglglessink\n")
            return -1

    sys.stdout.write("\n")
    sys.stdout.write(f"SOURCE: {SOURCE}\n")
    sys.stdout.write(f"INFER_CONFIG: {INFER_CONFIG}\n")
    sys.stdout.write(f"STREAMMUX_BATCH_SIZE: {STREAMMUX_BATCH_SIZE}\n")
    sys.stdout.write(f"STREAMMUX_WIDTH: {STREAMMUX_WIDTH}\n")
    sys.stdout.write(f"STREAMMUX_HEIGHT: {STREAMMUX_HEIGHT}\n")
    sys.stdout.write(f"GPU_ID: {GPU_ID}\n")
    sys.stdout.write(f"PERF_MEASUREMENT_INTERVAL_SEC: {PERF_MEASUREMENT_INTERVAL_SEC}\n")
    sys.stdout.write(f"JETSON: {'TRUE' if JETSON else 'FALSE'}\n")
    sys.stdout.write("\n")

    nvstreammux.set_property("batch-size", STREAMMUX_BATCH_SIZE)
    nvstreammux.set_property("batched-push-timeout", 25000)
    nvstreammux.set_property("width", STREAMMUX_WIDTH)
    nvstreammux.set_property("height", STREAMMUX_HEIGHT)
    nvstreammux.set_property("live-source", 1)
    nvinfer.set_property("config-file-path", INFER_CONFIG)
    nvinfer.set_property("qos", 0)
    nvosd.set_property("process-mode", int(pyds.MODE_GPU))
    nvosd.set_property("qos", 0)
    nvsink.set_property("async", 0)
    nvsink.set_property("sync", 0)
    nvsink.set_property("qos", 0)

    if SOURCE.startswith("file://"):
        nvstreammux.set_property("live-source", 0)

    if not JETSON:
        nvstreammux.set_property("nvbuf-memory-type", int(pyds.NVBUF_MEM_CUDA_DEVICE))
        nvstreammux.set_property("gpu_id", GPU_ID)
        nvinfer.set_property("gpu_id", GPU_ID)
        nvvidconv.set_property("nvbuf-memory-type", int(pyds.NVBUF_MEM_CUDA_DEVICE))
        nvvidconv.set_property("gpu_id", GPU_ID)
        nvosd.set_property("gpu_id", GPU_ID)

    nvstreammux.link(nvinfer)
    nvinfer.link(nvvidconv)
    nvvidconv.link(capsfilter)
    capsfilter.link(nvosd)
    nvosd.link(nvsink)

    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    nvosd_sink_pad = nvosd.get_static_pad("sink")
    if not nvosd_sink_pad:
        sys.stderr.write("ERROR - Failed to get nvosd sink pad\n")
        return -1

    nvosd_sink_pad.add_probe(Gst.PadProbeType.BUFFER, nvosd_sink_pad_buffer_probe, None)

    pipeline.set_state(Gst.State.PAUSED)

    if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
        sys.stderr.write("ERROR - Failed to set pipeline to playing\n")
        return -1

    sys.stdout.write("\n")

    try:
        loop.run()
    except:
        pass

    pipeline.set_state(Gst.State.NULL)

    sys.stdout.write("\n")

    return 0


def parse_args():
    global SOURCE, INFER_CONFIG, STREAMMUX_BATCH_SIZE, STREAMMUX_WIDTH, STREAMMUX_HEIGHT, GPU_ID, JETSON

    parser = argparse.ArgumentParser(description="DeepStream")
    parser.add_argument("-s", "--source", required=True, help="Source stream/file")
    parser.add_argument("-c", "--infer-config", required=True, help="Config infer file")
    parser.add_argument("-b", "--streammux-batch-size", type=int, default=1, help="Streammux batch-size (default 1)")
    parser.add_argument("-w", "--streammux-width", type=int, default=1920, help="Streammux width (default 1920)")
    parser.add_argument("-e", "--streammux-height", type=int, default=1080, help="Streammux height (default 1080)")
    parser.add_argument("-g", "--gpu-id", type=int, default=0, help="GPU id (default 0)")
    args = parser.parse_args()

    if args.source == "":
        sys.stderr.write("ERROR - Source not found\n")
        sys.exit(-1)

    if args.infer_config == "" or not os.path.isfile(args.infer_config):
        sys.stderr.write("ERROR - Config infer not found\n")
        sys.exit(-1)

    SOURCE = args.source
    INFER_CONFIG = args.infer_config
    STREAMMUX_BATCH_SIZE = args.streammux_batch_size
    STREAMMUX_WIDTH = args.streammux_width
    STREAMMUX_HEIGHT = args.streammux_height
    GPU_ID = args.gpu_id

    JETSON = is_aarch64()


if __name__ == "__main__":
    parse_args()
    sys.exit(main())
