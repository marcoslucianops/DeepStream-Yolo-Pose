/*
 * Copyright (c) 2018-2023, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 * Edited by Marcos Luciano
 * https://www.github.com/marcoslucianops
 */

#include "deepstream.h"

GOptionEntry entries[] = {
  {"source", 's', 0, G_OPTION_ARG_STRING, &SOURCE, "Source stream/file", NULL},
  {"config-infer", 'c', 0, G_OPTION_ARG_STRING, &CONFIG_INFER, "Config infer file", NULL},
  {"streammux-batch-size", 'b', 0, G_OPTION_ARG_INT, &STREAMMUX_BATCH_SIZE, "Streammux batch-size (default: 1)", NULL},
  {"streammux-width", 'w', 0, G_OPTION_ARG_INT, &STREAMMUX_WIDTH, "Streammux width (default: 1920)", NULL},
  {"streammux-height", 'e', 0, G_OPTION_ARG_INT, &STREAMMUX_HEIGHT, "Streammux height (default: 1080)", NULL},
  {"gpu-id", 'g', 0, G_OPTION_ARG_INT, &GPU_ID, "GPU id (default: 0)", NULL},
  {"fps-interval", 'f', 0, G_OPTION_ARG_INT, &PERF_MEASUREMENT_INTERVAL_SEC, "FPS measurement interval (default: 5)", NULL},
  {NULL}
};

static void
set_custom_bbox(NvDsObjectMeta *obj_meta)
{
  guint border_width = 6;
  guint font_size = 18;
  guint x_offset = MIN(STREAMMUX_WIDTH - 1, (guint) MAX(0, (gint) (obj_meta->rect_params.left - (border_width / 2))));
  guint y_offset = MIN(STREAMMUX_HEIGHT - 1, (guint) MAX(0, (gint) (obj_meta->rect_params.top - (font_size * 2) + 1)));

  obj_meta->rect_params.border_width = border_width;
  obj_meta->rect_params.border_color.red = 0.0;
  obj_meta->rect_params.border_color.green = 0.0;
  obj_meta->rect_params.border_color.blue = 1.0;
  obj_meta->rect_params.border_color.alpha = 1.0;
  obj_meta->text_params.font_params.font_name = (gchar *) "Ubuntu";
  obj_meta->text_params.font_params.font_size = font_size;
  obj_meta->text_params.x_offset = x_offset;
  obj_meta->text_params.y_offset = y_offset;
  obj_meta->text_params.font_params.font_color.red = 1.0;
  obj_meta->text_params.font_params.font_color.green = 1.0;
  obj_meta->text_params.font_params.font_color.blue = 1.0;
  obj_meta->text_params.font_params.font_color.alpha = 1.0;
  obj_meta->text_params.set_bg_clr = 1;
  obj_meta->text_params.text_bg_clr.red = 0.0;
  obj_meta->text_params.text_bg_clr.green = 0.0;
  obj_meta->text_params.text_bg_clr.blue = 1.0;
  obj_meta->text_params.text_bg_clr.alpha = 1.0;
}

static void
parse_pose_from_meta(NvDsFrameMeta *frame_meta, NvDsObjectMeta *obj_meta)
{
  guint num_joints = obj_meta->mask_params.size / (sizeof(float) * 3);

  gfloat gain = MIN((gfloat) obj_meta->mask_params.width / STREAMMUX_WIDTH,
      (gfloat) obj_meta->mask_params.height / STREAMMUX_HEIGHT);
  gfloat pad_x = (obj_meta->mask_params.width - STREAMMUX_WIDTH * gain) / 2.0;
  gfloat pad_y = (obj_meta->mask_params.height - STREAMMUX_HEIGHT * gain) / 2.0;

  NvDsBatchMeta *batch_meta = frame_meta->base_meta.batch_meta;
  NvDsDisplayMeta *display_meta = nvds_acquire_display_meta_from_pool(batch_meta);
  nvds_add_display_meta_to_frame(frame_meta, display_meta);

  for (guint i = 0; i < num_joints; ++i) {
    gfloat xc = (obj_meta->mask_params.data[i * 3 + 0] - pad_x) / gain;
    gfloat yc = (obj_meta->mask_params.data[i * 3 + 1] - pad_y) / gain;
    gfloat confidence = obj_meta->mask_params.data[i * 3 + 2];

    if (confidence < 0.5) {
      continue;
    }

    if (display_meta->num_circles == MAX_ELEMENTS_IN_DISPLAY_META) {
      display_meta = nvds_acquire_display_meta_from_pool(batch_meta);
      nvds_add_display_meta_to_frame(frame_meta, display_meta);
    }

    NvOSD_CircleParams *circle_params = &display_meta->circle_params[display_meta->num_circles];
    circle_params->xc = xc;
    circle_params->yc = yc;
    circle_params->radius = 6;
    circle_params->circle_color.red = 1.0;
    circle_params->circle_color.green = 1.0;
    circle_params->circle_color.blue = 1.0;
    circle_params->circle_color.alpha = 1.0;
    circle_params->has_bg_color = 1;
    circle_params->bg_color.red = 0.0;
    circle_params->bg_color.green = 0.0;
    circle_params->bg_color.blue = 1.0;
    circle_params->bg_color.alpha = 1.0;
    display_meta->num_circles++;
  }

  for (guint i = 0; i < num_joints + 2; ++i) {
    gfloat x1 = (obj_meta->mask_params.data[(skeleton[i][0] - 1) * 3 + 0] - pad_x) / gain;
    gfloat y1 = (obj_meta->mask_params.data[(skeleton[i][0] - 1) * 3 + 1] - pad_y) / gain;
    gfloat confidence1 = obj_meta->mask_params.data[(skeleton[i][0] - 1) * 3 + 2];
    gfloat x2 = (obj_meta->mask_params.data[(skeleton[i][1] - 1) * 3 + 0] - pad_x) / gain;
    gfloat y2 = (obj_meta->mask_params.data[(skeleton[i][1] - 1) * 3 + 1] - pad_y) / gain;
    gfloat confidence2 = obj_meta->mask_params.data[(skeleton[i][1] - 1) * 3 + 2];

    if (confidence1 < 0.5 || confidence2 < 0.5) {
      continue;
    }

    if (display_meta->num_lines == MAX_ELEMENTS_IN_DISPLAY_META) {
      display_meta = nvds_acquire_display_meta_from_pool(batch_meta);
      nvds_add_display_meta_to_frame(frame_meta, display_meta);
    }

    NvOSD_LineParams *line_params = &display_meta->line_params[display_meta->num_lines];
    line_params->x1 = x1;
    line_params->y1 = y1;
    line_params->x2 = x2;
    line_params->y2 = y2;
    line_params->line_width = 6;
    line_params->line_color.red = 0.0;
    line_params->line_color.green = 0.0;
    line_params->line_color.blue = 1.0;
    line_params->line_color.alpha = 1.0;
    display_meta->num_lines++;
  }

  g_free(obj_meta->mask_params.data);
  obj_meta->mask_params.width = 0;
  obj_meta->mask_params.height = 0;
  obj_meta->mask_params.size = 0;
}

static GstPadProbeReturn
tracker_src_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer user_data)
{
  GstBuffer *buf = (GstBuffer *) info->data;
  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);

  NvDsMetaList *l_frame = NULL;
  for (l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next) {
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) (l_frame->data);

    NvDsMetaList *l_obj = NULL;
    for (l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next) {
      NvDsObjectMeta *obj_meta = (NvDsObjectMeta *) (l_obj->data);

      parse_pose_from_meta(frame_meta, obj_meta);
      set_custom_bbox(obj_meta);
    }
  }
  return GST_PAD_PROBE_OK;
}

static void
decodebin_child_added(GstChildProxy *child_proxy, GObject *object, gchar *name, gpointer user_data)
{
  if (g_strrstr(name, "decodebin") == name) {
    g_signal_connect(object, "child-added", G_CALLBACK(decodebin_child_added), user_data);
  }
  if (g_strrstr(name, "nvv4l2decoder") == name) {
    g_object_set(object, "drop-frame-interval", 0, NULL);
    g_object_set(object, "num-extra-surfaces", 1, NULL);
    if (JETSON) {
      g_object_set(object, "enable-max-performance", 1, NULL);
    }
    else {
      g_object_set(object, "cudadec-memtype", 0, "gpu-id", GPU_ID, NULL);
    }
  }
}

static void
cb_newpad(GstElement *decodebin, GstPad *pad, gpointer user_data)
{
  GstPad *streammux_sink_pad = (GstPad *) user_data;
  GstCaps *caps = gst_pad_get_current_caps(pad);
  if (!caps) {
    caps = gst_pad_query_caps(pad, NULL);
  }
  const GstStructure *str = gst_caps_get_structure(caps, 0);
  const gchar *name = gst_structure_get_name(str);
  GstCapsFeatures *features = gst_caps_get_features(caps, 0);
  if (!strncmp(name, "video", 5)) {
    if (gst_caps_features_contains(features, "memory:NVMM")) {
      if (gst_pad_link(pad, streammux_sink_pad) != GST_PAD_LINK_OK) {
        g_printerr("ERROR: Failed to link source to streammux sink pad\n");
        gst_caps_unref(caps);
        return;
      }
    }
    else {
      g_printerr("ERROR: decodebin did not pick NVIDIA decoder plugin\n");
      gst_caps_unref(caps);
      return;
    }
  }
  gst_caps_unref(caps);
}

static GstElement *
create_uridecode_bin(guint stream_id, const gchar *uri, GstElement *streammux)
{
  gchar bin_name[32] = { };
  g_snprintf(bin_name, 32, "source-bin-%04d", stream_id);
  GstElement *bin = gst_element_factory_make("uridecodebin", bin_name);
  if (g_strrstr(uri, "rtsp://")) {
    configure_source_for_ntp_sync(bin);
  }
  g_object_set(G_OBJECT(bin), "uri", uri, NULL);
  gchar pad_name[16];
  g_snprintf(pad_name, 16, "sink_%u", stream_id);
  GstPad *streammux_sink_pad = gst_element_get_request_pad(streammux, pad_name);
  g_signal_connect(G_OBJECT(bin), "pad-added", G_CALLBACK(cb_newpad), streammux_sink_pad);
  g_signal_connect(G_OBJECT(bin), "child-added", G_CALLBACK(decodebin_child_added), NULL);
  gst_object_unref(streammux_sink_pad);
  return bin;
}

static gboolean
bus_call(GstBus *bus, GstMessage *msg, gpointer user_data)
{
  GMainLoop *loop = (GMainLoop *) user_data;
  switch (GST_MESSAGE_TYPE(msg)) {
    case GST_MESSAGE_EOS:
    {
      g_print("DEBUG: EOS\n");
      g_main_loop_quit(loop);
      break;
    }
    case GST_MESSAGE_WARNING:
    {
      gchar *debug;
      GError *error;
      gst_message_parse_warning(msg, &error, &debug);
      g_printerr("WARNING: %s: %s\n", GST_OBJECT_NAME(msg->src), error->message);
      g_free(debug);
      g_error_free(error);
      break;
    }
    case GST_MESSAGE_ERROR:
    {
      gchar *debug;
      GError *error;
      gst_message_parse_error(msg, &error, &debug);
      g_printerr("ERROR: %s: %s\n", GST_OBJECT_NAME(msg->src), error->message);
      g_free(debug);
      g_error_free(error);
      g_main_loop_quit(loop);
      break;
    }
    default:
      break;
  }
  return TRUE;
}

gint
main(gint argc, char *argv[])
{
  GOptionContext *ctx = g_option_context_new("DeepStream");
  GOptionGroup *group = g_option_group_new("deepstream", NULL, NULL, NULL, NULL);
  GError *error = NULL;
  g_option_group_add_entries(group, entries);
  g_option_context_set_main_group(ctx, group);
  g_option_context_add_group(ctx, gst_init_get_option_group());
  if (!g_option_context_parse(ctx, &argc, &argv, &error)) {
    g_option_context_free(ctx);
    g_printerr("ERROR: %s", error->message);
    return -1;
  }
  g_option_context_free(ctx);

  if (!SOURCE) {
    g_printerr("ERROR: Source not found\n");
    return -1;
  }

  if (!CONFIG_INFER) {
    g_printerr("ERROR: Config infer not found\n");
    return -1;
  }

  gint current_device = -1;
  cudaGetDevice(&current_device);
  struct cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, current_device);

  if (prop.integrated) {
    JETSON = TRUE;
  }

  GMainLoop *loop = g_main_loop_new(NULL, FALSE);

  _intr_setup();
  g_timeout_add(400, check_for_interrupt, &loop);

  GstElement *pipeline = gst_pipeline_new("deepstream");
  if (!pipeline) {
    g_printerr("ERROR: Failed to create pipeline\n");
    return -1;
  }
  GstElement *streammux = gst_element_factory_make("nvstreammux", "nvstreammux");
  if (!streammux) {
    g_printerr("ERROR: Failed to create nvstreammux\n");
    return -1;
  }
  gst_bin_add(GST_BIN(pipeline), streammux);

  GstElement *source_bin = create_uridecode_bin(0, SOURCE, streammux);
  if (!source_bin) {
    g_printerr("ERROR: Failed to create source_bin\n");
    return -1;
  }
  gst_bin_add(GST_BIN(pipeline), source_bin);

  GstElement *pgie = gst_element_factory_make("nvinfer", "nvinfer");
  if (!pgie) {
    g_printerr("ERROR: Failed to create nvinfer\n");
    return -1;
  }
  GstElement *tracker = gst_element_factory_make("nvtracker", "nvtracker");
  if (!tracker) {
    g_printerr("ERROR: Failed to create nvtracker\n");
    return -1;
  }
  GstElement *converter = gst_element_factory_make("nvvideoconvert", "nvvideoconvert");
  if (!converter) {
    g_printerr("ERROR: Failed to create nvvideoconvert\n");
    return -1;
  }
  GstElement *osd = gst_element_factory_make("nvdsosd", "nvdsosd");
  if (!osd) {
    g_printerr("ERROR: Failed to create nvdsosd\n");
    return -1;
  }

  GstElement *sink = NULL;
  if (JETSON) {
    sink = gst_element_factory_make("nv3dsink", "nv3dsink");
    if (!sink) {
      g_printerr("ERROR: Failed to create nv3dsink\n");
      return -1;
    }
  }
  else {
    sink = gst_element_factory_make("nveglglessink", "nveglglessink");
    if (!sink) {
      g_printerr("ERROR: Failed to create nveglglessink\n");
      return -1;
    }
  }

  g_print("\n");
  g_print("SOURCE: %s\n", SOURCE);
  g_print("CONFIG_INFER: %s\n", CONFIG_INFER);
  g_print("STREAMMUX_BATCH_SIZE: %d\n", STREAMMUX_BATCH_SIZE);
  g_print("STREAMMUX_WIDTH: %d\n", STREAMMUX_WIDTH);
  g_print("STREAMMUX_HEIGHT: %d\n", STREAMMUX_HEIGHT);
  g_print("GPU_ID: %d\n", GPU_ID);
  g_print("PERF_MEASUREMENT_INTERVAL_SEC: %d\n", PERF_MEASUREMENT_INTERVAL_SEC);
  g_print("JETSON: %s\n", JETSON ? "TRUE" : "FALSE");
  g_print("\n");

  g_object_set(G_OBJECT(streammux), "batch-size", STREAMMUX_BATCH_SIZE, "batched-push-timeout", 25000,
      "width", STREAMMUX_WIDTH, "height", STREAMMUX_HEIGHT, "enable-padding", 0, "live-source", 1, "attach-sys-ts", 1, NULL);
  g_object_set(G_OBJECT(pgie), "config-file-path", CONFIG_INFER, "qos", 0, NULL);
  g_object_set(G_OBJECT(tracker), "tracker-width", 640, "tracker-height", 384,
      "ll-lib-file", "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so",
      "ll-config-file", "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml",
      "display-tracking-id", 1, "qos", 0, NULL);
  g_object_set(G_OBJECT(osd), "process-mode", MODE_GPU, "qos", 0, NULL);
  g_object_set(G_OBJECT(sink), "async", 0, "sync", 0, "qos", 0, NULL);

  if (g_strrstr(SOURCE, "file://")) {
    g_object_set(G_OBJECT(streammux), "live-source", 0, NULL);
  }

  if (g_object_class_find_property(G_OBJECT_GET_CLASS(G_OBJECT(tracker)), "enable_batch_process")) {
    g_object_set(G_OBJECT(tracker), "enable_batch_process", 1, NULL);
  }

  if (g_object_class_find_property(G_OBJECT_GET_CLASS(G_OBJECT(tracker)), "enable_past_frame")) {
    g_object_set(G_OBJECT(tracker), "enable_past_frame", 1, NULL);
  }

  if (!JETSON) {
    g_object_set(G_OBJECT(streammux), "nvbuf-memory-type", 0, "gpu_id", GPU_ID, NULL);
    g_object_set(G_OBJECT(pgie), "gpu_id", GPU_ID, NULL);
    g_object_set(G_OBJECT(tracker), "gpu_id", GPU_ID, NULL);
    g_object_set(G_OBJECT(converter), "nvbuf-memory-type", 0, "gpu_id", GPU_ID, NULL);
    g_object_set(G_OBJECT(osd), "gpu_id", GPU_ID, NULL);
  }

  gst_bin_add_many(GST_BIN(pipeline), pgie, tracker, converter, osd, sink, NULL);
  if (!gst_element_link_many(streammux, pgie, tracker, converter, osd, sink, NULL)) {
    g_printerr("ERROR: Pipeline elements could not be linked\n");
    return -1;
  }

  GstBus *bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline));
  guint bus_watch_id = gst_bus_add_watch(bus, bus_call, loop);
  gst_object_unref(bus);

  GstPad *tracker_src_pad = gst_element_get_static_pad(tracker, "src");
  if (!tracker_src_pad) {
    g_printerr("ERROR: Failed to get tracker src pad\n");
    return -1;
  }
  else
    gst_pad_add_probe(tracker_src_pad, GST_PAD_PROBE_TYPE_BUFFER, tracker_src_pad_buffer_probe, NULL, NULL);
  gst_object_unref(tracker_src_pad);

  NvDsAppPerfStructInt *perf_struct;
  GstPad *converter_sink_pad = gst_element_get_static_pad(converter, "sink");
  if (!converter_sink_pad) {
    g_printerr("ERROR: Failed to get converter sink pad\n");
    return -1;
  }
  else {
    perf_struct = (NvDsAppPerfStructInt *) g_malloc0(sizeof(NvDsAppPerfStructInt));
    enable_perf_measurement(perf_struct, converter_sink_pad, 1, PERF_MEASUREMENT_INTERVAL_SEC, 0, perf_cb);
  }
  gst_object_unref(converter_sink_pad);

  gst_element_set_state(pipeline, GST_STATE_PAUSED);

  if (gst_element_set_state(pipeline, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
    g_printerr("ERROR: Failed to set pipeline to playing\n");
    return -1;
  }

  g_print("\n");
  g_main_loop_run(loop);

  gst_element_set_state(pipeline, GST_STATE_NULL);

  g_free(perf_struct);

  if (SOURCE) {
    g_free(SOURCE);
  }

  if (CONFIG_INFER) {
    g_free(CONFIG_INFER);
  }

  gst_object_unref(GST_OBJECT(pipeline));
  g_source_remove(bus_watch_id);
  g_main_loop_unref(loop);

  g_print("\n");
  return 0;
}
