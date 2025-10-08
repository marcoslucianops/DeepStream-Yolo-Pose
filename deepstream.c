#include "deepstream.h"

GOptionEntry entries[] = {
  {"source", 's', 0, G_OPTION_ARG_STRING, &SOURCE, "Source stream/file", NULL},
  {"infer-config", 'c', 0, G_OPTION_ARG_STRING, &INFER_CONFIG, "Config infer file", NULL},
  {"streammux-batch-size", 'b', 0, G_OPTION_ARG_INT, &STREAMMUX_BATCH_SIZE, "Streammux batch-size (default 1)", NULL},
  {"streammux-width", 'w', 0, G_OPTION_ARG_INT, &STREAMMUX_WIDTH, "Streammux width (default 1920)", NULL},
  {"streammux-height", 'e', 0, G_OPTION_ARG_INT, &STREAMMUX_HEIGHT, "Streammux height (default 1080)", NULL},
  {"gpu-id", 'g', 0, G_OPTION_ARG_INT, &GPU_ID, "GPU id (default 0)", NULL},
  {NULL}
};

static void
set_custom_bbox(NvDsObjectMeta *obj_meta)
{
  guint border_width = 6;
  guint font_size = 18;
  guint x_offset = MIN(STREAMMUX_WIDTH - 1, (guint) MAX(0, obj_meta->rect_params.left - border_width * 0.5f));
  guint y_offset = MIN(STREAMMUX_HEIGHT - 1, (guint) MAX(0, obj_meta->rect_params.top - font_size * 2 + border_width *
      0.5f + 1));

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
parse_pose_from_meta(NvDsBatchMeta *batch_meta, NvDsFrameMeta *frame_meta, NvDsObjectMeta *obj_meta)
{
  NvDsDisplayMeta *display_meta = NULL;

  guint num_joints = obj_meta->mask_params.size / (sizeof(gfloat) * 3);

  gfloat gain = MIN((gfloat) obj_meta->mask_params.width / STREAMMUX_WIDTH, (gfloat) obj_meta->mask_params.height /
      STREAMMUX_HEIGHT);

  gfloat pad_x = (obj_meta->mask_params.width - STREAMMUX_WIDTH * gain) * 0.5f;
  gfloat pad_y = (obj_meta->mask_params.height - STREAMMUX_HEIGHT * gain) * 0.5f;

  for (guint i = 0; i < num_joints; ++i) {
    gfloat xc = (obj_meta->mask_params.data[i * 3 + 0] - pad_x) / gain;
    gfloat yc = (obj_meta->mask_params.data[i * 3 + 1] - pad_y) / gain;
    gfloat confidence = obj_meta->mask_params.data[i * 3 + 2];

    if (confidence < 0.5) {
      continue;
    }

    if (!display_meta || display_meta->num_circles == MAX_ELEMENTS_IN_DISPLAY_META) {
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

    if (!display_meta || display_meta->num_lines == MAX_ELEMENTS_IN_DISPLAY_META) {
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
nvosd_sink_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer user_data)
{
  GstBuffer *buf = (GstBuffer *) info->data;
  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);

  NvDsMetaList *l_frame = NULL;
  for (l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next) {
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) (l_frame->data);

    NvDsMetaList *l_obj = NULL;
    for (l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next) {
      NvDsObjectMeta *obj_meta = (NvDsObjectMeta *) (l_obj->data);

      parse_pose_from_meta(batch_meta, frame_meta, obj_meta);
      set_custom_bbox(obj_meta);
    }
  }

  return GST_PAD_PROBE_OK;
}

static void
uridecodebin_child_added_callback(GstChildProxy *child_proxy, GObject *object, gchar *name, gpointer user_data)
{
  if (g_strrstr(name, "decodebin")) {
    g_signal_connect(object, "child-added", G_CALLBACK(uridecodebin_child_added_callback), user_data);
  }
  else if (g_strrstr(name, "nvv4l2decoder")) {
    g_object_set(object, "drop-frame-interval", 0, "num-extra-surfaces", 1, "qos", 0, NULL);
    if (JETSON) {
      g_object_set(object, "enable-max-performance", 1, NULL);
    }
    else {
      g_object_set(object, "cudadec-memtype", 0, "gpu-id", GPU_ID, NULL);
    }
  }
}

static void
uridecodebin_pad_added_callback(GstElement *decodebin, GstPad *pad, gpointer user_data)
{
  GstPad *nvstreammux_sink_pad = (GstPad *) user_data;

  GstCaps *caps = gst_pad_get_current_caps(pad);
  if (!caps) {
    caps = gst_pad_query_caps(pad, NULL);
  }

  const GstStructure *str = gst_caps_get_structure(caps, 0);
  const gchar *name = gst_structure_get_name(str);
  GstCapsFeatures *features = gst_caps_get_features(caps, 0);

  if (!strncmp(name, "video", 5)) {
    if (gst_caps_features_contains(features, "memory:NVMM")) {
      if (gst_pad_link(pad, nvstreammux_sink_pad) != GST_PAD_LINK_OK) {
        g_printerr("ERROR - Failed to link source to nvstreammux sink pad\n");
      }
    }
    else {
      g_printerr("ERROR - decodebin did not pick NVIDIA decoder plugin\n");
    }
  }

  gst_caps_unref(caps);
}

static GstElement *
create_uridecodebin(guint stream_id, const gchar *uri, GstElement *nvstreammux)
{
  gchar bin_name[32] = { };
  g_snprintf(bin_name, 32, "source-bin-%04d", stream_id);

  GstElement *uridecodebin = gst_element_factory_make("uridecodebin", bin_name);

  if (g_strrstr(uri, "rtsp://")) {
    configure_source_for_ntp_sync(uridecodebin);
  }

  g_object_set(G_OBJECT(uridecodebin), "uri", uri, NULL);

  gchar pad_name[16];
  g_snprintf(pad_name, 16, "sink_%u", stream_id);

  GstPad *nvstreammux_sink_pad = gst_element_get_request_pad(nvstreammux, pad_name);
  if (!nvstreammux_sink_pad) {
    g_printerr("ERROR - Failed to get nvstreammux %s pad\n", pad_name);
    return NULL;
  }

  g_signal_connect(G_OBJECT(uridecodebin), "pad-added", G_CALLBACK(uridecodebin_pad_added_callback),
      nvstreammux_sink_pad);
  g_signal_connect(G_OBJECT(uridecodebin), "child-added", G_CALLBACK(uridecodebin_child_added_callback), NULL);

  gst_object_unref(nvstreammux_sink_pad);

  return uridecodebin;
}

static gboolean
bus_call(GstBus *bus, GstMessage *message, gpointer user_data)
{
  GMainLoop *loop = (GMainLoop *) user_data;
  switch (GST_MESSAGE_TYPE(message)) {
    case GST_MESSAGE_EOS:
    {
      g_print("DEBUG - EOS\n");
      g_main_loop_quit(loop);
      break;
    }
    case GST_MESSAGE_WARNING:
    {
      gchar *debug;
      GError *error;
      gst_message_parse_warning(message, &error, &debug);
      g_printerr("WARNING - %s - %s\n", error->message, debug);
      g_free(debug);
      g_error_free(error);
      break;
    }
    case GST_MESSAGE_ERROR:
    {
      gchar *debug;
      GError *error;
      gst_message_parse_error(message, &error, &debug);
      g_printerr("ERROR - %s - %s\n", error->message, debug);
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
    g_printerr("ERROR - %s\n", error->message);
    g_error_free(error);
    return -1;
  }
  g_option_context_free(ctx);

  if (!SOURCE) {
    g_printerr("ERROR - Source not found\n");
    return -1;
  }

  if (!INFER_CONFIG) {
    g_printerr("ERROR - Config infer not found\n");
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
    g_printerr("ERROR - Failed to create pipeline\n");
    return -1;
  }

  GstElement *nvstreammux = gst_element_factory_make("nvstreammux", "nvstreammux");
  if (!nvstreammux || !gst_bin_add(GST_BIN(pipeline), nvstreammux)) {
    g_printerr("ERROR - Failed to create nvstreammux\n");
    return -1;
  }

  GstElement *uridecodebin = create_uridecodebin(0, SOURCE, nvstreammux);
  if (!uridecodebin || !gst_bin_add(GST_BIN(pipeline), uridecodebin)) {
    g_printerr("ERROR - Failed to create uridecodebin\n");
    return -1;
  }

  GstElement *nvinfer = gst_element_factory_make("nvinfer", "nvinfer");
  if (!nvinfer || !gst_bin_add(GST_BIN(pipeline), nvinfer)) {
    g_printerr("ERROR - Failed to create nvinfer\n");
    return -1;
  }

  GstElement *nvvidconv = gst_element_factory_make("nvvideoconvert", "nvvideoconvert");
  if (!nvvidconv || !gst_bin_add(GST_BIN(pipeline), nvvidconv)) {
    g_printerr("ERROR - Failed to create nvvideoconvert\n");
    return -1;
  }

  GstElement *capsfilter = gst_element_factory_make("capsfilter", "capsfilter");
  if (!capsfilter || !gst_bin_add(GST_BIN(pipeline), capsfilter)) {
    g_printerr("ERROR - Failed to create capsfilter\n");
    return -1;
  }

  GstElement *nvosd = gst_element_factory_make("nvdsosd", "nvdsosd");
  if (!nvosd || !gst_bin_add(GST_BIN(pipeline), nvosd)) {
    g_printerr("ERROR - Failed to create nvdsosd\n");
    return -1;
  }

  GstElement *nvsink = NULL;
  if (JETSON) {
    nvsink = gst_element_factory_make("nv3dsink", "nv3dsink");
    if (!nvsink || !gst_bin_add(GST_BIN(pipeline), nvsink)) {
      g_printerr("ERROR - Failed to create nv3dsink\n");
      return -1;
    }
  }
  else {
    nvsink = gst_element_factory_make("nveglglessink", "nveglglessink");
    if (!nvsink || !gst_bin_add(GST_BIN(pipeline), nvsink)) {
      g_printerr("ERROR - Failed to create nveglglessink\n");
      return -1;
    }
  }

  g_print("\n");
  g_print("SOURCE: %s\n", SOURCE);
  g_print("INFER_CONFIG: %s\n", INFER_CONFIG);
  g_print("STREAMMUX_BATCH_SIZE: %d\n", STREAMMUX_BATCH_SIZE);
  g_print("STREAMMUX_WIDTH: %d\n", STREAMMUX_WIDTH);
  g_print("STREAMMUX_HEIGHT: %d\n", STREAMMUX_HEIGHT);
  g_print("GPU_ID: %d\n", GPU_ID);
  g_print("PERF_MEASUREMENT_INTERVAL_SEC: %d\n", PERF_MEASUREMENT_INTERVAL_SEC);
  g_print("JETSON: %s\n", JETSON ? "TRUE" : "FALSE");
  g_print("\n");

  GstCaps *caps = gst_caps_from_string("video/x-raw(memory:NVMM), format=RGBA");
  g_object_set(G_OBJECT(capsfilter), "caps", caps, NULL);
  gst_caps_unref(caps);

  g_object_set(G_OBJECT(nvstreammux), "batch-size", STREAMMUX_BATCH_SIZE, "batched-push-timeout", 25000,
      "width", STREAMMUX_WIDTH, "height", STREAMMUX_HEIGHT, "live-source", 1, NULL);
  g_object_set(G_OBJECT(nvinfer), "config-file-path", INFER_CONFIG, "qos", 0, NULL);
  g_object_set(G_OBJECT(nvosd), "process-mode", MODE_GPU, "qos", 0, NULL);
  g_object_set(G_OBJECT(nvsink), "async", 0, "sync", 0, "qos", 0, NULL);

  if (g_strrstr(SOURCE, "file://")) {
    g_object_set(G_OBJECT(nvstreammux), "live-source", 0, NULL);
  }

  if (!JETSON) {
    g_object_set(G_OBJECT(nvstreammux), "nvbuf-memory-type", NVBUF_MEM_CUDA_DEVICE, "gpu_id", GPU_ID, NULL);
    g_object_set(G_OBJECT(nvinfer), "gpu_id", GPU_ID, NULL);
    g_object_set(G_OBJECT(nvvidconv), "nvbuf-memory-type", NVBUF_MEM_CUDA_DEVICE, "gpu_id", GPU_ID, NULL);
    g_object_set(G_OBJECT(nvosd), "gpu_id", GPU_ID, NULL);
  }

  if (!gst_element_link_many(nvstreammux, nvinfer, nvvidconv, capsfilter, nvosd, nvsink, NULL)) {
    g_printerr("ERROR - Failed to link pipeline elements\n");
    return -1;
  }

  GstBus *bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline));
  guint bus_watch_id = gst_bus_add_watch(bus, bus_call, loop);
  gst_object_unref(bus);

  GstPad *nvosd_sink_pad = gst_element_get_static_pad(nvosd, "sink");
  if (!nvosd_sink_pad) {
    g_printerr("ERROR - Failed to get nvosd sink pad\n");
    return -1;
  }

  gst_pad_add_probe(nvosd_sink_pad, GST_PAD_PROBE_TYPE_BUFFER, nvosd_sink_pad_buffer_probe, NULL, NULL);

  NvDsAppPerfStructInt *perf_struct = (NvDsAppPerfStructInt *) g_malloc0(sizeof(NvDsAppPerfStructInt));
  enable_perf_measurement(perf_struct, nvosd_sink_pad, 1, PERF_MEASUREMENT_INTERVAL_SEC, 0, perf_cb);

  gst_object_unref(nvosd_sink_pad);

  gst_element_set_state(pipeline, GST_STATE_PAUSED);

  if (gst_element_set_state(pipeline, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
    g_printerr("ERROR - Failed to set pipeline to playing\n");
    return -1;
  }

  g_print("\n");

  g_main_loop_run(loop);

  gst_element_set_state(pipeline, GST_STATE_NULL);

  g_free(perf_struct);

  if (SOURCE) {
    g_free(SOURCE);
  }

  if (INFER_CONFIG) {
    g_free(INFER_CONFIG);
  }

  gst_object_unref(GST_OBJECT(pipeline));
  g_source_remove(bus_watch_id);
  g_main_loop_unref(loop);

  g_print("\n");

  return 0;
}
