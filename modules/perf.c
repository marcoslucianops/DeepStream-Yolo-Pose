#include <math.h>
#include <sys/time.h>

#include "perf.h"

static GMutex fps_lock;
static gdouble fps[MAX_SOURCE_BINS];
static gdouble fps_avg[MAX_SOURCE_BINS];

void
perf_cb(gpointer context, NvDsAppPerfStruct *str)
{
  guint32 i;
  guint32 numf = str->num_instances;
  g_mutex_lock(&fps_lock);
  for (i = 0; i < numf; ++i) {
    fps[i] = str->fps[i];
    fps_avg[i] = str->fps_avg[i];
  }
  for (i = 0; i < numf; ++i) {
    g_print("DEBUG - FPS of stream %d: %.2f (%.2f)\n", i + 1, fps[i], fps_avg[i]);
  }
  g_mutex_unlock(&fps_lock);
}

static GstPadProbeReturn
sink_bin_buf_probe(GstPad *pad, GstPadProbeInfo *info, gpointer user_data)
{
  NvDsAppPerfStructInt *str = (NvDsAppPerfStructInt *) user_data;
  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(GST_BUFFER(info->data));

  if (!batch_meta) {
    return GST_PAD_PROBE_OK;
  }

  if (!str->stop) {
    g_mutex_lock(&str->struct_lock);
    for (NvDsMetaList *l_frame = batch_meta->frame_meta_list; l_frame; l_frame = l_frame->next) {
      NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) l_frame->data;
      NvDsInstancePerfStruct *str1 = &str->instance_str[frame_meta->pad_index];
      gettimeofday(&str1->last_fps_time, NULL);
      if (str1->start_fps_time.tv_sec == 0 && str1->start_fps_time.tv_usec == 0) {
        str1->start_fps_time = str1->last_fps_time;
      }
      else {
        str1->buffer_cnt++;
      }
    }
    g_mutex_unlock(&str->struct_lock);
  }
  return GST_PAD_PROBE_OK;
}

static gboolean
perf_measurement_callback(gpointer user_data)
{
  NvDsAppPerfStructInt *str = (NvDsAppPerfStructInt *) user_data;
  guint buffer_cnt[MAX_SOURCE_BINS];
  NvDsAppPerfStruct perf_struct;
  struct timeval current_fps_time;
  guint i;

  g_mutex_lock(&str->struct_lock);
  if (str->stop) {
    g_mutex_unlock(&str->struct_lock);
    return FALSE;
  }

  for (i = 0; i < str->num_instances; i++) {
    buffer_cnt[i] = str->instance_str[i].buffer_cnt / str->dewarper_surfaces_per_frame;
    str->instance_str[i].buffer_cnt = 0;
  }

  perf_struct.num_instances = str->num_instances;
  gettimeofday(&current_fps_time, NULL);

  for (i = 0; i < str->num_instances; i++) {
    NvDsInstancePerfStruct *str1 = &str->instance_str[i];

    gdouble time1 =
        (str1->total_fps_time.tv_sec + str1->total_fps_time.tv_usec / 1000000.0) +
        (current_fps_time.tv_sec + current_fps_time.tv_usec / 1000000.0) - (str1->start_fps_time.tv_sec +
        str1->start_fps_time.tv_usec / 1000000.0);

    gdouble time2;

    if (str1->last_sample_fps_time.tv_sec == 0 && str1->last_sample_fps_time.tv_usec == 0) {
      time2 =
          (str1->last_fps_time.tv_sec + str1->last_fps_time.tv_usec / 1000000.0) -
          (str1->start_fps_time.tv_sec + str1->start_fps_time.tv_usec / 1000000.0);
    }
    else {
      time2 =
          (str1->last_fps_time.tv_sec + str1->last_fps_time.tv_usec / 1000000.0) -
          (str1->last_sample_fps_time.tv_sec + str1->last_sample_fps_time.tv_usec / 1000000.0);
    }

    str1->total_buffer_cnt += buffer_cnt[i];
    perf_struct.fps[i] = buffer_cnt[i] / time2;

    if (isnan(perf_struct.fps[i])) {
      perf_struct.fps[i] = 0;
    }

    perf_struct.fps_avg[i] = str1->total_buffer_cnt / time1;

    if (isnan(perf_struct.fps_avg[i])) {
      perf_struct.fps_avg[i] = 0;
    }

    str1->last_sample_fps_time = str1->last_fps_time;
  }

  g_mutex_unlock(&str->struct_lock);

  str->callback(str->context, &perf_struct);

  return TRUE;
}

void
resume_perf_measurement(NvDsAppPerfStructInt *str)
{
  guint i;

  g_mutex_lock(&str->struct_lock);
  if (!str->stop) {
    g_mutex_unlock(&str->struct_lock);
    return;
  }

  str->stop = FALSE;

  for (i = 0; i < str->num_instances; i++) {
    str->instance_str[i].buffer_cnt = 0;
  }

  if (!str->perf_measurement_timeout_id) {
    str->perf_measurement_timeout_id = g_timeout_add(str->measurement_interval_ms, perf_measurement_callback, str);
  }

  g_mutex_unlock(&str->struct_lock);
}

gboolean
enable_perf_measurement(NvDsAppPerfStructInt *str, GstPad *sink_bin_pad, guint num_sources, gulong interval_sec,
    guint num_surfaces_per_frame, perf_callback callback)
{
  guint i;

  if (!callback) {
    return FALSE;
  }

  str->num_instances = num_sources;
  str->measurement_interval_ms = interval_sec * 1000;
  str->callback = callback;
  str->stop = TRUE;

  if (num_surfaces_per_frame) {
    str->dewarper_surfaces_per_frame = num_surfaces_per_frame;
  }
  else {
    str->dewarper_surfaces_per_frame = 1;
  }

  for (i = 0; i < num_sources; i++) {
    str->instance_str[i].buffer_cnt = 0;
  }

  str->sink_bin_pad = sink_bin_pad;
  str->fps_measure_probe_id = gst_pad_add_probe(sink_bin_pad, GST_PAD_PROBE_TYPE_BUFFER, sink_bin_buf_probe, str, NULL);

  resume_perf_measurement(str);

  return TRUE;
}
