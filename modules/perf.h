#ifndef __PERF_H__
#define __PERF_H__

#include "gstnvdsmeta.h"

#define MAX_SOURCE_BINS 1024

typedef struct
{
  gdouble fps[MAX_SOURCE_BINS];
  gdouble fps_avg[MAX_SOURCE_BINS];
  guint num_instances;
} NvDsAppPerfStruct;

typedef void (*perf_callback) (gpointer ctx, NvDsAppPerfStruct *str);

typedef struct
{
  guint buffer_cnt;
  guint total_buffer_cnt;
  struct timeval total_fps_time;
  struct timeval start_fps_time;
  struct timeval last_fps_time;
  struct timeval last_sample_fps_time;
} NvDsInstancePerfStruct;

typedef struct
{
  gulong measurement_interval_ms;
  gulong perf_measurement_timeout_id;
  guint num_instances;
  gboolean stop;
  gpointer context;
  GMutex struct_lock;
  perf_callback callback;
  GstPad *sink_bin_pad;
  gulong fps_measure_probe_id;
  NvDsInstancePerfStruct instance_str[MAX_SOURCE_BINS];
  guint dewarper_surfaces_per_frame;
} NvDsAppPerfStructInt;

void perf_cb(gpointer context, NvDsAppPerfStruct *str);

gboolean enable_perf_measurement(NvDsAppPerfStructInt *str, GstPad *sink_bin_pad, guint num_sources, gulong interval_sec,
    guint num_surfaces_per_frame, perf_callback callback);

#endif
