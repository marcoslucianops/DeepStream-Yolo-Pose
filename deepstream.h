#ifndef __DEEPSTREAM_H__
#define __DEEPSTREAM_H__

#include <nvdsgstutils.h>
#include <cuda_runtime_api.h>

#include "gstnvdsmeta.h"
#include "nvbufsurface.h"

#include "modules/interrupt.h"
#include "modules/perf.h"

static gchar *SOURCE = NULL;
static gchar *INFER_CONFIG = NULL;
static guint STREAMMUX_BATCH_SIZE = 1;
static guint STREAMMUX_WIDTH = 1920;
static guint STREAMMUX_HEIGHT = 1080;
static guint GPU_ID = 0;

static guint PERF_MEASUREMENT_INTERVAL_SEC = 5;
static gboolean JETSON = FALSE;

static gint skeleton[][2] = {{16, 14}, {14, 12}, {17, 15}, {15, 13}, {12, 13}, {6, 12}, {7, 13}, {6, 7}, {6, 8}, {7, 9},
    {8, 10}, {9, 11}, {2, 3}, {1, 2}, {1, 3}, {2, 4}, {3, 5}, {4, 6}, {5, 7}};

#endif
