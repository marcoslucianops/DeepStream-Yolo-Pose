#include <cassert>
#include <algorithm>
#include <iostream>

#include "nvdsinfer_custom_impl.h"

#define NMS_THRESH 0.45;

extern "C" bool
NvDsInferParseYoloPose(std::vector<NvDsInferLayerInfo> const& outputLayersInfo, NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams, std::vector<NvDsInferInstanceMaskInfo>& objectList);

static float
clamp(float val, float minVal, float maxVal)
{
  assert(minVal <= maxVal);
  return std::min(maxVal, std::max(minVal, val));
}

static float
overlap1D(float x1min, float x1max, float x2min, float x2max)
{
  if (x1min > x2min) {
    std::swap(x1min, x2min);
    std::swap(x1max, x2max);
  }
  return x1max < x2min ? 0 : std::min(x1max, x2max) - x2min;
}

static float
computeIoU(NvDsInferInstanceMaskInfo& bbox1, NvDsInferInstanceMaskInfo& bbox2)
{
  float overlapX = overlap1D(bbox1.left, bbox1.left + bbox1.width, bbox2.left, bbox2.left + bbox2.width);
  float overlapY = overlap1D(bbox1.top, bbox1.top + bbox1.height, bbox2.top, bbox2.top + bbox2.height);
  float area1 = (bbox1.width) * (bbox1.height);
  float area2 = (bbox2.width) * (bbox2.height);
  float overlap2D = overlapX * overlapY;
  float u = area1 + area2 - overlap2D;
  return u == 0 ? 0 : overlap2D / u;
}

static std::vector<NvDsInferInstanceMaskInfo>
nonMaximumSuppression(std::vector<NvDsInferInstanceMaskInfo> binfo)
{
  std::stable_sort(binfo.begin(), binfo.end(), [](const NvDsInferInstanceMaskInfo& b1,
      const NvDsInferInstanceMaskInfo& b2) {
    return b1.detectionConfidence > b2.detectionConfidence;
  });

  std::vector<NvDsInferInstanceMaskInfo> out;

  for (auto i : binfo) {
    bool keep = true;
    for (auto j : out) {
      if (keep) {
        float overlap = computeIoU(i, j);
        keep = overlap <= NMS_THRESH;
      }
      else {
        break;
      }
    }
    if (keep) {
      out.push_back(i);
    }
    else {
      delete[] i.mask;
    }
  }

  return out;
}

static std::vector<NvDsInferInstanceMaskInfo>
nmsAllClasses(std::vector<NvDsInferInstanceMaskInfo>& binfo)
{
  std::vector<NvDsInferInstanceMaskInfo> result = nonMaximumSuppression(binfo);
  return result;
}

static void
addPoseProposal(const float* output, size_t channelsSize, uint netW, uint netH, size_t n, NvDsInferInstanceMaskInfo& b)
{
  size_t kptsSize = channelsSize - 5;
  b.mask = new float[kptsSize];
  for (size_t p = 0; p < kptsSize / 3; ++p) {
    b.mask[p * 3 + 0] = clamp(output[n * channelsSize + p * 3 + 5], 0, netW);
    b.mask[p * 3 + 1] = clamp(output[n * channelsSize + p * 3 + 6], 0, netH);
    b.mask[p * 3 + 2] = output[n * channelsSize + p * 3 + 7];
  }
  b.mask_width = netW;
  b.mask_height = netH;
  b.mask_size = sizeof(float) * kptsSize;
}

static NvDsInferInstanceMaskInfo
convertBBox(float x1, float y1, float x2, float y2, uint netW, uint netH)
{
  NvDsInferInstanceMaskInfo b;

  x1 = clamp(x1, 0, netW);
  y1 = clamp(y1, 0, netH);
  x2 = clamp(x2, 0, netW);
  y2 = clamp(y2, 0, netH);

  b.left = x1;
  b.width = clamp(x2 - x1, 0, netW);
  b.top = y1;
  b.height = clamp(y2 - y1, 0, netH);

  return b;
}

static void
addBBoxProposal(float x1, float y1, float x2, float y2, uint netW, uint netH, int maxIndex, float maxProb,
    NvDsInferInstanceMaskInfo& b)
{
  b = convertBBox(x1, y1, x2, y2, netW, netH);

  if (b.width < 1 || b.height < 1) {
      return;
  }

  b.detectionConfidence = maxProb;
  b.classId = maxIndex;
}

static std::vector<NvDsInferInstanceMaskInfo>
decodeTensorYoloPose(const float* output, size_t outputSize, size_t channelsSize, uint netW, uint netH,
    const std::vector<float>& preclusterThreshold)
{
  std::vector<NvDsInferInstanceMaskInfo> objects;

  for (size_t n = 0; n < outputSize; ++n) {
    float maxProb = output[n * channelsSize + 4];

    if (maxProb < preclusterThreshold[0]) {
      continue;
    }

    float x1 = output[n * channelsSize + 0];
    float y1 = output[n * channelsSize + 1];
    float x2 = output[n * channelsSize + 2];
    float y2 = output[n * channelsSize + 3];

    NvDsInferInstanceMaskInfo b;

    addBBoxProposal(x1, y1, x2, y2, netW, netH, 0, maxProb, b);
    addPoseProposal(output, channelsSize, netW, netH, n, b);

    objects.push_back(b);
  }

  return objects;
}

static bool
NvDsInferParseCustomYoloPose(std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo, NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferInstanceMaskInfo>& objectList)
{
  if (outputLayersInfo.empty()) {
    std::cerr << "ERROR - Could not find output layer" << std::endl;
    return false;
  }

  const NvDsInferLayerInfo& output = outputLayersInfo[0];

  size_t outputSize = output.inferDims.d[0];
  size_t channelsSize = output.inferDims.d[1];

  std::vector<NvDsInferInstanceMaskInfo> objects = decodeTensorYoloPose((const float*) (output.buffer), outputSize,
      channelsSize, networkInfo.width, networkInfo.height, detectionParams.perClassPreclusterThreshold);

  objectList.clear();
  objectList = nmsAllClasses(objects);

  return true;
}

extern "C" bool
NvDsInferParseYoloPose(std::vector<NvDsInferLayerInfo> const& outputLayersInfo, NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams, std::vector<NvDsInferInstanceMaskInfo>& objectList)
{
  return NvDsInferParseCustomYoloPose(outputLayersInfo, networkInfo, detectionParams, objectList);
}

CHECK_CUSTOM_INSTANCE_MASK_PARSE_FUNC_PROTOTYPE(NvDsInferParseYoloPose);
