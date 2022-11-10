// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <assert.h>
// #include <png.h>
#include <stdio.h>

#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>
#include <opencv4/opencv2/imgproc.hpp>

#include "onnxruntime_c_api.h"
#ifdef _WIN32
#ifdef USE_DML
#include "providers.h"
#endif
#include <objbase.h>
#endif

#ifdef _WIN32
#define tcscmp wcscmp
#else
#define tcscmp strcmp
#endif

const OrtApi *g_ort = NULL;

#define ORT_ABORT_ON_ERROR(expr)                                               \
  do {                                                                         \
    OrtStatus *onnx_status = (expr);                                           \
    if (onnx_status != NULL) {                                                 \
      const char *msg = g_ort->GetErrorMessage(onnx_status);                   \
      fprintf(stderr, "%s\n", msg);                                            \
      g_ort->ReleaseStatus(onnx_status);                                       \
      abort();                                                                 \
    }                                                                          \
  } while (0);

#define LOG(x) do { std::cout << x << std::endl; } while(0);


/**
 * convert input from HWC format to CHW format
 * \param input float32 single image. The array has length of 3*h*w
 * \param h image height
 * \param w image width
 * \param output A float array. should be freed by caller after use
 * \param output_count Array length of the `output` param
 */
template <typename T>
static T* hwc_to_chw(const T *input, size_t h, size_t w) {
  size_t stride = h * w;
  size_t output_count = stride * 3;
  T *output_data = (T *)malloc(output_count * sizeof(T));
  for (size_t i = 0; i != stride; ++i) {
    for (size_t c = 0; c != 3; ++c) {
      output_data[c * stride + i] = input[i * 3 + c];
    }
  }
  return output_data;
}

// Create expected ORT format (channel blocks by row): https://answers.opencv.org/question/64837
void mat_hwc2chw(cv::Mat image_bgr, float *input_tensor_values, int64_t input_tensor_size) {
    LOG("doing mat hwc2chw")
    std::vector<cv::Mat> channels;
    cv::Mat image_by_channel;

    LOG("split")
    cv::split(image_bgr, channels);

    LOG("building image_by_channel")
    for (size_t i=0; i<channels.size(); i++) {
        image_by_channel.push_back(channels[i]);
    }
    if (!image_by_channel.isContinuous()) {
        LOG("making continuous")
        image_by_channel = image_by_channel.clone();
    }
    LOG("dividing values")
    for (size_t i = 0; i < input_tensor_size; i++) {
        input_tensor_values[i] = image_by_channel.data[i] / 255.0;
    }

}

/**
 * convert input from CHW format to HWC format
 * \param input A single image. This float array has length of 3*h*w
 * \param h image height
 * \param w image width
 * \param output A byte array. should be freed by caller after use
 *
 * \return caller should free returned pointer
 */
template <typename T>
T* chw_to_hwc(const T *input, size_t h, size_t w) {
  LOG("chw_to_hwc: h="<<h<<", w="<<w)
  size_t stride = h * w;
  LOG("Stride: "<<stride)
  LOG("allocing "<<stride*3)
  T* output_data = (T *) malloc(stride * 3*sizeof(T));
  for (int c = 0; c != 3; ++c) {
    size_t t = c * stride;
    for (size_t i = 0; i != stride; ++i) {
      T f = (T) input[t + i];
      output_data[i * 3 + c] = f;
    }
  }
  return output_data;
}

/**
 * @brief Operator overloading for printing vectors
 * @tparam T
 * @param os
 * @param v
 * @return std::ostream&
 */
template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
    os << "[";
    for (int i = 0; i < v.size(); ++i)
    {
        os << v[i];
        if (i != v.size() - 1)
        {
            os << ", ";
        }
    }
    os << "]";
    return os;
}


// from https://stackoverflow.com/questions/48207847/how-to-create-cvmat-from-buffer-array-of-t-data-using-a-template-function
// only changed chs default to 3
template <typename T>
cv::Mat mat_from_buffer(T* data, int rows, int cols, int chs = 3) {
    // Create Mat from buffer 
    cv::Mat mat(rows, cols, CV_MAKETYPE(cv::DataType<T>::type, chs));
    memcpy(mat.data, data, rows*cols*chs * sizeof(T));
    return mat;
}

// }

/**
 * \param tensor should be a float tensor in [N,C,H,W] format
 */
static int write_tensor_to_img_file(OrtValue *tensor, const char *output_file) {
  struct OrtTensorTypeAndShapeInfo *shape_info;
  ORT_ABORT_ON_ERROR(g_ort->GetTensorTypeAndShape(tensor, &shape_info));
  size_t dim_count;
  ORT_ABORT_ON_ERROR(g_ort->GetDimensionsCount(shape_info, &dim_count));
  if (dim_count != 4) {
    printf("output tensor must have 4 dimensions");
    return -1;
  }
  int64_t dims[4];
  ORT_ABORT_ON_ERROR(
      g_ort->GetDimensions(shape_info, dims, sizeof(dims) / sizeof(dims[0])));
  if (dims[0] != 1 || dims[1] != 3) {
    printf("output tensor shape error");
    return -1;
  }
  float *tensor_ptr;
  ORT_ABORT_ON_ERROR(g_ort->GetTensorMutableData(tensor, (void **)&tensor_ptr));
  tensor_ptr = chw_to_hwc(tensor_ptr, dims[2], dims[3]);
  cv::Mat out_img = mat_from_buffer<float>(tensor_ptr, dims[2], dims[3]);
  cv::Mat out_img_uint;
  out_img.convertTo(out_img_uint, CV_8UC3, 255.0);

  cv::imwrite(output_file, out_img_uint);

  return 0;
}

static void usage() {
  printf("usage: <model_path> <input_file> <output_file> [cpu|cuda|dml] \n");
}

#ifdef _WIN32
static char *convert_string(const wchar_t *input) {
  size_t src_len = wcslen(input) + 1;
  if (src_len > INT_MAX) {
    printf("size overflow\n");
    abort();
  }
  const int len =
      WideCharToMultiByte(CP_ACP, 0, input, (int)src_len, NULL, 0, NULL, NULL);
  assert(len > 0);
  char *ret = (char *)malloc(len);
  assert(ret != NULL);
  const int r =
      WideCharToMultiByte(CP_ACP, 0, input, (int)src_len, ret, len, NULL, NULL);
  assert(len == r);
  return ret;
}
#endif

cv::Mat bgr_uint2rgb_float(cv::Mat input_image) {
  cv::Mat result(input_image.rows, input_image.cols, input_image.type());
  cv::cvtColor(input_image, result, cv::COLOR_BGR2RGB);
  input_image.convertTo(result, CV_32FC3, 1/255.0);
  return result;
}

int run_inference(OrtSession *session, const ORTCHAR_T *input_file,
                  const ORTCHAR_T *output_file) {
  int64_t input_height;
  int64_t input_width;
  size_t model_input_ele_count;
#ifdef _WIN32
  const char *output_file_p = convert_string(output_file);
  const char *input_file_p = convert_string(input_file);
#else
  const char *output_file_p = output_file;
  const char *input_file_p = input_file;
#endif

  // legge input
  cv::Mat input_image = cv::imread(input_file_p, cv::IMREAD_COLOR);
  input_height = input_image.rows;
  input_width = input_image.cols;
  // Normalize image from uint8 in [0-255] to float in [0-1]: 
  // this code...
  ///*
  int64_t input_size = 3*input_height*input_width;
  float * model_input = (float*) calloc(input_size, sizeof(float));
  assert(model_input != NULL);
  mat_hwc2chw(input_image, model_input, input_size);
  //*/
  // ... and this code both produce the same identical result:
  /*
  input_image = bgr_uint2rgb_float(input_image);
  float * model_input = hwc_to_chw((float*)input_image.ptr(), input_height, input_width);
  */


  OrtMemoryInfo *memory_info;
  ORT_ABORT_ON_ERROR(g_ort->CreateCpuMemoryInfo(
      OrtArenaAllocator, OrtMemTypeDefault, &memory_info));
  const int64_t input_shape[] = {1, 3, input_height, input_width};
  const size_t input_shape_len = sizeof(input_shape) / sizeof(input_shape[0]);
  model_input_ele_count = input_shape[0]*input_shape[1]*input_shape[2]*input_shape[3];
  const size_t model_input_len = model_input_ele_count * sizeof(float);

  LOG("input shape: [" << input_shape[0] <<", "<<input_shape[1] <<", "<<input_shape[2]<<", "<<input_shape[3]<<"]")

  OrtValue *input_tensor = NULL;
  ORT_ABORT_ON_ERROR(g_ort->CreateTensorWithDataAsOrtValue(
      memory_info, model_input, model_input_len, input_shape, input_shape_len,
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor));
  assert(input_tensor != NULL);
  int is_tensor;
  ORT_ABORT_ON_ERROR(g_ort->IsTensor(input_tensor, &is_tensor));
  assert(is_tensor);
  g_ort->ReleaseMemoryInfo(memory_info);
  const char *input_names[] = {"input"};
  const char *output_names[] = {"output"};
  OrtValue *output_tensor = NULL;

  LOG("running session")
  ORT_ABORT_ON_ERROR(g_ort->Run(session, NULL, input_names,
                                (const OrtValue *const *)&input_tensor, 1,
                                output_names, 1, &output_tensor));
  assert(output_tensor != NULL);
  ORT_ABORT_ON_ERROR(g_ort->IsTensor(output_tensor, &is_tensor));
  assert(is_tensor);
  int ret = 0;
  LOG("writing image to file")
  if (write_tensor_to_img_file(output_tensor, output_file_p) != 0) {
    ret = -1;
  }
  g_ort->ReleaseValue(output_tensor);
  g_ort->ReleaseValue(input_tensor);
#ifdef _WIN32
  free(input_file_p);
  free(output_file_p);
#endif // _WIN32
  return ret;
}

void verify_input_output_count(OrtSession *session) {
  size_t count;
  ORT_ABORT_ON_ERROR(g_ort->SessionGetInputCount(session, &count));
  assert(count == 1);
  ORT_ABORT_ON_ERROR(g_ort->SessionGetOutputCount(session, &count));
  assert(count == 1);
}

int enable_cuda(OrtSessionOptions *session_options) {
  // OrtCUDAProviderOptions is a C struct. C programming language doesn't have
  // constructors/destructors.
  OrtCUDAProviderOptions o;
  // Here we use memset to initialize every field of the above data struct to
  // zero.
  memset(&o, 0, sizeof(o));
  // But is zero a valid value for every variable? Not quite. It is not
  // guaranteed. In the other words: does every enum type contain zero? The
  // following line can be omitted because EXHAUSTIVE is mapped to zero in
  // onnxruntime_c_api.h.
  o.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
  o.gpu_mem_limit = SIZE_MAX;
  OrtStatus *onnx_status =
      g_ort->SessionOptionsAppendExecutionProvider_CUDA(session_options, &o);
  if (onnx_status != NULL) {
    const char *msg = g_ort->GetErrorMessage(onnx_status);
    fprintf(stderr, "%s\n", msg);
    g_ort->ReleaseStatus(onnx_status);
    return -1;
  }
  return 0;
}

#ifdef USE_DML
void enable_dml(OrtSessionOptions *session_options) {
  ORT_ABORT_ON_ERROR(
      OrtSessionOptionsAppendExecutionProvider_DML(session_options, 0));
}
#endif

#ifdef _WIN32
int wmain(int argc, wchar_t *argv[]) {
#else
int main(int argc, char *argv[]) {
#endif
  if (argc < 4) {
    usage();
    return -1;
  }

  g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  if (!g_ort) {
    fprintf(stderr, "Failed to init ONNX Runtime engine.\n");
    return -1;
  }
#ifdef _WIN32
  // CoInitializeEx is only needed if Windows Image Component will be used in
  // this program for image loading/saving.
  HRESULT hr = CoInitializeEx(NULL, COINIT_MULTITHREADED);
  if (!SUCCEEDED(hr))
    return -1;
#endif
  ORTCHAR_T *model_path = argv[1];
  ORTCHAR_T *input_file = argv[2];
  ORTCHAR_T *output_file = argv[3];
  // By default it will try CUDA first. If CUDA is not available, it will run
  // all the things on CPU. But you can also explicitly set it to DML(directml)
  // or CPU(which means cpu-only).
  ORTCHAR_T *execution_provider = (argc >= 5) ? argv[4] : NULL;
  OrtEnv *env;
  ORT_ABORT_ON_ERROR(g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "test", &env));
  assert(env != NULL);
  int ret = 0;
  OrtSessionOptions *session_options;
  ORT_ABORT_ON_ERROR(g_ort->CreateSessionOptions(&session_options));

  if (execution_provider) {
    if (tcscmp(execution_provider, ORT_TSTR("cpu")) == 0) {
      // Nothing; this is the default
    } else if (tcscmp(execution_provider, ORT_TSTR("dml")) == 0) {
#ifdef USE_DML
      enable_dml(session_options);
#else
      puts("DirectML is not enabled in this build.");
      return -1;
#endif
    } else {
      usage();
      puts("Invalid execution provider option.");
      return -1;
    }
  } else {
    printf("Try to enable CUDA first\n");
    ret = enable_cuda(session_options);
    if (ret) {
      fprintf(stderr, "CUDA is not available\n");
    } else {
      printf("CUDA is enabled\n");
    }
  }

  OrtSession *session;
  ORT_ABORT_ON_ERROR(
      g_ort->CreateSession(env, model_path, session_options, &session));
  verify_input_output_count(session);
  ret = run_inference(session, input_file, output_file);
  g_ort->ReleaseSessionOptions(session_options);
  g_ort->ReleaseSession(session);
  g_ort->ReleaseEnv(env);
  if (ret != 0) {
    fprintf(stderr, "fail\n");
  }
#ifdef _WIN32
  CoUninitialize();
#endif
  return ret;
}
