#ifdef _WIN32
#include <wchar.h> // _wgetenv for nvtx
#endif

#ifndef ROCM_ON_WINDOWS
#ifdef TORCH_CUDA_USE_NVTX3
#include <nvtx3/nvtx3.hpp>
#else // TORCH_CUDA_USE_NVTX3
#include <nvToolsExt.h>
#endif // TORCH_CUDA_USE_NVTX3
#else // ROCM_ON_WINDOWS
#include <c10/util/Exception.h>
#endif // ROCM_ON_WINDOWS

#ifdef USE_ROCM
#include <roctracer/roctx.h>
#include <hip/hip_runtime.h>
#else
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#endif

#include <torch/csrc/utils/pybind.h>

namespace torch::cuda::shared {

#if !defined(ROCM_ON_WINDOWS) || defined(USE_ROCM)
struct RangeHandle {
#ifdef USE_ROCM
  roctx_range_id_t id;
#else
  nvtxRangeId_t id;
#endif
  const char* msg;
};

static void device_callback_range_end(void* userData) {
  RangeHandle* handle = ((RangeHandle*)userData);
#ifdef USE_ROCM
  roctxRangeStop(handle->id);
#else
  nvtxRangeEnd(handle->id);
#endif
  free((void*)handle->msg);
  free((void*)handle);
}

static void device_nvtxRangeEnd(void* handle, std::intptr_t stream) {
#ifdef USE_ROCM
  TORCH_CHECK(hipLaunchHostFunc(
      (hipStream_t)stream, device_callback_range_end, handle) == hipSuccess,
      "Failed to launch HIP host function for range end");
#else
  C10_CUDA_CHECK(cudaLaunchHostFunc(
      (cudaStream_t)stream, device_callback_range_end, handle));
#endif
}

static void device_callback_range_start(void* userData) {
  RangeHandle* handle = ((RangeHandle*)userData);
#ifdef USE_ROCM
  handle->id = roctxRangeStartA(handle->msg);
#else
  handle->id = nvtxRangeStartA(handle->msg);
#endif
}

static void* device_nvtxRangeStart(const char* msg, std::intptr_t stream) {
  RangeHandle* handle = (RangeHandle*)calloc(sizeof(RangeHandle), 1);
  handle->msg = strdup(msg);
  handle->id = 0;
#ifdef USE_ROCM
  TORCH_CHECK(
      hipLaunchHostFunc(
          (hipStream_t)stream, device_callback_range_start, (void*)handle) ==
      hipSuccess,
      "Failed to launch HIP host function for range start");
#else
  TORCH_CHECK(
      cudaLaunchHostFunc(
          (cudaStream_t)stream, device_callback_range_start, (void*)handle) ==
      cudaSuccess);
#endif
  return handle;
}

void initNvtxBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

#ifdef USE_ROCM
  auto nvtx = m.def_submodule("_nvtx", "roctracer/roctx bindings");
  nvtx.def("rangePushA", roctxRangePushA);
  nvtx.def("rangePop", roctxRangePop);
  nvtx.def("rangeStartA", roctxRangeStartA);
  nvtx.def("rangeEnd", roctxRangeStop);
  nvtx.def("markA", roctxMarkA);
#else
#ifdef TORCH_CUDA_USE_NVTX3
  auto nvtx = m.def_submodule("_nvtx", "nvtx3 bindings");
#else
  auto nvtx = m.def_submodule("_nvtx", "libNvToolsExt.so bindings");
#endif
  nvtx.def("rangePushA", nvtxRangePushA);
  nvtx.def("rangePop", nvtxRangePop);
  nvtx.def("rangeStartA", nvtxRangeStartA);
  nvtx.def("rangeEnd", nvtxRangeEnd);
  nvtx.def("markA", nvtxMarkA);
#endif
  nvtx.def("deviceRangeStart", device_nvtxRangeStart);
  nvtx.def("deviceRangeEnd", device_nvtxRangeEnd);
}

#else // !defined(ROCM_ON_WINDOWS) || defined(USE_ROCM)

static void printUnavailableWarning() {
  TORCH_WARN_ONCE("Warning: roctracer isn't available on Windows");
}

static int rangePushA(const std::string&) {
  printUnavailableWarning();
  return 0;
}

static int rangePop() {
  printUnavailableWarning();
  return 0;
}

static int rangeStartA(const std::string&) {
  printUnavailableWarning();
  return 0;
}

static void rangeEnd(int) {
  printUnavailableWarning();
}

static void markA(const std::string&) {
  printUnavailableWarning();
}

static py::object deviceRangeStart(const std::string&, std::intptr_t) {
  printUnavailableWarning();
  return py::none(); // Return an appropriate default object
}

static void deviceRangeEnd(py::object, std::intptr_t) {
  printUnavailableWarning();
}

void initNvtxBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();
  auto nvtx = m.def_submodule("_nvtx", "unavailable");

  nvtx.def("rangePushA", rangePushA);
  nvtx.def("rangePop", rangePop);
  nvtx.def("rangeStartA", rangeStartA);
  nvtx.def("rangeEnd", rangeEnd);
  nvtx.def("markA", markA);
  nvtx.def("deviceRangeStart", deviceRangeStart);
  nvtx.def("deviceRangeEnd", deviceRangeEnd);
}
#endif // !defined(ROCM_ON_WINDOWS) || defined(USE_ROCM)

} // namespace torch::cuda::shared
