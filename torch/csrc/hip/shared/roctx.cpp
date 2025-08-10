#ifdef _WIN32
#include <wchar.h> // _wgetenv for roctx
#endif

#include <c10/util/Exception.h>
#ifdef USE_ROCM
#include <roctracer/roctx.h>
#include <hip/hip_runtime.h>
#endif // USE_ROCM
#include <torch/csrc/utils/pybind.h>

namespace torch::hip::shared {

#ifdef USE_ROCM
struct RangeHandle {
  roctx_range_id_t id;
  const char* msg;
};

static void device_callback_range_end(void* userData) {
  RangeHandle* handle = ((RangeHandle*)userData);
  roctxRangeStop(handle->id);
  free((void*)handle->msg);
  free((void*)handle);
}

static void device_roctxRangeEnd(void* handle, std::intptr_t stream) {
  // Use TORCH_CHECK since we don't have HIP exception macros readily available
  auto result = hipLaunchHostFunc(
      (hipStream_t)stream, device_callback_range_end, handle);
  TORCH_CHECK(result == hipSuccess,
      "Failed to launch HIP host function for range end: ", result);
}

static void device_callback_range_start(void* userData) {
  RangeHandle* handle = ((RangeHandle*)userData);
  handle->id = roctxRangeStartA(handle->msg);
}

static void* device_roctxRangeStart(const char* msg, std::intptr_t stream) {
  RangeHandle* handle = (RangeHandle*)calloc(sizeof(RangeHandle), 1);
  handle->msg = strdup(msg);
  handle->id = 0;
  auto result = hipLaunchHostFunc(
      (hipStream_t)stream, device_callback_range_start, (void*)handle);
  TORCH_CHECK(result == hipSuccess,
      "Failed to launch HIP host function for range start: ", result);
  return handle;
}

void initRoctxBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  auto roctx = m.def_submodule("_roctx", "roctracer/roctx bindings");
  roctx.def("rangePushA", roctxRangePushA);
  roctx.def("rangePop", roctxRangePop);
  roctx.def("rangeStartA", roctxRangeStartA);
  roctx.def("rangeEnd", roctxRangeStop);
  roctx.def("markA", roctxMarkA);
  roctx.def("deviceRangeStart", device_roctxRangeStart);
  roctx.def("deviceRangeEnd", device_roctxRangeEnd);
}

#else // USE_ROCM

static void printUnavailableWarning() {
  TORCH_WARN_ONCE("Warning: roctracer isn't available");
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

void initRoctxBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();
  auto roctx = m.def_submodule("_roctx", "unavailable");

  roctx.def("rangePushA", rangePushA);
  roctx.def("rangePop", rangePop);
  roctx.def("rangeStartA", rangeStartA);
  roctx.def("rangeEnd", rangeEnd);
  roctx.def("markA", markA);
  roctx.def("deviceRangeStart", deviceRangeStart);
  roctx.def("deviceRangeEnd", deviceRangeEnd);
}
#endif // USE_ROCM

} // namespace torch::hip::shared
