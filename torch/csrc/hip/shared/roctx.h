#pragma once

#include <torch/csrc/utils/pybind.h>

namespace torch::hip::shared {

void initRoctxBindings(PyObject* module);

} // namespace torch::hip::shared
