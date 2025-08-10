#include <ATen/core/PythonOpRegistrationTrampoline.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <chrono>
#include <list>
#include <sstream>
#include <utility>
// Patch
#include <ATen/Context.h>
#include <nvtx3/nvtx3.hpp>
#ifdef USE_ROCM
#include <roctracer/roctx.h>
#endif
#include <sys/syscall.h>
#include <unistd.h>
#include <zlib.h>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <queue>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <random>
#ifdef FBCODE_CAFFE2
#include <c10/util/static_tracepoint.h>
#endif

namespace c10 {

#ifdef FBCODE_CAFFE2
TORCH_SDT_DEFINE_SEMAPHORE(operator_start)
TORCH_SDT_DEFINE_SEMAPHORE(operator_end)
#endif

bool show_dispatch_trace() {
  static auto envar = std::getenv("TORCH_SHOW_DISPATCH_TRACE");

  if (envar) {
    if (strcmp(envar, "0") == 0) {
      return false;
    }
    if (strcmp(envar, "1") == 0) {
      return true;
    }
    TORCH_WARN(
        "ignoring invalid value for TORCH_SHOW_DISPATCH_TRACE: ",
        envar,
        " valid values are 0 or 1.");
  }

  return false;
}

static thread_local int64_t dispatch_trace_nesting_value_;

// Patch, the following logic required log.
#define CUDA_TIME_MEASUREMENT 1
static thread_local uint64_t log_sequence = 0;

// We duplicate it to prevent potential interfence with the existing profile
// framework
static thread_local int16_t a10_dispatch_trace_nesting_value_;

// RNG for CUDA NVTX suffixes (1–10000)
static thread_local std::mt19937_64 rng{std::random_device{}()};
static thread_local std::uniform_int_distribution<uint64_t> dist(1, 10000);

A10TraceNestingGuard::A10TraceNestingGuard() {
  ++a10_dispatch_trace_nesting_value_;
}

A10TraceNestingGuard::~A10TraceNestingGuard() {
  --a10_dispatch_trace_nesting_value_;
}

int16_t A10TraceNestingGuard::nesting_level() {
  return a10_dispatch_trace_nesting_value_;
}

class ZlibFileWriter {
 public:
  ZlibFileWriter() : file_(nullptr), stop_thread_(false) {
    // Construct filename using PID
    pid_t pid = getpid();
    std::string filename = "trace_log_" + std::to_string(pid) + ".bin.gz";

    // Use highest compression level
    mode_str_ = "wb9";
    file_ = gzopen(filename.c_str(), mode_str_.c_str());
    if (!file_) {
      throw std::runtime_error(
          "Failed to open gz file for writing: " + filename);
    }

    // Start the background writer thread
    worker_ = std::thread(&ZlibFileWriter::backgroundWriter, this);
  }

  ~ZlibFileWriter() {
    {
      std::lock_guard<std::mutex> lock(queue_mtx_);
      stop_thread_ = true;
    }
    cv_.notify_one();
    if (worker_.joinable()) {
      worker_.join();
    }
    if (file_) {
      gzclose(file_);
    }
  }

  ZlibFileWriter(const ZlibFileWriter&) = delete;
  ZlibFileWriter& operator=(const ZlibFileWriter&) = delete;

  void write(const std::byte* data, size_t size) {
    std::lock_guard<std::mutex> lock(queue_mtx_);
    queue_.emplace(data, data + size);
    cv_.notify_one();
  }

 private:
  gzFile file_;
  std::string mode_str_;

  std::mutex queue_mtx_;
  std::condition_variable cv_;
  std::queue<std::vector<std::byte>> queue_;
  std::thread worker_;
  std::atomic<bool> stop_thread_;

  void backgroundWriter() {
    while (true) {
      std::vector<std::byte> data;
      {
        std::unique_lock<std::mutex> lock(queue_mtx_);
        cv_.wait(lock, [this] { return !queue_.empty() || stop_thread_; });

        if (stop_thread_ && queue_.empty()) {
          break;
        }

        data = std::move(queue_.front());
        queue_.pop();
      }

      if (gzwrite(file_, data.data(), static_cast<unsigned int>(data.size())) <=
          0) {
        throw std::runtime_error("Failed to write to gz file");
      }
    }
  }
};

static ZlibFileWriter traceLogWriter;

A10LoggingGuard::A10LoggingGuard(
    const A10LoggingTraceType call_type,
    const std::string& op_name,
    const DispatchKeySet& dispatchKeySet) {
  // Dispatch key ID (fixed enum mapping)
  this->dispatch_key = dispatchKeySet.highestPriorityTypeId();

  // Store the op_name via copy
  this->op_name = op_name;

  // Trace Type
  this->trace_type = call_type;

  // Autograd mode
  this->autograd_mode = static_cast<uint16_t>(at::GradMode::is_enabled());

  // Is In BackwardPass
  this->in_backward_pass =
      static_cast<uint16_t>(at::A10BackwardPassGuard::is_in_backward_pass());

  // Thread ID
  this->tid = syscall(SYS_gettid);

  // Timestamp
  auto now = std::chrono::system_clock::now().time_since_epoch();
  this->timestamp_start =
      std::chrono::duration_cast<std::chrono::microseconds>(now).count();

  // Unique Sequence Number
  this->nesting_value = A10TraceNestingGuard::nesting_level();

  if (this->nesting_value == 1) {
    log_sequence++;
  }
  this->sequence = log_sequence;

  
  // Cuda/HIP Push
#if CUDA_TIME_MEASUREMENT
  if (this->trace_type == A10LoggingTraceType::CALL ||
      this->dispatch_key == DispatchKey::CUDA ||
      this->dispatch_key == DispatchKey::HIP) {
    uint64_t rnd = dist(rng);
    this->random_suffix = rnd;
    std::string combined_id =
        std::to_string(this->tid) + "_" +
        std::to_string(this->sequence) + "_" +
        std::to_string(this->nesting_value) + "_" +
        std::to_string(rnd);
#ifdef USE_ROCM
    if (this->dispatch_key == DispatchKey::HIP) {
      ::roctxRangePushA(combined_id.c_str());
    } else
#endif
    {
      ::nvtxRangePushA(combined_id.c_str());
    }
  }
#endif
}

A10LoggingGuard::~A10LoggingGuard() {
  // End Timestamp
  auto now = std::chrono::system_clock::now().time_since_epoch();
  this->timestamp_end =
      std::chrono::duration_cast<std::chrono::microseconds>(now).count();

  // Cuda/HIP Pop
#if CUDA_TIME_MEASUREMENT
  if (this->trace_type == A10LoggingTraceType::CALL ||
      this->dispatch_key == DispatchKey::CUDA ||
      this->dispatch_key == DispatchKey::HIP) {
#ifdef USE_ROCM
    if (this->dispatch_key == DispatchKey::HIP) {
      ::roctxRangePop();
    } else
#endif
    {
      ::nvtxRangePop();
    }
  }
#endif

  // Write the log
  auto bin_log = this->serialize();
  traceLogWriter.write(reinterpret_cast<const std::byte*>(bin_log.data()), bin_log.size());
}

void A10LoggingGuard::_recordInputTensorShapeAndType(
    const uint64_t tensorID,
    const at::Tensor& value) {
  if (!value.defined()) {
    input_sizes.emplace_back(tensorID, std::vector<int64_t>());
    input_strides.emplace_back(tensorID, std::vector<int64_t>());
    input_types.emplace_back(tensorID, "undef");
  } else if (value.is_nested()) {
    input_sizes.emplace_back(tensorID, std::vector<int64_t>());
    input_strides.emplace_back(tensorID, std::vector<int64_t>());
    input_types.emplace_back(tensorID, "nested");
  } else {
    input_sizes.emplace_back(tensorID, value.sizes().vec());
    input_strides.emplace_back(tensorID, value.strides().vec());
    input_types.emplace_back(tensorID, value.device().str() + "_" + std::string(value.dtype().name()));
  }
}

void A10LoggingGuard::recordInputs(
    at::RecordFunction::schema_ref_t schema_ref,
    c10::ArrayRef<const c10::IValue> args) {
  this->schema_str = toString(schema_ref.get());

  for (const c10::IValue& input : args) {
    if (input.isTensor()) {
      const auto& tensor = input.toTensor();
      uint64_t tensorID =
          reinterpret_cast<uint64_t>(tensor.unsafeGetTensorImpl()->unique_id);
      _recordInputTensorShapeAndType(tensorID, tensor);
    } else if (input.isTensorList()) {
      const auto& tensor_list = input.toTensorList();
      for (size_t j = 0; j < tensor_list.size(); j++) {
        const auto& tensor = tensor_list[j];
        uint64_t tensorID =
            reinterpret_cast<uint64_t>(tensor.unsafeGetTensorImpl()->unique_id);
        _recordInputTensorShapeAndType(tensorID, tensor);
      }
    } else if (input.isOptionalTensorList()) {
      const auto& optional_tensor_list = input.toOptionalTensorList();
      for (size_t j = 0; j < optional_tensor_list.size(); j++) {
        const auto& tensor_option = optional_tensor_list[j];
        if (tensor_option.has_value()) {
          auto const& tensor = *tensor_option;
          uint64_t tensor_id =
              reinterpret_cast<uint64_t>(tensor.unsafeGetTensorImpl()->unique_id);
          _recordInputTensorShapeAndType(tensor_id, tensor);
        } else {
          // Trick: We assume it is undef, for 0, it is always undef, we just
          // keep our format consistent
          input_sizes.emplace_back(0, std::vector<int64_t>());
          input_strides.emplace_back(0, std::vector<int64_t>());
          input_types.emplace_back(0, "undef");
        }
      }
    }
  }
}

void A10LoggingGuard::_recordOutputTensorShapeAndType(
    const uint64_t tensorID,
    const at::Tensor& value) {
  if (!value.defined()) {
    output_sizes.emplace_back(tensorID, std::vector<int64_t>());
    output_strides.emplace_back(tensorID, std::vector<int64_t>());
    output_types.emplace_back(tensorID, "undef");
  } else if (value.is_nested()) {
    output_sizes.emplace_back(tensorID, std::vector<int64_t>());
    output_strides.emplace_back(tensorID, std::vector<int64_t>());
    output_types.emplace_back(tensorID, "nested");
  } else {
    output_sizes.emplace_back(tensorID, value.sizes().vec());
    output_strides.emplace_back(tensorID, value.strides().vec());
    output_types.emplace_back(tensorID, value.device().str() + "_" + std::string(value.dtype().name()));
  }
}

void A10LoggingGuard::recordOutputs(std::vector<c10::IValue>&& outputs) {
  for (const c10::IValue& output : outputs) {
    if (output.isTensor()) {
      const auto& tensor = output.toTensor();
      uint64_t tensorID =
          reinterpret_cast<uint64_t>(tensor.unsafeGetTensorImpl()->unique_id);
      _recordOutputTensorShapeAndType(tensorID, tensor);
    } else if (output.isTensorList()) {
      const auto& tensor_list = output.toTensorList();
      for (size_t j = 0; j < tensor_list.size(); j++) {
        const auto& tensor = tensor_list[j];
        uint64_t tensorID =
            reinterpret_cast<uint64_t>(tensor.unsafeGetTensorImpl()->unique_id);
        _recordOutputTensorShapeAndType(tensorID, tensor);
      }
    }
  }
}

std::vector<std::uint8_t> A10LoggingGuard::serialize() {
  std::vector<uint8_t> buffer;

  auto append_bytes = [&](const void* data, size_t size) {
    const uint8_t* ptr = static_cast<const uint8_t*>(data);
    buffer.insert(buffer.end(), ptr, ptr + size);
  };

  auto append_string = [&](const std::string& str) {
    size_t len = str.size();
    append_bytes(&len, sizeof(len));
    append_bytes(str.data(), len);
  };

  auto append_vector = [&](auto const& vec) {
    using ElemT = typename std::decay_t<decltype(vec)>::value_type;
    size_t len = vec.size();
    append_bytes(&len, sizeof(len));
    for (const ElemT& v : vec) {
      append_bytes(&v, sizeof(v));
    }
  };

  // Serialize enum as int32_t
  int32_t trace_typeInt = static_cast<int32_t>(trace_type);
  append_bytes(&trace_typeInt, sizeof(trace_typeInt));

  // Serialize op_name
  append_string(op_name);

  // Serialize other fields
  std::string dispatch_key_str = toString(dispatch_key);
  append_string(dispatch_key_str);
  append_bytes(&autograd_mode, sizeof(autograd_mode));
  append_bytes(&nesting_value, sizeof(nesting_value));
  append_bytes(&in_backward_pass, sizeof(in_backward_pass));

  // We did this to enforce int32_t type
  int32_t tid_serialized = static_cast<int32_t>(tid);
  append_bytes(&tid_serialized, sizeof(tid_serialized));

  append_bytes(&timestamp_start, sizeof(timestamp_start));
  append_bytes(&timestamp_end, sizeof(timestamp_end));
  append_bytes(&sequence, sizeof(sequence));
  
  append_bytes(&this->random_suffix, sizeof(this->random_suffix));

  append_string(schema_str);

  // Serialize input_sizes vector
  {
    size_t shape_count = input_sizes.size();
    append_bytes(&shape_count, sizeof(shape_count));
    for (auto const& pair : input_sizes) {
      uint64_t tensor_id = pair.first;
      const std::vector<int64_t>& shape = pair.second;
      append_bytes(&tensor_id, sizeof(tensor_id));
      append_vector(shape);
    }
  }

  // Serialize input_stride
  {
    size_t shape_count = input_strides.size();
    append_bytes(&shape_count, sizeof(shape_count));
    for (auto const& pair : input_strides) {
      uint64_t tensor_id = pair.first;
      const std::vector<int64_t>& shape = pair.second;
      append_bytes(&tensor_id, sizeof(tensor_id));
      append_vector(shape);
    }
  }

  // Serialize input_types
  {
    size_t type_count = input_types.size();
    append_bytes(&type_count, sizeof(type_count));
    for (auto const& pair : input_types) {
      uint64_t tensor_id = pair.first;
      const std::string& type_str = pair.second;
      append_bytes(&tensor_id, sizeof(tensor_id));
      append_string(type_str);
    }
  }

  // Serialize output_sizes vector
  {
    size_t shape_count = output_sizes.size();
    append_bytes(&shape_count, sizeof(shape_count));
    for (auto const& pair : output_sizes) {
      uint64_t tensor_id = pair.first;
      const std::vector<int64_t>& shape = pair.second;
      append_bytes(&tensor_id, sizeof(tensor_id));
      append_vector(shape);
    }
  }

  // Serialize output_stride
  {
    size_t shape_count = output_strides.size();
    append_bytes(&shape_count, sizeof(shape_count));
    for (auto const& pair : output_strides) {
      uint64_t tensor_id = pair.first;
      const std::vector<int64_t>& shape = pair.second;
      append_bytes(&tensor_id, sizeof(tensor_id));
      append_vector(shape);
    }
  }

  // Serialize output_types
  {
    size_t type_count = output_types.size();
    append_bytes(&type_count, sizeof(type_count));
    for (auto const& pair : output_types) {
      uint64_t tensor_id = pair.first;
      const std::string& type_str = pair.second;
      append_bytes(&tensor_id, sizeof(tensor_id));
      append_string(type_str);
    }
  }

  return buffer;
}

void dispatch_trace_nesting_incr() {
  ++dispatch_trace_nesting_value_;
}
void dispatch_trace_nesting_decr() {
  --dispatch_trace_nesting_value_;
}
int64_t dispatch_trace_nesting_value() {
  return dispatch_trace_nesting_value_;
}

namespace detail {

class RegistrationListenerList final {
 public:
  std::function<void()> addListener(
      std::unique_ptr<OpRegistrationListener> listener) {
    listeners_.push_back(std::move(listener));
    auto delete_it = --listeners_.end();
    return [this, delete_it] { listeners_.erase(delete_it); };
  }

  void callOnOperatorRegistered(const OperatorHandle& op) {
    for (auto& listener : listeners_) {
      listener->onOperatorRegistered(op);
    }
  }

  void callOnOperatorDeregistered(const OperatorHandle& op) {
    for (auto& listener : listeners_) {
      listener->onOperatorDeregistered(op);
    }
  }

 private:
  std::list<std::unique_ptr<OpRegistrationListener>> listeners_;
};

void _print_dispatch_trace(
    const std::string& label,
    const std::string& op_name,
    const DispatchKeySet& dispatchKeySet) {
  auto nesting_value = dispatch_trace_nesting_value();
  for (int64_t i = 0; i < nesting_value; ++i)
    std::cerr << " ";
  std::cerr << label << " op=[" << op_name << "], key=["
            << toString(dispatchKeySet.highestPriorityTypeId()) << "]"
            << std::endl;
}
} // namespace detail

OpRegistrationListener::~OpRegistrationListener() = default;

Dispatcher::Dispatcher()
    : operators_(),
      operatorLookupTable_(),
      backendFallbackKernels_(),
      listeners_(std::make_unique<detail::RegistrationListenerList>()),
      cond_var_(),
      guard_(std::make_shared<Guard>()) {}

Dispatcher::~Dispatcher() {
  std::lock_guard<std::mutex> lock(guard_->mutex);
  guard_->alive.store(false);
}

C10_EXPORT Dispatcher& Dispatcher::realSingleton() {
  static Dispatcher _singleton;
  return _singleton;
}

std::optional<OperatorHandle> Dispatcher::findOp(
    const OperatorName& overload_name) {
  return operatorLookupTable_.read(
      [&](const ska::flat_hash_map<OperatorName, OperatorHandle>&
              operatorLookupTable) -> std::optional<OperatorHandle> {
        auto found = operatorLookupTable.find(overload_name);
        if (found == operatorLookupTable.end()) {
          return std::nullopt;
        }
        return found->second;
      });
}

// NB: If you add more waitFor* implementations, you also have to add
// appropriate notify_all() calls to the relevant register calls

void Dispatcher::waitForDef(const FunctionSchema& schema) {
  using namespace std::chrono_literals;
  std::unique_lock<std::mutex> lock(guard_->mutex);
  bool r = cond_var_.wait_for(
      lock, 2s, [&] { return findOp(schema.operator_name()).has_value(); });
  TORCH_INTERNAL_ASSERT(
      r,
      "Expected main interpreter to define ",
      schema.operator_name(),
      ", but this didn't happen within timeout.  Are you trying to load "
      "different models in the same torchdeploy/multipy instance?  You "
      "must warmup each interpreter identically, e.g., import all "
      "the same dependencies.");
}

void Dispatcher::waitForImpl(
    const OperatorName& op_name,
    std::optional<c10::DispatchKey> maybe_dk) {
  using namespace std::chrono_literals;
  std::unique_lock<std::mutex> lock(guard_->mutex);
  auto dk = maybe_dk.value_or(DispatchKey::CompositeImplicitAutograd);
  auto op = findOrRegisterName_(op_name);
  bool r = cond_var_.wait_for(lock, 2s, [&] {
    // NB: this is slightly unsound for overrides, but overrides are
    // funny business anyway
    return op.hasKernelForDispatchKey(dk);
  });
  TORCH_INTERNAL_ASSERT(
      r,
      "Expected main interpreter to implement ",
      dk,
      " for ",
      op_name,
      ", but this didn't happen within timeout.  Are you trying to load "
      "different models in the same torchdeploy/multipy instance?  You "
      "must warmup each interpreter identically, e.g., import all "
      "the same dependencies.");
}

std::optional<OperatorHandle> Dispatcher::findSchema(
    const OperatorName& overload_name) {
  auto it = findOp(overload_name);
  if (it.has_value()) {
    if (it->hasSchema()) {
      return it;
    } else {
      return std::nullopt;
    }
  } else {
    return it;
  }
}

OperatorHandle Dispatcher::findSchemaOrThrow(
    const char* name,
    const char* overload_name) {
  auto it = findSchema({name, overload_name});
  if (!it.has_value()) {
    // Check if we have ANYTHING; if that's the case, that means you're
    // missing schema
    auto it2 = findOp({name, overload_name});
    if (!it2.has_value()) {
      TORCH_CHECK(
          false, "Could not find schema for ", name, ".", overload_name);
    } else {
      TORCH_CHECK(
          false,
          "Could not find schema for ",
          name,
          ".",
          overload_name,
          " but we found an implementation; did you forget to def() the operator?");
    }
  }
  return it.value();
}

const std::vector<OperatorName> Dispatcher::getAllOpNames() {
  return operatorLookupTable_.read(
      [&](const ska::flat_hash_map<OperatorName, OperatorHandle>&
              operatorLookupTable) -> std::vector<OperatorName> {
        std::vector<OperatorName> allOpNames;
        for (const auto& op : operatorLookupTable) {
          allOpNames.push_back(op.first);
        }
        return allOpNames;
      });
}

// Postcondition: caller is responsible for disposing of registration when they
// are done
OperatorHandle Dispatcher::findOrRegisterName_(const OperatorName& op_name) {
  const auto found = findOp(op_name);
  if (found.has_value()) {
    return *found;
  }

  operators_.emplace_back(OperatorName(op_name));
  OperatorHandle handle(--operators_.end());
  operatorLookupTable_.write(
      [&](ska::flat_hash_map<OperatorName, OperatorHandle>&
              operatorLookupTable) {
        operatorLookupTable.emplace(op_name, handle);
      });

  return handle;
}

// Adding explicit destructor definition in the cpp to over linker error in
// Windows builds. Windows build doesn't produce the destructor symbol in
// PyTorch libs causing a linker failure in downstream projects. x-ref
// https://github.com/pytorch/pytorch/issues/70032
OperatorHandle::~OperatorHandle() = default;

RegistrationHandleRAII Dispatcher::registerLibrary(
    std::string ns,
    std::string debug) {
  std::lock_guard<std::mutex> lock(guard_->mutex);
  auto found = libraries_.find(ns);
  TORCH_CHECK(
      found == libraries_.end(),
      "Only a single TORCH_LIBRARY can be used to register the namespace ",
      ns,
      "; please put all of your definitions in a single TORCH_LIBRARY block.  "
      "If you were trying to specify implementations, consider using TORCH_LIBRARY_IMPL "
      "(which can be duplicated).  If you really intended to define operators for a "
      "single namespace in a distributed way, you can use TORCH_LIBRARY_FRAGMENT to "
      "explicitly indicate this.  "
      "Previous registration of TORCH_LIBRARY was ",
      found->second,
      "; latest registration was ",
      debug);
  libraries_.emplace(ns, std::move(debug));
  return RegistrationHandleRAII([guard = this->guard_, this, ns] {
    std::lock_guard<std::mutex> lock(guard->mutex);
    if (!guard->alive.load()) {
      return;
    }
    deregisterLibrary_(ns);
  });
}

void Dispatcher::deregisterLibrary_(const std::string& ns) {
  // we need a lock to avoid concurrent writes
  libraries_.erase(ns);
}

RegistrationHandleRAII Dispatcher::registerDef(
    FunctionSchema schema,
    std::string debug,
    std::vector<at::Tag> tags) {
  // we need a lock to avoid concurrent writes
  std::lock_guard<std::mutex> lock(guard_->mutex);

  OperatorName op_name = schema.operator_name();
  auto op = findOrRegisterName_(op_name);

  TORCH_CHECK(
      op.operatorDef_->def_count == 0,
      "Tried to register an operator (",
      schema,
      ") with the same name and overload name multiple times.",
      " Each overload's schema should only be registered with a single call to def().",
      " Duplicate registration: ",
      debug,
      ". Original registration: ",
      op.operatorDef_->op.debug());
  op.operatorDef_->op.registerSchema(
      std::move(schema), std::move(debug), std::move(tags));
  listeners_->callOnOperatorRegistered(op);

  // NB: do not increment the counts until AFTER error checking
  ++op.operatorDef_->def_count;
  ++op.operatorDef_->def_and_impl_count;

  cond_var_.notify_all();

  return RegistrationHandleRAII([guard = this->guard_, this, op, op_name] {
    // we need a lock to avoid concurrent writes
    std::lock_guard<std::mutex> lock(guard->mutex);
    if (!guard->alive.load()) {
      return;
    }
    deregisterDef_(op, op_name);
  });
}

void Dispatcher::deregisterDef_(
    const OperatorHandle& op,
    const OperatorName& op_name) {
  TORCH_INTERNAL_ASSERT(op.schema().operator_name() == op_name);

  // reduce def_count and actually deregister if no references left
  TORCH_INTERNAL_ASSERT(op.operatorDef_->def_count > 0);
  TORCH_INTERNAL_ASSERT(op.operatorDef_->def_and_impl_count > 0);

  --op.operatorDef_->def_count;
  --op.operatorDef_->def_and_impl_count;
  if (0 == op.operatorDef_->def_count) {
    // note: call listeners *before* operator is removed, i.e. dispatcher is
    // still valid for removed op
    // TODO: check that listeners are not relying on prepareForDeregistration()
    // invariant
    listeners_->callOnOperatorDeregistered(op);
    op.operatorDef_->op.deregisterSchema();
  }

  cleanup(op, op_name);
}

namespace {

// Maps OperatorName to (python module name, description) tuple.
using PythonModuleMapType =
    std::unordered_map<at::OperatorName, std::pair<const char*, const char*>>;
PythonModuleMapType& pythonModulesSingleton() {
  static PythonModuleMapType _data;
  return _data;
}

} // namespace

std::optional<std::pair<const char*, const char*>> Dispatcher::getPyStub(
    OperatorName op_name) {
  std::lock_guard<std::mutex> lock(guard_->mutex);
  auto found = pythonModulesSingleton().find(op_name);
  if (found == pythonModulesSingleton().end()) {
    return std::nullopt;
  }
  return found->second;
}

RegistrationHandleRAII Dispatcher::registerPythonModule(
    const OperatorName& op_name,
    const char* pymodule,
    const char* context) {
  std::lock_guard<std::mutex> lock(guard_->mutex);
  // If there are duplicates, we just let it through and warn about it.
  // Throwing an error during static initialization causes a crash that
  // doesn't give any sign of what happened.
  auto found = pythonModulesSingleton().find(op_name);
  if (found != pythonModulesSingleton().end()) {
    TORCH_WARN(
        "Tried to register an python registration stub (pystub) for ",
        op_name,
        " ",
        "that specifies the Python module ",
        pymodule,
        " "
        "but there already was a pystub that specifies the Python module ",
        found->second.first,
        ". We will override the existing pystub.");
  }
  pythonModulesSingleton()[op_name] = std::make_pair(pymodule, context);
  return RegistrationHandleRAII([guard = this->guard_, op_name] {
    std::lock_guard<std::mutex> lock(guard->mutex);
    if (!guard->alive.load()) {
      return;
    }
    pythonModulesSingleton().erase(op_name);
  });
}

void Dispatcher::throwIfHasPythonModule(OperatorName op_name) {
  std::lock_guard<std::mutex> lock(guard_->mutex);
  auto elt = pythonModulesSingleton().find(op_name);
  if (elt == pythonModulesSingleton().end()) {
    return;
  }
  const char* pymodule = elt->second.first;
  const char* context = elt->second.second;
  auto* interpreter =
      at::impl::PythonOpRegistrationTrampoline::getInterpreter();
  TORCH_CHECK(
      interpreter != nullptr,
      op_name,
      ": while attempting to run this operator with Meta Tensors: "
      "Either there is no meta kernel for this operator, or it is located "
      "in the python module ",
      pymodule,
      " which is not available "
      "because Python isn't available.")
  (*interpreter)
      ->throw_abstract_impl_not_imported_error(
          toString(op_name), pymodule, context);
}

RegistrationHandleRAII Dispatcher::registerImpl(
    OperatorName op_name,
    std::optional<DispatchKey> dispatch_key,
    KernelFunction kernel,
    std::optional<impl::CppSignature> cpp_signature,
    std::unique_ptr<FunctionSchema> inferred_function_schema,
    std::string debug) {
  std::lock_guard<std::mutex> lock(guard_->mutex);

  auto op = findOrRegisterName_(op_name);

  auto handle = op.operatorDef_->op.registerKernel(
      *this,
      dispatch_key,
      std::move(kernel),
      std::move(cpp_signature),
      std::move(inferred_function_schema),
      std::move(debug));

  ++op.operatorDef_->def_and_impl_count;

  cond_var_.notify_all();

  return RegistrationHandleRAII(
      [guard = this->guard_, this, op, op_name, dispatch_key, handle] {
        std::lock_guard<std::mutex> lock(guard->mutex);
        if (!guard->alive.load()) {
          return;
        }
        deregisterImpl_(op, op_name, dispatch_key, handle);
      });
}

void Dispatcher::deregisterImpl_(
    const OperatorHandle& op,
    const OperatorName& op_name,
    std::optional<DispatchKey> dispatch_key,
    impl::OperatorEntry::AnnotatedKernelContainerIterator handle) {
  op.operatorDef_->op.deregisterKernel_(*this, dispatch_key, handle);

  TORCH_INTERNAL_ASSERT(op.operator_name() == op_name);

  TORCH_INTERNAL_ASSERT(op.operatorDef_->def_and_impl_count > 0);
  --op.operatorDef_->def_and_impl_count;

  cleanup(op, op_name);
}

RegistrationHandleRAII Dispatcher::registerName(OperatorName op_name) {
  std::lock_guard<std::mutex> lock(guard_->mutex);
  auto op = findOrRegisterName_(op_name);
  ++op.operatorDef_->def_and_impl_count;

  return RegistrationHandleRAII([guard = this->guard_, this, op, op_name] {
    std::lock_guard<std::mutex> lock(guard->mutex);
    if (!guard->alive.load()) {
      return;
    }
    deregisterName_(op, op_name);
  });
}

void Dispatcher::deregisterName_(
    const OperatorHandle& op,
    const OperatorName& op_name) {
  TORCH_INTERNAL_ASSERT(op.operator_name() == op_name);
  TORCH_INTERNAL_ASSERT(op.operatorDef_->def_and_impl_count > 0);
  --op.operatorDef_->def_and_impl_count;
  cleanup(op, op_name);
}

// Test if the operator entry is completely dead, and if so remove it completely
void Dispatcher::cleanup(
    const OperatorHandle& op,
    const OperatorName& op_name) {
  if (0 == op.operatorDef_->def_and_impl_count) {
    // NOTE: Making this call fast is the only reason OperatorHandle
    // stores operatorIterator_!
    operators_.erase(op.operatorIterator_);
    operatorLookupTable_.write(
        [&](ska::flat_hash_map<OperatorName, OperatorHandle>&
                operatorLookupTable) { operatorLookupTable.erase(op_name); });
  }
}

RegistrationHandleRAII Dispatcher::registerFallback(
    DispatchKey dispatchKey,
    KernelFunction kernel,
    std::string debug) {
  std::lock_guard<std::mutex> lock(guard_->mutex);

  auto idx = getDispatchTableIndexForDispatchKey(dispatchKey);
  TORCH_CHECK(
      idx >= 0 && static_cast<uint64_t>(idx) < backendFallbackKernels_.size(),
      "idx=",
      idx);
  TORCH_CHECK(
      !backendFallbackKernels_[idx].kernel.isValid(),
      "Tried to register multiple backend fallbacks for the same dispatch key ",
      dispatchKey,
      "; previous registration ",
      backendFallbackKernels_[idx].debug,
      ", new registration ",
      debug);
  // NB: inferred function schema is always nullptr for fallbacks, as fallbacks
  // cannot be unboxed
  backendFallbackKernels_[idx] =
      impl::AnnotatedKernel(std::move(kernel), nullptr, std::move(debug));

  for (auto& op : operators_) {
    op.op.updateFallback(*this, dispatchKey);
  }

  return RegistrationHandleRAII([guard = this->guard_, this, dispatchKey] {
    std::lock_guard<std::mutex> lock(guard->mutex);
    if (!guard->alive.load()) {
      return;
    }
    deregisterFallback_(dispatchKey);
  });
}

void Dispatcher::deregisterFallback_(DispatchKey dispatchKey) {
  auto idx = getDispatchTableIndexForDispatchKey(dispatchKey);
  backendFallbackKernels_[idx] = {};

  for (auto& op : operators_) {
    op.op.updateFallback(*this, dispatchKey);
  }
}

RegistrationHandleRAII Dispatcher::addRegistrationListener(
    std::unique_ptr<OpRegistrationListener> listener) {
  std::lock_guard<std::mutex> lock(guard_->mutex);

  for (auto iter = operators_.begin(); iter != operators_.end(); ++iter) {
    if (iter->def_count > 0) {
      listener->onOperatorRegistered(OperatorHandle(iter));
    }
  }

  auto removeListener = listeners_->addListener(std::move(listener));
  return RegistrationHandleRAII([guard = this->guard_, this, removeListener] {
    std::lock_guard<std::mutex> lock(guard_->mutex);
    if (!guard->alive.load()) {
      return;
    }
    removeListener();
  });
}

void Dispatcher::checkInvariants() const {
  for (const auto& op : operators_) {
    op.op.checkInvariants();
  }
}

std::vector<OperatorHandle> Dispatcher::findDanglingImpls() const {
  return operatorLookupTable_.read(
      [&](const ska::flat_hash_map<OperatorName, OperatorHandle>&
              operatorLookupTable) -> std::vector<OperatorHandle> {
        std::vector<OperatorHandle> opsWithDanglingImpls;
        for (const auto& op : operatorLookupTable) {
          if (!op.second.hasSchema()) {
            opsWithDanglingImpls.push_back(op.second);
          }
        }
        return opsWithDanglingImpls;
      });
}

std::vector<OperatorName> Dispatcher::getRegistrationsForDispatchKey(
    std::optional<DispatchKey> k) const {
  return operatorLookupTable_.read(
      [&](const ska::flat_hash_map<OperatorName, OperatorHandle>&
              operatorLookupTable) -> std::vector<OperatorName> {
        std::vector<OperatorName> op_names;
        for (const auto& op : operatorLookupTable) {
          // If no DispatchKey is specified, print all of the operators.
          if (!k || op.second.hasKernelForDispatchKey(*k)) {
            op_names.push_back(op.first);
          }
        }
        return op_names;
      });
}

int64_t Dispatcher::sequenceNumberForRunningRecordFunction(
    DispatchKey dispatchKey,
    DispatchKeySet dispatchKeySet) {
  int64_t seq_num = -1;
  // Setting sequence number in the Autograd case to associate
  // the forward range with the corresponding Autograd's node

  // Note: this records a sequence number for both Autograd keys, and for
  // non-Autograd keys where the dispatchKeySet still contains an autograd key.
  // This means that we might collect the same sequence nubmer two different
  // events if they all occurred above Autograd and still had the Autograd
  // dispatch key in the dispatch key set.
  // However, this usually doesn't happen: normally the first call will
  // go through the call() or callBoxed() path in the dispatcher, while
  // subsequent redispatches go through redispatch() or redispatchBoxed().
  // `call` has profiler instrumentation, whereas `redispatch` doesn't.
  // So usually, we'll collect a sequence number on the first call() if the
  // dispatch keys contain autograd, and not on subsequent redispatches.
  bool dispatchHasAutograd =
      !(dispatchKeySet & autograd_dispatch_keyset).empty();

  if (dispatchHasAutograd && at::GradMode::is_enabled()) {
    seq_num = at::sequence_number::peek();
  }
  return seq_num;
}

void Dispatcher::runRecordFunction(
    at::RecordFunction& guard,
    at::RecordFunction::schema_ref_t schema_ref,
    DispatchKey dispatchKey,
    DispatchKeySet dispatchKeySet,
    c10::ArrayRef<const c10::IValue> args) {
  guard.before(
      schema_ref,
      args,
      sequenceNumberForRunningRecordFunction(dispatchKey, dispatchKeySet));
}

void Dispatcher::runRecordFunction(
    at::RecordFunction& guard,
    at::RecordFunction::schema_ref_t schema_ref,
    DispatchKey dispatchKey,
    DispatchKeySet dispatchKeySet) {
  // Setting sequence number in the Autograd case to associate
  // the forward range with the corresponding Autograd's node
  guard.before(
      schema_ref,
      sequenceNumberForRunningRecordFunction(dispatchKey, dispatchKeySet));
}
#ifdef FBCODE_CAFFE2
bool Dispatcher::profilingOperatorEvents() {
  return TORCH_SDT_IS_ENABLED(operator_start) ||
      TORCH_SDT_IS_ENABLED(operator_end);
}

C10_NOINLINE void Dispatcher::fireOpStartUSDT(
    at::RecordFunction::schema_ref_t schema_ref) {
  if (TORCH_SDT_IS_ENABLED(operator_start)) {
    TORCH_SDT_WITH_SEMAPHORE(operator_start, schema_ref.get().name().c_str());
  }
}

C10_NOINLINE void Dispatcher::fireOpEndUSDT(
    at::RecordFunction::schema_ref_t schema_ref) {
  if (TORCH_SDT_IS_ENABLED(operator_end)) {
    TORCH_SDT_WITH_SEMAPHORE(operator_end, schema_ref.get().name().c_str());
  }
}
#endif

} // namespace c10
