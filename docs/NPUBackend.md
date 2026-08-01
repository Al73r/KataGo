# Design Doc: Hexagon NPU Backend (Snapdragon X Elite / X1E80100) via ONNX Runtime + QNN

Status: Draft / proposal
Scope: Add a new KataGo neural-net backend that runs inference on the Qualcomm Hexagon
NPU (HTP — Hexagon Tensor Processor) of a Snapdragon X Elite (X1E80100) machine running
Windows on ARM64, using **ONNX Runtime** with the **QNN Execution Provider**.

This document is an engineering design. No backend code is committed as part of it; the
CMake snippets and function signatures below are illustrative but are grounded in the
current source tree so that implementation can follow directly.

---

## 1. Overview, goals and non-goals

### 1.1 Motivation

The Snapdragon X Elite (X1E80100) integrates a Qualcomm Hexagon NPU capable of tens of
TOPS at low power. KataGo today has no backend that targets this device: on such a
machine users fall back to the OpenCL backend (Adreno GPU) or the Eigen CPU backend, both
of which leave the NPU idle. Adding NPU support gives an additional — and potentially more
power-efficient — inference target for Windows-on-ARM laptops.

KataGo already contains the two ingredients that make this tractable:

- `cpp/neuralnet/onnxmodelbuilder.cpp` emits a **self-contained ONNX ModelProto** for any
  KataGo model (weights baked in as initializers). It is currently consumed only by the
  TensorRT backend, but the graph is backend-agnostic.
- The Metal backend (`cpp/neuralnet/metalbackend.*`, `cpp/external/katagocoreml`) is a
  working precedent for a "compile the whole model to an NPU accelerator" backend — it
  converts KataGo models to CoreML and runs them on Apple's Neural Engine (ANE), with a
  GPU/NPU multiplexer keyed off `gpuIdx`.

The ONNX Runtime QNN Execution Provider (QNN EP) loads a standard ONNX model and offloads
supported subgraphs to the Hexagon HTP. Reusing the existing ONNX emitter, a QNN backend
is largely a matter of "emit ONNX → run it in an ORT session configured for QNN HTP →
decode outputs with the code we already have".

### 1.2 Goals

- New build target `-DUSE_BACKEND=QNN` producing a `katago` that runs NN inference on the
  Hexagon HTP via ORT + QNN EP on Windows ARM64.
- Reuse `onnxmodelbuilder` and KataGo's existing input construction and output decoding
  (`getOutput`) with no changes to search, GTP, benchmark, or model format.
- **FP16 on HTP** as the initial precision (correct results within an acceptable tolerance
  vs. the CPU/Eigen reference).
- Persist a compiled **QNN HTP context binary** to disk so startup does not pay full
  compilation cost every run (analogous to the existing TensorRT plan cache).
- Pass `runnnevalcanarytests` (NN numeric correctness vs an Eigen reference) and produce sane
  `katago benchmark` numbers.

### 1.3 Non-goals (initial version)

- **INT8/INT16 quantization** for maximum HTP throughput. This is the biggest future
  performance lever and is designed for but deferred to a later phase (see §6.3 and §13).
- Multi-model / mixed GPU+NPU multiplexing beyond the minimum needed for device selection.
  (We keep the door open by following the Metal `gpuIdx` convention, but do not build a
  full multiplexer initially.)
- Non-Windows QNN targets (Android, Linux-on-Snapdragon). The code should not be gratuitously
  Windows-only, but only Windows ARM64 is validated here.
- Changing the KataGo model/training format or the `.bin.gz` on-disk representation.

---

## 2. Target hardware and software stack

| Layer | Choice | Notes |
|-------|--------|-------|
| SoC | Snapdragon X Elite **X1E80100** | Hexagon NPU (HTP), Adreno GPU, 12x Oryon CPU |
| Accelerator | Hexagon **HTP** | Fixed-point + FP16 tensor engine; the QNN "htp" backend |
| OS | Windows 11 on ARM64 (aarch64) | Native ARM64 build required for best perf |
| Inference runtime | **ONNX Runtime** (ARM64), QNN EP enabled | `onnxruntime.dll` + `onnxruntime_providers_qnn.dll` |
| Vendor runtime | **Qualcomm AI Engine Direct (QNN) SDK** | provides `QnnHtp.dll`, `QnnHtpV73Stub.dll`, `libqnnhtpv73.cat`, skel, etc. |
| Compiler | MSVC ARM64 (VS 2022) or clang-cl ARM64 | Must match ORT/QNN ABI |

Notes:

- The HTP is a **fixed-shape** engine. Like TensorRT, QNN EP wants fully static input
  shapes; dynamic batch is not natively supported on HTP (see §7).
- The **QNN SDK version** and the ORT version are coupled (ORT is built against a specific
  QNN API level). We pin both and record them in `Compiling.md`.
- HTP FP16 support depends on the QNN version and the specific HTP architecture version
  (V73 on X1E80100). The design assumes a QNN/ORT combination new enough to support
  `enable_htp_fp16_precision`.

---

## 3. Background: KataGo NN backend architecture

Understanding the existing structure is essential because the QNN backend slots into it
exactly like the others.

### 3.1 The backend contract

Every backend implements the `NeuralNet::` interface declared in
`cpp/neuralnet/nninterface.h`. The key entry points (with their real signatures) are:

```cpp
namespace NeuralNet {
  void globalInitialize();
  void globalCleanup();
  void printDevices();

  LoadedModel* loadModelFile(const std::string& file, const std::string& expectedSha256);
  void freeLoadedModel(LoadedModel* loadedModel);
  const ModelDesc& getModelDesc(const LoadedModel* loadedModel);

  ComputeContext* createComputeContext(
    const std::vector<int>& gpuIdxs, Logger* logger, int nnXLen, int nnYLen,
    const std::string& homeDataDirOverride, enabled_t useFP16Mode,
    const LoadedModel* loadedModel, ConfigParser& cfg);
  void freeComputeContext(ComputeContext* computeContext);

  ComputeHandle* createComputeHandle(
    ComputeContext* context, const LoadedModel* loadedModel, Logger* logger,
    int maxBatchSize, bool requireExactNNLen, bool inputsUseNHWC,
    int gpuIdxForThisThread, int serverThreadIdx);
  void freeComputeHandle(ComputeHandle* computeHandle);
  bool isUsingFP16(const ComputeHandle* computeHandle);
  bool setIsWarmup(const ComputeHandle* computeHandle, bool isWarmup);

  InputBuffers* createInputBuffers(const LoadedModel* loadedModel, int maxBatchSize, int nnXLen, int nnYLen);
  void freeInputBuffers(InputBuffers* buffers);

  void getOutput(ComputeHandle* handle, InputBuffers* buffers, int numBatchEltsFilled,
                 NNResultBuf** inputBufs, std::vector<NNOutput*>& outputs);

  // Optional single-op test hooks (may return false if unimplemented):
  bool testEvaluateConv(...); bool testEvaluateBatchNorm(...);
  bool testEvaluateResidualBlock(...); bool testEvaluateGlobalPoolingResidualBlock(...);
}
```

`ComputeContext`, `ComputeHandle`, `InputBuffers`, `LoadedModel` are **opaque structs each
backend defines privately**. There is exactly one backend compiled into a given `katago`
binary, so there is no vtable/registry — the linker picks the one `.cpp` that
`NEURALNET_BACKEND_SOURCES` names.

`LoadedModel` construction is identical across backends and is what we reuse verbatim:

```cpp
struct LoadedModel {
  ModelDesc modelDesc;
  LoadedModel(const string& fileName, const string& expectedSha256) {
    ModelDesc::loadFromFileMaybeGZipped(fileName, modelDesc, expectedSha256);
    modelDesc.applyScale8ToReduceActivations();
  }
};
```

### 3.2 Backend selection in CMake

`cpp/CMakeLists.txt` exposes `USE_BACKEND` (`CUDA`/`TENSORRT`/`OPENCL`/`EIGEN`/`METAL`, or
empty = dummy). It:

1. sets `NEURALNET_BACKEND_SOURCES` to the backend's `.cpp`(s) (the first `if/elseif`
   ladder, ~lines 60–171), and
2. later (a second `if/elseif` ladder, ~lines 373–535) adds a `USE_<BACKEND>_BACKEND`
   compile definition and links the backend's libraries.

The ONNX emitter today lives **inside the TENSORRT arm** of the second ladder
(`protobuf_generate_cpp` on `cpp/external/onnx/onnx.proto`, compiling
`neuralnet/onnxmodelbuilder.cpp` and linking our own `protobuf::libprotobuf`, ~lines
452–478). Decoupling this is the main CMake refactor (see §5.1).

### 3.3 The ONNX emitter and the input/output contract

`OnnxModelBuilder::build` (declared in `cpp/neuralnet/onnxmodelbuilder.h`) has signature:

```cpp
Result build(const ModelDesc& desc, int nnXLen, int nnYLen,
             bool requireExactNNLen, bool transformerNHWC, Logger* logger);

struct Result {
  std::string serializedModel;                        // serialized ONNX ModelProto (FP32)
  std::vector<std::string> trunkTipAndHeadNodeNames;  // regions to force to FP32
  std::vector<std::string> rmsNormNodeNames;          // every RMSNorm op
};
```

The emitted graph is **FP32, NCHW**, weights baked as initializers, with this fixed I/O
contract (the same tensor names the TensorRT backend binds). **All tensors are rank-4**, and
the **batch dimension is a symbolic `dim_param("batch")` — i.e. dynamic** (see
`onnxmodelbuilder.cpp` `addInput`/`addInputNC11`/`markOutput`). This dynamic batch is a key
constraint for the HTP (see §5.2 and §7).

| Tensor | Role | Shape (N = symbolic `batch`) |
|--------|------|-------------------|
| `InputMask` | board validity mask | `[N, 1, nnYLen, nnXLen]` |
| `InputSpatial` | spatial features | `[N, numInputChannels, nnYLen, nnXLen]` |
| `InputGlobal` | global features | `[N, numInputGlobalChannels, 1, 1]` |
| `OutputPolicyPass` | pass-move logits | `[N, numPolicyChannels, 1, 1]` |
| `OutputPolicy` | spatial policy logits | `[N, numPolicyChannels, nnYLen, nnXLen]` |
| `OutputValue` | value head logits | `[N, numValueChannels, 1, 1]` |
| `OutputScoreValue` | score-value head logits | `[N, numScoreValueChannels, 1, 1]` |
| `OutputOwnership` | ownership map logits | `[N, numOwnershipChannels, nnYLen, nnXLen]` |

Two important corrections about what the emitter does **not** do:

- **`InputMeta` is not emitted.** `OnnxModelBuilder::build` throws
  `"OnnxModelBuilder: SGF metadata encoder not yet supported"` when `desc.metaEncoderVersion
  > 0` (`onnxmodelbuilder.cpp:817`). So SGF-metadata models are **out of scope** for the QNN
  backend until the emitter learns to emit `InputMeta` (they already fail on TensorRT via the
  ONNX path too). Phase 1 targets non-metadata models only.
- **No activation functions are in the graph, and `getOutput` does not apply them either.**
  The graph outputs **raw logits**; `getOutput` (in every backend) only reorders spatial
  outputs by symmetry and copies logits into `NNOutput` fields — it explicitly notes "these
  are in logits, the client does the postprocessing" (`trtbackend.cpp` getOutput). Softmax /
  tanh / value transforms are applied **later**, when the search consumes `NNOutput`
  (`nneval.cpp`), per the `nninterface.h` contract ("All outputs are in logits … NOT
  applied"). So the QNN backend reuses the **logit-copy + symmetry** decode, not any softmax.

### 3.4 The TensorRT backend as the structural template

`cpp/neuralnet/trtbackend.cpp` is the closest analog to what we build. Its shape:

- `createComputeContext` stashes `nnXLen/nnYLen/useFP16Mode/homeDataDirOverride` and reads
  a few custom keys off `cfg` (e.g. `trtDisableOnnx`, `trtTransformerNHWC`).
- `createComputeHandle` (per server thread) emits ONNX via `OnnxModelBuilder::build`, builds
  an engine, and (when `USE_CACHE_TENSORRT_PLAN` is enabled) **caches the serialized engine
  ("plan") in the home data dir**, keyed on model SHA-256 + a parameter string; otherwise it
  caches only a timing cache. Either way subsequent runs are cheaper.
- `InputBuffers` holds host-side `float[]` staging arrays sized `maxBatchSize * singleXxxElts`.
- `getOutput` copies each row's features (with symmetry) into the staging arrays, runs the
  engine, copies **logit** results back, and reorders spatial outputs by symmetry into
  `NNOutput` (no activation — see §3.3).

The QNN backend mirrors this one-to-one, swapping "TensorRT engine + plan cache" for "ORT
session + QNN EPContext cache" (§5.5). Note the plan-cache analogy is structural, not literal:
QNN's EPContext cache is a generated ONNX wrapper model, not an opaque byte blob (§5.5).

### 3.5 The Metal/CoreML backend as the NPU precedent

The Metal backend proves KataGo can drive an NPU by compiling the whole model to a vendor
graph. Two ideas we borrow:

- **`gpuIdx` as a device/mode selector.** Metal defines `METAL_MUX_GPU = 0` and
  `METAL_MUX_ANE = 100`; a `gpuIdx` of 100 routes a server thread to the ANE.
- **Per-thread handles** each own their own compiled graph.

We adopt the same convention so a future multiplexer (HTP vs. CPU-EP vs. GPU) is a natural
extension rather than a rewrite.

---

## 4. High-level design

```
                KataGo model (.bin.gz)
                        │  ModelDesc::loadFromFileMaybeGZipped   (shared)
                        ▼
                   ModelDesc  ──────────────────────────────────┐
                        │                                        │
   OnnxModelBuilder::build(desc, nnXLen, nnYLen, exact, …)       │ (reused, unchanged)
                        │  serialized ONNX ModelProto (FP32)     │
                        ▼                                        │
        ┌───────────────────────────────────────────────┐       │
        │  Ort::Session(onnx_bytes, session_options)     │       │
        │    provider: QNN EP  →  QnnHtp.dll (HTP)        │       │
        │    enable_htp_fp16_precision = 1                │       │
        │    EP-context cache  →  <homeData>/qnncache/... │       │
        │    (batch dim specialized to static N — §5.2)   │       │
        └───────────────────────────────────────────────┘       │
                        │  run(InputMask/Spatial/Global)         │
                        ▼                                        │
        Output{Policy,PolicyPass,Value,ScoreValue,Ownership} (logits)
                        │                                        │
        getOutput(): logit copy + symmetry (NO activation) ◄─────┘  (shared)
                        ▼
        NNOutput (logits) → search applies softmax/tanh later (nneval.cpp)
```

New/changed components:

| Component | Change |
|-----------|--------|
| `cpp/CMakeLists.txt` | add `QNN` backend arm; factor ONNX-proto compilation into a shared helper used by both TENSORRT and QNN; find/link ONNX Runtime; deploy runtime DLLs |
| `cpp/neuralnet/onnxmodelbuilder.*` | reused; specialize batch to static `N` via ORT free-dim override (or optional emitter change) + opset check (§5.2) |
| `cpp/neuralnet/qnnbackend.cpp` | **new** — the entire `NeuralNet::` implementation for QNN |
| `cpp/program/setup.cpp` | **required** — add `"qnn"` backend prefix, `USE_QNN_BACKEND` selection, NCHW default (§5.6) |
| `cpp/main.cpp` | add QNN to backend/version reporting (§5.6) |
| `cpp/configs/gtp_example.cfg` | document new `qnn*` config keys |
| `Compiling.md` | Windows-ARM64 + ORT/QNN build instructions |
| `docs/NPUBackend.md` | this document |

Everything above `getOutput` (search, GTP, nnEval batching, symmetry) is unchanged.

---

## 5. Detailed design (file by file)

### 5.1 `cpp/CMakeLists.txt`

**(a) Backend source selection** — add to the first ladder (near the `EIGEN`/empty arms):

```cmake
elseif(USE_BACKEND STREQUAL "QNN")
  message(STATUS "-DUSE_BACKEND=QNN, using ONNX Runtime + Qualcomm QNN (Hexagon HTP) backend.")
  set(NEURALNET_BACKEND_SOURCES
    neuralnet/qnnbackend.cpp
    )
```

Register it in the cache property so GUIs list it:

```cmake
set_property(CACHE USE_BACKEND PROPERTY STRINGS "" CUDA TENSORRT OPENCL EIGEN METAL QNN)
```

**(b) Factor out ONNX-proto compilation.** Today the `protobuf_generate_cpp` call and the
`onnxmodelbuilder.cpp` source are inside the `TENSORRT` arm of the second ladder. Extract a
function so both backends share it:

```cmake
function(katago_enable_onnx_emitter target)
  find_package(Protobuf REQUIRED)
  set(ONNX_PROTO_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/onnx")
  protobuf_generate_cpp(ONNX_PROTO_SRCS ONNX_PROTO_HDRS "${ONNX_PROTO_DIR}/onnx.proto")
  set_source_files_properties(${ONNX_PROTO_SRCS} PROPERTIES COMPILE_OPTIONS "-w")
  target_sources(${target} PRIVATE ${ONNX_PROTO_SRCS} neuralnet/onnxmodelbuilder.cpp)
  target_include_directories(${target} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
  target_link_libraries(${target} protobuf::libprotobuf)
endfunction()
```

The `TENSORRT` arm calls `katago_enable_onnx_emitter(katago)` (replacing its inline copy),
and the new `QNN` arm calls it too.

> Symbol-isolation note: for TensorRT this protobuf must be isolated from the protobuf that
> `nvonnxparser` statically links. For QNN there is no such conflict — ONNX Runtime consumes
> the **serialized bytes**, not our protobuf types — so our vendored protobuf is used purely
> to *build* the byte string and never crosses the ORT boundary.

**(c) Find and link ONNX Runtime** — add to the second ladder:

```cmake
elseif(USE_BACKEND STREQUAL "QNN")
  target_compile_definitions(katago PRIVATE USE_QNN_BACKEND)

  # ONNX Runtime with the QNN EP (prebuilt ARM64 package or self-built).
  find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
    HINTS ${ONNXRUNTIME_ROOT_DIR} PATH_SUFFIXES include include/onnxruntime)
  find_library(ONNXRUNTIME_LIBRARY NAMES onnxruntime
    HINTS ${ONNXRUNTIME_ROOT_DIR} PATH_SUFFIXES lib)
  if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
    message(FATAL_ERROR "ONNX Runtime not found; set -DONNXRUNTIME_ROOT_DIR=<dir>.")
  endif()
  include_directories(SYSTEM ${ONNXRUNTIME_INCLUDE_DIR})
  target_link_libraries(katago ${ONNXRUNTIME_LIBRARY})

  katago_enable_onnx_emitter(katago)

  # QNN EP loads QnnHtp.dll + friends at runtime via provider option "backend_path".
  # Deploy the ORT + QNN runtime DLLs next to katago.exe so they are found.
  if(ONNXRUNTIME_RUNTIME_DLLS)
    add_custom_command(TARGET katago POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${ONNXRUNTIME_RUNTIME_DLLS} $<TARGET_FILE_DIR:katago>)
  endif()
```

**(d) ARM64 hygiene.** The existing block that forces signed `char` on ARM
(`-fsigned-char`, ~lines 644–648) already covers ARM64. Confirm the MSVC-vs-clang toolchain
path and that `NEON`/vectorized core code compiles under the ARM64 toolchain (it already
does for the Metal/ARM path).

### 5.2 `onnxmodelbuilder` reuse (and the static-batch requirement)

The emitter is backend-agnostic, but two things need explicit handling:

- **Dynamic batch → static for HTP (blocking).** The emitter marks the batch dim of every
  input and output as a symbolic `dim_param("batch")` (`onnxmodelbuilder.cpp` `addInput` /
  `addInputNC11` / `markOutput`). The QNN HTP backend requires **fully static shapes**, so
  the graph must be specialized to a concrete `N` *before session initialization*. Two ways:
  1. **ORT free-dimension override (no emitter change):** call
     `sessionOptions.AddFreeDimensionOverrideByName("batch", N)` before creating the session,
     then assert every graph input/output resolved to a static shape. Simplest; preferred for
     Phase 1 with `N = 1`.
  2. **Emitter change:** add an optional fixed-`N` parameter to `OnnxModelBuilder::build` that
     writes `set_dim_value(N)` instead of `set_dim_param("batch")`. More invasive but removes
     any doubt about residual dynamic dims.

  Runtime zero-padding **cannot** substitute for this — a session created with a dynamic (or
  wrong fixed) batch will not compile onto the HTP. Whatever `N` is chosen is part of the
  EPContext cache key (§5.5).
- **Metadata models are rejected by the emitter.** `build` throws on
  `metaEncoderVersion > 0` (`onnxmodelbuilder.cpp:817`). The backend must surface a clear
  "QNN backend does not support SGF-metadata models" error and Phase 1 scope excludes them.

Other items to verify (minor):

- **Opset compatibility.** The emitter finalizes the `ModelProto` with
  `ir_version = onnx::IR_VERSION_2023_5_5` and `opset_import` version **20**
  (`onnxmodelbuilder.cpp`, `model.set_ir_version(...)` / `opset->set_version(20)`). Confirm
  the pinned ORT + QNN EP accept opset 20; if not, adjust that one line. The op set is
  intentionally primitive (Conv, MatMul, Add/Mul/Sub, ReduceSum/ReduceMean, Sqrt/Div,
  Reshape/Transpose; RMSNorm as primitive ops; transformer attention as MatMul + `Softmax` +
  Reshape/Transpose). The only `Softmax` nodes are *inside* transformer attention blocks;
  head softmax/tanh are not in the graph (§3.3).
- **FP16 region hints are informational only.** `Result::trunkTipAndHeadNodeNames` and
  `rmsNormNodeNames` name numerically-sensitive regions the *TensorRT* backend force-pins to
  FP32 via `setPrecision`. QNN EP has **no equivalent documented per-node force-FP32 API**, so
  these lists are not directly actionable on QNN (see §6.2) — keep them for diagnostics, not
  as a precision-control mechanism.
- **`transformerNHWC`.** The QNN backend passes a fixed choice (default NCHW for simplicity;
  revisit if HTP prefers channel-last). Whatever is chosen is part of the cache key (§5.5).

### 5.3 `cpp/neuralnet/qnnbackend.cpp` — the new backend

A single translation unit implementing the whole `NeuralNet::` interface. Structure mirrors
`trtbackend.cpp`.

**Global lifecycle**

```cpp
void NeuralNet::globalInitialize();  // create the shared Ort::Env; for a plugin-style QNN EP
                                     // build, also register the EP plugin here (see §9)
void NeuralNet::globalCleanup();     // release the shared Ort::Env
void NeuralNet::printDevices();      // log resolved QNN backend path + HTP availability
```

We hold one process-wide `Ort::Env` (thread-safe). Note `globalInitialize` is **not
necessarily a no-op**: depending on the ORT/QNN packaging (§9), the QNN EP may need explicit
plugin registration before any session can append it.

**LoadedModel** — identical to the shared version in §3.1 (load `ModelDesc`, apply
`applyScale8ToReduceActivations`).

**How device selection and FP16 actually arrive.** These do **not** come from custom `cfg`
reads in the backend. `cpp/program/setup.cpp` parses the generic per-backend keys and passes
them into the interface: `qnnDeviceToUse[...]` → the `gpuIdxs` vector / `gpuIdxForThisThread`
argument, and `qnnUseFP16` (an `enabled_t`: `true`/`false`/`auto`) → the `useFP16Mode`
argument of `createComputeContext`. So the backend consumes `gpuIdxForThisThread` and
`useFP16Mode`; only the genuinely QNN-specific options below are read directly off `cfg`
(matching how `trtbackend` reads `trt*` keys). **This requires adding `"qnn"` to
`Setup::getBackendPrefixes()` — see §5.6.**

**ComputeContext** — parsed once in `createComputeContext`:

```cpp
struct ComputeContext {
  int nnXLen, nnYLen;
  enabled_t useFP16Mode;        // from the framework (qnnUseFP16=auto|true|false)
  string homeDataDirOverride;

  // Genuinely QNN-specific, read directly off cfg (all optional with defaults):
  string qnnBackendPath;        // path to QnnHtp.dll (default: resolve next to exe / SDK)
  string htpPerformanceMode;    // "burst"|"sustained_high_performance"|"balanced"|... (default "burst")
  bool   useContextCache;       // persist the compiled EPContext model (default true)
  string contextCacheDirOverride;
  int    vtcmMb;                // optional HTP vtcm size hint (0 = default)
  bool   transformerNHWC;       // ONNX emitter layout (default false for HTP)
};
```

FP16 on HTP is derived from `useFP16Mode` (treat `auto` as "on for HTP"): the backend sets
`enable_htp_fp16_precision` accordingly rather than owning a separate boolean.

**ComputeHandle** — one per server thread; owns the ORT session:

```cpp
struct ComputeHandle {
  const ComputeContext* ctx;
  int modelVersion;
  int maxBatchSize;
  int staticBatch;      // the concrete N the session was specialized to (§5.2/§7)
  bool requireExactNNLen;
  bool inputsUseNHWC;   // as passed by the framework; QNN defaults NCHW via setup.cpp (§5.6)
  bool usingFP16;
  int  qnnDevice;       // gpuIdx: 0 = HTP (default), 100 = plain ORT CPU EP (reference) — Metal-style

  std::unique_ptr<Ort::Session> session;
  Ort::MemoryInfo memInfo;               // Cpu, since we stage host buffers
  std::vector<const char*> inputNames;   // InputMask, InputSpatial, InputGlobal
  std::vector<const char*> outputNames;  // Output{PolicyPass,Policy,Value,ScoreValue,Ownership}

  size_t getBufferRowElts(const char* name) const;   // parity helpers like the TRT backend
};
```

`createComputeHandle`:

1. Resolve `qnnDevice` from `gpuIdxForThisThread` (`-1 → 0`), Metal-style. `0` = HTP; `100`
   = **plain ORT CPU EP (no QNN EP appended)** for a correctness reference (see §6.2 note —
   `QnnCpu.dll` is *not* the ORT CPU EP; device 100 simply skips the QNN EP). With device 100,
   `usingFP16 = false`.
2. Reject metadata models early (`getModelDesc(...).metaEncoderVersion > 0`).
3. Build the ONNX bytes: `OnnxModelBuilder::build(getModelDesc(loadedModel), ctx->nnXLen,
   ctx->nnYLen, requireExactNNLen, ctx->transformerNHWC, logger)`.
4. Pick `staticBatch` (Phase 1: `1`; later `maxBatchSize`, §7) and **make the graph static**
   via `AddFreeDimensionOverrideByName("batch", staticBatch)` (§5.2); assert all I/O is static.
5. Configure `Ort::SessionOptions`. For device 0 append the QNN EP with provider options:
   `backend_type = "htp"` (or `backend_path = ctx->qnnBackendPath`),
   `htp_performance_mode`, `enable_htp_fp16_precision = usingFP16 ? "1" : "0"`, optional
   `vtcm_mb`, `htp_graph_finalization_optimization_mode`. Wire EPContext caching per §5.5.
6. Construct the session from the ONNX bytes or from a cached EPContext model (§5.5).
7. Record `usingFP16`, `staticBatch`, model version / channel counts.

`isUsingFP16` returns `handle->usingFP16`. `setIsWarmup` returns false (no-op; satisfies the
interface).

**InputBuffers** — identical layout to the TRT backend: host `float[]` staging arrays sized
`maxBatchSize * singleXxxElts` for mask/spatial/global inputs and the five outputs, plus the
`singleXxxElts`/`singleXxxBytes` bookkeeping. (Copy the TRT `InputBuffers` struct essentially
verbatim; it has no CUDA dependency. Drop the meta arrays for Phase 1.)

**getOutput** — the hot path:

1. For each of `numBatchEltsFilled` rows, copy `rowGlobal` and (via
   `SymmetryHelpers::copyInputsWithSymmetry`) `rowSpatial` into the staging arrays, and derive
   the mask from spatial channel 0 — as the TRT backend does. **The TRT copy assumes NCHW**;
   since QNN defaults `inputsUseNHWC = false` via §5.6, that copy is correct as-is. If NHWC is
   ever enabled, an explicit NHWC→NCHW conversion is required first.
2. If `staticBatch > numBatchEltsFilled`, pad up to `staticBatch`. **Do not pad with an
   all-zero mask when `requireExactNNLen == false`:** an all-zero `InputMask` makes the
   emitter's mask-sum zero and divides by zero in gpool/RMSNorm. Pad by **duplicating the last
   valid row** (or run with `requireExactNNLen` semantics). For `staticBatch == 1`, each eval
   is a single row and this is moot.
3. Wrap the staging arrays as `Ort::Value` tensors of the static shapes and
   `session->Run(...)` into preallocated output `Ort::Value`s backed by the result arrays.
4. Copy back **logits** and reorder spatial outputs by symmetry into `NNOutput`, reusing the
   `trtbackend.cpp` `getOutput` decode verbatim (policy optimism blend, `whiteWinProb` etc.,
   ownership symmetry, score-value channels by model version). **No softmax/tanh** — the
   client applies those later (§3.3).

**Test hooks** (`testEvaluateConv`, etc.) may return `false` initially (allowed by the
interface). Correctness is validated end-to-end via `runnnevalcanarytests` (§10).

### 5.4 Config keys (`cpp/configs/gtp_example.cfg`)

Follow the existing device-key convention (`trtDeviceToUse`, `cudaDeviceToUseThread<N>`,
`metalDeviceToUseThread<N>`, `openclDeviceToUse`, `<prefix>UseFP16`). Once `"qnn"` is a
registered backend prefix (§5.6), `qnnDeviceToUse*` and `qnnUseFP16` are parsed **generically
by `setup.cpp`** — do not reinvent them. Only `qnnBackendPath`, `qnnHtpPerformanceMode`,
`qnnUseContextCache`, `qnnContextCacheDir`, and `qnnVtcmMb` are read directly off `cfg` in
`createComputeContext`. Documented block:

```ini
# ===== QNN (Qualcomm Hexagon NPU) backend =====
# Device select (0 = Hexagon HTP/NPU [default], 100 = plain CPU EP reference for bring-up):
# qnnDeviceToUse = 0
# Per-thread (with numNNServerThreadsPerModel > 1):
# qnnDeviceToUseThread0 = 0
# qnnDeviceToUseThread1 = 0

# Run the NPU in FP16 (recommended). auto = on for HTP. (framework key, like openclUseFP16)
# qnnUseFP16 = auto

# Path to the QNN HTP backend library (QnnHtp.dll). Default: auto-detect next to katago.exe.
# qnnBackendPath = C:\path\to\QnnHtp.dll

# HTP performance mode: burst | sustained_high_performance | high_performance | balanced | power_saver
# qnnHtpPerformanceMode = burst

# Cache the compiled EPContext model to speed up subsequent startups (§5.5).
# qnnUseContextCache = true
# qnnContextCacheDir = <homeDataDir>/qnncache
```

`nnMaxBatchSize` and `numNNServerThreadsPerModel` keep their existing meaning.

### 5.5 QNN HTP context caching (ORT EPContext)

Compiling an ONNX graph for the HTP is slow (seconds–minutes). ORT's **EPContext** feature
caches the compiled context — but the mechanism is **not** "append bytes to an opaque plan"
like TensorRT. Instead, with `ep.context_enable=1`, ORT **generates a new ONNX "EPContext"
wrapper model** (default name `<model>_ctx.onnx`) whose node(s) embed or reference the
precompiled QNN context; on the next run you **load that generated model** to skip
compilation. Concretely:

- **Location:** `HomeData::getHomeDataDir(true, ctx->homeDataDirOverride)` + `/qnncache/`,
  overridable via `qnnContextCacheDir`.
- **Cache key (filename):** SHA-256 of the **finalized static ONNX bytes** (post free-dim
  override) ⊕ a parameter string of `{nnXLen, nnYLen, staticBatch, requireExactNNLen, fp16,
  transformerNHWC, qnnDevice, htpPerformanceMode, ORT version, QNN/QAIRT version, HTP arch
  version}`. Any change ⇒ different filename ⇒ rebuild.
- **Miss path:** create the session from the in-memory static ONNX with
  `ep.context_enable = "1"`, `ep.context_embed_mode = "1"` (single self-contained artifact),
  and `ep.context_file_path = <uniqueTempPath>`. ORT compiles and writes the EPContext model.
  Then rename it atomically into the keyed cache path (mirror `writeFileAtomically`).
- **Hit path:** create the session directly from the cached EPContext model file (still
  appending the QNN EP). ORT loads the precompiled context instead of recompiling.
- **Do not** append our own SHA/param bytes to the file (it is a valid ONNX protobuf, unlike
  the TRT plan). Encode the key in the filename and/or a sidecar `.meta` file; on any load
  failure, discard and rebuild.
- **Embed mode** (`ep.context_embed_mode=1`) yields one file for atomic replace; non-embed
  mode produces an ONNX wrapper **plus** an external `.bin` you must keep together.

> The EPContext binary is device/SDK/driver-specific; the version fields in the key prevent a
> context built under a different QNN/HTP/ORT from being loaded.

### 5.6 Required non-CMake integration (`setup.cpp`, `main.cpp`, tests)

CMake wiring alone is insufficient — the framework must learn about the backend:

- **`cpp/program/setup.cpp`:**
  - Add `"qnn"` to `Setup::getBackendPrefixes()` (currently
    `cuda/trt/metal/opencl/eigen/dummybackend`). Without this, `qnnDeviceToUse`/`qnnUseFP16`
    are treated as unused keys and the generic device/FP16 parsing never runs for QNN.
  - Add a `#elif defined(USE_QNN_BACKEND)` arm that sets `backendPrefix = "qnn"`.
  - Add `qnn` to the **NCHW default** list for `inputsUseNHWC` (the line
    `backendPrefix == "opencl" || ... == "trt" || ... == "metal" ? false : true`). QNN's ONNX
    contract is NCHW; if this is missed, `inputsUseNHWC` defaults **true** and the TRT-style
    NCHW copy in `getOutput` silently misinterprets inputs.
- **`cpp/main.cpp`:** add QNN to the backend/version reporting (`--version`, startup banner),
  alongside the other `USE_*_BACKEND` cases.
- **`cpp/tests/testcommon.cpp`:** if any test hard-codes the set of backends / NHWC
  expectations, add the QNN case (NCHW).

---

## 6. Precision strategy

### 6.1 FP16 on HTP (initial)

The HTP executes FP16 (and fixed-point) natively; FP32 is not its fast path. We emit an
FP32 ONNX graph (unchanged) and let the QNN EP run it in FP16 by setting
`enable_htp_fp16_precision = 1`. This keeps the emitter and the on-disk model untouched
while getting NPU-native math. `isUsingFP16` reports true so logs/benchmark are accurate.

### 6.2 Numerically sensitive regions

KataGo's global-pooling reductions, RMSNorm, and head layers can lose accuracy in FP16. The
TensorRT backend explicitly pins these (`trunkTipAndHeadNodeNames`, `rmsNormNodeNames`) to
FP32 via `setPrecision(kFLOAT)` — evidence that whole-graph FP16 is **not** assumed safe.

The catch: **QNN EP exposes no documented TensorRT-style API to force named nodes to FP32**,
so the emitter's region lists are not directly actionable on QNN. That makes whole-graph FP16
a **go/no-go experiment**, not an expected pass:

- **Phase 1:** run whole-graph FP16 and *measure* accuracy vs. the Eigen reference (§10).
  KataGo tolerates FP16 on OpenCL/CUDA/Metal, so this may well pass — but treat it as
  unverified until measured, especially value/score/ownership heads.
- **If a head is out of tolerance**, the levers are coarser than TRT's: (a) partition the
  sensitive tail onto CPU (steer ORT so those nodes run on the CPU EP rather than HTP), (b)
  keep the whole model on the CPU EP for those outputs, or (c) move to a QDQ/quantized model
  with higher-precision head (§6.3). Node-level FP32 pinning on HTP is not a given.

### 6.3 INT8/INT16 quantization (future — see §13)

Full HTP throughput (and possibly transformer-`MatMul` support at all, §8) needs
quantization. This is more than "another context flavor": the `.bin.gz` model carries **no
calibration scales**, so it requires building a **calibrated QDQ ONNX model** (or a
precompiled quantized artifact) from a representative dataset of board positions, plus careful
per-head accuracy validation (Go strength is sensitive to value/score error). The emitter and
cache key already parameterize precision, so a quantized path slots in as a new precision
flavor — but the calibration/QDQ pipeline itself is substantial net-new work.

---

## 7. Static shapes and batching

The HTP requires **fully static input shapes**, but the emitter marks the batch dim symbolic
(`dim_param("batch")`, §3.3). So the graph **must be specialized to a concrete `N` before
session creation** (free-dimension override or emitter change, §5.2); runtime padding of a
still-dynamic session does not work. `nnXLen`/`nnYLen` are already fixed per process. Options:

- **Static batch = 1 (Phase 1).** Specialize `N = 1`; each eval runs one row. Lowest latency,
  simplest, no padding hazards. Throughput-limited but the correct first target.
- **Static batch = `maxBatchSize` (Phase 2).** Specialize `N = maxBatchSize`. When
  `numBatchEltsFilled < N`, pad the staging arrays up to `N` — but **never with an all-zero
  mask when `requireExactNNLen == false`**, because the emitter divides by the mask sum in
  gpool/RMSNorm (all-zero ⇒ divide-by-zero / NaNs). Pad by duplicating the last valid row and
  discard its output. `nnMaxBatchSize` sizes this; "≈ numSearchThreads" guidance still applies.
- **Multiple static contexts / buckets.** Build several fixed-`N` contexts (e.g. 1, 8,
  `maxBatchSize`) and dispatch to the smallest that fits — more cache artifacts and memory;
  deferred.

`requireExactNNLen` is passed straight through to `OnnxModelBuilder::build` and is part of the
cache key. When exact, the mask is all-ones and the emitter elides masking (also sidestepping
the padding hazard above).

Recommendation: **static N = 1 for Phase 1**, then **static N = maxBatchSize with
last-row-duplication padding** for Phase 2; buckets only if profiling demands it.

---

## 8. Operator coverage and CPU-fallback audit (do this in Phase 0)

The single biggest performance/feasibility risk: any ONNX node the QNN HTP backend does not
support is **partitioned onto the CPU EP**, and each HTP↔CPU boundary forces a round trip. A
trunk full of fallbacks can be slower than pure CPU — and if a whole family (e.g.
transformers) falls back, the backend is pointless for it. **This must be measured before
writing the full backend, not discovered in a late phase.**

Specific known hazard: current QNN HTP op tables restrict `MatMul` largely to **quantized**
input combinations. KataGo transformer **attention emits floating-point `MatMul`**
(`onnxmodelbuilder.cpp:663,674,717,729`), and NHWC channel projections add more `MatMul`s.
So transformer nets may not run usefully on HTP in FP16 at all — quantization (§6.3) could be
a **prerequisite**, not an optional future. Convnets (Conv-dominated) are the safe MVP target.

| KataGo construct | ONNX ops emitted | HTP risk |
|------------------|------------------|----------|
| Conv (kxk, dilated, SAME pad) | `Conv` | low — core HTP op |
| 1x1 conv / channel matmul | `Conv` / `MatMul` | low–medium (`MatMul` float support varies) |
| Residual add, scale/bias | `Add`, `Mul`, `Sub` | low |
| Global pooling | `ReduceSum`, `ReduceMean` + broadcast `Mul`/`Add` | medium — check reduce+broadcast |
| Mask application | `Mul` with mask, `Reshape`/`Transpose` | medium — layout ops sometimes fall back |
| RMSNorm | `Mul`/`ReduceMean`/`Sqrt`/`Div` (no single op) | medium — many small ops |
| Transformer attention | float `MatMul`+`Softmax`+`Reshape`+`Transpose` | **high — float MatMul may be HTP-unsupported** |

Method to produce the audit (Phase 0, standalone — no KataGo backend needed yet):

1. Emit ONNX for a representative convnet and a representative transformer net (small harness
   calling `OnnxModelBuilder::build`, or the TRT debug-dump path), free-dim-overridden to
   `N = 1`.
2. Create an ORT QNN(HTP) session with **`session.disable_cpu_ep_fallback = "1"`** so any
   unsupported node is a hard error that names itself, and enable QNN EP partition/verbose
   logging. Record the exact HTP-vs-CPU split.
3. For heavy fallbacks, either rewrite the emitter's expression into an HTP-friendly
   equivalent (emitter-local, benefits TRT too), accept a tail-only (heads) fallback, or
   conclude the family needs quantization.

Deliverable: an op-coverage table with the real HTP/CPU split per model family, feeding the
Phase-1 scope decision (which model families ship first).

---

## 9. Build, dependencies and packaging

### 9.1 Dependencies

- **ONNX Runtime (ARM64) with the QNN EP.** Provides `onnxruntime.dll`, the QNN EP
  (`onnxruntime_providers_qnn.dll`), and headers (`onnxruntime_cxx_api.h`). Obtain via a
  prebuilt NuGet package or a self-built ORT (`--use_qnn`).
  - **Pin a compatible {ORT, QNN EP, QAIRT/QNN SDK, HTP arch/driver} set** per ORT's QNN EP
    compatibility matrix — these are tightly coupled and the packaging is evolving. Confirm at
    implementation time: the classic in-box path used the `Microsoft.ML.OnnxRuntime.QNN`
    package and `SessionOptionsAppendExecutionProvider("QNN", opts)`; newer ORT is moving to a
    **standalone/plugin QNN EP** with device enumeration + `AppendExecutionProvider_V2`. Which
    one you target changes (a) whether `globalInitialize` must register the EP plugin, and (b)
    which DLLs ship. Record the exact chosen versions in `Compiling.md` and in the cache key.
- **Qualcomm AI Engine Direct (QNN/QAIRT) SDK** matching the ORT build. Provides `QnnHtp.dll`
  and the HTP stub/skel for the X1E80100's HTP arch version (e.g. V73), plus `QnnSystem.dll`.
- **Protobuf** — already vendored/used by the ONNX emitter (unchanged).

### 9.2 Build invocation (Windows ARM64)

```powershell
cmake . -G "Visual Studio 17 2022" -A ARM64 `
  -DUSE_BACKEND=QNN `
  -DONNXRUNTIME_ROOT_DIR="C:\deps\onnxruntime-qnn-arm64" `
  -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

### 9.3 Runtime layout

`katago.exe` must find, at runtime, `onnxruntime.dll`, `onnxruntime_providers_qnn.dll`, and
the QNN HTP DLLs. The CMake `POST_BUILD` copy (§5.1c) stages the ORT DLLs next to the exe;
the QNN SDK DLLs are either copied alongside or referenced via `qnnBackendPath`. Document
the required DLL set in the release packaging notes.

### 9.4 `Compiling.md`

Add a Windows-on-ARM64 subsection under `## Windows` describing: prerequisites (VS 2022
ARM64, ORT+QNN package, QNN SDK), the `-DUSE_BACKEND=QNN -DONNXRUNTIME_ROOT_DIR=...` cmake
line, DLL deployment, and a note that the HTP path is FP16 and validated on X1E80100.

---

## 10. Testing and validation

`runoutputtests` does **not** exercise any NN backend (it runs deterministic board/rules
tests — `command/runtests.cpp:74`). The correctness gates that actually run inference are:

- **`katago runnnevalcanarytests -model <net> ...`** — NN-eval canary sanity checks on a
  model (`runtests.cpp:408`). Primary correctness gate: run with the QNN backend and compare
  against an **Eigen-generated reference** for the same net. Because HTP is FP16, use an
  FP16-appropriate tolerance (cf. `runsearchtestsfp16.sh` and the OpenCL/Metal FP16 paths) on
  policy/value/score/ownership.
- **`katago runnnbatchingtest`** (`runtests.cpp:317`) — exercises batched eval; run once
  static batch > 1 is enabled (Phase 2).
- **Symmetry + search tests:** `runsearchtests*.sh` to confirm MCTS behaves and no NaNs
  propagate through the (external) softmax/tanh.
- **Device sanity:** `qnnDeviceToUse = 100` (plain ORT CPU EP, no QNN) as an FP32-ish
  reference to isolate "NPU numerics vs. wiring bug" during bring-up.
- **No-fallback gate:** during benchmarking, run the HTP session with
  `disable_cpu_ep_fallback` to prove the trunk is genuinely on the NPU (§8) while collecting
  `katago benchmark` visits/sec.
- **Test hooks:** `testEvaluate*` may stay unimplemented initially; the canary/batching tests
  are the real gate.

---

## 11. Performance considerations

- **Latency vs throughput.** The HTP favors steady, appropriately-sized batches. Tune
  `nnMaxBatchSize` (fixed-batch context) and thread count; start with the "half of
  numSearchThreads" guidance used elsewhere and measure.
- **HTP performance mode.** `burst`/`sustained_high_performance` for max speed; `balanced`/
  `power_saver` for battery. Exposed via `qnnHtpPerformanceMode`.
- **Warmup.** First HTP execution after context load incurs one-time cost; run a warmup eval
  when the handle is created so the first real search move isn't penalized (the framework
  already warms handles up).
- **Context caching.** With §5.5, only the first-ever run per model/params pays compilation.
- **Fallback partitions** (from §8) are the dominant perf risk; minimizing them is the main
  tuning work in Phase 3.
- **VTCM / graph finalization.** Optional `vtcm_mb` and graph-finalization-optimization
  provider options can materially affect HTP throughput; expose and tune.

---

## 12. Risks and open questions

| # | Risk / question | Mitigation |
|---|-----------------|------------|
| R1 | HTP op coverage: transformer float `MatMul` / RMSNorm fall back to CPU (or don't run) | **Audit in Phase 0** with `disable_cpu_ep_fallback` (§8); ship convnets first; transformers may need quantization (§6.3) |
| R2 | FP16 accuracy on value/score/ownership heads degrades strength; **no QNN per-node FP32-force API** | Whole-graph FP16 is a go/no-go experiment vs Eigen (§6.2/§10); if it fails, partition tail to CPU or quantize |
| R3 | Emitter's **symbolic batch** dim won't run on HTP | Specialize `N` before session init via free-dim override / emitter change (§5.2); Phase 1 `N=1` |
| R4 | ORT/QNN EP/QAIRT/HTP-driver version coupling + evolving packaging (in-box vs plugin) on Win-ARM64 | Pin a compatible set; `globalInitialize` may need plugin registration; document + cache-key the versions (§9.1) |
| R5 | EPContext cache is a **generated ONNX wrapper model**, not a byte-appended plan; staleness/corruption | Generate to temp + atomic rename; embed mode; key on static-ONNX SHA + params + versions; rebuild on load failure (§5.5) |
| R6 | Padding partial batches with an all-zero mask ⇒ divide-by-zero in gpool/RMSNorm | Duplicate last valid row (or exact-NN-len); never zero-mask-pad (§7) |
| R7 | Missing non-CMake wiring (`setup.cpp` prefix/NHWC, `main.cpp`) ⇒ keys ignored, inputs mis-laid-out | Add `"qnn"` prefix + NCHW default + reporting (§5.6) |
| R8 | Metadata (SGF) models unsupported by the emitter (throws) | Reject early with a clear error; out of Phase-1 scope (§3.3/§5.2) |
| R9 | DLL discovery at runtime (QnnHtp + skel/stub + ORT QNN EP) | POST_BUILD copy + `qnnBackendPath`; documented DLL set (§9.3) |
| Q1 | NCHW vs NHWC emitter layout best for HTP? | Benchmark both; winner becomes default (cache-keyed) |
| Q2 | Prebuilt ORT-QNN package vs self-build? | Prefer a pinned prebuilt ARM64 package; self-build if EP options lag |

---

## 13. Phased implementation roadmap

**Phase 0 — Version pin + standalone op-coverage probe (feasibility gate)**
- Pin a compatible {ORT, QNN EP, QAIRT, HTP driver} set (§9.1). Build a tiny standalone
  harness that calls `OnnxModelBuilder::build`, free-dim-overrides to `N=1`, and creates an
  ORT QNN(HTP) session with `disable_cpu_ep_fallback=1` for a representative convnet **and**
  transformer net. Produce the HTP/CPU op-coverage report (§8). **Decide which model families
  are viable in FP16** before investing in the full backend. Also factor
  `katago_enable_onnx_emitter` out of the TENSORRT CMake arm and wire the `QNN` arm so a
  `-DUSE_BACKEND=QNN` binary links (§5.1).

**Phase 1 — FP16 HTP MVP, static N=1, full KataGo integration (convnets)**
- Implement `qnnbackend.cpp` end-to-end (load model → emit ONNX → free-dim `N=1` → ORT+QNN
  HTP session with `enable_htp_fp16_precision` → logit decode). Add the **`setup.cpp` /
  `main.cpp` integration (§5.6)** — this is required, not optional. Reject metadata models.
  No cache yet. Gate: `runnnevalcanarytests` within FP16 tolerance vs Eigen, on a
  non-metadata convnet; first `benchmark` numbers with no CPU fallback.

**Phase 2 — Batching + EPContext caching**
- Static `N = maxBatchSize` with last-row-duplication padding (§7); EPContext cache with the
  generate-to-temp/atomic-rename workflow and version-keyed filenames (§5.5); config keys
  (§5.4) and warmup; `runnnbatchingtest`. Tune threads/batch/perf-mode.

**Phase 3 — Broaden model families & performance hardening**
- Reduce fallbacks via emitter-local rewrites; tune performance mode / VTCM / graph
  finalization; validate additional families. For transformers, **decide whether HTP support
  requires QDQ quantization** (from the Phase-0 finding). Ship updated benchmarks + op table.

**Phase 4 — INT8/INT16 quantization (future)**
- Build the calibrated QDQ pipeline: representative board-position dataset, calibration,
  per-head accuracy validation; add a quantized precision flavor to the context/cache key.
  Gate on measured strength parity. (Larger than a "context flavor" — see §6.3.)

---

## Appendix A — Key source references

- Backend interface: `cpp/neuralnet/nninterface.h`
- Backend selection & linking: `cpp/CMakeLists.txt` (`USE_BACKEND` ladders; ONNX proto at
  the TENSORRT arm)
- ONNX emitter: `cpp/neuralnet/onnxmodelbuilder.{h,cpp}` (I/O contract, `Result` region lists)
- Structural template: `cpp/neuralnet/trtbackend.cpp` (`createComputeContext`/`Handle`,
  `InputBuffers`, `getOutput`, plan cache via `HomeData::getHomeDataDir` + `writeFileAtomically`)
- NPU precedent & `gpuIdx` device convention: `cpp/neuralnet/metalbackend.{h,cpp}`
  (`METAL_MUX_GPU=0`, `METAL_MUX_ANE=100`)
- Config conventions: `cpp/configs/gtp_example.cfg` (`*DeviceToUse*`,
  `numNNServerThreadsPerModel`, `nnMaxBatchSize`)
- Build docs: `Compiling.md`
- Correctness/perf harnesses: `katago runnnevalcanarytests`, `katago runnnbatchingtest`,
  `cpp/runsearchtests*.sh`, `katago benchmark` (note: `runoutputtests` does not run the NN)
