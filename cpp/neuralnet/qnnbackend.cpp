#ifdef USE_QNN_BACKEND

// KataGo neural-net backend that runs inference on the Qualcomm Hexagon NPU (HTP) via ONNX Runtime
// with the QNN Execution Provider. It reuses the existing ONNX emitter (OnnxModelBuilder) and the
// exact input construction / output decoding of the other backends, so only the "emit ONNX -> run
// it in an ORT session configured for the QNN HTP -> copy logits back" middle is new. The structure
// deliberately mirrors trtbackend.cpp (ComputeContext / ComputeHandle / InputBuffers / getOutput),
// swapping "TensorRT engine + plan cache" for "ORT session + QNN EPContext cache". See
// docs/NPUBackend.md for the full design.

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <random>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "../core/fileutils.h"
#include "../core/makedir.h"
#include "../core/sha2.h"
#include "../core/test.h"
#include "../dataio/homedata.h"
#include "../neuralnet/desc.h"
#include "../neuralnet/modelversion.h"
#include "../neuralnet/nneval.h"
#include "../neuralnet/nninputs.h"
#include "../neuralnet/nninterface.h"
#include "../neuralnet/onnxmodelbuilder.h"

using namespace std;

// gpuIdx convention, mirroring the Metal backend's GPU/ANE multiplexer: 0 selects the Hexagon HTP
// (the real target), 100 selects a plain ORT CPU EP (no QNN EP appended) as an FP32 correctness
// reference for bring-up. Any other value is treated as HTP.
static constexpr int QNN_DEVICE_HTP = 0;
static constexpr int QNN_DEVICE_CPU_REFERENCE = 100;

// The fixed I/O tensor names emitted by OnnxModelBuilder (see docs/NPUBackend.md 3.3). Phase 1
// excludes SGF-metadata models, so InputMeta is never bound.
static const char* const INPUT_NAMES[3] = {"InputMask", "InputSpatial", "InputGlobal"};
static const char* const OUTPUT_NAMES[5] = {
  "OutputPolicyPass", "OutputPolicy", "OutputValue", "OutputScoreValue", "OutputOwnership"};

//------------------------------------------------------------------------------------------------
// Global lifecycle
//------------------------------------------------------------------------------------------------

// One process-wide Ort::Env (thread-safe, shared across all handles). Created lazily on first use:
// NeuralNet::globalInitialize() is NOT guaranteed to be called before handles are built (e.g.
// runtinynntests / testgpuerror construct an NNEvaluator without it), so every access must go
// through getOrtEnv() rather than assuming the env already exists.
static std::mutex g_ortEnvMutex;
static unique_ptr<Ort::Env> g_ortEnv;

static Ort::Env& getOrtEnv() {
  std::lock_guard<std::mutex> lock(g_ortEnvMutex);
  if(g_ortEnv == nullptr)
    g_ortEnv = make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "katago-qnn");
  return *g_ortEnv;
}

void NeuralNet::globalInitialize() {
  getOrtEnv();
}

void NeuralNet::globalCleanup() {
  std::lock_guard<std::mutex> lock(g_ortEnvMutex);
  g_ortEnv.reset();
}

//------------------------------------------------------------------------------------------------
// LoadedModel (identical to every other backend)
//------------------------------------------------------------------------------------------------

struct LoadedModel {
  ModelDesc modelDesc;

  LoadedModel(const string& fileName, const string& expectedSha256) {
    ModelDesc::loadFromFileMaybeGZipped(fileName, modelDesc, expectedSha256);
    modelDesc.applyScale8ToReduceActivations();
  }

  LoadedModel() = delete;
  LoadedModel(const LoadedModel&) = delete;
  LoadedModel& operator=(const LoadedModel&) = delete;
};

LoadedModel* NeuralNet::loadModelFile(const string& file, const string& expectedSha256) {
  return new LoadedModel(file, expectedSha256);
}

void NeuralNet::freeLoadedModel(LoadedModel* loadedModel) {
  delete loadedModel;
}

const ModelDesc& NeuralNet::getModelDesc(const LoadedModel* loadedModel) {
  return loadedModel->modelDesc;
}

//------------------------------------------------------------------------------------------------
// ComputeContext
//------------------------------------------------------------------------------------------------

struct ComputeContext {
  int nnXLen;
  int nnYLen;
  enabled_t useFP16Mode;  // framework key qnnUseFP16 (auto|true|false); auto == on for the HTP
  string homeDataDirOverride;

  // Genuinely QNN-specific options, read directly off cfg (all optional with defaults), mirroring how
  // trtbackend reads trt* keys. Device selection and FP16 arrive through the framework, not here.
  string qnnBackendPath;         // path to QnnHtp.dll ("" => "QnnHtp.dll", resolved via the exe dir)
  string htpPerformanceMode;     // burst|sustained_high_performance|high_performance|balanced|...
  bool useContextCache;          // persist the compiled EPContext model (default true)
  string contextCacheDirOverride;
  int vtcmMb;                    // optional HTP VTCM size hint (0 => leave to the EP default)
  bool transformerNHWC;          // ONNX emitter layout (default false => NCHW for the HTP)
  string batchBucketsSpec;       // qnnBatchBuckets: comma-separated static-N buckets ("" => default)
};

ComputeContext* NeuralNet::createComputeContext(
  const vector<int>& gpuIdxs,
  Logger* logger,
  int nnXLen,
  int nnYLen,
  const string& homeDataDirOverride,
  enabled_t useFP16Mode,
  const LoadedModel* loadedModel,
  ConfigParser& cfg) {
  (void)gpuIdxs;
  (void)logger;
  (void)loadedModel;

  ComputeContext* context = new ComputeContext();
  context->nnXLen = nnXLen;
  context->nnYLen = nnYLen;
  context->useFP16Mode = useFP16Mode;
  context->homeDataDirOverride = homeDataDirOverride;

  context->qnnBackendPath = cfg.contains("qnnBackendPath") ? cfg.getString("qnnBackendPath") : "";
  context->htpPerformanceMode =
    cfg.contains("qnnHtpPerformanceMode") ? cfg.getString("qnnHtpPerformanceMode") : "burst";
  context->useContextCache = cfg.contains("qnnUseContextCache") ? cfg.getBool("qnnUseContextCache") : true;
  context->contextCacheDirOverride = cfg.contains("qnnContextCacheDir") ? cfg.getString("qnnContextCacheDir") : "";
  context->vtcmMb = cfg.contains("qnnVtcmMb") ? cfg.getInt("qnnVtcmMb", 0, 1024) : 0;
  // The HTP's ONNX contract is NCHW. Keep this false unless someone deliberately benchmarks NHWC.
  context->transformerNHWC = cfg.contains("qnnTransformerNHWC") ? cfg.getBool("qnnTransformerNHWC") : false;
  // Optional explicit static-batch bucket set (parsed per-handle since it depends on maxBatchSize).
  context->batchBucketsSpec = cfg.contains("qnnBatchBuckets") ? cfg.getString("qnnBatchBuckets") : "";
  return context;
}

void NeuralNet::freeComputeContext(ComputeContext* computeContext) {
  delete computeContext;
}

//------------------------------------------------------------------------------------------------
// Helpers
//------------------------------------------------------------------------------------------------

static bool deviceIsHtp(int qnnDevice) {
  return qnnDevice != QNN_DEVICE_CPU_REFERENCE;
}

// Resolve the effective FP16 setting. The HTP runs FP16 natively (enable_htp_fp16_precision); the CPU
// reference EP is always FP32.
static bool resolveUseFP16(enabled_t useFP16Mode, int qnnDevice) {
  if(!deviceIsHtp(qnnDevice))
    return false;
  return useFP16Mode == enabled_t::True || useFP16Mode == enabled_t::Auto;
}

// SHA-256 hex of an arbitrary byte buffer.
static string sha256Hex(const string& bytes) {
  char hash[65];
  SHA2::get256(reinterpret_cast<const uint8_t*>(bytes.data()), bytes.size(), hash);
  return string(hash);
}

// Compute the sorted, deduped set of static-batch bucket sizes for a handle. Each bucket is a separate
// compiled HTP context specialized to a fixed N; getOutput dispatches each eval to the smallest bucket
// that fits, so partial batches waste at most (nextBucket/batchSize - 1) rows instead of padding all
// the way up to maxBatchSize. spec is the qnnBatchBuckets config ("" => default {1, 8, maxBatchSize}).
// The result is clamped to [1, maxBatchSize], always includes maxBatchSize (so any batch fits), and is
// sorted ascending.
static vector<int> computeBucketSizes(int maxBatchSize, const string& spec) {
  if(maxBatchSize < 1)
    maxBatchSize = 1;
  std::set<int> sizes;
  if(Global::trim(spec).empty()) {
    sizes.insert(1);
    sizes.insert(8);
  } else {
    for(const string& tok: Global::split(spec, ',')) {
      string t = Global::trim(tok);
      if(t.empty())
        continue;
      int n = 0;
      if(Global::tryStringToInt(t, n) && n >= 1)
        sizes.insert(std::min(n, maxBatchSize));
    }
  }
  sizes.insert(maxBatchSize);  // the largest bucket must cover the maximum batch the server can send
  return vector<int>(sizes.begin(), sizes.end());
}

//------------------------------------------------------------------------------------------------
// ComputeHandle
//------------------------------------------------------------------------------------------------

struct ComputeHandle {
  // A single compiled HTP context specialized to a fixed static batch N. We hold several (buckets) and
  // dispatch each eval to the smallest N that fits, so partial batches don't pad all the way up to
  // maxBatchSize.
  struct Bucket {
    int n;
    unique_ptr<Ort::Session> session;
  };

  const ComputeContext* ctx;

  int modelVersion;
  int maxBatchSize;
  bool requireExactNNLen;
  bool inputsUseNHWC;
  bool usingFP16;
  int qnnDevice;  // gpuIdx: 0 = HTP (default), 100 = plain ORT CPU EP (FP32 reference)

  int numSpatialFeatures;
  int numGlobalFeatures;
  int numPolicyChannels;
  int numValueChannels;
  int numScoreValueChannels;
  int numOwnershipChannels;

  size_t singleMaskElts;
  size_t singleSpatialElts;
  size_t singleGlobalElts;

  Ort::MemoryInfo memInfo;
  vector<Bucket> buckets;  // ascending by n; buckets.back().n == maxBatchSize

  ComputeHandle(
    Logger* logger,
    const ComputeContext* context,
    const LoadedModel* loadedModel,
    int maxBatchSz,
    bool requireExactNNLenArg,
    bool inputsUseNHWCArg,
    int gpuIdxForThisThread,
    int serverThreadIdx)
    : memInfo(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU)) {
    ctx = context;
    maxBatchSize = maxBatchSz;
    requireExactNNLen = requireExactNNLenArg;
    inputsUseNHWC = inputsUseNHWCArg;

    // Metal-style device resolution: -1 defaults to the HTP.
    qnnDevice = (gpuIdxForThisThread == -1) ? QNN_DEVICE_HTP : gpuIdxForThisThread;

    if(inputsUseNHWC)
      throw StringError("QNN backend: inputsUseNHWC=true is not supported; the ONNX contract is NCHW");

    const ModelDesc& desc = loadedModel->modelDesc;
    modelVersion = desc.modelVersion;

    // SGF-metadata models are rejected: the ONNX emitter throws on metaEncoderVersion > 0, so they
    // cannot run through this backend (they also fail on TensorRT via the ONNX path).
    if(desc.metaEncoderVersion > 0)
      throw StringError("QNN backend does not support SGF-metadata models (metaEncoderVersion > 0)");

    numSpatialFeatures = NNModelVersion::getNumSpatialFeatures(modelVersion);
    numGlobalFeatures = NNModelVersion::getNumGlobalFeatures(modelVersion);
    numPolicyChannels = desc.numPolicyChannels;
    numValueChannels = desc.numValueChannels;
    numScoreValueChannels = desc.numScoreValueChannels;
    numOwnershipChannels = desc.numOwnershipChannels;

    usingFP16 = resolveUseFP16(ctx->useFP16Mode, qnnDevice);

    singleMaskElts = (size_t)ctx->nnXLen * ctx->nnYLen;
    singleSpatialElts = (size_t)numSpatialFeatures * ctx->nnXLen * ctx->nnYLen;
    singleGlobalElts = (size_t)numGlobalFeatures;

    // Emit the FP32, NCHW ONNX graph (weights baked in) once; every bucket compiles from these bytes.
    if(logger != NULL)
      logger->write(
        "QNN backend thread " + Global::intToString(serverThreadIdx) + ": emitting ONNX for model version " +
        Global::intToString(modelVersion));
    OnnxModelBuilder::Result onnxResult =
      OnnxModelBuilder::build(desc, ctx->nnXLen, ctx->nnYLen, requireExactNNLen, ctx->transformerNHWC, logger);
    const string& onnxBytes = onnxResult.serializedModel;

    // The HTP requires fully static shapes. Build one compiled context per static-batch bucket; each is
    // a separate weight-carrying HTP context (memory scales with the number of buckets). getOutput
    // dispatches each eval to the smallest bucket that fits; partial batches are padded up to that
    // bucket's N by duplicating the last valid row (never an all-zero mask, which would divide by zero
    // in gpool/RMSNorm).
    vector<int> bucketSizes = computeBucketSizes(maxBatchSize, ctx->batchBucketsSpec);
    string bucketList;
    for(int n: bucketSizes) {
      Bucket bucket;
      bucket.n = n;
      bucket.session = createSession(logger, onnxBytes, n, serverThreadIdx);
      buckets.push_back(std::move(bucket));
      bucketList += (bucketList.empty() ? "" : ",") + Global::intToString(n);
    }

    if(logger != NULL)
      logger->write(
        "QNN backend thread " + Global::intToString(serverThreadIdx) + ": ready (device " +
        (deviceIsHtp(qnnDevice) ? string("HTP") : string("CPU-reference")) + ", batch buckets {" +
        bucketList + "}, useFP16 " + Global::boolToString(usingFP16) + ")");

    for(const Bucket& bucket: buckets)
      runWarmup(logger, bucket, serverThreadIdx);
  }

  ComputeHandle() = delete;
  ComputeHandle(const ComputeHandle&) = delete;
  ComputeHandle& operator=(const ComputeHandle&) = delete;

  // Configure the session options. When compiling a fresh ONNX graph (forCompile=true) we specialize
  // the symbolic "batch" dim to the static N and let ORT optimize. When loading an already-compiled
  // EPContext model (forCompile=false) the shapes are already static and the graph is precompiled, so
  // we must NOT re-apply a free-dim override or re-optimize -- doing so makes ORT re-partition and hunt
  // for a (nonexistent) external context binary. Either way, for the HTP device we append the QNN EP.
  void configureSessionOptions(Ort::SessionOptions& so, bool forCompile, int staticN) const {
    so.SetIntraOpNumThreads(1);
    so.SetGraphOptimizationLevel(
      forCompile ? GraphOptimizationLevel::ORT_ENABLE_ALL : GraphOptimizationLevel::ORT_DISABLE_ALL);

    // Specialize the emitter's symbolic batch dim (dim_param "batch") to a concrete N *before*
    // session creation. Runtime padding cannot substitute for this: a dynamic-batch session will not
    // compile onto the HTP. Only needed when compiling from the raw ONNX; the cached model is static.
    if(forCompile)
      so.AddFreeDimensionOverrideByName("batch", (int64_t)staticN);

    if(deviceIsHtp(qnnDevice)) {
      unordered_map<string, string> qnnOptions;
      qnnOptions["backend_path"] = ctx->qnnBackendPath.empty() ? string("QnnHtp.dll") : ctx->qnnBackendPath;
      qnnOptions["htp_performance_mode"] = ctx->htpPerformanceMode;
      qnnOptions["enable_htp_fp16_precision"] = usingFP16 ? "1" : "0";
      qnnOptions["htp_graph_finalization_optimization_mode"] = "3";
      if(ctx->vtcmMb > 0)
        qnnOptions["vtcm_mb"] = Global::intToString(ctx->vtcmMb);
      so.AppendExecutionProvider("QNN", qnnOptions);
    }
    // Device 100 appends no EP: the default ORT CPU EP runs the graph as an FP32 reference.
  }

  // Build the ORT session for a given static batch N, using the on-disk EPContext cache when enabled.
  // Compiling an ONNX graph for the HTP is slow, so ORT's EPContext feature persists the compiled
  // context as a generated ONNX wrapper model, keyed on the static-ONNX SHA + the parameters (incl. N)
  // that affect compilation, so each bucket caches in its own subdirectory.
  unique_ptr<Ort::Session> createSession(Logger* logger, const string& onnxBytes, int staticN, int serverThreadIdx) {
    const bool cacheEnabled = ctx->useContextCache && deviceIsHtp(qnnDevice);

    // The EPContext cache lives in a per-key SUBDIRECTORY: qnncache/qnnctx_<hash>/model_ctx.onnx. Using
    // a directory (rather than a bare file) keeps the generated EPContext ONNX together with any sibling
    // context binary ORT emits, and lets us publish the whole thing atomically with a directory rename.
    string keyDir, keyModel, cacheDir;
    if(cacheEnabled) {
      const string paramStr = Global::strprintf(
        "nnx%d_nny%d_N%d_exact%d_fp16%d_nhwc%d_dev%d_htp%s_perf%s_ort%s",
        ctx->nnXLen, ctx->nnYLen, staticN, requireExactNNLen ? 1 : 0, usingFP16 ? 1 : 0,
        ctx->transformerNHWC ? 1 : 0, qnnDevice, ctx->qnnBackendPath.c_str(),
        ctx->htpPerformanceMode.c_str(), Ort::GetVersionString().c_str());
      const string key = sha256Hex(onnxBytes + "\n" + paramStr);
      cacheDir = ctx->contextCacheDirOverride.empty()
        ? (HomeData::getHomeDataDir(true, ctx->homeDataDirOverride) + "/qnncache")
        : ctx->contextCacheDirOverride;
      try {
        MakeDir::make(cacheDir);
        keyDir = cacheDir + "/qnnctx_" + key.substr(0, 32);
        keyModel = keyDir + "/model_ctx.onnx";
      } catch(const exception& e) {
        if(logger != NULL)
          logger->write(string("QNN backend: could not create context cache dir, proceeding without cache: ") + e.what());
      }
    }

    // Cache hit: load the precompiled EPContext model from its FILE PATH so ORT can resolve the
    // context (and any sibling binary) relative to the model directory. Published via directory rename,
    // so a present model_ctx.onnx is always complete; on any load failure we discard it and rebuild.
    if(!keyModel.empty() && FileUtils::exists(keyModel)) {
      try {
        Ort::SessionOptions so;
        configureSessionOptions(so, /*forCompile=*/false, staticN);
        std::filesystem::path cachePath(keyModel);
        auto sess = make_unique<Ort::Session>(getOrtEnv(), cachePath.c_str(), so);
        if(logger != NULL)
          logger->write(
            "QNN backend thread " + Global::intToString(serverThreadIdx) + ": loaded cached HTP context " + keyModel);
        return sess;
      } catch(const exception& e) {
        if(logger != NULL)
          logger->write(string("QNN backend: cached context failed to load, rebuilding: ") + e.what());
        std::error_code ec;
        std::filesystem::remove_all(keyDir, ec);
      }
    }

    // Cache miss (or caching disabled): compile the static ONNX for the HTP. When caching, we write the
    // ONNX to a temp file inside a build directory and compile from THAT FILE (not from memory) so ORT
    // has a real base directory for the generated EPContext model + its context binary; then we publish
    // the build directory into the keyed cache with an atomic rename.
    Ort::SessionOptions so;
    configureSessionOptions(so, /*forCompile=*/true, staticN);

    string buildDir, srcModelPath, ctxModelPath;
    if(!keyDir.empty()) {
      static const uint64_t randBase = std::random_device{}();
      static std::atomic<uint64_t> counter{0};
      buildDir = Global::strprintf(
        "%s.building_%llx_%llu", keyDir.c_str(), (unsigned long long)randBase,
        (unsigned long long)counter.fetch_add(1));
      std::error_code ec;
      std::filesystem::remove_all(buildDir, ec);
      try {
        MakeDir::make(buildDir);
        srcModelPath = buildDir + "/src.onnx";
        ctxModelPath = buildDir + "/model_ctx.onnx";
        ofstream ofs;
        FileUtils::open(ofs, srcModelPath, ios::out | ios::binary);
        ofs.write(onnxBytes.data(), (std::streamsize)onnxBytes.size());
        ofs.close();
        if(ofs.fail())
          throw StringError("failed to write temp ONNX for HTP compilation");
        so.AddConfigEntry("ep.context_enable", "1");
        // embed_mode=0: write the compiled context to a sibling .bin referenced by relative path. We
        // keep it co-located with model_ctx.onnx in the keyed cache dir, so loading by file path
        // resolves it. (embed_mode=1 wrote an empty in-model reference that ORT could not reload.)
        so.AddConfigEntry("ep.context_embed_mode", "0");
        so.AddConfigEntry("ep.context_file_path", ctxModelPath.c_str());
      } catch(const exception& e) {
        if(logger != NULL)
          logger->write(string("QNN backend: could not stage context cache build dir, compiling without cache: ") + e.what());
        std::filesystem::remove_all(buildDir, ec);
        buildDir.clear();
        srcModelPath.clear();
      }
    }

    unique_ptr<Ort::Session> sess;
    try {
      if(!srcModelPath.empty()) {
        std::filesystem::path srcPath(srcModelPath);
        sess = make_unique<Ort::Session>(getOrtEnv(), srcPath.c_str(), so);
      } else {
        sess = make_unique<Ort::Session>(getOrtEnv(), onnxBytes.data(), onnxBytes.size(), so);
      }
    } catch(const exception& e) {
      if(!buildDir.empty()) {
        std::error_code ec;
        std::filesystem::remove_all(buildDir, ec);
      }
      throw StringError(string("QNN backend: failed to create ORT session: ") + e.what());
    }

    // Publish the freshly-generated EPContext model into the cache via a directory rename. The source
    // ONNX is dropped first so only the compiled context is cached. Failures here are non-fatal: the
    // in-memory session we just built is fully usable regardless.
    if(!buildDir.empty()) {
      std::error_code ec;
      FileUtils::tryRemoveFile(srcModelPath);
      if(FileUtils::exists(ctxModelPath)) {
        std::filesystem::remove_all(keyDir, ec);
        std::filesystem::rename(buildDir, keyDir, ec);
        if(ec) {
          // Another thread/process may have already published this key, or a cross-device move failed.
          std::error_code ec2;
          std::filesystem::remove_all(buildDir, ec2);
        } else if(logger != NULL) {
          logger->write(
            "QNN backend thread " + Global::intToString(serverThreadIdx) + ": cached HTP context to " + keyModel);
        }
      } else {
        std::filesystem::remove_all(buildDir, ec);
      }
    }
    return sess;
  }

  // One dummy eval per bucket so the one-time HTP finalization/first-run cost is paid at handle
  // creation rather than on the first real search move. Uses an all-ones mask (never an all-zero mask,
  // which would divide by zero in the emitter's gpool/RMSNorm). Non-fatal on failure.
  void runWarmup(Logger* logger, const Bucket& bucket, int serverThreadIdx) {
    try {
      vector<float> mask((size_t)bucket.n * singleMaskElts, 1.0f);
      vector<float> spatial((size_t)bucket.n * singleSpatialElts, 0.0f);
      vector<float> global((size_t)bucket.n * singleGlobalElts, 0.0f);
      // Mark on-board via spatial channel 0 so the derived mask semantics are consistent.
      for(int n = 0; n < bucket.n; n++) {
        float* s0 = &spatial[(size_t)n * singleSpatialElts];
        for(size_t i = 0; i < singleMaskElts; i++)
          s0[i] = 1.0f;
      }
      runSession(bucket, mask.data(), spatial.data(), global.data(), nullptr, nullptr, nullptr, nullptr, nullptr);
    } catch(const exception& e) {
      if(logger != NULL)
        logger->write(
          "QNN backend thread " + Global::intToString(serverThreadIdx) + ": warmup eval failed (non-fatal): " + e.what());
    }
  }

  // Run one bucket's session over its full static batch (bucket.n rows). Inputs point at staging arrays
  // holding at least bucket.n rows. Each non-null output pointer receives the first copyRows rows of
  // that output; passing null skips copying that output (used by warmup).
  void runSession(
    const Bucket& bucket,
    const float* maskData,
    const float* spatialData,
    const float* globalData,
    float* policyPassOut,
    float* policyOut,
    float* valueOut,
    float* scoreValueOut,
    float* ownershipOut,
    int copyRows = 0) {
    const int64_t N = bucket.n;
    const int64_t H = ctx->nnYLen;
    const int64_t W = ctx->nnXLen;

    const int64_t maskShape[4] = {N, 1, H, W};
    const int64_t spatialShape[4] = {N, numSpatialFeatures, H, W};
    const int64_t globalShape[4] = {N, numGlobalFeatures, 1, 1};

    Ort::Value inputs[3] = {
      Ort::Value::CreateTensor<float>(
        memInfo, const_cast<float*>(maskData), (size_t)N * singleMaskElts, maskShape, 4),
      Ort::Value::CreateTensor<float>(
        memInfo, const_cast<float*>(spatialData), (size_t)N * singleSpatialElts, spatialShape, 4),
      Ort::Value::CreateTensor<float>(
        memInfo, const_cast<float*>(globalData), (size_t)N * singleGlobalElts, globalShape, 4)};

    vector<Ort::Value> results =
      bucket.session->Run(Ort::RunOptions{nullptr}, INPUT_NAMES, inputs, 3, OUTPUT_NAMES, 5);

    if(copyRows <= 0)
      return;

    auto copyOut = [&](const Ort::Value& v, float* dst, size_t singleElts) {
      if(dst == nullptr)
        return;
      const float* src = v.GetTensorData<float>();
      std::copy(src, src + (size_t)copyRows * singleElts, dst);
    };
    copyOut(results[0], policyPassOut, (size_t)numPolicyChannels);
    copyOut(results[1], policyOut, (size_t)numPolicyChannels * singleMaskElts);
    copyOut(results[2], valueOut, (size_t)numValueChannels);
    copyOut(results[3], scoreValueOut, (size_t)numScoreValueChannels);
    copyOut(results[4], ownershipOut, (size_t)numOwnershipChannels * singleMaskElts);
  }

  // Pick the smallest bucket whose static N is >= batchSize. Buckets are sorted ascending and the
  // largest is maxBatchSize >= batchSize, so a fit always exists.
  const Bucket& pickBucket(int batchSize) const {
    for(const Bucket& bucket: buckets) {
      if(bucket.n >= batchSize)
        return bucket;
    }
    return buckets.back();
  }
};

ComputeHandle* NeuralNet::createComputeHandle(
  ComputeContext* context,
  const LoadedModel* loadedModel,
  Logger* logger,
  int maxBatchSize,
  bool requireExactNNLen,
  bool inputsUseNHWC,
  int gpuIdxForThisThread,
  int serverThreadIdx) {
  if(logger != NULL)
    logger->write(
      "QNN backend thread " + Global::intToString(serverThreadIdx) + ": initializing (may take a long time)");
  return new ComputeHandle(
    logger, context, loadedModel, maxBatchSize, requireExactNNLen, inputsUseNHWC, gpuIdxForThisThread,
    serverThreadIdx);
}

void NeuralNet::freeComputeHandle(ComputeHandle* computeHandle) {
  delete computeHandle;
}

bool NeuralNet::isUsingFP16(const ComputeHandle* computeHandle) {
  return computeHandle->usingFP16;
}

bool NeuralNet::setIsWarmup(const ComputeHandle* computeHandle, bool isWarmup) {
  (void)computeHandle;
  (void)isWarmup;
  return false;
}

void NeuralNet::printDevices() {
  cout << "QNN backend: inference via ONNX Runtime " << Ort::GetVersionString()
       << " + Qualcomm QNN Execution Provider" << endl;
  cout << "  device 0   = Hexagon HTP (NPU) [default]" << endl;
  cout << "  device 100 = plain ORT CPU EP (FP32 reference for bring-up)" << endl;
}

//------------------------------------------------------------------------------------------------
// InputBuffers (host-side staging arrays, TensorRT layout minus the CUDA/meta bits)
//------------------------------------------------------------------------------------------------

struct InputBuffers {
  int maxBatchSize;

  size_t singleMaskElts;
  size_t singleInputElts;
  size_t singleInputGlobalElts;
  size_t singlePolicyPassResultElts;
  size_t singlePolicyResultElts;
  size_t singleValueResultElts;
  size_t singleScoreValueResultElts;
  size_t singleOwnershipResultElts;

  unique_ptr<float[]> maskInputs;
  unique_ptr<float[]> spatialInputs;
  unique_ptr<float[]> globalInputs;
  unique_ptr<float[]> policyPassResults;
  unique_ptr<float[]> policyResults;
  unique_ptr<float[]> valueResults;
  unique_ptr<float[]> scoreValueResults;
  unique_ptr<float[]> ownershipResults;

  InputBuffers(const LoadedModel* loadedModel, int maxBatchSz, int nnXLen, int nnYLen) {
    const ModelDesc& m = loadedModel->modelDesc;

    if(nnXLen > NNPos::MAX_BOARD_LEN)
      throw StringError(
        Global::strprintf("nnXLen (%d) is greater than NNPos::MAX_BOARD_LEN (%d)", nnXLen, NNPos::MAX_BOARD_LEN));
    if(nnYLen > NNPos::MAX_BOARD_LEN)
      throw StringError(
        Global::strprintf("nnYLen (%d) is greater than NNPos::MAX_BOARD_LEN (%d)", nnYLen, NNPos::MAX_BOARD_LEN));

    maxBatchSize = maxBatchSz;
    singleMaskElts = (size_t)nnXLen * nnYLen;
    singleInputElts = (size_t)m.numInputChannels * nnXLen * nnYLen;
    singleInputGlobalElts = (size_t)m.numInputGlobalChannels;
    singlePolicyPassResultElts = (size_t)m.numPolicyChannels;
    singlePolicyResultElts = (size_t)m.numPolicyChannels * nnXLen * nnYLen;
    singleValueResultElts = (size_t)m.numValueChannels;
    singleScoreValueResultElts = (size_t)m.numScoreValueChannels;
    singleOwnershipResultElts = (size_t)m.numOwnershipChannels * nnXLen * nnYLen;

    testAssert(NNModelVersion::getNumSpatialFeatures(m.modelVersion) == m.numInputChannels);
    testAssert(NNModelVersion::getNumGlobalFeatures(m.modelVersion) == m.numInputGlobalChannels);

    maskInputs = make_unique<float[]>(maxBatchSize * singleMaskElts);
    spatialInputs = make_unique<float[]>(maxBatchSize * singleInputElts);
    globalInputs = make_unique<float[]>(maxBatchSize * singleInputGlobalElts);
    policyPassResults = make_unique<float[]>(maxBatchSize * singlePolicyPassResultElts);
    policyResults = make_unique<float[]>(maxBatchSize * singlePolicyResultElts);
    valueResults = make_unique<float[]>(maxBatchSize * singleValueResultElts);
    scoreValueResults = make_unique<float[]>(maxBatchSize * singleScoreValueResultElts);
    ownershipResults = make_unique<float[]>(maxBatchSize * singleOwnershipResultElts);
  }

  InputBuffers() = delete;
  InputBuffers(const InputBuffers&) = delete;
  InputBuffers& operator=(const InputBuffers&) = delete;
};

InputBuffers* NeuralNet::createInputBuffers(const LoadedModel* loadedModel, int maxBatchSize, int nnXLen, int nnYLen) {
  return new InputBuffers(loadedModel, maxBatchSize, nnXLen, nnYLen);
}

void NeuralNet::freeInputBuffers(InputBuffers* inputBuffers) {
  delete inputBuffers;
}

//------------------------------------------------------------------------------------------------
// getOutput (the hot path). Input fill + logit/symmetry decode are identical to trtbackend.cpp; only
// the "run the model" middle differs (ORT session Run instead of a CUDA engine enqueue).
//------------------------------------------------------------------------------------------------

void NeuralNet::getOutput(
  ComputeHandle* gpuHandle,
  InputBuffers* inputBuffers,
  int numBatchEltsFilled,
  NNResultBuf** inputBufs,
  vector<NNOutput*>& outputs) {
  assert(numBatchEltsFilled <= inputBuffers->maxBatchSize);
  assert(numBatchEltsFilled > 0);

  const int batchSize = numBatchEltsFilled;
  const int nnXLen = gpuHandle->ctx->nnXLen;
  const int nnYLen = gpuHandle->ctx->nnYLen;
  const int modelVersion = gpuHandle->modelVersion;

  const int numSpatialFeatures = NNModelVersion::getNumSpatialFeatures(modelVersion);
  const int numGlobalFeatures = NNModelVersion::getNumGlobalFeatures(modelVersion);
  assert(numSpatialFeatures * nnXLen * nnYLen == (int)inputBuffers->singleInputElts);
  assert(numGlobalFeatures == (int)inputBuffers->singleInputGlobalElts);

  // Fill the host staging arrays row by row, applying the input symmetry, exactly as trtbackend does.
  for(int nIdx = 0; nIdx < batchSize; nIdx++) {
    float* rowMaskInput = &inputBuffers->maskInputs[inputBuffers->singleMaskElts * nIdx];
    float* rowSpatialInput = &inputBuffers->spatialInputs[inputBuffers->singleInputElts * nIdx];
    float* rowGlobalInput = &inputBuffers->globalInputs[inputBuffers->singleInputGlobalElts * nIdx];

    const float* rowGlobal = inputBufs[nIdx]->rowGlobalBuf.data();
    const float* rowSpatial = inputBufs[nIdx]->rowSpatialBuf.data();
    std::copy(rowGlobal, rowGlobal + numGlobalFeatures, rowGlobalInput);
    SymmetryHelpers::copyInputsWithSymmetry(
      rowSpatial, rowSpatialInput, 1, nnYLen, nnXLen, numSpatialFeatures, false, inputBufs[nIdx]->symmetry);
    // The mask is spatial feature channel 0 (1.0 on-board), just like the TensorRT backend derives it.
    std::copy(rowSpatialInput, rowSpatialInput + inputBuffers->singleMaskElts, rowMaskInput);
  }

  // Dispatch to the smallest bucket whose static N covers this batch, then pad up to that N by
  // duplicating the last valid row. Never pad with an all-zero mask: the emitter divides by the mask
  // sum in gpool/RMSNorm, so a zero mask would produce NaNs. Duplicated rows' outputs are discarded.
  const ComputeHandle::Bucket& bucket = gpuHandle->pickBucket(batchSize);
  const int staticBatch = bucket.n;
  assert(batchSize <= staticBatch);
  for(int nIdx = batchSize; nIdx < staticBatch; nIdx++) {
    const int src = batchSize - 1;
    std::copy(
      &inputBuffers->maskInputs[inputBuffers->singleMaskElts * src],
      &inputBuffers->maskInputs[inputBuffers->singleMaskElts * (src + 1)],
      &inputBuffers->maskInputs[inputBuffers->singleMaskElts * nIdx]);
    std::copy(
      &inputBuffers->spatialInputs[inputBuffers->singleInputElts * src],
      &inputBuffers->spatialInputs[inputBuffers->singleInputElts * (src + 1)],
      &inputBuffers->spatialInputs[inputBuffers->singleInputElts * nIdx]);
    std::copy(
      &inputBuffers->globalInputs[inputBuffers->singleInputGlobalElts * src],
      &inputBuffers->globalInputs[inputBuffers->singleInputGlobalElts * (src + 1)],
      &inputBuffers->globalInputs[inputBuffers->singleInputGlobalElts * nIdx]);
  }

  const int numPolicyChannels = inputBuffers->singlePolicyPassResultElts;
  assert(inputBuffers->singlePolicyResultElts == (size_t)numPolicyChannels * nnXLen * nnYLen);

  // Run the model. We read back the first batchSize rows (padding rows are discarded).
  gpuHandle->runSession(
    bucket,
    inputBuffers->maskInputs.get(),
    inputBuffers->spatialInputs.get(),
    inputBuffers->globalInputs.get(),
    inputBuffers->policyPassResults.get(),
    inputBuffers->policyResults.get(),
    inputBuffers->valueResults.get(),
    inputBuffers->scoreValueResults.get(),
    inputBuffers->ownershipResults.get(),
    batchSize);

  assert((int)outputs.size() == batchSize);

  float policyProbsTmp[NNPos::MAX_NN_POLICY_SIZE];

  for(int row = 0; row < batchSize; row++) {
    NNOutput* output = outputs[row];

    assert(output->nnXLen == nnXLen);
    assert(output->nnYLen == nnYLen);
    float policyOptimism = (float)inputBufs[row]->policyOptimism;

    const float* policyPassSrcBuf = &inputBuffers->policyPassResults[row * inputBuffers->singlePolicyPassResultElts];
    const float* policySrcBuf = &inputBuffers->policyResults[row * inputBuffers->singlePolicyResultElts];
    float* policyProbs = output->policyProbs;

    // These are in logits; the client (nneval) does the softmax / value transforms later.
    if(numPolicyChannels == 2 || (numPolicyChannels == 4 && modelVersion >= 16)) {
      for(int i = 0; i < nnXLen * nnYLen; i++) {
        float p = policySrcBuf[i];
        float pOpt = policySrcBuf[i + nnXLen * nnYLen];
        policyProbsTmp[i] = p + (pOpt - p) * policyOptimism;
      }
      SymmetryHelpers::copyOutputsWithSymmetry(policyProbsTmp, policyProbs, 1, nnYLen, nnXLen, inputBufs[row]->symmetry);
      policyProbs[nnXLen * nnYLen] = policyPassSrcBuf[0] + (policyPassSrcBuf[1] - policyPassSrcBuf[0]) * policyOptimism;
    } else {
      assert(numPolicyChannels == 1);
      SymmetryHelpers::copyOutputsWithSymmetry(policySrcBuf, policyProbs, 1, nnYLen, nnXLen, inputBufs[row]->symmetry);
      policyProbs[nnXLen * nnYLen] = policyPassSrcBuf[0];
    }

    int numValueChannels = inputBuffers->singleValueResultElts;
    assert(numValueChannels == 3);
    output->whiteWinProb = inputBuffers->valueResults[row * numValueChannels];
    output->whiteLossProb = inputBuffers->valueResults[row * numValueChannels + 1];
    output->whiteNoResultProb = inputBuffers->valueResults[row * numValueChannels + 2];

    if(output->whiteOwnerMap != NULL) {
      const float* ownershipSrcBuf = &inputBuffers->ownershipResults[row * nnXLen * nnYLen];
      assert(inputBuffers->singleOwnershipResultElts == (size_t)nnXLen * nnYLen);
      SymmetryHelpers::copyOutputsWithSymmetry(
        ownershipSrcBuf, output->whiteOwnerMap, 1, nnYLen, nnXLen, inputBufs[row]->symmetry);
    }

    int numScoreValueChannels = inputBuffers->singleScoreValueResultElts;
    if(modelVersion >= 9) {
      assert(numScoreValueChannels == 6);
      output->whiteScoreMean = inputBuffers->scoreValueResults[row * numScoreValueChannels];
      output->whiteScoreMeanSq = inputBuffers->scoreValueResults[row * numScoreValueChannels + 1];
      output->whiteLead = inputBuffers->scoreValueResults[row * numScoreValueChannels + 2];
      output->varTimeLeft = inputBuffers->scoreValueResults[row * numScoreValueChannels + 3];
      output->shorttermWinlossError = inputBuffers->scoreValueResults[row * numScoreValueChannels + 4];
      output->shorttermScoreError = inputBuffers->scoreValueResults[row * numScoreValueChannels + 5];
    } else if(modelVersion >= 8) {
      assert(numScoreValueChannels == 4);
      output->whiteScoreMean = inputBuffers->scoreValueResults[row * numScoreValueChannels];
      output->whiteScoreMeanSq = inputBuffers->scoreValueResults[row * numScoreValueChannels + 1];
      output->whiteLead = inputBuffers->scoreValueResults[row * numScoreValueChannels + 2];
      output->varTimeLeft = inputBuffers->scoreValueResults[row * numScoreValueChannels + 3];
      output->shorttermWinlossError = 0;
      output->shorttermScoreError = 0;
    } else if(modelVersion >= 4) {
      assert(numScoreValueChannels == 2);
      output->whiteScoreMean = inputBuffers->scoreValueResults[row * numScoreValueChannels];
      output->whiteScoreMeanSq = inputBuffers->scoreValueResults[row * numScoreValueChannels + 1];
      output->whiteLead = output->whiteScoreMean;
      output->varTimeLeft = 0;
      output->shorttermWinlossError = 0;
      output->shorttermScoreError = 0;
    } else if(modelVersion >= 3) {
      assert(numScoreValueChannels == 1);
      output->whiteScoreMean = inputBuffers->scoreValueResults[row * numScoreValueChannels];
      output->whiteScoreMeanSq = output->whiteScoreMean * output->whiteScoreMean;
      output->whiteLead = output->whiteScoreMean;
      output->varTimeLeft = 0;
      output->shorttermWinlossError = 0;
      output->shorttermScoreError = 0;
    } else {
      ASSERT_UNREACHABLE;
    }
  }
}

//------------------------------------------------------------------------------------------------
// Optional single-op test hooks. Not implemented for QNN; end-to-end correctness is validated via
// runnnevalcanarytests. Returning false is explicitly allowed by the interface.
//------------------------------------------------------------------------------------------------

bool NeuralNet::testEvaluateConv(
  const ConvLayerDesc*, int, int, int, bool, bool, const vector<float>&, vector<float>&) {
  return false;
}

bool NeuralNet::testEvaluateBatchNorm(
  const BatchNormLayerDesc*, int, int, int, bool, bool, const vector<float>&, const vector<float>&, vector<float>&) {
  return false;
}

bool NeuralNet::testEvaluateResidualBlock(
  const ResidualBlockDesc*, int, int, int, bool, bool, const vector<float>&, const vector<float>&, vector<float>&) {
  return false;
}

bool NeuralNet::testEvaluateGlobalPoolingResidualBlock(
  const GlobalPoolingResidualBlockDesc*, int, int, int, bool, bool, const vector<float>&, const vector<float>&,
  vector<float>&) {
  return false;
}

#endif  // USE_QNN_BACKEND
