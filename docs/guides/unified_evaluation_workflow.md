# Unified Evaluation Workflow Support

This document identifies which strategies from the unified evaluation workflow are supported by GuideLLM. A strategy is considered "supported" only if GuideLLM provides it natively in its full installation—meaning that once the harness is fully installed, the strategy can be executed directly without implementing custom modules or integrating external libraries.

## Phase 0: Provisioning (The Runtime)

### Step A: Harness Installation

**Strategy 1: Git Clone** ✅ **SUPPORTED**

- GuideLLM can be installed from source by cloning the Git repository
- Documented in: [Install Guide](../getting-started/install.md#3-install-from-source-on-the-main-branch)
- Command: `pip install git+https://github.com/vllm-project/guidellm.git`

**Strategy 2: PyPI Packages** ✅ **SUPPORTED**

- GuideLLM is available on PyPI and can be installed via pip
- Documented in: [Install Guide](../getting-started/install.md#1-install-the-latest-release-from-pypi)
- Commands:
  - `pip install guidellm[recommended]`
  - `pip install guidellm==0.2.0` (specific version)

**Strategy 3: Node Package** ❌ **NOT SUPPORTED**

- GuideLLM is a Python-based tool and does not provide Node.js packages

**Strategy 4: Binary Packages** ❌ **NOT SUPPORTED**

- GuideLLM does not distribute standalone executable binaries

**Strategy 5: Container Images** ✅ **SUPPORTED**

- GuideLLM provides prebuilt Docker/OCI container images
- Documented in: [README](../../README.md#install-guidellm) and [Install Guide](../getting-started/install.md)
- Registry: `ghcr.io/vllm-project/guidellm:latest`
- Command: `podman run --rm -it ghcr.io/vllm-project/guidellm:latest`

### Step B: Credential Configuration

**Strategy 1: Model API Authentication** ✅ **SUPPORTED**

- GuideLLM supports configuring API endpoints for remote inference
- OpenAI-compatible API endpoints can be specified via `--target` parameter
- Environment variables and configuration can be used for authentication
- Documented in: [Backends Guide](./backends.md)

**Strategy 2: Artifact Repository Authentication** ✅ **SUPPORTED**

- GuideLLM integrates with Hugging Face Hub for datasets and processors
- Authentication via Hugging Face CLI login or environment variables
- Documented in: [Datasets Guide](./datasets.md#hugging-face-datasets)
- Supports gated/private models and datasets through HF authentication

**Strategy 3: Evaluation Platform Authentication** ❌ **NOT SUPPORTED**

- GuideLLM does not provide native integration with evaluation platform leaderboard submission APIs
- Results can be exported but submission to external platforms requires manual integration

## Phase I: Specification (The Contract)

### Step A: SUT Preparation

**Strategy 1: Model-as-a-Service (Remote Inference)** ✅ **SUPPORTED**

- GuideLLM supports OpenAI-compatible HTTP endpoints for remote inference
- Documented in: [Backends Guide](./backends.md)
- Parameter: `--target http://localhost:8000`
- Supported endpoints: `/completions`, `/chat/completions`, `/audio/translation`, `/audio/transcription`

**Strategy 2: Model-in-Process (Local Inference)** ✅ **SUPPORTED**

- GuideLLM's supported backends (vLLM, TGI) load model weights directly into memory for local inference
- Documented in: [Backends Guide](./backends.md)
- Examples:
  - vLLM: `vllm serve "neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w4a16"`
  - TGI: Loads models via `MODEL_ID` environment variable
- Note: GuideLLM orchestrates the benchmarking via HTTP API, but the backend servers perform local model inference

**Strategy 3: Algorithm Implementation (In-Memory Structures)** ❌ **NOT SUPPORTED**

- GuideLLM does not provide ANN algorithms, ranking algorithms, or signal processing pipelines
- Focus is on benchmarking LLM inference endpoints

**Strategy 4: Policy/Agent Instantiation (Stateful Controllers)** ❌ **NOT SUPPORTED**

- GuideLLM does not support reinforcement learning policies or autonomous agents
- No support for multi-agent systems or robot controllers

### Step B: Benchmark Preparation (Inputs)

**Strategy 1: Benchmark Data Preparation (Offline)** ✅ **SUPPORTED**

- GuideLLM supports loading pre-existing datasets from multiple sources
- Documented in: [Datasets Guide](./datasets.md)
- Sources supported:
  - Hugging Face datasets (e.g., `--data "garage-bAInd/Open-Platypus"`)
  - Local files (`.txt`, `.csv`, `.json`, `.jsonl`, `.parquet`, `.arrow`, `.hdf5`)
  - In-memory datasets (Python objects)
- Preprocessing capabilities via `guidellm preprocess dataset` command

**Strategy 2: Synthetic Data Generation (Generative)** ✅ **SUPPORTED**

- GuideLLM supports synthetic data generation on the fly
- Documented in: [Datasets Guide](./datasets.md#synthetic-data)
- Configuration options:
  - `--data "prompt_tokens=256,output_tokens=128"`
  - Supports token count distributions (mean, stdev, min, max)
  - Generates prompts from source text with specified token lengths

**Strategy 3: Simulation Environment Setup (Simulated)** ❌ **NOT SUPPORTED**

- GuideLLM does not support 3D virtual environments, scene construction, or multi-agent simulations
- Focus is on LLM API benchmarking, not interactive environments

**Strategy 4: Production Traffic Sampling (Online)** ❌ **NOT SUPPORTED**

- GuideLLM does not support sampling real-world production traffic
- Benchmarks use predefined or synthetic datasets, not live traffic streams

### Step C: Benchmark Preparation (References)

**Strategy 1: Ground Truth Preparation** ⚠️ **PARTIALLY SUPPORTED**

- GuideLLM can load datasets with reference data in columns
- However, it does not use ground truth for correctness evaluation
- Ground truth can be included in datasets but is not actively used for scoring
- Focus is on performance metrics (latency, throughput) rather than accuracy metrics

**Strategy 2: Judge Preparation** ❌ **NOT SUPPORTED**

- GuideLLM does not support training or loading judge models
- No LLM-as-judge functionality for quality assessment
- No support for fine-tuning discriminative models or reward models

## Phase II: Execution (The Run)

### Step A: SUT Invocation

**Strategy 1: Batch Inference** ✅ **SUPPORTED**

- GuideLLM executes multiple input samples through a single SUT instance
- Documented in: [README](../../README.md#common-use-cases-and-configurations)
- Multiple execution profiles supported:
  - `--profile synchronous` (sequential requests)
  - `--profile concurrent` (parallel requests)
  - `--profile throughput` (maximum capacity)
  - `--profile constant` (fixed requests/sec)
  - `--profile poisson` (randomized requests/sec)
  - `--profile sweep` (automatic rate exploration)

**Strategy 2: Arena Battle** ❌ **NOT SUPPORTED**

- GuideLLM does not support executing the same input across multiple SUTs simultaneously
- Each benchmark run targets a single endpoint
- Comparison requires separate benchmark runs

**Strategy 3: Interactive Loop** ❌ **NOT SUPPORTED**

- GuideLLM does not support stateful stepping through state transitions
- No support for tool-based reasoning, physics simulation, or multi-agent coordination
- Benchmarks are request-response based, not interactive loops

**Strategy 4: Production Streaming** ❌ **NOT SUPPORTED**

- GuideLLM does not support continuously processing live production traffic
- No real-time drift monitoring or interactive feedback collection

## Phase III: Assessment (The Score)

### Step A: Individual Scoring

**Strategy 1: Deterministic Measurement** ⚠️ **PARTIALLY SUPPORTED**

- GuideLLM does not perform correctness checking (no BLEU, ROUGE, METEOR)
- Does not compare outputs against ground truth for quality metrics
- Focus is on performance measurements, not correctness

**Strategy 2: Embedding Measurement** ❌ **NOT SUPPORTED**

- GuideLLM does not compute semantic similarity or embedding-based metrics
- No BERTScore, sentence embeddings, or cross-modal embeddings support

**Strategy 3: Subjective Measurement** ❌ **NOT SUPPORTED**

- GuideLLM does not use LLMs or classifiers as evaluators
- No pairwise comparison of outputs
- No model-based judgments for subjective quality assessment

**Strategy 4: Performance Measurement** ✅ **SUPPORTED**

- GuideLLM's primary focus is performance measurement
- Documented in: [Metrics Guide](./metrics.md)
- Metrics collected:
  - **Time costs**: Request latency, TTFT (Time to First Token), ITL (Inter-Token Latency), Time Per Output Token
  - **Throughput**: Request rate, output tokens per second, total tokens per second
  - **Concurrency**: Request concurrency levels
  - **Token counts**: Prompt tokens, output tokens
  - **Request status**: Successful, incomplete, and error requests

### Step B: Aggregate Scoring

**Strategy 1: Distributional Statistics** ✅ **SUPPORTED**

- GuideLLM computes comprehensive statistical summaries
- Documented in: [Metrics Guide](./metrics.md#statistical-summaries)
- Statistics computed:
  - Mean, median, mode
  - Variance, standard deviation
  - Min, max, count, sum
  - Percentiles: p0.1, p1, p5, p10, p25, p50, p75, p90, p95, p99, p99.9

**Strategy 2: Uncertainty Quantification** ❌ **NOT SUPPORTED**

- GuideLLM does not estimate confidence bounds around metrics
- No bootstrap resampling or Prediction-Powered Inference (PPI)
- Reports point estimates and distributions but not confidence intervals

## Phase IV: Reporting (The Output)

### Step A: Insight Presentation

**Strategy 1: Execution Tracing** ⚠️ **PARTIALLY SUPPORTED**

- GuideLLM captures progress and status information during execution
- Console output shows benchmark progress and intermediate states
- However, detailed step-by-step execution logs with intermediate computational states are not captured
- Documented in: [Outputs Guide](./outputs.md#console-output)

**Strategy 2: Subgroup Analysis** ❌ **NOT SUPPORTED**

- GuideLLM does not break down performance by demographic groups, data domains, or task categories
- No built-in stratification or subgroup analysis capabilities
- Analysis is at the benchmark level, not stratified by data characteristics

**Strategy 3: Regression Alerting** ❌ **NOT SUPPORTED**

- GuideLLM does not automatically compare results against historical baselines
- No performance degradation detection or automatic alerting
- Historical comparison requires manual analysis of saved results

**Strategy 4: Chart Generation** ✅ **SUPPORTED**

- GuideLLM generates HTML reports with visualizations
- Documented in: [Outputs Guide](./outputs.md#file-based-outputs)
- Output formats include:
  - HTML interactive reports with tables and charts
  - Performance metrics visualizations
- Parameter: `--outputs html`

**Strategy 5: Dashboard Creation** ⚠️ **PARTIALLY SUPPORTED**

- GuideLLM generates HTML reports with tabular data and visualizations
- However, these are static reports, not interactive web dashboards with filtering
- No dedicated dashboard server or web interface
- Reports must be manually opened and viewed
- Documented in: [Outputs Guide](./outputs.md#file-based-outputs)

**Strategy 6: Leaderboard Publication** ❌ **NOT SUPPORTED**

- GuideLLM does not support submitting results to public or private leaderboards
- Results can be exported but require manual submission to external platforms
- No integration with leaderboard APIs

## Summary

### Supported Strategies: 14 out of 50

GuideLLM is a specialized LLM benchmarking tool focused on performance evaluation of inference endpoints. Its strengths lie in:

**Fully Supported (✅):**

01. Git Clone installation
02. PyPI package installation
03. Container image deployment
04. Model API authentication (OpenAI-compatible endpoints)
05. Artifact repository authentication (Hugging Face)
06. Remote inference (Model-as-a-Service)
07. Local model inference (via backends like vLLM and TGI)
08. Offline benchmark data preparation
09. Synthetic data generation
10. Batch inference with multiple execution profiles
11. Performance measurement (latency, throughput, token metrics)
12. Distributional statistics
13. Chart generation (HTML reports)

**Partially Supported (⚠️):**

1. Ground truth preparation (can load but doesn't use for scoring)
2. Deterministic measurement (performance only, not correctness)
3. Execution tracing (progress tracking but not detailed traces)
4. Dashboard creation (static HTML reports, not interactive dashboards)

**Not Supported (❌):**

- Quality/correctness evaluation (no ground truth comparison, embedding metrics, or LLM judges)
- Interactive environments and agents
- Production traffic sampling
- Multi-SUT comparison (arena battles)
- Uncertainty quantification
- Regression alerting
- Leaderboard submission
- Advanced analysis (subgroup stratification)

GuideLLM is purpose-built for performance benchmarking of LLM inference servers and excels in measuring latency, throughput, and resource utilization under various load patterns. It is not designed for evaluating model quality, accuracy, or correctness against ground truth references.
