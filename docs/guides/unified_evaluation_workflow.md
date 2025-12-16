# Unified Evaluation Workflow Support

This document identifies which strategies from the unified evaluation workflow are supported by GuideLLM, using the following classification framework:

## Classification Framework

### Native Support (Natively Supported)

Strategies that meet ALL of the following requirements:

- Available immediately after installing the evaluation harness
- Requires only import statements and minimal configuration (≤2 lines)
- No external dependencies beyond the harness itself
- No custom implementation or glue code required

### Integrated Support (Supported via Third-Party Integration)

Strategies that meet ALL of the following requirements:

- Requires installing ≥1 external package(s) beyond GuideLLM
- Requires glue code or configuration
- Has documented integration pattern or official example
- Functionality enabled through third-party tools rather than the harness alone

### Not Supported

Strategies that do not meet either of the above criteria.

## Phase 0: Provisioning (The Runtime)

### Step A: Harness Installation

**Strategy 1: Git Clone** ✅ **NATIVELY SUPPORTED**

- GuideLLM can be installed from source by cloning the Git repository
- Documented in: [Install Guide](../getting-started/install.md#3-install-from-source-on-the-main-branch)
- Command: `pip install git+https://github.com/vllm-project/guidellm.git`

**Strategy 2: PyPI Packages** ✅ **NATIVELY SUPPORTED**

- GuideLLM is available on PyPI and can be installed via pip
- Documented in: [Install Guide](../getting-started/install.md#1-install-the-latest-release-from-pypi)
- Commands:
  - `pip install guidellm[recommended]`
  - `pip install guidellm==0.2.0` (specific version)

**Strategy 3: Node Package** ❌ **NOT SUPPORTED**

- GuideLLM is a Python-based tool and does not provide Node.js packages

**Strategy 4: Binary Packages** ❌ **NOT SUPPORTED**

- GuideLLM does not distribute standalone executable binaries

**Strategy 5: Container Images** ✅ **NATIVELY SUPPORTED**

- GuideLLM provides prebuilt Docker/OCI container images
- Documented in: [README](../../README.md#install-guidellm) and [Install Guide](../getting-started/install.md)
- Registry: `ghcr.io/vllm-project/guidellm:latest`
- Command: `podman run --rm -it ghcr.io/vllm-project/guidellm:latest`

### Step B: Credential Configuration

**Strategy 1: Model API Authentication** 🔗 **SUPPORTED VIA THIRD-PARTY INTEGRATION**

- GuideLLM can benchmark OpenAI-compatible API endpoints
- Requires external backend server (vLLM, TGI, or commercial API providers)
- Configuration: `--target http://localhost:8000` parameter
- Integration documented in: [Backends Guide](./backends.md)
- Examples: vLLM server, TGI server, OpenAI API, Anthropic API

**Strategy 2: Artifact Repository Authentication** ✅ **NATIVELY SUPPORTED**

- GuideLLM includes Hugging Face Hub integration as a core dependency
- Documented in: [Datasets Guide](./datasets.md#hugging-face-datasets)
- Authentication via Hugging Face CLI login or environment variables
- Supports gated/private models and datasets through HF authentication
- No additional packages required beyond `pip install guidellm`

**Strategy 3: Evaluation Platform Authentication** ❌ **NOT SUPPORTED**

- GuideLLM does not provide native integration with evaluation platform leaderboard submission APIs
- Results can be exported but submission to external platforms requires manual integration

## Phase I: Specification (The Contract)

### Step A: SUT Preparation

**Strategy 1: Model-as-a-Service (Remote Inference)** 🔗 **SUPPORTED VIA THIRD-PARTY INTEGRATION**

- GuideLLM benchmarks OpenAI-compatible HTTP endpoints for remote inference
- Requires external backend server (vLLM, TGI, commercial APIs)
- Documented in: [Backends Guide](./backends.md)
- Configuration: `--target http://localhost:8000`
- Supported endpoints: `/completions`, `/chat/completions`, `/audio/translation`, `/audio/transcription`

**Strategy 2: Model-in-Process (Local Inference)** 🔗 **SUPPORTED VIA THIRD-PARTY INTEGRATION**

- Requires installing and running backend servers (vLLM, TGI) that load model weights
- Documented in: [Backends Guide](./backends.md)
- Integration examples:
  - vLLM: `pip install vllm && vllm serve "model-name"`
  - TGI: Docker image with `MODEL_ID` environment variable
- GuideLLM benchmarks these backends via HTTP API

**Strategy 3: Algorithm Implementation (In-Memory Structures)** ❌ **NOT SUPPORTED**

- GuideLLM does not provide ANN algorithms, ranking algorithms, or signal processing pipelines
- Focus is on benchmarking LLM inference endpoints

**Strategy 4: Policy/Agent Instantiation (Stateful Controllers)** ❌ **NOT SUPPORTED**

- GuideLLM does not support reinforcement learning policies or autonomous agents
- No support for multi-agent systems or robot controllers

### Step B: Benchmark Preparation (Inputs)

**Strategy 1: Benchmark Data Preparation (Offline)** ✅ **NATIVELY SUPPORTED**

- GuideLLM includes support for loading pre-existing datasets from multiple sources
- Documented in: [Datasets Guide](./datasets.md)
- Built-in support via `datasets` dependency:
  - Hugging Face datasets (e.g., `--data "garage-bAInd/Open-Platypus"`)
  - Local files (`.txt`, `.csv`, `.json`, `.jsonl`, `.parquet`, `.arrow`, `.hdf5`)
  - In-memory datasets (Python objects)
- Preprocessing capabilities via `guidellm preprocess dataset` command

**Strategy 2: Synthetic Data Generation (Generative)** ✅ **NATIVELY SUPPORTED**

- GuideLLM includes built-in synthetic data generation
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

**Strategy 1: Ground Truth Preparation** ❌ **NOT SUPPORTED**

- GuideLLM can load datasets with various columns (including output_tokens_count_column)
- However, it does not use ground truth, reference answers, or expected outputs for evaluation
- No comparison of generated outputs against reference data
- Focus is exclusively on performance metrics (latency, throughput) rather than correctness

**Strategy 2: Judge Preparation** ❌ **NOT SUPPORTED**

- GuideLLM does not support training or loading judge models
- No LLM-as-judge functionality for quality assessment
- No support for fine-tuning discriminative models or reward models

## Phase II: Execution (The Run)

### Step A: SUT Invocation

**Strategy 1: Batch Inference** ✅ **NATIVELY SUPPORTED**

- GuideLLM executes multiple input samples through benchmarking profiles
- Documented in: [README](../../README.md#common-use-cases-and-configurations)
- Built-in execution profiles:
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

**Strategy 1: Deterministic Measurement** ❌ **NOT SUPPORTED**

- GuideLLM does not perform correctness checking or quality evaluation
- No support for equality checks, answer extraction, or unit test pass/fail
- No token-based text metrics (BLEU, ROUGE, METEOR)
- No distance metrics for comparing outputs against ground truth
- Focus is exclusively on performance measurement (latency, throughput, token counts)

**Strategy 2: Embedding Measurement** ❌ **NOT SUPPORTED**

- GuideLLM does not compute semantic similarity or embedding-based metrics
- No BERTScore, sentence embeddings, or cross-modal embeddings support

**Strategy 3: Subjective Measurement** ❌ **NOT SUPPORTED**

- GuideLLM does not use LLMs or classifiers as evaluators
- No pairwise comparison of outputs
- No model-based judgments for subjective quality assessment

**Strategy 4: Performance Measurement** ✅ **NATIVELY SUPPORTED**

- GuideLLM's core functionality is measuring resource consumption and efficiency
- Documented in: [Metrics Guide](./metrics.md)
- Built-in metrics collected:
  - **Time costs**: Request latency, TTFT (Time to First Token), ITL (Inter-Token Latency), Time Per Output Token
  - **Throughput**: Request rate, output tokens per second, total tokens per second
  - **Concurrency**: Request concurrency levels
  - **Token counts**: Prompt tokens, output tokens
  - **Request status**: Successful, incomplete, and error requests

### Step B: Aggregate Scoring

**Strategy 1: Distributional Statistics** ✅ **NATIVELY SUPPORTED**

- GuideLLM computes comprehensive statistical summaries for all metrics
- Documented in: [Metrics Guide](./metrics.md#statistical-summaries)
- Built-in statistics computed:
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

**Strategy 1: Execution Tracing** ✅ **NATIVELY SUPPORTED**

- GuideLLM captures detailed step-by-step execution information for each request
- Source code reference: [request_stats.py](../../src/guidellm/schemas/request_stats.py)
- Built-in timing data captured:
  - Queue timing (targeted_start, queued, dequeued)
  - Execution timing (resolve_start, request_start, request_end, resolve_end)
  - Token-level timing (first_token_iteration, last_token_iteration, token_iterations)
  - Request iterations and scheduler processing times
- Request-level statistics saved in JSON/YAML outputs with sampled requests
- Progress display shows real-time execution state during benchmarks

**Strategy 2: Subgroup Analysis** ❌ **NOT SUPPORTED**

- GuideLLM does not break down performance by demographic groups, data domains, or task categories
- No built-in stratification or subgroup analysis capabilities
- Analysis is at the benchmark level, not stratified by data characteristics

**Strategy 3: Regression Alerting** ❌ **NOT SUPPORTED**

- GuideLLM does not automatically compare results against historical baselines
- No performance degradation detection or automatic alerting
- Historical comparison requires manual analysis of saved results

**Strategy 4: Chart Generation** ✅ **NATIVELY SUPPORTED**

- GuideLLM generates HTML reports with built-in visualizations
- Documented in: [Outputs Guide](./outputs.md#file-based-outputs)
- Built-in output formats:
  - HTML interactive reports with tables and charts
  - Performance metrics visualizations
- Configuration: `--outputs html`

**Strategy 5: Dashboard Creation** ✅ **NATIVELY SUPPORTED**

- GuideLLM provides an interactive web dashboard built with Next.js
- Documented in: [Outputs Guide](./outputs.md#file-based-outputs)
- Built-in dashboard features:
  - Interactive visualizations using @nivo charts (bar, line charts)
  - Redux-based state management for interactivity
  - Material-UI components for rich interface
  - Metric comparisons and result tables
  - Client-side data exploration and filtering
- HTML output embeds the full interactive dashboard application
- Can be served as a standalone web interface

**Strategy 6: Leaderboard Publication** ❌ **NOT SUPPORTED**

- GuideLLM does not support submitting results to public or private leaderboards
- Results can be exported but require manual submission to external platforms
- No integration with leaderboard APIs

## Summary

### Supported Strategies: 15 out of 50

GuideLLM is a specialized LLM benchmarking tool focused on performance evaluation of inference endpoints.

**Natively Supported (12 strategies):**

01. Git Clone installation
02. PyPI package installation
03. Container image deployment
04. Artifact repository authentication (Hugging Face)
05. Offline benchmark data preparation
06. Synthetic data generation
07. Batch inference with multiple execution profiles
08. Performance measurement (latency, throughput, token metrics)
09. Distributional statistics
10. Chart generation (HTML reports)
11. Execution tracing (detailed request-level timing data)
12. Dashboard creation (interactive Next.js web dashboard)

**Supported via Third-Party Integration (3 strategies):**

1. Model API authentication (requires backend server like vLLM/TGI)
2. Remote inference Model-as-a-Service (requires OpenAI-compatible backend)
3. Local model inference (requires vLLM/TGI installation and setup)

**Not Supported (35 strategies):**

- Quality/correctness evaluation (no ground truth comparison, embedding metrics, or LLM judges)
- Ground truth preparation for scoring (can load data but doesn't use for evaluation)
- Deterministic measurement for correctness (no BLEU, ROUGE, answer checking)
- Interactive environments and agents
- Production traffic sampling
- Multi-SUT comparison (arena battles)
- Uncertainty quantification
- Regression alerting
- Leaderboard submission
- Advanced analysis (subgroup stratification)
- And 25+ additional strategies across evaluation workflow phases

**Key Insight**: GuideLLM measures *how fast* models run, not *how well* they perform. It excels at performance benchmarking (latency, throughput, resource utilization) and provides detailed execution traces with interactive dashboards. Backend integration (vLLM, TGI) enables local and remote inference benchmarking, but these require separate installation and setup.
