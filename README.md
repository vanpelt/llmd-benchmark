# LLM-D Benchmarking

A benchmarking suite for LLM deployments with a focus on caching and performance optimization.

## Repository Structure

- `main.py`: Custom benchmark for testing cacheability. Modifies prompts to test cached vs. non-cached performance.
- `vllm/`: Git submodule containing vllm's benchmark tools
- `Justfile`: Command runner for benchmark operations

## Basic Benchmark

The main.py script is meant to test prefix caching and prefill. Run with:

```bash
uv sync
uv run main.py
```

Or use the Justfile:

```bash
just benchmark-cache-basic
```

## Setup for vllm Benchmarks

Initialize the vllm benchmarks submodule:

```bash
just setup-vllm-benchmarks
```

Install dependencies:

```bash
just install-deps
```

## Running vllm Benchmarks

### Standard Benchmarks

Run the standard benchmark against the PD (Prefill Disaggregation) server:

```bash
just benchmark
```

Run against the basic server:

```bash
just benchmark-basic
```

Compare both servers:

```bash
just benchmark-compare
```

### Dataset-specific Benchmarks

Run benchmarks with the ShareGPT dataset:

```bash
just benchmark-sharegpt
```

Run with the Sonnet dataset:

```bash
just benchmark-sonnet
```

## Configuration

Modify the variables at the top of the Justfile to change:

- `MODEL`: The model to benchmark (default: Llama-3.2-3B-Instruct)
- `BASE_URL`: The server base URL
- `RR`: Request rate (requests per second)
- `NUM_REQUESTS`: Total number of requests to send
- `INPUT_LEN`: Input token length
- `OUTPUT_LEN`: Output token length