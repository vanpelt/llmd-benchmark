# Justfile for running vllm benchmarks
# "meta-llama/Llama-3.2-3B-Instruct")
export MODEL := env("MODEL", "RedHatAI/Llama-3.3-70B-Instruct-quantized.w8a8")
export SERVED_MODEL_NAME := env("SERVED_MODEL_NAME", MODEL)
export BASE_URL := env("BASE_URL", "http://llm-d.cw4637-llm-d.coreweave.app/pd")
export RR := env("RR", "10")
export MC := env("MC", "50")
export NUM_REQUESTS := env("NUM_REQUESTS", "300")
export DATASET_PATH := env("DATASET_PATH", "vllm_openui_300.json")
export INPUT_LEN := env("INPUT_LEN", "1000")
export OUTPUT_LEN := env("OUTPUT_LEN", "250")

# Default recipe to show help
default:
    @just --list

# Setup the vllm benchmarks submodule and download required datasets
download-sharegpt:
    # Download ShareGPT dataset if not already present
    @if [ ! -f "ShareGPT_V3_unfiltered_cleaned_split.json" ]; then \
        echo "Downloading ShareGPT dataset..."; \
        curl -L https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json \
            -o ShareGPT_V3_unfiltered_cleaned_split.json; \
        echo "ShareGPT dataset downloaded."; \
    else \
        echo "ShareGPT dataset already exists. Skipping download."; \
    fi

# Install required dependencies
install-deps:
    uv sync

# Run benchmark with current settings
benchmark:
    python benchmark.py \
        --base-url {{BASE_URL}} \
        --model {{MODEL}} \
        --served-model-name {{SERVED_MODEL_NAME}} \
        --endpoint-type openai \
        --endpoint /v1/completions \
        --dataset-name random \
        --random-input-len {{INPUT_LEN}} \
        --random-output-len {{OUTPUT_LEN}} \
        --request-rate {{RR}} \
        --max-concurrency {{MC}} \
        --seed $(date +%M%H%M%S) \
        --num-prompts {{NUM_REQUESTS}} \
        --ignore-eos

# Run benchmark with ShareGPT dataset
benchmark-sharegpt:
    python benchmark.py \
        --base-url {{BASE_URL}} \
        --model {{MODEL}} \
        --served-model-name {{SERVED_MODEL_NAME}} \
        --request-rate {{RR}} \
        --max-concurrency {{MC}} \
        --seed $(date +%M%H%M%S) \
        --endpoint-type openai \
        --endpoint /v1/completions \
        --dataset-name sharegpt \
        --dataset-path {{DATASET_PATH}} \
        --num-prompts {{NUM_REQUESTS}}

# Run benchmark with basic server
benchmark-basic:
    BASE_URL="http://llm-d.cw4637-llm-d.coreweave.app/basic" SERVED_MODEL_NAME="{{MODEL}}-basic" just benchmark

# Run ShareGPT benchmark with basic server
benchmark-basic-sharegpt:
    BASE_URL="http://llm-d.cw4637-llm-d.coreweave.app/basic" SERVED_MODEL_NAME="{{MODEL}}-basic" just benchmark-sharegpt

# Run a comparison benchmark between both servers (just run both back to back)
benchmark-compare: benchmark-basic benchmark
    @echo "Benchmark comparison completed."

benchmark-sharegpt-compare: benchmark-sharegpt benchmark-basic-sharegpt
    @echo "Benchmark comparison completed."

setup-helm:
    cd ./helm
    helm plugin install https://github.com/databus23/helm-diff
    helm dependency update .

helm-upgrade-fancy *extra_args:
    helm upgrade --install -f helm/fancy.yaml test ./helm {{extra_args}}

helm-diff-fancy:
    helm diff upgrade test ./helm -f helm/fancy.yaml

helm-upgrade-basic *extra_args:
    HELM_NAMESPACE=test-inference-basic helm upgrade --install -f helm/basic.yaml basic ./helm {{extra_args}}

helm-diff-basic:
    HELM_NAMESPACE=test-inference-basic helm diff upgrade basic ./helm -f helm/basic.yaml --allow-unreleased