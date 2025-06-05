#!/usr/bin/env python3
"""
Wrapper around vLLM's serve benchmarking script to instrument it
with Weave EvaluationLogger for detailed per-request statistics and overall summary.
"""
import argparse
import json
import os
import random
import logging
import numpy as np

import weave
from weave import EvaluationLogger  # ensure weave in PYTHONPATH
import vllm.benchmarks.serve as bench_serve

# Keep reference to original request function
original_request_func = None
# Global EvaluationLogger instance
eval_logger: EvaluationLogger

async def instrumented_request_func(request_func_input, pbar=None):
    """
    Wrap the original request_func to log detailed metrics per request.
    """
    # Begin logging this prediction

    # Execute the real request
    output = await original_request_func(request_func_input=request_func_input, pbar=pbar)
    pred_logger = eval_logger.log_prediction(
        inputs={
            "prompt": request_func_input.prompt,
        },
        output=output.generated_text
    )

    """RequestFuncOutput:
    generated_text: str = ""
    success: bool = False
    latency: float = 0.0
    output_tokens: int = 0
    ttft: float = 0.0  # Time to first token
    itl: list[float] = field(
        default_factory=list)  # list of inter-token latencies
    tpot: float = 0.0  # avg next-token latencies
    prompt_len: int = 0
    error: str = ""
    """
    if output.error != "":
        print(f"!!! Error: {output.error}")
    try:
        # Log success flag
        pred_logger.log_score(scorer="success", score=bool(output.success))
        # End-to-end latency (client send to full response)
        pred_logger.log_score(scorer="latency_ms", score=output.latency * 1000)
        # Time to first token
        pred_logger.log_score(scorer="ttft_ms", score=output.ttft * 1000)
        # Tokens generated
        token_count = getattr(output, "output_tokens", None)
        if token_count is not None:
            # Time per output token excluding first token
            if token_count > 1:
                tpot_ms = (output.latency - output.ttft) / (token_count - 1) * 1000
                pred_logger.log_score(scorer="tpot_ms", score=tpot_ms)
        # Mean inter-token latency
        if output.itl:
            mean_itl_ms = float(np.mean(output.itl)) * 1000
            pred_logger.log_score(scorer="itl_ms", score=mean_itl_ms)
    except Exception as e:
        logging.exception(f"Error logging prediction: {e}")
    finally:
        pred_logger.finish()

    return output


def main():
    parser = argparse.ArgumentParser(
        prog="serve_bench_wrapped",
        description="vLLM bench serve with Weave EvaluationLogger instrumentation"
    )
    # Reuse original CLI args
    bench_serve.add_cli_args(parser)
    args = parser.parse_args()

    # Seed RNGs
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Initialize EvaluationLogger
    client = weave.init("vllm-benchmark")
    global eval_logger, original_request_func
    model_name = args.model.split("/")[-1].replace("-", "_").replace(".", "_")
    eval_logger = EvaluationLogger(
        model=model_name,
        dataset=args.dataset_name,
    )

    # Monkey-patch the request function
    # TODO: maybe wrap in a weave.op?  Got recursion error
    global original_request_func # Ensure we are modifying the global
    original_request_func = bench_serve.ASYNC_REQUEST_FUNCS[args.endpoint_type]
    bench_serve.ASYNC_REQUEST_FUNCS[args.endpoint_type] = instrumented_request_func

    output_path = "results.json"
    args.save_result = True
    args.result_filename = output_path

    # Remove previous results
    if os.path.exists(output_path):
        os.remove(output_path)

    print(f"Benchmarking {args.served_model_name}")
    # Run the benchmark and capture summary metrics
    bench_serve.main(args)

    # Log overall summary stats
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            result_json = json.load(f)
        eval_logger.log_summary(result_json)
        print("Logged summary metrics to EvaluationLogger.")
    else:
        # Fallback: just flush summary without details
        eval_logger.log_summary()
        print("Logged summary (no metrics returned) to EvaluationLogger.")

    print("Evaluation logging complete. View results in the Weave UI.")
    client.finish()


if __name__ == "__main__":
    main()
