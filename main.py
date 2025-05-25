import argparse
import time
import random
import openai
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import string

# Dummy tool implementations
def dummy_tool_1(input_text):
    # Simulate processing
    return f"Result of tool1 on '{input_text}'"

def dummy_tool_2(input_text):
    return f"Result of tool2 on '{input_text}'"

# Define function schema for OpenAI tool calling
FUNCTIONS = [
    {
        "name": "dummy_tool_1",
        "description": "Performs dummy processing A",
        "parameters": {
            "type": "object",
            "properties": {"input_text": {"type": "string"}},
            "required": ["input_text"],
        }
    },
    {
        "name": "dummy_tool_2",
        "description": "Performs dummy processing B",
        "parameters": {
            "type": "object",
            "properties": {"input_text": {"type": "string"}},
            "required": ["input_text"],
        }
    }
]

# Generate dummy system prompts
SYSTEM_PROMPTS = [
    "You are a helpful assistant that summarizes conversation.",
    "You are an analytical agent that processes tool calls and returns summaries.",
]

QUESTIONS = [
    "How do I put Cursor into Agent Mode?",
    "What are the coolest features of Cursor?",
]

basic_client = openai.OpenAI(api_key="sk-proj-1234567890", base_url="http://llm-d.cw4637-llm-d.coreweave.app/basic/v1")
fancy_client = openai.OpenAI(api_key="sk-proj-1234567890", base_url="http://llm-d.cw4637-llm-d.coreweave.app/pd/v1")

def load_llm_context(path="cursor-llm.txt"):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def mutate_context(context):
    # Mutate the first line by inserting a random string at the start
    lines = context.splitlines()
    if not lines:
        return context
    rand_str = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
    lines[0] = f"[MUTATED-{rand_str}] " + lines[0]
    return '\n'.join(lines)

def make_request(client, payload, n_requests, mutate_prob=0.5):
    timings = []
    cacheable_flags = []
    error_flags = []
    for _ in range(n_requests):
        is_cacheable = random.random() > mutate_prob
        if is_cacheable:
            context = payload
        else:
            context = mutate_context(payload)
        system_prompt = random.choice(SYSTEM_PROMPTS)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": context+"\n"+random.choice(QUESTIONS)},
        ]
        start = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model="meta-llama/Llama-3.2-3B-Instruct",
                messages=messages,
                functions=FUNCTIONS,
                function_call="auto"
            )
            msg = response.choices[0].message
            if msg.function_call:
                fn_name = msg.function_call.name
                args = json.loads(msg.function_call.arguments) if msg.function_call.arguments else {}
                if fn_name == "dummy_tool_1":
                    fn_response = dummy_tool_1(args.get("input_text", ""))
                elif fn_name == "dummy_tool_2":
                    fn_response = dummy_tool_2(args.get("input_text", ""))
                else:
                    fn_response = ""
                followup = client.chat.completions.create(
                    model="meta-llama/Llama-3.2-3B-Instruct",
                    messages=[
                        *messages,
                        {
                            "role": "assistant",
                            "content": None,
                            "function_call": {
                                "name": fn_name,
                                "arguments": msg.function_call.arguments
                            }
                        },
                        {"role": "function", "name": fn_name, "content": fn_response}
                    ]
                )
            elapsed = time.perf_counter() - start
            timings.append(elapsed)
            cacheable_flags.append(is_cacheable)
            error_flags.append(False)
        except Exception as e:
            print(f"Error: {e}")
            elapsed = time.perf_counter() - start
            timings.append(elapsed)
            cacheable_flags.append(is_cacheable)
            error_flags.append(True)
    return timings, cacheable_flags, error_flags

def main():
    parser = argparse.ArgumentParser(description="Benchmark using OpenAI SDK with tool calling.")
    parser.add_argument(
        "--threads", type=int, default=4,
        help="Number of concurrent threads"
    )
    parser.add_argument(
        "--requests", type=int, default=20,
        help="Total requests per model"
    )
    args = parser.parse_args()

    # Load long context from llm.txt
    payload = load_llm_context("cursor-llm.txt")

    results = {}
    for client in [basic_client, fancy_client]:
        print(f"\nBenchmarking model: {client.base_url}")
        n = args.requests
        pool = ThreadPoolExecutor(max_workers=args.threads)
        futures = [
            pool.submit(make_request, client, payload, n // args.threads)
            for _ in range(args.threads)
        ]
        all_times = []
        all_cacheable = []
        all_errors = []
        for fut in as_completed(futures):
            times, cacheable_flags, error_flags = fut.result()
            all_times.extend(times)
            all_cacheable.extend(cacheable_flags)
            all_errors.extend(error_flags)
        pool.shutdown()

        # Split timings and errors by cacheable/uncacheable
        cacheable_times = [t for t, c in zip(all_times, all_cacheable) if c]
        uncacheable_times = [t for t, c in zip(all_times, all_cacheable) if not c]
        cacheable_errors = [e for e, c in zip(all_errors, all_cacheable) if c]
        uncacheable_errors = [e for e, c in zip(all_errors, all_cacheable) if not c]
        def stats(times, errors):
            count = len(times)
            err_count = sum(errors)
            ok_count = count - err_count
            ok_times = [t for t, e in zip(times, errors) if not e]
            avg = sum(ok_times) / ok_count if ok_count else 0
            mn = min(ok_times) if ok_times else 0
            mx = max(ok_times) if ok_times else 0
            return count, ok_count, err_count, avg, mn, mx
        c_count, c_ok, c_err, c_avg, c_min, c_max = stats(cacheable_times, cacheable_errors)
        u_count, u_ok, u_err, u_avg, u_min, u_max = stats(uncacheable_times, uncacheable_errors)
        print(f"  Cacheable:   Calls: {c_count}, Success: {c_ok}, Errors: {c_err}, Avg: {c_avg:.3f}s, Min: {c_min:.3f}s, Max: {c_max:.3f}s")
        print(f"  Uncacheable: Calls: {u_count}, Success: {u_ok}, Errors: {u_err}, Avg: {u_avg:.3f}s, Min: {u_min:.3f}s, Max: {u_max:.3f}s")
        results[client.base_url] = {
            "cacheable": (c_count, c_ok, c_err, c_avg, c_min, c_max),
            "uncacheable": (u_count, u_ok, u_err, u_avg, u_min, u_max)
        }

    print("\nSummary:")
    for model_url, stats_dict in results.items():
        c = stats_dict["cacheable"]
        u = stats_dict["uncacheable"]
        print(f"- {model_url}: cacheable avg {c[3]:.3f}s ({c[1]}/{c[0]} success, {c[2]} errors), uncacheable avg {u[3]:.3f}s ({u[1]}/{u[0]} success, {u[2]} errors)")

if __name__ == "__main__":
    main()
