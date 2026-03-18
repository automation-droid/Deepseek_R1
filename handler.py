import runpod
from vllm import LLM, SamplingParams

# Load model once (cold start)
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

llm = LLM(
    model=MODEL_NAME,
    dtype="auto",
    max_model_len=32768
)

def handler(event):
    try:
        input_data = event["input"]

        prompt = input_data.get("prompt", "Hello")
        max_tokens = input_data.get("max_tokens", 200)
        temperature = input_data.get("temperature", 0.7)

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature
        )

        outputs = llm.generate(prompt, sampling_params)

        text = outputs[0].outputs[0].text

        return {
            "output": text
        }

    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
