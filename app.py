################## STREAMING + INTERPUT #############
import torch
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer,
    StoppingCriteria,
    StoppingCriteriaList
)
from threading import Thread

model_id = r".\Llama-3.2-1B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

app = FastAPI()

# ----------------------------
# Request schema
# ----------------------------
class ChatRequest(BaseModel):
    messages: list
    max_new_tokens: int = 256
    temperature: float = 0.7

# ----------------------------
# Sliding window
# ----------------------------
def build_prompt(messages, max_chars=6000):
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    return text[-max_chars:]

# ----------------------------
# STOPPING CRITERIA
# ----------------------------
class StopOnFlag(StoppingCriteria):
    def __init__(self, stop_flag):
        self.stop_flag = stop_flag

    def __call__(self, input_ids, scores, **kwargs):
        return self.stop_flag["stop"]

# ----------------------------
# STREAM ENDPOINT
# ----------------------------
@app.post("/generate")
async def generate(req: ChatRequest, request: Request):

    prompt = build_prompt(req.messages)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )

    stop_flag = {"stop": False}
    stopping_criteria = StoppingCriteriaList([StopOnFlag(stop_flag)])

    def generate_thread():
        try:
            model.generate(
                **inputs,
                streamer=streamer,
                max_new_tokens=req.max_new_tokens,
                temperature=req.temperature,
                do_sample=True,
                use_cache=True,
                stopping_criteria=stopping_criteria
            )
        except Exception as e:
            print("Generation stopped:", e)

    thread = Thread(target=generate_thread)
    thread.start()

    async def event_stream():
        try:
            for token in streamer:
                if await request.is_disconnected():
                    print("⚠️ Client disconnected → stopping generation")
                    stop_flag["stop"] = True
                    break
                yield f"data: {json.dumps({'token': token})}\n\n"

        finally:
            stop_flag["stop"] = True

    return StreamingResponse(event_stream(), media_type="text/event-stream")