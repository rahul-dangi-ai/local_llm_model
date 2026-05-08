# LLM STREAMING SERVER (FASTAPI + LLAMA)

- Author  : Rahul Dangi
- Type    : Local LLM Inference Server
- Mode    : Streaming (SSE - Server Sent Events)

--------------------------------------------------------------
OVERVIEW
--------------------------------------------------------------

A high-performance FastAPI server that serves a local LLM
(LLaMA 3.2 Instruct) with real-time token streaming.

Designed for:
  - RAG pipelines
  - Chatbots
  - Voice assistants
  - Low-latency applications

--------------------------------------------------------------
FEATURES
--------------------------------------------------------------

  * ⚡ Token-by-token streaming (SSE)
  * 🧠 HuggingFace Transformers backend
  * 💻 GPU auto-detection (device_map="auto")
  * 🔁 Sliding window prompt trimming
  * 🛑 Interrupt support (client disconnect)
  * 🧵 Threaded generation (non-blocking)

--------------------------------------------------------------
MODEL
--------------------------------------------------------------
```
  Model:
    LLaMA-3.2-1B-Instruct

  Path:
    .\Llama-3.2-1B-Instruct

  Precision:
    bfloat16
```
--------------------------------------------------------------
REQUIREMENTS
--------------------------------------------------------------
```
  Python 3.9+
```
Install dependencies:
```
  pip install fastapi uvicorn torch transformers
```
--------------------------------------------------------------
RUN SERVER [NON-DOCKER]
--------------------------------------------------------------

  Start the API:
```
    uvicorn main:app --host 0.0.0.0 --port 8000
```
  Output:

    Server running at http://localhost:8000

--------------------------------------------------------------
RUN SERVER [DOCKER]
--------------------------------------------------------------
## Build Container

```bash
docker compose build
```

---

## Start Server

```bash
docker compose up -d
```

---

## Build with docker compose

```bash
docker compose up -d --build
```

## Stop Server

```bash
docker compose down
```

---

# NVIDIA Docker Support

Install NVIDIA Container Toolkit:

[NVIDIA Container Toolkit Docs](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html?utm_source=chatgpt.com)

Verify GPU access:

```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-runtime-ubuntu22.04 nvidia-smi
```



--------------------------------------------------------------
ENDPOINTS
--------------------------------------------------------------
```
  [POST] /generate
    Streams generated tokens
```
--------------------------------------------------------------
REQUEST FORMAT
--------------------------------------------------------------
```
  POST /generate

  {
    "messages": [
      {"role": "system", "content": "You are helpful"},
      {"role": "user", "content": "Explain AI simply"}
    ],
    "max_new_tokens": 256,
    "temperature": 0.7
  }
```
--------------------------------------------------------------
STREAMING RESPONSE (SSE)
--------------------------------------------------------------
```
  data: {"token": "Artificial"}
  
  data: {"token": " intelligence"}
  
  data: {"token": " is"}
  
  ...

  (ends when generation completes or client disconnects)
```
--------------------------------------------------------------
CURL EXAMPLES
--------------------------------------------------------------

  1. Basic streaming request:

    curl -N -X POST http://localhost:8000/generate \
         -H "Content-Type: application/json" \
         -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}]}"

--------------------------------------------------------------

  2. With full parameters:

    curl -N -X POST http://localhost:8000/generate \
         -H "Content-Type: application/json" \
         -d "{
               \"messages\": [
                 {\"role\":\"system\",\"content\":\"You are helpful\"},
                 {\"role\":\"user\",\"content\":\"Explain transformers\"}
               ],
               \"max_new_tokens\": 100,
               \"temperature\": 0.5
             }"

--------------------------------------------------------------

  3. Pretty stream (Linux/macOS):

    curl -N http://localhost:8000/generate \
         -H "Content-Type: application/json" \
         -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}]}" \
    | while read line; do echo $line; done

--------------------------------------------------------------
ARCHITECTURE
--------------------------------------------------------------
```
  Client Request
        |
        v
  FastAPI Endpoint (/generate)
        |
        v
  Prompt Builder (chat template + trimming)
        |
        v
  Tokenizer → Model.generate()
        |
        v
  TextIteratorStreamer
        |
        v
  SSE Output (data: {token})
        |
        v
  Client (real-time tokens)
```
--------------------------------------------------------------
KEY COMPONENTS
--------------------------------------------------------------
```
  TextIteratorStreamer
    → streams tokens as they are generated

  StoppingCriteria
    → allows interruption when client disconnects

  Threaded Generation
    → prevents blocking main async loop
```
--------------------------------------------------------------
INTERRUPT HANDLING
--------------------------------------------------------------

  If client disconnects:

    ⚠️ Client disconnected → stopping generation

  → Model generation stops immediately
  → Saves compute resources

--------------------------------------------------------------
CONFIGURATION
--------------------------------------------------------------
```
  model_id        = local model path
  max_new_tokens  = response length
  temperature     = randomness
  max_chars       = 6000 (prompt window)
```
--------------------------------------------------------------
NOTES
--------------------------------------------------------------

  - Uses chat template from tokenizer
  - Automatically trims long conversations
  - Optimized for local deployment
  - Works with your RAG + Voice pipeline

--------------------------------------------------------------
TROUBLESHOOTING
--------------------------------------------------------------
```
  CUDA not used:
    → Check torch.cuda.is_available()

  Slow inference:
    → Use GPU or smaller model

  No streaming:
    → Ensure curl uses -N flag
```

## Slow Inference

- Use GPU acceleration
- Reduce `max_new_tokens`
- Use smaller models
- Enable quantization

==============================================================