
# OpenAI Python API Library

## Overview

The OpenAI Python library provides convenient access to the OpenAI REST API from any Python 3.7+ application. The library includes type definitions for all request parameters and response fields, and offers both synchronous and asynchronous clients powered by httpx.

This document will guide you on how to install, use, and update the OpenAI Python library.

## Installation

To install the OpenAI library from PyPI:

```bash
pip install openai
```

For upgrading to the latest version:

```bash
pip install --upgrade openai
```

### Important
The SDK was rewritten in v1, which was released November 6th, 2023. If you are upgrading from a previous version, you should see the v1 migration guide, which includes scripts to automatically update your code.

## Usage Example

Here’s a basic usage example of how to create a client and make a request:

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

chat_completion = client.chat.completions.create(
    messages=[
        {"role": "user", "content": "Say this is a test"}
    ],
    model="gpt-3.5-turbo",
)

print(chat_completion)
```

## Vision (Image Processing)

With a hosted image:

```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"{img_url}"}},
            ],
        }
    ],
)
```

With the image as a base64 encoded string:

```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{img_type};base64,{img_b64_str}"}},
            ],
        }
    ],
)
```

## Polling Helpers

The SDK includes helper functions which will poll the status of actions until they reach a terminal state. Example:

```python
run = client.beta.threads.runs.create_and_poll(
    thread_id=thread.id,
    assistant_id=assistant.id,
)
```

## Bulk Upload Helpers

Example for uploading multiple files to vector stores:

```python
from pathlib import Path

sample_files = [Path("sample-paper.pdf"), ...]

batch = await client.vector_stores.file_batches.upload_and_poll(
    store.id,
    files=sample_files,
)
```

## Streaming Responses

Support for streaming responses using Server-Side Events (SSE):

```python
from openai import OpenAI

client = OpenAI()

stream = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Say this is a test"}],
    stream=True,
)

for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="")
```

## Async Usage

```python
import os
import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

async def main() -> None:
    chat_completion = await client.chat.completions.create(
        messages=[{"role": "user", "content": "Say this is a test"}],
        model="gpt-3.5-turbo",
    )
    print(chat_completion)

asyncio.run(main())
```

## Microsoft Azure OpenAI

To use this library with Azure OpenAI, use the `AzureOpenAI` class:

```python
from openai import AzureOpenAI

client = AzureOpenAI(
    api_version="2023-07-01-preview",
    azure_endpoint="https://example-endpoint.openai.azure.com",
)

completion = client.chat.completions.create(
    model="deployment-name",
    messages=[{"role": "user", "content": "How do I output all files in a directory using Python?"}],
)

print(completion.to_json())
```

## Error Handling

Example of handling errors when interacting with the API:

```python
from openai import APIError, APIConnectionError

try:
    client.fine_tuning.jobs.create(model="gpt-3.5-turbo", training_file="file-abc123")
except APIConnectionError as e:
    print("The server could not be reached:", e)
except APIError as e:
    print("API returned an error:", e.status_code)
```

## Retries

Certain errors are automatically retried 2 times by default, with a short exponential backoff. To configure retry settings:

```python
client.with_options(max_retries=5).chat.completions.create(
    messages=[{"role": "user", "content": "How can I get the name of the current day in Node.js?"}],
    model="gpt-3.5-turbo",
)
```

## Timeouts

By default, requests timeout after 10 minutes. You can configure this with a `timeout` option:

```python
client = OpenAI(timeout=20.0)

client.with_options(timeout=5.0).chat.completions.create(
    messages=[{"role": "user", "content": "How can I list all files in a directory using Python?"}],
    model="gpt-3.5-turbo",
)
```

## Conclusion

This guide covers the essential updates and functionality for using the OpenAI Python API library, including installation, usage examples, error handling, retries, and timeouts. For more details, refer to the [official OpenAI API documentation](https://platform.openai.com/docs).

## Model Overview

The OpenAI API is powered by a diverse set of models with different capabilities and price points. You can customize these models for specific use cases with fine-tuning.

| Model                | Description                                                                 | Context Window   | Max Output Tokens  | Training Data |
|----------------------|-----------------------------------------------------------------------------|------------------|-------------------|---------------|
| **GPT-4o**           | High-intelligence flagship model for complex, multi-step tasks               | 128,000 tokens   | 4,096 tokens      | Up to Oct 2023 |
| **GPT-4o mini**      | Affordable, lightweight model for fast tasks                                | 128,000 tokens   | 16,384 tokens     | Up to Oct 2023 |
| **o1-preview**       | Reasoning model for solving hard problems                                   | 128,000 tokens   | 32,768 tokens     | Up to Oct 2023 |
| **o1-mini**          | Faster and cheaper version of the o1-preview model                          | 128,000 tokens   | 65,536 tokens     | Up to Oct 2023 |
| **GPT-4 Turbo**      | Multimodal model with vision capabilities                                   | 128,000 tokens   | 4,096 tokens      | Up to Dec 2023 |
| **GPT-3.5 Turbo**    | Inexpensive model for simple tasks                                          | 16,385 tokens    | 4,096 tokens      | Up to Sep 2021 |
| **DALL·E**           | Image generation model from natural language descriptions                    | N/A              | N/A               | N/A           |
| **TTS**              | Converts text into natural-sounding spoken audio                            | N/A              | N/A               | N/A           |
| **Whisper**          | Speech recognition and transcription model                                  | N/A              | N/A               | N/A           |
| **Embeddings**       | Converts text into numerical form for relatedness                           | N/A              | N/A               | N/A           |
| **Moderation**       | Detects whether text may be sensitive or unsafe                             | N/A              | N/A               | N/A           |
| **GPT base**         | Non-instruction following GPT models for natural language or code generation | N/A              | N/A               | N/A           |

### Continuous Model Upgrades
Models like GPT-4o, GPT-4o-mini, and GPT-3.5 Turbo are frequently updated to reflect the latest advancements. Pinned versions are also available for continued usage for at least three months.

### GPT-4o
GPT-4o is the most advanced multimodal model, generating text faster and cheaper than previous models, with high performance across non-English languages. Available in API for paying customers.

| Model Version                   | Description                                                                 | Max Output Tokens  |
|----------------------------------|-----------------------------------------------------------------------------|--------------------|
| **gpt-4o-2024-08-06**            | Latest snapshot that supports Structured Outputs                            | 16,384 tokens      |
| **chatgpt-4o-latest**            | Continuously updated model for research and evaluation                      | 16,384 tokens      |

### GPT-4o Mini
GPT-4o Mini is an affordable multimodal model for lightweight tasks, outperforming GPT-3.5 Turbo in most cases.

| Model Version                   | Description                                                                 | Max Output Tokens  |
|----------------------------------|-----------------------------------------------------------------------------|--------------------|
| **gpt-4o-mini-2024-07-18**       | Latest snapshot of GPT-4o Mini                                              | 16,384 tokens      |

### o1-preview and o1-mini
The o1 series models are specialized in complex reasoning tasks, producing a long internal chain of thought before responding.

| Model Version                   | Description                                                                 | Max Output Tokens  |
|----------------------------------|-----------------------------------------------------------------------------|--------------------|
| **o1-preview-2024-09-12**        | Latest o1-preview model snapshot                                            | 32,768 tokens      |
| **o1-mini-2024-09-12**           | Latest o1-mini model snapshot                                               | 65,536 tokens      |

### Data Usage Policy
OpenAI's data policy ensures that data sent to the API will not be used for model training or improvement unless explicitly opted in.

| Endpoint                             | Data used for training | Default retention | Eligible for zero retention |
|--------------------------------------|------------------------|-------------------|-----------------------------|
| **/v1/chat/completions**             | No                     | 30 days           | Yes                         |
| **/v1/assistants**                   | No                     | 30 days **        | No                          |
| **/v1/threads**                      | No                     | 30 days **        | No                          |
| **/v1/vector_stores**                | No                     | 30 days **        | No                          |
| **/v1/audio/transcriptions**         | No                     | Zero retention    | -                           |
| **/v1/audio/speech**                 | No                     | 30 days           | Yes                         |

For more details on OpenAI's data usage policy, refer to the official documentation.
