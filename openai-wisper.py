import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, WhisperProcessor
from datasets import load_dataset
import time
import os
from openai import OpenAI
from dotenv import load_dotenv
import sys

print(sys.argv[0])
print(sys.argv[1])

load_dotenv()

audio_path = sys.argv[1]
# audio_path = "/Users/chenpin-han/Desktop/output.flac"

# Record the start time
start_time = time.time()

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# model_id = "openai/whisper-large-v3"
model_id = "openai/whisper-medium"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)

model.to(device)

processor = WhisperProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps="word",
    torch_dtype=torch_dtype,
    device=device,
    generate_kwargs={"language": "zh"},
)

result = pipe(audio_path)
print(result)
print(result["text"])

# Record the end time
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time

# Print the elapsed time
print("Elapsed time:", elapsed_time, "seconds")

# client = OpenAI(
#     # This is the default and can be omitted
#     api_key=os.environ.get("OPENAI_API_KEY"),
# )

# chat_completion = client.chat.completions.create(
#     messages=[
#         {
#             "role": "user",
#             "content": result["text"] + "\n條列式寫出重點摘要",
#         }
#     ],
#     model="gpt-3.5-turbo",
# )

# print(chat_completion)
# items = chat_completion.choices[0].message.content.split("\n")
# for item in items:
#     print(item)
#     if item.startswith("摘要"):
#         print(item[4:])
#         break