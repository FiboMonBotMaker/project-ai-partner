import whisper
import json
import time
import torch

load_time = time.time()

model = whisper.load_model("medium", device="cpu")

load_end = time.time() - load_time

_ = model.half()
_ = model.cuda()

for m in model.modules():
    if isinstance(m, whisper.model.LayerNorm):
        m.float()

ts_time = time.time()

result = model.transcribe("./test_voice/sampleSuper.mp3", verbose=True, fp16=False, language="ja")

ts_end = time.time() - ts_time

print(result['text'])
print(f"Load model time:{round(load_end, 3)}s -- Transcribe time: {round(ts_end, 3)}s")

f = open('transcription-long.txt', 'w', encoding='UTF-8')
f.write(json.dumps(result['text'], sort_keys=True, indent=4, ensure_ascii=False))
f.close()
