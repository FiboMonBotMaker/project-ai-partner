from io import BytesIO

import numpy as np
import soundfile as sf
import speech_recognition as sr
import whisper

r = sr.Recognizer()

r.pause_threshold = 0.5
with sr.Microphone(sample_rate=16_000) as source:
    print("なにか話してください")
    audio = r.listen(source)

    print("音声処理中 ...")
    wav_bytes = audio.get_wav_data()
    wav_stream = BytesIO(wav_bytes)
    audio_array, sampling_rate = sf.read(wav_stream)
    audio_fp32 = audio_array.astype(np.float32)
    model = whisper.load_model("medium", device="cpu")

    _ = model.half()
    _ = model.cuda()

    for m in model.modules():
        if isinstance(m, whisper.model.LayerNorm):
            m.float()

    result = model.transcribe(audio_fp32, verbose=True, fp16=False, language="ja")
    print(result['text'])
