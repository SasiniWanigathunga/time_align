from whisper import transcribe,load_model
import whisper
import torch
from datasets import load_dataset
import torch
from scipy.io import wavfile

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.random.manual_seed(0)

ter = 0
count = 0

ds = load_dataset("asapp/slue-phase-2", "vp_nel", split = 'test')

model_size = "large-v2"
audio_path = "audio.wav"
language = "en"
task = "transcribe"
initial_prompt = ""

transcribe_args={
    'task': task, 'language': language, 'patience': None,
    'length_penalty': None, 'suppress_tokens': '-1',
    'initial_prompt': initial_prompt, 'fp16': False,
    'condition_on_previous_text':True,'word_timestamps':True,
    }

if torch.cuda.is_available():
    model_transcribe = whisper.load_model(model_size).cuda().eval()
else:
    model_transcribe = whisper.load_model(model_size)

for i in range(len(ds)):
    sample = ds[i]
    original_audio = sample['audio']
    text = sample['text']
    word_timestamps = sample['word_timestamps']

    words = []
    starts = []
    ends = []

    for i in range(len(word_timestamps['word'])):
        if word_timestamps['word'][i] != "":
            words.append(word_timestamps['word'][i].lower())
            starts.append(word_timestamps['start_sec'][i])
            ends.append(word_timestamps['end_sec'][i])
    print(words)

    output_path = f"./audio_files/audio_{i}.wav"
    print(output_path)
    wavfile.write(output_path, original_audio['sampling_rate'], original_audio['array'])

    text = whisper.transcribe(model_transcribe,output_path,**transcribe_args)
    print(text)

    break

#     for i in range(len(transcript.split("|"))):
#         w, s, e = display_segment(i)

#         if w in words:
#             index = words.index(w)
#             ter += (starts[index] - s)**2 + (ends[index] - e)**2
#             print(starts[index], s)
#             print(ends[index], e)
#             count += 2
#             # remove the word for index to avoid double counting
#             words.pop(index)
#             starts.pop(index)   
#             ends.pop(index)

# # rms
# print((ter/count)**0.5)

# # 0.17103070008113913 validation
# # 0.10732642767195283 test