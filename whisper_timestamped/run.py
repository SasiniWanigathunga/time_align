import whisper_timestamped 
from datasets import load_dataset
from scipy.io import wavfile

model = whisper_timestamped.load_model("large", device="cuda")

ter = 0
count = 0

ds = load_dataset("asapp/slue-phase-2", "vp_nel", split = 'test')

for i in range(len(ds)):
    sample = ds[i]
    original_audio = sample['audio']
    text = sample['text']
    word_timestamps = sample['word_timestamps']
    start = word_timestamps['start_sec'][0]
    end = word_timestamps['end_sec'][-1]

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
    audio = whisper_timestamped.load_audio(output_path)
    results = whisper_timestamped.transcribe(model, audio, temperature = 1.0, language="en", detect_disfluencies=True)
    for k in range(len(results['segments'])):
        for j in range(len(results['segments'][k]['words'])):
            w, s, e = results['segments'][k]['words'][j]['text'], results['segments'][k]['words'][j]['start'], results['segments'][k]['words'][j]['end']
            print(w, words)
            if w in words:
                index = words.index(w)
                ter += (starts[index] - s)**2 + (ends[index] - e)**2
                print(starts[index], s)
                print(ends[index], e)
                count += 2
                # remove the word for index to avoid double counting
                words.pop(index)
                starts.pop(index)   
                ends.pop(index)

# rms
print((ter/count)**0.5)

# 0.41261733037809845 test
# 0.4464182927739309 validation