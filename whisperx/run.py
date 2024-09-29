import whisperx
from datasets import load_dataset
from scipy.io import wavfile

device = "cpu" 

ter = 0
count = 0

ds = load_dataset("asapp/slue-phase-2", "vp_nel", split = 'validation')
model_a, metadata = whisperx.load_align_model(language_code='en', device=device)

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
    audio = whisperx.load_audio(output_path)

    result = {'segments': [{'text': text, 'start': start, 'end': end}], 'language': 'en'}
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    timestamps = result["word_segments"]
    print(timestamps)

    for i in range(len(timestamps)):
        w, s, e = timestamps[i]['word'], timestamps[i]['start'], timestamps[i]['end']

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