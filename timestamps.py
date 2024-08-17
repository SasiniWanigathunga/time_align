import json

ter = 0
count = 0
 
def read_manifest(manifest_filepath):
    manifest_data = []
    with open(manifest_filepath, 'r') as f:
        for line in f:
            manifest_data.append(json.loads(line))
    return manifest_data
 
def extract_timestamps_with_names_from_ctm(ctm_file):
    timestamps_with_names = []
    with open(ctm_file, 'r') as f:
        for line in f:
            parts = line.split()  # Split line by whitespace
            if len(parts) >= 5:  # Ensure the line has at least 5 parts
                name = parts[4]  # Extract the name (assuming it's the fifth part)
                start_time = float(parts[2])  # Extract the start time (assuming it's the third part)
                end_time = float(parts[3])
                timestamps_with_names.append((start_time, end_time, name))
    return timestamps_with_names
 
def print_original_audio_path_tokens_and_timestamps(manifest_data):
    for item in range(len(manifest_data)):
        audio_filepath = manifest_data[item]['audio_filepath']
        if 'words_level_ctm_filepath' in manifest_data[item] and 'words_level_ass_filepath' in manifest_data[item]:
            audio.append(str(audio_filepath)) 
            global x
            global ter
            global count
            x+=1
            ctm_file = manifest_data[item]['words_level_ctm_filepath']
            print(ctm_file)
            ass_file = manifest_data[item]['words_level_ass_filepath']
            print("Original Audio Path:", audio_filepath)
            print(manifest_data[item]['text'])
            tokens_with_timestamps = extract_timestamps_with_names_from_ctm(ctm_file)
            print(tokens_with_timestamps)
            
            words = dataset_words[item]
            starts = dataset_starts[item]
            ends = dataset_ends[item]
            for s, e, w in tokens_with_timestamps:
                if w in words:
                    i = words.index(w)
                    ter += (starts[i] - s)**2 + (ends[i] - (s+e))**2
                    count += 2
                    # remove the word for index to avoid double counting
                    words.pop(i)
                    starts.pop(i)   
                    ends.pop(i)

x=0
manifest_filepath = "nfa_output\manifest_with_output_file_paths.json"
manifest_data = read_manifest(manifest_filepath)
print(len(manifest_data))
 
audio = []
dataset_words = []
dataset_starts = []
dataset_ends = []

with open('manifest.json', 'r') as f:
    lines = f.readlines()

for line in lines:
    dataset_words.append(json.loads(line)['words'])
    dataset_starts.append(json.loads(line)['starts'])
    dataset_ends.append(json.loads(line)['ends'])
 
print_original_audio_path_tokens_and_timestamps(manifest_data)
print(x)

# rms
print((ter/count)**0.5)