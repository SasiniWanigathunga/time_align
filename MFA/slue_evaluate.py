from datasets import load_dataset
import torch
import torchaudio
from dataclasses import dataclass
import soundfile as sf
import numpy as np
import re

# # stage 1: save the audio and text to a file
# audio_id = 0
# ds = load_dataset("asapp/slue-phase-2", "vp_nel", split = 'test')
# for i in range(len(ds)):
#     sample = ds[i]
#     original_audio = sample['audio']
#     text = sample['text']
#     audio_id += 1
#     path = '/home/sasini/Documents/input_english/'
#     audio_path = path + 'audio_' + str(audio_id) + '.wav'
#     sf.write(audio_path, original_audio['array'], original_audio['sampling_rate'])
#     text_path = path + 'audio_' + str(audio_id) + '.txt'
#     with open(text_path, 'w') as f:
#         f.write(text)

# stage 2: calculate the TER
path = '/home/sasini/Documents/'
ds = load_dataset("asapp/slue-phase-2", "vp_nel", split = 'test')
ter = 0
count = 0
for i in range(len(ds)):
    word_timestamps = ds[i]['word_timestamps']
    target_tokens_temp = word_timestamps['word']
    target_start_seconds_temp = word_timestamps['start_sec']
    target_end_seconds_temp = word_timestamps['end_sec']
    target_tokens = []
    target_start_seconds = []
    target_end_seconds = []

    for j in range(len(target_tokens_temp)):
        if target_tokens_temp[j] != "" and target_tokens_temp[j] != " ":
            if target_tokens_temp[j] == "'s":
                target_tokens[-1] = target_tokens[-1] + target_tokens_temp[j]
                target_end_seconds[-1] = target_end_seconds_temp[j]
            else:
                target_tokens.append(target_tokens_temp[j])
                target_start_seconds.append(target_start_seconds_temp[j])
                target_end_seconds.append(target_end_seconds_temp[j])

    text_file_path = path + f'input_english/audio_{i+1}.txt'  
    textgrid_file_path = path + f'output_english/audio_{i+1}.TextGrid'

    with open(text_file_path, "r", encoding="utf-8") as text_file:
        transcription = text_file.read().strip()

    try:
        with open(textgrid_file_path, "r", encoding="utf-8") as textgrid_file:
            textgrid_content = textgrid_file.read()
    except FileNotFoundError:
        print(f"Error: TextGrid file not found for audio_{i+1}.")
        with open(f'error_{i+1}.txt', 'w') as f:
            f.write("TextGrid file not found for audio_{i+1}.")
        continue

    # Regular expression to extract intervals with non-empty text from the "words" tier
    pattern = re.compile(r'xmin = ([0-9.]+)\s+xmax = ([0-9.]+)\s+text = "(.*?)"')

    # Extract tokens, start seconds, and end seconds
    tokens = []
    start_seconds = []
    end_seconds = []
    
    for match in pattern.finditer(textgrid_content):
        xmin, xmax, text = match.groups()
        if text.strip():  # Ignore empty texts
            tokens.append(text.strip())
            start_seconds.append(float(xmin))
            end_seconds.append(float(xmax))
            # Stop if all expected tokens are matched
            if len(tokens) == len(target_tokens):
                break

    # Verify if all expected tokens are matched
    if len(tokens) < len(target_tokens):
        print("Warning: Not all expected tokens were found in the TextGrid file.")

    # Calculate the TER
    if tokens != target_tokens:
        print(tokens)
        print(target_tokens)
        print(i+1, "Error: The number of tokens in the TextGrid file does not match the number of tokens in the target.")
        # save tokens, start_seconds, end_seconds, target_tokens, target_start_seconds, target_end_seconds to a file
        with open(f'error_{i+1}.txt', 'w') as f:
            f.write("Tokens: " + str(tokens) + "\n")
            f.write("Start Seconds: " + str(start_seconds) + "\n")
            f.write("End Seconds: " + str(end_seconds) + "\n")
            f.write("Target Tokens: " + str(target_tokens) + "\n")
            f.write("Target Start Seconds: " + str(target_start_seconds) + "\n")
            f.write("Target End Seconds: " + str(target_end_seconds) + "\n")
        continue
    for k in range(len(tokens)):
        ter += (start_seconds[k] - target_start_seconds[k])**2 + (end_seconds[k] - target_end_seconds[k])**2
        count += 2
    
# rms
print((ter/count)**0.5)
print(count)

# validation = 0.34285611890741474
# test = 0.31748279167616233