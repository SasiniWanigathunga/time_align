# import json
# import os

# text = "If you do not address this problem, the ground is there for populist nationalist forces to go on growing all over Europe."

# manifest_filepath = f"./manifest.json"
# manifest_data = {
#     "audio_filepath": f"./audio.wav",
#     "text": text
# }
# with open(manifest_filepath, 'w') as f:
#   line = json.dumps(manifest_data)
#   f.write(line + "\n")

# run this then
# python NeMo/tools/nemo_forced_aligner/align.py pretrained_name="stt_en_fastconformer_hybrid_large_pc" manifest_filepath=manifest.json output_dir=nfa_output/ additional_segment_grouping_separator="|" ass_file_config.vertical_alignment="bottom" ass_file_config.text_already_spoken_rgb=[66,245,212] ass_file_config.text_being_spoken_rgb=[242,222,44] ass_file_config.text_not_yet_spoken_rgb=[223,242,239]

import json
from datasets import load_dataset
from scipy.io import wavfile

ds = load_dataset("asapp/slue-phase-2", "vp_nel", split = 'validation') # validation/ test

with open('manifest.json', 'w',encoding = 'utf-8') as f:
   for i in range(len(ds)):
      row = []
      sample = ds[i]
      original_audio = sample['audio']
      print(original_audio)
      text = sample['text']

      output_path = f"./audio_files/audio_{i}.wav"
      print(output_path)

      wavfile.write(output_path, original_audio['sampling_rate'], original_audio['array'])

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

      manifest_data = {
      "audio_filepath": output_path,
      "text": text,
      'words': words,
      'starts': starts,
      'ends': ends
      }
      
      line = json.dumps(manifest_data)
      f.write(line + "\n")
