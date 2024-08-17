from datasets import load_dataset
import torch
import torchaudio
from dataclasses import dataclass
import IPython
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.random.manual_seed(0)

bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
model = bundle.get_model().to(device)
labels = bundle.get_labels()

ter = 0
count = 0

ds = load_dataset("asapp/slue-phase-2", "vp_nel", split = 'validation')

def get_trellis(emission, tokens, blank_id=0):
    num_frame = emission.size(0)
    num_tokens = len(tokens)

    trellis = torch.zeros((num_frame, num_tokens))
    trellis[1:, 0] = torch.cumsum(emission[1:, blank_id], 0)
    trellis[0, 1:] = -float("inf")
    trellis[-num_tokens + 1 :, 0] = float("inf")

    for t in range(num_frame - 1):
        trellis[t + 1, 1:] = torch.maximum(
            # Score for staying at the same token
            trellis[t, 1:] + emission[t, blank_id],
            # Score for changing to the next token
            trellis[t, :-1] + emission[t, tokens[1:]],
        )
    return trellis

@dataclass
class Point:
    token_index: int
    time_index: int
    score: float


def backtrack(trellis, emission, tokens, blank_id=0):
    t, j = trellis.size(0) - 1, trellis.size(1) - 1

    path = [Point(j, t, emission[t, blank_id].exp().item())]
    while j > 0:
        # Should not happen but just in case
        assert t > 0

        # 1. Figure out if the current position was stay or change
        # Frame-wise score of stay vs change
        p_stay = emission[t - 1, blank_id]
        p_change = emission[t - 1, tokens[j]]

        # Context-aware score for stay vs change
        stayed = trellis[t - 1, j] + p_stay
        changed = trellis[t - 1, j - 1] + p_change

        # Update position
        t -= 1
        if changed > stayed:
            j -= 1

        # Store the path with frame-wise probability.
        prob = (p_change if changed > stayed else p_stay).exp().item()
        path.append(Point(j, t, prob))

    # Now j == 0, which means, it reached the SoS.
    # Fill up the rest for the sake of visualization
    while t > 0:
        prob = emission[t - 1, blank_id].exp().item()
        path.append(Point(j, t - 1, prob))
        t -= 1

    return path[::-1]

@dataclass
class Segment:
    label: str
    start: int
    end: int
    score: float

    def __repr__(self):
        return f"{self.label}\t({self.score:4.2f}): [{self.start:5d}, {self.end:5d})"

    @property
    def length(self):
        return self.end - self.start

def merge_repeats(path):
    i1, i2 = 0, 0
    segments = []
    while i1 < len(path):
        while i2 < len(path) and path[i1].token_index == path[i2].token_index:
            i2 += 1
        score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
        segments.append(
            Segment(
                transcript[path[i1].token_index],
                path[i1].time_index,
                path[i2 - 1].time_index + 1,
                score,
            )
        )
        i1 = i2
    return segments

def merge_words(segments, separator="|"):
    words = []
    i1, i2 = 0, 0
    while i1 < len(segments):
        if i2 >= len(segments) or segments[i2].label == separator:
            if i1 != i2:
                segs = segments[i1:i2]
                word = "".join([seg.label for seg in segs])
                score = sum(seg.score * seg.length for seg in segs) / sum(seg.length for seg in segs)
                words.append(Segment(word, segments[i1].start, segments[i2 - 1].end, score))
            i1 = i2 + 1
            i2 = i1
        else:
            i2 += 1
    return words

def display_segment(i):
    ratio = waveform.size(1) / trellis.size(0)
    word = word_segments[i]
    x0 = int(ratio * word.start)
    x1 = int(ratio * word.end)
    # print(f"{word.label} ({word.score:.2f}): {x0 / bundle.sample_rate:.3f} - {x1 / bundle.sample_rate:.3f} sec")
    x0 = x0 / bundle.sample_rate
    x1 = x1 / bundle.sample_rate
    return word.label, x0, x1

for i in range(2):
    sample = ds[i]
    original_audio = sample['audio']
    text = sample['text']
    word_timestamps = sample['word_timestamps']

    waveform = torch.tensor(sample['audio']['array']).unsqueeze(0).to(device).float()
    with torch.inference_mode():
        emissions, _ = model(waveform)
        emissions = torch.log_softmax(emissions, dim=-1)
    emission = emissions[0].cpu().detach()

    # We enclose the transcript with space tokens, which represent SOS and EOS.
    transcript = "".join([c for c in text.upper() if c.isalpha() or c == " "])
    transcript = "|".join(transcript.split())
    dictionary = {c: i for i, c in enumerate(labels)}
    tokens = [dictionary[c] for c in transcript]

    trellis = get_trellis(emission, tokens)
    path = backtrack(trellis, emission, tokens)
    # Merge the labels
    segments = merge_repeats(path)
    # Merge words
    word_segments = merge_words(segments)
    # Generate the audio for each segment

    words = []
    starts = []
    ends = []

    for i in range(len(word_timestamps['word'])):
        if word_timestamps['word'][i] != "":
            words.append(word_timestamps['word'][i].upper())
            starts.append(word_timestamps['start_sec'][i])
            ends.append(word_timestamps['end_sec'][i])
    print(words)
    for i in range(len(transcript.split("|"))):
        w, s, e = display_segment(i)

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

# 0.17103070008113913 validation
# 0.10732642767195283 test