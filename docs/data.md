# Data

CDD-AVDL consumes pre-extracted feature sequences, not raw media. This keeps
training independent of ffmpeg, torchaudio and the OpenGraphAU checkpoint, and
it makes the repository usable on any corpus that provides the same two
streams.

## Directory layout

    data/
      labels.csv
      audio/
        <sample_id>.npy      float32, shape (T_audio, 768)
      visual/
        <sample_id>.npy      float32, shape (T_visual, 41)

## Label file

labels.csv has a header and at least two columns:

    sample_id,label
    wch_0001,1
    wch_0002,0

label is 1 for a depressed participant and 0 otherwise. An optional fold
column pins a sample to a cross-validation fold:

    sample_id,label,fold
    wch_0001,1,0
    wch_0002,0,3

Without a fold column, split_folds assigns folds deterministically from
train.seed.

## Feature streams

Audio: wav2vec2-base hidden states, 768 dimensions, one frame every 20 ms
(50 Hz). The paper uses the last hidden layer.

Visual: OpenGraphAU facial action units, 41 dimensions, one frame per video
frame (30 Hz).

## Choosing audio_len and visual_len

These are the only two numbers that must be measured. Every width in the
network follows from them, so a corpus with different recording lengths needs
no code change.

    python -c "import glob, numpy as np; print(max(np.load(p).shape[0] for p in glob.glob('data/audio/*.npy')))"
    python -c "import glob, numpy as np; print(max(np.load(p).shape[0] for p in glob.glob('data/visual/*.npy')))"

Put the two values into data.audio_len and data.visual_len. Shorter samples are
zero-padded, longer samples are truncated, and the true visual length is passed
to the LSTM so padding does not leak into the recurrent state.

Constraints enforced by Config.validate:

    audio_len >= audio_conv_kernel
    visual_len >= window_size
    audio_conv_out % fusion_patch_sizes[0] == 0
    mlstm_conv_channels[i] % fusion_patch_sizes[i] == 0

## Extracting features

    python scripts/extract_features.py --manifest manifest.csv --root data \
        --graphau-root /path/to/OpenGraphAU

manifest.csv has the columns sample_id,wav,video. Audio extraction needs
torch, torchaudio and transformers; visual extraction needs opencv-python and a
checkout of OpenGraphAU. Both are optional dependencies, so a machine that
already has the .npy files never imports them.

## Corpora used in the paper

The paper evaluates on a corpus recorded at the Mental Health Center, West
China Hospital, Sichuan University: 413 participants (103 depressed, 310
non-depressed), each reading the same 17-sentence, 617-character text, with
videos of 105 to 238 seconds.

The corpus is not redistributed with this repository. It contains clinical
recordings of identifiable participants and is bound by the ethics approval and
the data-sharing agreement of the recording institution.

The agreement permits academic, non-commercial research only. Commercial use,
product development, clinical deployment, and any further redistribution of the
recordings or of features derived from them fall outside it. The restriction
covers the corpus; the code in this repository is MIT licensed.

Requests go to the recording institution, the Mental Health Center, West China
Hospital, Sichuan University, addressed to the corresponding author listed on
the first page of the paper. State your institution, the academic research
purpose the data would serve, your own ethics approval or IRB reference, and
the expected duration of the collaboration.

Two public corpora with the same two feature streams can be used with the
provided configurations:

    configs/daic_woz.yaml       DAIC-WOZ
    configs/eatd_corpus.yaml    EATD-Corpus

Both files document how to set the two sequence lengths from the extracted
features.
