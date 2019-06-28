# -*- coding: utf-8 -*-

""" ================ ^^^
    Vocal Separation
^^^ ================ """


# Imports
from __future__ import print_function
import numpy as np
import librosa


if __name__ == "__main__":

    # y, sr = librosa.load('audio/source/hunter.wav', duration=10)
    y, sr = librosa.load('audio/source/piano.mp3', duration=10)

    S_full, phase = librosa.magphase(librosa.stft(y))

    S_filter = librosa.decompose.nn_filter(S_full,
                                           aggregate=np.median,
                                           metric='cosine',
                                           width=int(librosa.time_to_frames(2, sr=sr)))

    S_filter = np.minimum(S_full, S_filter)

    margin_i, margin_v = 2, 10
    power = 2

    mask_i = librosa.util.softmask(S_filter,
                                   margin_i * (S_full - S_filter),
                                   power=power)

    mask_v = librosa.util.softmask(S_full - S_filter,
                                   margin_v * S_filter,
                                   power=power)

    S_background = mask_i * S_full
    S_foreground = mask_v * S_full

    librosa.output.write_wav("audio/result/music.wav", librosa.istft(S_background * phase), sr)
    librosa.output.write_wav("audio/result/lyrics.wav", librosa.istft(S_foreground * phase), sr)

    print("")
    print("      Done! ")
