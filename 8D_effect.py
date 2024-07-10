import os
import numpy as np
import librosa
import soundfile as sf
import sox

def song_features(file_name):
    wav_mono, sampling_rate = librosa.load(file_name, duration=270)
    wav_stereo, sampling_rate = librosa.load(file_name, mono=False, duration=270)
    tempo, beat_frames = librosa.beat.beat_track(y=wav_stereo[0], sr=sampling_rate)
    return wav_mono, wav_stereo, sampling_rate, tempo, beat_frames

def save_song(name, wav, sampling_rate):
    sf.write(name, wav.T, sampling_rate)
    return

def rotate_left_right(wav_mono, wav_stereo, tempo, sampling_rate):
    length = wav_mono.shape[0]
    end_of_beat = int((tempo / 120) * sampling_rate)
    down_value = .15
    amplitude_down = np.linspace(1, down_value, 2*end_of_beat)
    amplitude_up = np.linspace(down_value, 1, 2*end_of_beat)
    amplitude_down_slower = np.linspace(1, down_value, 8*end_of_beat)
    amplitude_up_slower = np.linspace(down_value, 1, 8*end_of_beat)

    left_up = False
    right_up = False
    left_maintain = False
    right_maintain = True
    i = 0
    while i < length - 8*end_of_beat:
        fast = np.random.choice([True, False])
        if left_up:
            if fast:
                wav_stereo[0, i:i+(2*end_of_beat)] = wav_mono[i:i+(2*end_of_beat)]*amplitude_up
                wav_stereo[1, i:i+(2*end_of_beat)] = wav_mono[i:i+(2*end_of_beat)]*amplitude_down
                left_up = False
                left_maintain = True
                i += (2 * end_of_beat)
            else:
                wav_stereo[0, i:i+(8*end_of_beat)] = wav_mono[i:i+(8*end_of_beat)]*amplitude_up_slower
                wav_stereo[1, i:i+(8*end_of_beat)] = wav_mono[i:i+(8*end_of_beat)]*amplitude_down_slower
                left_up = False
                left_maintain = True
                i += (8 * end_of_beat)
        elif right_up:
            if fast:
                wav_stereo[1, i:i+(2*end_of_beat)] = wav_mono[i:i+(2*end_of_beat)]*amplitude_up
                wav_stereo[0, i:i+(2*end_of_beat)] = wav_mono[i:i+(2*end_of_beat)]*amplitude_down
                right_up = False
                right_maintain = True
                i += (2 * end_of_beat)
            else:
                wav_stereo[1, i:i+(8*end_of_beat)] = wav_mono[i:i+(8*end_of_beat)]*amplitude_up_slower
                wav_stereo[0, i:i+(8*end_of_beat)] = wav_mono[i:i+(8*end_of_beat)]*amplitude_down_slower
                right_up = False
                right_maintain = True
                i += (8 * end_of_beat)
        elif left_maintain:
            wav_stereo[0, i:i+end_of_beat] = wav_mono[i:i+end_of_beat]
            wav_stereo[1, i:i+end_of_beat] = wav_mono[i:i+end_of_beat]*down_value
            right_up = True
            left_maintain = False
            i += end_of_beat
        elif right_maintain:
            wav_stereo[1, i:i + end_of_beat] = wav_mono[i:i + end_of_beat]
            wav_stereo[0, i:i + end_of_beat] = wav_mono[i:i+end_of_beat]*down_value
            right_maintain = False
            left_up = True
            i += end_of_beat

    wav_stereo[0, (length//(8*end_of_beat))*(8*end_of_beat):] *= 0
    wav_stereo[1, (length//(8*end_of_beat))*(8*end_of_beat):] *= 0
    return wav_stereo

def add_effects(input_file, output_file):
    tfm = sox.Transformer()
    tfm.reverb(reverberance=25)
    tfm.treble(gain_db=5, slope=.3)
    tfm.bass(gain_db=5, slope=0.3)
    tfm.build(input_file, output_file)
    return



def process_audio(url_or_path, input_file, output_file, effect_file, is_url=False):

    if not os.path.exists(url_or_path):
        raise FileNotFoundError("The provided file path does not exist.")
    os.rename(url_or_path, input_file)

    wav_mono, wav_stereo, sampling_rate, tempo, beat_frame = song_features(input_file)
    wav = rotate_left_right(wav_mono, wav_stereo, tempo, sampling_rate)
    save_song(output_file, wav, sampling_rate)
    add_effects(output_file, effect_file)
    return effect_file

audio_url_or_path = 'sway.wav'
temp_dir = 'temp_results'
input_file = os.path.join(temp_dir, 'test.wav')
output_file = os.path.join(temp_dir, 'in.wav')
effect_file = os.path.join(temp_dir, 'sway_8D.wav')

output_file_path = process_audio(audio_url_or_path, input_file, output_file, effect_file, is_url=False)
print(f"Processed audio saved to: {output_file_path}")
