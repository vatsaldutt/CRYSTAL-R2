from pedalboard import (
    Pedalboard, Reverb, HighpassFilter, LowShelfFilter, HighShelfFilter,
    Compressor, Chorus, Delay, PitchShift, Distortion
)
from pedalboard.io import AudioFile
import numpy as np
import soundfile as sf

def add_background_hum(audio, sr, hum_file="server_hum.wav", mix=0.08):
    """Mix a low-volume background hum under the voice."""
    hum, hum_sr = sf.read(hum_file)
    if hum_sr != sr:
        raise ValueError("Hum sample rate must match voice sample rate")
    # Ensure hum is at least 2D
    if hum.ndim == 1:
        hum = np.expand_dims(hum, axis=0)
    if audio.ndim == 1:
        audio = np.expand_dims(audio, axis=0)
    # Match channel count
    if hum.shape[0] != audio.shape[0]:
        if hum.shape[0] == 1:
            hum = np.repeat(hum, audio.shape[0], axis=0)
        else:
            raise ValueError("Hum and audio must have same number of channels")
    # Loop or crop hum to match audio length
    hum_len = hum.shape[1]
    audio_len = audio.shape[1]
    if hum_len < audio_len:
        repeats = int(np.ceil(audio_len / hum_len))
        hum = np.tile(hum, (1, repeats))
    hum = hum[:, :audio_len]
    return audio * (1 - mix) + hum * mix


def add_monster_layer(audio, sr, semitones=-9, mix=0.25):
    """Add a deep growl-like undertone."""
    # Drop pitch for the "monster" effect
    monster = PitchShift(semitones=semitones)(audio, sr)
    # Low-pass filter it to keep only rumble frequencies
    monster = HighpassFilter(cutoff_frequency_hz=50)(monster, sr)
    monster = LowShelfFilter(gain_db=6.0, cutoff_frequency_hz=150)(monster, sr)
    return (audio * (1 - mix)) + (monster * mix)


# --- Main AI processing chain ---
board = Pedalboard([
    HighpassFilter(cutoff_frequency_hz=100),
    LowShelfFilter(gain_db=-2.0, cutoff_frequency_hz=200),
    HighShelfFilter(gain_db=3.0, cutoff_frequency_hz=8000),
    Compressor(threshold_db=-18, ratio=3, attack_ms=15, release_ms=100),
    PitchShift(semitones=-0.5),
    Chorus(rate_hz=1.5, depth=0.2, centre_delay_ms=15, feedback=0.1),
    Delay(delay_seconds=0.035, mix=0.15),
    Reverb(room_size=0.15, damping=0.5, wet_level=0.15, dry_level=0.85),
    Distortion(drive_db=7.0)
])

def process_ai_voice():
    # --- Load voice ---
    with AudioFile("voice.wav") as f:
        audio = f.read(f.frames)
        samplerate = f.samplerate

    # Apply effects
    effected = board(audio, samplerate)

    # Bitcrush (digital edge)
    bit_depth = 12
    max_val = np.max(np.abs(effected))
    effected = np.round(effected / max_val * (2**(bit_depth - 1))) / (2**(bit_depth - 1)) * max_val

    # Add background hum
    # effected = add_background_hum(effected, samplerate, hum_file="server_hum.wav", mix=0.05)

    # Monster undertone
    # effected = add_monster_layer(effected, samplerate, semitones=1, mix=0.2)
    # effected = add_monster_layer(effected, samplerate, semitones=-3, mix=0.3)

    # Save result
    with AudioFile("voice_ai.wav", 'w', samplerate, effected.shape[0]) as f:
        f.write(effected)

    print("Processed AI voice saved as voice_ai.wav")

if __name__ == "__main__":
    process_ai_voice()