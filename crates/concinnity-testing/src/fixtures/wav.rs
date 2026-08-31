//! RIFF/WAVE bytes.

/// A 16-bit mono PCM WAV carrying `samples` at `sample_rate`.
pub fn pcm16_mono(sample_rate: u32, samples: &[i16]) -> Vec<u8> {
    let data_len = (samples.len() * 2) as u32;
    let mut wav = Vec::with_capacity(44 + data_len as usize);
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&(36 + data_len).to_le_bytes());
    wav.extend_from_slice(b"WAVE");
    wav.extend_from_slice(b"fmt ");
    wav.extend_from_slice(&16u32.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes()); // PCM
    wav.extend_from_slice(&1u16.to_le_bytes()); // mono
    wav.extend_from_slice(&sample_rate.to_le_bytes());
    wav.extend_from_slice(&(sample_rate * 2).to_le_bytes()); // byte rate
    wav.extend_from_slice(&2u16.to_le_bytes()); // block align
    wav.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_len.to_le_bytes());
    for sample in samples {
        wav.extend_from_slice(&sample.to_le_bytes());
    }
    wav
}

/// `millis` of 16-bit mono silence at `sample_rate`.
pub fn silence(sample_rate: u32, millis: u32) -> Vec<u8> {
    let count = (sample_rate as usize * millis as usize) / 1000;
    pcm16_mono(sample_rate, &vec![0i16; count])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_header_describes_the_samples_that_follow() {
        let wav = pcm16_mono(8000, &[0, 1, -1]);

        assert_eq!(&wav[..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        assert_eq!(&wav[36..40], b"data");
        assert_eq!(u32::from_le_bytes(wav[40..44].try_into().unwrap()), 6);
        assert_eq!(
            u32::from_le_bytes(wav[4..8].try_into().unwrap()) as usize,
            wav.len() - 8
        );
        assert_eq!(wav.len(), 44 + 6);
    }

    #[test]
    fn silence_runs_for_the_duration_asked_for() {
        let wav = silence(8000, 100);

        // 0.1 s at 8 kHz is 800 samples of two bytes each.
        assert_eq!(wav.len(), 44 + 1600);
        assert!(wav[44..].iter().all(|&b| b == 0));
    }
}
