# Noise sources

The formal experiment uses:

- white noise generated deterministically in MATLAB;
- DEMAND OOFFICE channel 1 at 16 kHz;
- DEMAND STRAFFIC channel 1 at 16 kHz.

Download DEMAND from <https://zenodo.org/records/1227121>. The downloaded `STRAFFIC_16k.zip` used in the completed run matched the published MD5 checksum `2efa87262f272bbf9ba578088e81939c`. The full third-party recordings are not redistributed in the repository.

After download, extract/prepare mono files as:

```text
data/noise/office.wav
data/noise/traffic.wav
```
