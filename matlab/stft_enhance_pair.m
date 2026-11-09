function [spectralOut, wienerOut, reconstructionError] = stft_enhance_pair(...
    noisyWithPrefix, prefixSamples, frameLength, hopLength, ...
    subtractionAlpha, floorRatio)
%STFT_ENHANCE_PAIR Apply power spectral subtraction and a basic Wiener gain.

x = double(noisyWithPrefix(:));
window = sqrt(hann(frameLength, "periodic"));
offset = frameLength;
padded = [zeros(offset, 1); x; zeros(frameLength, 1)];
frameCount = ceil((numel(padded) - frameLength) / hopLength) + 1;
paddedLength = (frameCount - 1) * hopLength + frameLength;
padded(end+1:paddedLength, 1) = 0;

spectrum = zeros(frameLength, frameCount);
for k = 1:frameCount
    index = (k - 1) * hopLength + (1:frameLength);
    spectrum(:, k) = fft(padded(index) .* window);
end

frameStart = (0:frameCount-1) * hopLength + 1;
noiseFrames = frameStart > offset & ...
    frameStart + frameLength - 1 <= offset + prefixSamples;
assert(any(noiseFrames), "No complete noise-only STFT frames were found.");

powerSpectrum = abs(spectrum).^2;
noisePower = mean(powerSpectrum(:, noiseFrames), 2);
subtractedPower = max(powerSpectrum - subtractionAlpha * noisePower, ...
    floorRatio * noisePower);
subtractionGain = min(sqrt(subtractedPower ./ max(powerSpectrum, eps)), 1);
estimatedSpeechPower = max(powerSpectrum - noisePower, 0);
wienerGain = estimatedSpeechPower ./ max(estimatedSpeechPower + noisePower, eps);

spectralFull = invertSpectrum(spectrum .* subtractionGain, window, ...
    hopLength, paddedLength, offset, numel(x));
wienerFull = invertSpectrum(spectrum .* wienerGain, window, ...
    hopLength, paddedLength, offset, numel(x));
bypass = invertSpectrum(spectrum, window, hopLength, paddedLength, ...
    offset, numel(x));
reconstructionError = norm(bypass - x) / max(norm(x), eps);

spectralOut = spectralFull(prefixSamples+1:end);
wienerOut = wienerFull(prefixSamples+1:end);
end


function output = invertSpectrum(spectrum, window, hopLength, ...
    paddedLength, offset, originalLength)
frameLength = numel(window);
outputPadded = zeros(paddedLength, 1);
weight = zeros(paddedLength, 1);
for k = 1:size(spectrum, 2)
    index = (k - 1) * hopLength + (1:frameLength);
    frame = real(ifft(spectrum(:, k))) .* window;
    outputPadded(index) = outputPadded(index) + frame;
    weight(index) = weight(index) + window.^2;
end
valid = weight > 1e-12;
outputPadded(valid) = outputPadded(valid) ./ weight(valid);
output = outputPadded(offset + (1:originalLength));
end
