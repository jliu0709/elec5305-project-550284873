function [speechOut, reconstructionError] = stft_enhance(noisyWithPrefix, ...
    prefixSamples, frameLength, hopLength, method, subtractionAlpha, floorRatio)
%STFT_ENHANCE Apply one classical STFT enhancement method.

arguments
    noisyWithPrefix (:,1) double
    prefixSamples (1,1) double {mustBeInteger,mustBePositive}
    frameLength (1,1) double {mustBeInteger,mustBePositive}
    hopLength (1,1) double {mustBeInteger,mustBePositive}
    method (1,1) string {mustBeMember(method,["spectral_subtraction","wiener"])}
    subtractionAlpha (1,1) double {mustBePositive} = 1
    floorRatio (1,1) double {mustBeNonnegative} = 0.02
end

x = noisyWithPrefix(:);
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
assert(any(noiseFrames), "No complete noise-only frames were found.");

powerSpectrum = abs(spectrum).^2;
noisePower = mean(powerSpectrum(:, noiseFrames), 2);
if method == "spectral_subtraction"
    enhancedPower = max(powerSpectrum - subtractionAlpha * noisePower, ...
        floorRatio * noisePower);
    gain = min(sqrt(enhancedPower ./ max(powerSpectrum, eps)), 1);
else
    speechPower = max(powerSpectrum - noisePower, 0);
    gain = speechPower ./ max(speechPower + noisePower, eps);
end

fullOut = invert(spectrum .* gain, window, hopLength, ...
    paddedLength, offset, numel(x));
bypass = invert(spectrum, window, hopLength, paddedLength, ...
    offset, numel(x));
reconstructionError = norm(bypass - x) / max(norm(x), eps);
speechOut = fullOut(prefixSamples+1:end);
end


function output = invert(spectrum, window, hopLength, ...
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
