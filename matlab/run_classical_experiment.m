function mixtureManifest = run_classical_experiment(projectRoot, splitName, runAlphaSweep)
%RUN_CLASSICAL_EXPERIMENT Create mixtures and classical enhanced outputs.
% The same saved noisy-with-prefix signal is used by both classical methods.

arguments
    projectRoot (1,1) string
    splitName (1,1) string {mustBeMember(splitName,["development","test"])}
    runAlphaSweep (1,1) logical = false
end

fs = 16000;
prefixSamples = fs;
frameLength = 512;
hopLength = 128;
floorRatio = 0.02;
baselineAlpha = 1.0;
alphaCandidates = [0.75 1.00 1.25 1.50 2.00];
noiseNames = ["white", "office", "traffic"];
snrTargets = [0 5 10];

manifestPath = fullfile(projectRoot, "data", "splits", ...
    splitName + "_manifest.csv");
manifest = readtable(manifestPath, Delimiter=",", TextType="string", ...
    VariableNamingRule="preserve");
office = readNoise(fullfile(projectRoot, "data", "noise", "office.wav"), fs);
traffic = readNoise(fullfile(projectRoot, "data", "noise", "traffic.wav"), fs);

resultRoot = fullfile(projectRoot, "results", "formal", splitName);
audioRoot = fullfile(resultRoot, "audio");
processingRoot = fullfile(resultRoot, "processing");
if ~isfolder(audioRoot), mkdir(audioRoot); end
if ~isfolder(processingRoot), mkdir(processingRoot); end

nRows = height(manifest) * numel(noiseNames) * numel(snrTargets);
utteranceId = strings(nRows, 1); projectSplit = repmat(splitName, nRows, 1);
noiseType = strings(nRows, 1); targetSNRdB = zeros(nRows, 1);
measuredInputSNRdB = zeros(nRows, 1); randomSeed = zeros(nRows, 1);
noiseStartSample16k = nan(nRows, 1); noiseGain = zeros(nRows, 1);
commonScale = zeros(nRows, 1); prefixLengthSamples = repmat(prefixSamples, nRows, 1);
cleanRelativePath = strings(nRows, 1); noisyRelativePath = strings(nRows, 1);
prefixRelativePath = strings(nRows, 1); spectralRelativePath = strings(nRows, 1);
wienerRelativePath = strings(nRows, 1); reconstructionError = zeros(nRows, 1);
spectralSeconds = zeros(nRows, 1); wienerSeconds = zeros(nRows, 1);
durationSeconds = zeros(nRows, 1);

row = 0;
for u = 1:height(manifest)
    id = string(manifest.utterance_id(u));
    [clean, cleanFs] = audioread(fullfile(projectRoot, manifest.clean_wav(u)));
    assert(cleanFs == fs && size(clean, 2) == 1);
    clean = double(clean(:)) - mean(clean);
    speechLength = numel(clean);
    utteranceDir = fullfile(audioRoot, id);
    prefixDir = fullfile(processingRoot, id);
    if ~isfolder(utteranceDir), mkdir(utteranceDir); end
    if ~isfolder(prefixDir), mkdir(prefixDir); end
    cleanPath = fullfile(utteranceDir, "clean.wav");
    if ~isfile(cleanPath)
        audiowrite(cleanPath, clean, fs, BitsPerSample=24);
    end

    for n = 1:numel(noiseNames)
        for q = 1:numel(snrTargets)
            row = row + 1;
            seed = 5305 + u * 100 + n * 10 + q;
            rng(seed, "twister");
            totalLength = prefixSamples + speechLength;
            if noiseNames(n) == "white"
                baseNoise = randn(totalLength, 1);
                startSample = NaN;
            else
                sourceNoise = office;
                if noiseNames(n) == "traffic", sourceNoise = traffic; end
                startSample = randi(numel(sourceNoise) - totalLength + 1);
                baseNoise = sourceNoise(startSample:startSample + totalLength - 1);
            end
            baseNoise = baseNoise - mean(baseNoise);
            speechNoise = baseNoise(prefixSamples+1:end);
            gain = sqrt(mean(clean.^2) / ...
                (mean(speechNoise.^2) * 10^(snrTargets(q)/10) + eps));
            scaledNoise = baseNoise * gain;
            paddedClean = [zeros(prefixSamples, 1); clean];
            mixture = paddedClean + scaledNoise;
            scale = min(1, 0.95 / max(abs([paddedClean; mixture])));
            mixture = mixture * scale;
            scaledReference = clean * scale;
            noisy = mixture(prefixSamples+1:end);

            stem = noiseNames(n) + "_" + snrTargets(q) + "dB";
            noisyPath = fullfile(utteranceDir, stem + "_noisy.wav");
            prefixPath = fullfile(prefixDir, stem + "_noisy_with_prefix.wav");
            spectralPath = fullfile(utteranceDir, stem + "_spectral_subtraction.wav");
            wienerPath = fullfile(utteranceDir, stem + "_wiener.wav");
            if ~isfile(noisyPath), audiowrite(noisyPath, noisy, fs, BitsPerSample=24); end
            if ~isfile(prefixPath), audiowrite(prefixPath, mixture, fs, BitsPerSample=24); end

            storedNoisy = audioread(noisyPath);
            inputSNR = 10 * log10(sum(scaledReference.^2) / ...
                (sum((storedNoisy - scaledReference).^2) + eps));
            assert(abs(inputSNR - snrTargets(q)) < 0.01, ...
                "Stored input SNR is incorrect.");

            tic;
            [spectral, recError1] = stft_enhance(mixture, prefixSamples, ...
                frameLength, hopLength, "spectral_subtraction", ...
                baselineAlpha, floorRatio);
            spectralTime = toc;
            tic;
            [wiener, recError2] = stft_enhance(mixture, prefixSamples, ...
                frameLength, hopLength, "wiener", baselineAlpha, floorRatio);
            wienerTime = toc;
            assert(max(recError1, recError2) < 1e-9, "STFT reconstruction failed.");
            assert(all(isfinite([spectral; wiener])));
            assert(max(abs([spectral; wiener])) < 1);
            if ~isfile(spectralPath), audiowrite(spectralPath, spectral, fs, BitsPerSample=24); end
            if ~isfile(wienerPath), audiowrite(wienerPath, wiener, fs, BitsPerSample=24); end

            if runAlphaSweep
                for alpha = alphaCandidates
                    alphaName = replace(sprintf("%.2f", alpha), ".", "p");
                    alphaPath = fullfile(utteranceDir, ...
                        stem + "_spectral_alpha_" + alphaName + ".wav");
                    if ~isfile(alphaPath)
                        alphaAudio = stft_enhance(mixture, prefixSamples, ...
                            frameLength, hopLength, "spectral_subtraction", ...
                            alpha, floorRatio);
                        audiowrite(alphaPath, alphaAudio, fs, BitsPerSample=24);
                    end
                end
            end

            utteranceId(row) = id; noiseType(row) = noiseNames(n);
            targetSNRdB(row) = snrTargets(q); measuredInputSNRdB(row) = inputSNR;
            randomSeed(row) = seed; noiseStartSample16k(row) = startSample;
            noiseGain(row) = gain; commonScale(row) = scale;
            cleanRelativePath(row) = relative(projectRoot, cleanPath);
            noisyRelativePath(row) = relative(projectRoot, noisyPath);
            prefixRelativePath(row) = relative(projectRoot, prefixPath);
            spectralRelativePath(row) = relative(projectRoot, spectralPath);
            wienerRelativePath(row) = relative(projectRoot, wienerPath);
            reconstructionError(row) = max(recError1, recError2);
            spectralSeconds(row) = spectralTime; wienerSeconds(row) = wienerTime;
            durationSeconds(row) = speechLength / fs;
            fprintf("%s %s %d dB complete\n", id, noiseNames(n), snrTargets(q));
        end
    end
end

mixtureManifest = table(utteranceId, projectSplit, noiseType, targetSNRdB, ...
    measuredInputSNRdB, randomSeed, noiseStartSample16k, noiseGain, commonScale, ...
    prefixLengthSamples, durationSeconds, cleanRelativePath, noisyRelativePath, ...
    prefixRelativePath, spectralRelativePath, wienerRelativePath, ...
    reconstructionError, spectralSeconds, wienerSeconds);
writetable(mixtureManifest, fullfile(resultRoot, "mixture_manifest.csv"));

settings = struct(fs=fs, prefix_samples=prefixSamples, frame_length=frameLength, ...
    hop_length=hopLength, window="sqrt periodic Hann", floor_ratio=floorRatio, ...
    spectral_subtraction_alpha=baselineAlpha, alpha_candidates=alphaCandidates, ...
    snr_targets_db=snrTargets, random_seed_formula="5305 + 100*u + 10*n + q", ...
    clipping_peak_limit=0.95);
fid = fopen(fullfile(resultRoot, "classical_settings.json"), "w");
fprintf(fid, "%s", jsonencode(settings, PrettyPrint=true)); fclose(fid);
end


function noise = readNoise(path, expectedFs)
[noise, fs] = audioread(path);
assert(fs == expectedFs && size(noise, 2) == 1, ...
    "Noise must be mono 16 kHz audio: %s", path);
noise = double(noise(:));
end


function path = relative(root, absolutePath)
root = replace(string(root), "\", "/");
absolutePath = replace(string(absolutePath), "\", "/");
path = extractAfter(absolutePath, strlength(root) + 1);
if startsWith(path, "/"), path = extractAfter(path, 1); end
end
