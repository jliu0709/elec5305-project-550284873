function timing = generate_modified_spectral_test(projectRoot, selectedAlpha)
%GENERATE_MODIFIED_SPECTRAL_TEST Apply the development-selected alpha to test.

arguments
    projectRoot (1,1) string
    selectedAlpha (1,1) double {mustBePositive}
end

splitName = "test";
resultRoot = fullfile(projectRoot, "results", "formal", splitName);
mixtures = readtable(fullfile(resultRoot, "mixture_manifest.csv"), ...
    Delimiter=",", TextType="string", VariableNamingRule="preserve");
token = replace(sprintf("%.2f", selectedAlpha), ".", "p");
processingSeconds = zeros(height(mixtures), 1);
realTimeFactor = zeros(height(mixtures), 1);
outputRelativePath = strings(height(mixtures), 1);

for i = 1:height(mixtures)
    prefixPath = fullfile(projectRoot, mixtures.prefixRelativePath(i));
    [mixture, fs] = audioread(prefixPath);
    assert(fs == 16000 && size(mixture, 2) == 1);
    stem = mixtures.noiseType(i) + "_" + mixtures.targetSNRdB(i) + "dB";
    outputPath = fullfile(projectRoot, "results", "formal", splitName, ...
        "audio", mixtures.utteranceId(i), ...
        stem + "_spectral_alpha_" + token + ".wav");
    if ~isfile(outputPath)
        tic;
        output = stft_enhance(double(mixture), 16000, 512, 128, ...
            "spectral_subtraction", selectedAlpha, 0.02);
        processingSeconds(i) = toc;
        audiowrite(outputPath, output, fs, BitsPerSample=24);
    end
    duration = mixtures.durationSeconds(i);
    realTimeFactor(i) = processingSeconds(i) / duration;
    outputRelativePath(i) = relative(projectRoot, outputPath);
    fprintf("[%d/%d] %s %s %g dB alpha %.2f\n", i, height(mixtures), ...
        mixtures.utteranceId(i), mixtures.noiseType(i), ...
        mixtures.targetSNRdB(i), selectedAlpha);
end

timing = table(mixtures.utteranceId, mixtures.noiseType, ...
    mixtures.targetSNRdB, repmat(selectedAlpha, height(mixtures), 1), ...
    processingSeconds, realTimeFactor, outputRelativePath, ...
    'VariableNames', {'utterance_id', 'noise_type', 'target_snr_db', ...
    'selected_alpha', 'processing_seconds', 'real_time_factor', ...
    'output_relative_path'});
writetable(timing, fullfile(resultRoot, "modified_spectral_manifest.csv"));
end


function path = relative(root, absolutePath)
root = replace(string(root), "\", "/");
absolutePath = replace(string(absolutePath), "\", "/");
path = extractAfter(absolutePath, strlength(root) + 1);
if startsWith(path, "/"), path = extractAfter(path, 1); end
end
