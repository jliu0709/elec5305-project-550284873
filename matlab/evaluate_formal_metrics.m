function [perFile, summary] = evaluate_formal_metrics(projectRoot, splitName, includeAlphaSweep, includeModified)
%EVALUATE_FORMAL_METRICS Compute aligned waveform SNR and STOI.

arguments
    projectRoot (1,1) string
    splitName (1,1) string {mustBeMember(splitName,["development","test"])}
    includeAlphaSweep (1,1) logical = false
    includeModified (1,1) logical = false
end

resultRoot = fullfile(projectRoot, "results", "formal", splitName);
mixtures = readtable(fullfile(resultRoot, "mixture_manifest.csv"), ...
    Delimiter=",", TextType="string", VariableNamingRule="preserve");
deepfilter = readtable(fullfile(resultRoot, "deepfilter_manifest.csv"), ...
    Delimiter=",", TextType="string", VariableNamingRule="preserve");
modified = table;
methods = ["noisy", "spectral_subtraction", "wiener", "deepfilternet3"];
alphas = [0.75 1.00 1.25 1.50 2.00];
if includeAlphaSweep
    for alpha = alphas
        methods(end+1) = "spectral_alpha_" + replace(sprintf("%.2f", alpha), ".", "p"); %#ok<AGROW>
    end
end
if includeModified
    modified = readtable(fullfile(resultRoot, "modified_spectral_manifest.csv"), ...
        Delimiter=",", TextType="string", VariableNamingRule="preserve");
    methods(end+1) = "spectral_subtraction_modified";
end

nRows = height(mixtures) * numel(methods);
utteranceId = strings(nRows, 1); noiseType = strings(nRows, 1);
targetSNRdB = zeros(nRows, 1); method = strings(nRows, 1);
outputSNRdB = zeros(nRows, 1); snrImprovementdB = zeros(nRows, 1);
stoiScore = zeros(nRows, 1); estimatedLagSamples = zeros(nRows, 1);
durationSeconds = zeros(nRows, 1); processingSeconds = nan(nRows, 1);
realTimeFactor = nan(nRows, 1); peakAmplitude = zeros(nRows, 1);
audioRelativePath = strings(nRows, 1);

pool = gcp("nocreate");
if isempty(pool)
    parpool("Processes", min(8, feature("numcores")));
end
nMethods = numel(methods);
parfor rowIndex = 1:nRows
    i = ceil(rowIndex / nMethods);
    m = mod(rowIndex - 1, nMethods) + 1;
    id = mixtures.utteranceId(i);
    [baseClean, fs] = audioread(fullfile(projectRoot, mixtures.cleanRelativePath(i)));
    assert(fs == 16000 && size(baseClean, 2) == 1);
    reference = double(baseClean(:)) * mixtures.commonScale(i);
    duration = numel(reference) / fs;
    deepUse = deepfilter.utterance_id == id & ...
        deepfilter.noise_type == mixtures.noiseType(i) & ...
        deepfilter.target_snr_db == mixtures.targetSNRdB(i);
    assert(nnz(deepUse) == 1, "Missing DeepFilterNet timing row.");

        currentMethod = methods(m);
        if currentMethod == "noisy"
            relativePath = mixtures.noisyRelativePath(i); elapsed = 0;
        elseif currentMethod == "spectral_subtraction"
            relativePath = mixtures.spectralRelativePath(i); elapsed = mixtures.spectralSeconds(i);
        elseif currentMethod == "wiener"
            relativePath = mixtures.wienerRelativePath(i); elapsed = mixtures.wienerSeconds(i);
        elseif currentMethod == "deepfilternet3"
            relativePath = deepfilter.deepfilter_relative_path(deepUse);
            elapsed = deepfilter.processing_seconds(deepUse);
        elseif currentMethod == "spectral_subtraction_modified"
            modifiedUse = modified.utterance_id == id & ...
                modified.noise_type == mixtures.noiseType(i) & ...
                modified.target_snr_db == mixtures.targetSNRdB(i);
            assert(nnz(modifiedUse) == 1, "Missing modified spectral timing row.");
            relativePath = modified.output_relative_path(modifiedUse);
            elapsed = modified.processing_seconds(modifiedUse);
        else
            token = extractAfter(currentMethod, "spectral_alpha_");
            stem = mixtures.noiseType(i) + "_" + mixtures.targetSNRdB(i) + "dB";
            relativePath = fullfile("results", "formal", splitName, "audio", id, ...
                stem + "_spectral_alpha_" + token + ".wav");
            relativePath = replace(relativePath, "\", "/"); elapsed = NaN;
        end
        [processed, processedFs] = audioread(fullfile(projectRoot, relativePath));
        processed = double(processed(:));
        assert(processedFs == fs && numel(processed) == numel(reference));
        assert(all(isfinite(processed)));
        lag = finddelay(reference, processed, round(0.01 * fs));
        currentSNR = 10 * log10(sum(reference.^2) / ...
            (sum((reference - processed).^2) + eps));

        utteranceId(rowIndex) = id; noiseType(rowIndex) = mixtures.noiseType(i);
        targetSNRdB(rowIndex) = mixtures.targetSNRdB(i); method(rowIndex) = currentMethod;
        outputSNRdB(rowIndex) = currentSNR;
        snrImprovementdB(rowIndex) = currentSNR - mixtures.measuredInputSNRdB(i);
        stoiScore(rowIndex) = stoi(processed, reference, fs);
        estimatedLagSamples(rowIndex) = lag; durationSeconds(rowIndex) = duration;
        processingSeconds(rowIndex) = elapsed; realTimeFactor(rowIndex) = elapsed / duration;
        peakAmplitude(rowIndex) = max(abs(processed)); audioRelativePath(rowIndex) = relativePath;
end

perFile = table(utteranceId, noiseType, targetSNRdB, method, outputSNRdB, ...
    snrImprovementdB, stoiScore, estimatedLagSamples, durationSeconds, ...
    processingSeconds, realTimeFactor, peakAmplitude, audioRelativePath);
writetable(perFile, fullfile(resultRoot, "metrics_per_file.csv"));

[groups, noise, snr, groupMethod] = findgroups(perFile.noiseType, ...
    perFile.targetSNRdB, perFile.method);
summary = table(noise, snr, groupMethod, splitapply(@numel, perFile.outputSNRdB, groups), ...
    splitapply(@mean, perFile.outputSNRdB, groups), ...
    splitapply(@mean, perFile.snrImprovementdB, groups), ...
    splitapply(@mean, perFile.stoiScore, groups), ...
    splitapply(@(x) max(abs(x)), perFile.estimatedLagSamples, groups), ...
    splitapply(@(x) mean(x, "omitnan"), perFile.realTimeFactor, groups), ...
    'VariableNames', {'noise_type', 'target_snr_db', 'method', 'n_files', ...
    'mean_output_snr_db', 'mean_snr_improvement_db', 'mean_stoi', ...
    'max_abs_lag_samples', 'mean_real_time_factor'});
writetable(summary, fullfile(resultRoot, "metrics_summary.csv"));
disp(summary);
end
