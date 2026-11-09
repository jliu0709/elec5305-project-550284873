function [perUtterance, summary] = evaluate_pilot_metrics(projectRoot)
%EVALUATE_PILOT_METRICS Evaluate the five completed pilot conditions.
% Use only the aligned speech part. Do not score the noise-only prefix.

arguments
    projectRoot (1,1) string
end

fsExpected = 16000;
conditions = ["clean", "noisy", "spectral_subtraction", ...
    "wiener", "deepfilter"];
manifest = readtable(fullfile(projectRoot, "utterances.csv"), ...
    Delimiter=",", ReadVariableNames=true, TextType="string", ...
    VariableNamingRule="preserve");
audioDir = fullfile(projectRoot, "results", "audio");
analysisDir = fullfile(projectRoot, "results", "analysis");
if ~isfolder(analysisDir)
    mkdir(analysisDir);
end

nRows = height(manifest) * numel(conditions);
utteranceId = strings(nRows, 1);
condition = strings(nRows, 1);
sampleRate = zeros(nRows, 1);
referenceSamples = zeros(nRows, 1);
processedSamples = zeros(nRows, 1);
estimatedLagSamples = zeros(nRows, 1);
outputSNRdB = nan(nRows, 1);
snrImprovementdB = nan(nRows, 1);
stoiScore = nan(nRows, 1);
peakAmplitude = nan(nRows, 1);

rowIndex = 0;
for u = 1:height(manifest)
    % Use the first column by position because the CSV may include a UTF-8 BOM.
    id = string(manifest{u, 1});
    cleanPath = fullfile(audioDir, id + "_clean.wav");
    [clean, fs] = audioread(cleanPath);
    assert(fs == fsExpected && size(clean, 2) == 1, ...
        "Clean reference must be mono 16 kHz audio.");
    clean = double(clean(:));

    noisyPath = fullfile(audioDir, id + "_noisy.wav");
    noisy = audioread(noisyPath);
    assert(numel(noisy) == numel(clean), "Noisy audio length mismatch.");
    inputSNR = 10 * log10(sum(clean.^2) / ...
        (sum((double(noisy(:)) - clean).^2) + eps));

    for c = 1:numel(conditions)
        rowIndex = rowIndex + 1;
        currentCondition = conditions(c);
        audioPath = fullfile(audioDir, id + "_" + currentCondition + ".wav");
        [processed, processedFs] = audioread(audioPath);
        processed = double(processed(:));

        assert(processedFs == fsExpected, "Unexpected sample rate: %s", audioPath);
        assert(numel(processed) == numel(clean), "Length mismatch: %s", audioPath);
        assert(all(isfinite(processed)), "Non-finite samples: %s", audioPath);

        lag = finddelay(clean, processed, round(0.05 * fsExpected));
        errorSignal = clean - processed;
        if currentCondition == "clean"
            currentSNR = Inf;
            currentImprovement = NaN;
        else
            currentSNR = 10 * log10(sum(clean.^2) / ...
                (sum(errorSignal.^2) + eps));
            currentImprovement = currentSNR - inputSNR;
        end

        utteranceId(rowIndex) = id;
        condition(rowIndex) = currentCondition;
        sampleRate(rowIndex) = processedFs;
        referenceSamples(rowIndex) = numel(clean);
        processedSamples(rowIndex) = numel(processed);
        estimatedLagSamples(rowIndex) = lag;
        outputSNRdB(rowIndex) = currentSNR;
        snrImprovementdB(rowIndex) = currentImprovement;
        stoiScore(rowIndex) = stoi(processed, clean, fsExpected);
        peakAmplitude(rowIndex) = max(abs(processed));
    end
end

perUtterance = table(utteranceId, condition, sampleRate, referenceSamples, ...
    processedSamples, estimatedLagSamples, outputSNRdB, snrImprovementdB, ...
    stoiScore, peakAmplitude);

summaryCondition = conditions(:);
nUtterances = zeros(numel(conditions), 1);
meanOutputSNRdB = nan(numel(conditions), 1);
meanSNRImprovementdB = nan(numel(conditions), 1);
meanSTOI = nan(numel(conditions), 1);
maxAbsLagSamples = zeros(numel(conditions), 1);
maxPeakAmplitude = zeros(numel(conditions), 1);

for c = 1:numel(conditions)
    use = perUtterance.condition == conditions(c);
    nUtterances(c) = nnz(use);
    meanOutputSNRdB(c) = mean(perUtterance.outputSNRdB(use), "omitnan");
    meanSNRImprovementdB(c) = mean( ...
        perUtterance.snrImprovementdB(use), "omitnan");
    meanSTOI(c) = mean(perUtterance.stoiScore(use), "omitnan");
    maxAbsLagSamples(c) = max(abs(perUtterance.estimatedLagSamples(use)));
    maxPeakAmplitude(c) = max(perUtterance.peakAmplitude(use));
end

summary = table(summaryCondition, nUtterances, meanOutputSNRdB, ...
    meanSNRImprovementdB, meanSTOI, maxAbsLagSamples, maxPeakAmplitude, ...
    'VariableNames', {'condition', 'n_utterances', 'mean_output_snr_db', ...
    'mean_snr_improvement_db', 'mean_stoi', 'max_abs_lag_samples', ...
    'max_peak_amplitude'});

writetable(perUtterance, fullfile(analysisDir, ...
    "pilot_metrics_per_utterance.csv"));
writetable(summary, fullfile(analysisDir, "pilot_metrics_summary.csv"));

disp(summary);
fprintf("Pilot metrics saved in %s\n", analysisDir);
end
