function [development, test] = prepare_data_splits(projectRoot, corpusRoot)
%PREPARE_DATA_SPLITS Create separate development and test sets.
% Both sets use LibriSpeech dev-clean-2. They are not the official test set.

arguments
    projectRoot (1,1) string
    corpusRoot (1,1) string
end

splitDir = fullfile(projectRoot, "data", "splits");
devManifestPath = fullfile(splitDir, "development_manifest.csv");
testManifestPath = fullfile(splitDir, "test_manifest.csv");
if isfile(devManifestPath) && isfile(testManifestPath)
    development = readtable(devManifestPath, Delimiter=",", ...
        TextType="string", VariableNamingRule="preserve");
    test = readtable(testManifestPath, Delimiter=",", ...
        TextType="string", VariableNamingRule="preserve");
    return;
end

if ~isfolder(splitDir), mkdir(splitDir); end
speechFiles = dir(fullfile(corpusRoot, "**", "*.flac"));
textFiles = dir(fullfile(corpusRoot, "**", "*.trans.txt"));
assert(~isempty(speechFiles) && ~isempty(textFiles), ...
    "LibriSpeech audio or transcript files were not found.");

transcriptMap = containers.Map(KeyType="char", ValueType="char");
for i = 1:numel(textFiles)
    lines = splitlines(string(fileread(fullfile(textFiles(i).folder, textFiles(i).name))));
    for j = 1:numel(lines)
        token = regexp(char(strtrim(lines(j))), '^(\S+)\s+(.+)$', 'tokens', 'once');
        if ~isempty(token)
            transcriptMap(token{1}) = token{2};
        end
    end
end

paths = sort(string(fullfile({speechFiles.folder}, {speechFiles.name}))');
utteranceId = strings(0, 1);
speakerId = strings(0, 1);
referenceText = strings(0, 1);
sourcePath = strings(0, 1);
durationSeconds = zeros(0, 1);

for i = 1:numel(paths)
    [~, id] = fileparts(paths(i));
    parts = split(string(id), "-");
    info = audioinfo(paths(i));
    if info.SampleRate ~= 16000 || info.NumChannels ~= 1 || ...
            info.Duration < 2 || info.Duration > 15 || ~isKey(transcriptMap, id)
        continue;
    end
    utteranceId(end+1, 1) = string(id); %#ok<AGROW>
    speakerId(end+1, 1) = parts(1); %#ok<AGROW>
    referenceText(end+1, 1) = string(transcriptMap(id)); %#ok<AGROW>
    sourcePath(end+1, 1) = paths(i); %#ok<AGROW>
    durationSeconds(end+1, 1) = info.Duration; %#ok<AGROW>
end

candidates = table(utteranceId, speakerId, referenceText, sourcePath, durationSeconds);
candidates = sortrows(candidates, ["speakerId", "utteranceId"]);
speakers = unique(candidates.speakerId, "stable");
assert(numel(speakers) >= 15, "Too few speakers for a disjoint split.");

devSpeakers = speakers(1:5);
devRows = zeros(5, 1);
for i = 1:5
    devRows(i) = find(candidates.speakerId == devSpeakers(i), 1, "first");
end

testSpeakers = speakers(6:end);
testRows = zeros(0, 1);
roundIndex = 1;
while numel(testRows) < 50
    added = false;
    for i = 1:numel(testSpeakers)
        speakerRows = find(candidates.speakerId == testSpeakers(i));
        if numel(speakerRows) >= roundIndex
            testRows(end+1, 1) = speakerRows(roundIndex); %#ok<AGROW>
            added = true;
            if numel(testRows) == 50, break; end
        end
    end
    assert(added, "Could not select 50 test utterances.");
    roundIndex = roundIndex + 1;
end

development = buildManifest(candidates(devRows, :), "development", projectRoot);
test = buildManifest(candidates(testRows, :), "test", projectRoot);
assert(isempty(intersect(development.speaker_id, test.speaker_id)), ...
    "Development and test speakers overlap.");

writetable(development, devManifestPath);
writetable(test, testManifestPath);
fprintf("Development: %d utterances, %d speakers\n", ...
    height(development), numel(unique(development.speaker_id)));
fprintf("Test: %d utterances, %d speakers\n", ...
    height(test), numel(unique(test.speaker_id)));
end


function manifest = buildManifest(rows, splitName, projectRoot)
cleanDir = fullfile(projectRoot, "data", "clean", splitName);
if ~isfolder(cleanDir), mkdir(cleanDir); end
relativePath = strings(height(rows), 1);
for i = 1:height(rows)
    outputPath = fullfile(cleanDir, rows.utteranceId(i) + ".wav");
    if ~isfile(outputPath)
        [audio, fs] = audioread(rows.sourcePath(i));
        assert(fs == 16000 && size(audio, 2) == 1);
        audiowrite(outputPath, audio, fs, BitsPerSample=24);
    end
    relativePath(i) = "data/clean/" + splitName + "/" + rows.utteranceId(i) + ".wav";
end
manifest = table(rows.utteranceId, relativePath, rows.referenceText, ...
    repmat(string(splitName), height(rows), 1), ...
    repmat("LibriSpeech dev-clean-2", height(rows), 1), ...
    rows.sourcePath, rows.speakerId, rows.durationSeconds, ...
    'VariableNames', {'utterance_id', 'clean_wav', 'reference_text', ...
    'project_split', 'source_partition', 'source_path', 'speaker_id', ...
    'duration_seconds'});
end
