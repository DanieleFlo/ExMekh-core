param([Parameter(Mandatory=$true)][string]$Output)
$ErrorActionPreference = 'Stop'
# Development fixture generation only; no microphone or audio playback.
Add-Type -AssemblyName System.Speech
$suiteSpeechRoot = [System.IO.Path]::GetFullPath($Output)
[System.IO.Directory]::CreateDirectory($suiteSpeechRoot) | Out-Null
$suiteVoice = New-Object System.Speech.Synthesis.SpeechSynthesizer
try {
    $suiteVoice.SetOutputToNull()
    $suiteVoice.SelectVoice('Microsoft Zira Desktop')
    $suiteLines = @('A red square beside a blue circle.', 'A green triangle on a white page.',
        'A yellow star above a purple box.', 'Two orange circles below a black line.',
        'A blue square beside a yellow circle.', 'A purple triangle on a grey page.',
        'A green star above an orange box.', 'Two red circles below a white line.')
    for ($suiteIndex = 0; $suiteIndex -lt $suiteLines.Count; $suiteIndex++) {
        $suiteFile = Join-Path $suiteSpeechRoot ('speech-{0}.wav' -f $suiteIndex)
        if (Test-Path -LiteralPath $suiteFile) { throw 'Fixture already exists' }
        $suiteFormat = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
        $suiteVoice.SetOutputToWaveFile($suiteFile, $suiteFormat)
        $suiteVoice.Speak($suiteLines[$suiteIndex])
        $suiteVoice.SetOutputToNull()
    }
    @{voice=$suiteVoice.Voice.Name; version=$suiteVoice.Voice.AdditionalInfo; text=$suiteLines} |
        ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $suiteSpeechRoot 'speech.json') -Encoding utf8
} finally { $suiteVoice.Dispose() }
