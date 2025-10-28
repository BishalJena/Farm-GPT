# Indian Languages Support - Deepgram Nova-2 STT

## Overview
The FarmChat application now supports Speech-to-Text (STT) transcription for multiple Indian languages using Deepgram's Nova-2 model.

## Supported Indian Languages

The following Indian languages are fully supported for voice transcription:

| Language | Code | Deepgram Code | Native Name |
|----------|------|---------------|-------------|
| English | `en` | `en` | English |
| Hindi | `hi` | `hi` | हिन्दी |
| Punjabi | `pa` | `pa-IN` | ਪੰਜਾਬੀ |
| Tamil | `ta` | `ta` | தமிழ் |
| Telugu | `te` | `te` | తెలుగు |
| Marathi | `mr` | `mr` | मराठी |
| Bengali | `bn` | `bn` | বাংলা |
| Gujarati | `gu` | `gu` | ગુજરાતી |
| Kannada | `kn` | `kn` | ಕನ್ನಡ |
| Malayalam | `ml` | `ml` | മലയാളം |
| Odia | `or` | `or` | ଓଡ଼ିଆ |
| Urdu | `ur` | `ur` | اردو |

## How It Works

### Backend Implementation

The `VoiceSTTService` class in `voice_stt_service.py` handles the transcription:

1. **Automatic Language Detection**: Deepgram Nova-2 automatically detects the spoken language (no manual selection needed!)
2. **Transcription Options**: Uses Nova-2 model with smart formatting and punctuation enabled
3. **Multi-language Support**: Works seamlessly across all 12 supported languages without any configuration

### Frontend Implementation

The `ChatInterface` component handles voice input:

1. **Voice Button**: Users tap the microphone button to start recording
2. **Automatic Language Detection**: No language selection needed - speak in any supported language!
3. **Text Appending**: New transcriptions are appended to existing text in the input box (not replaced)
4. **Offline Support**: Voice recordings made offline are processed when connection is restored

## Voice Input Behavior

### Appending Text
When you use voice input multiple times:
- The first voice input adds text to the empty input box
- Subsequent voice inputs **append** to the existing text with a space separator
- This allows building complex messages through multiple voice inputs
- You can edit the transcribed text before sending

Example:
1. First voice input: "मेरे खेत में कीट"
2. Input box shows: "मेरे खेत में कीट"
3. Second voice input: "लग गए हैं"
4. Input box shows: "मेरे खेत में कीट लग गए हैं"

## API Usage

### Voice Transcription Endpoint

**POST** `/api/voice/transcribe`

**Request:**
```json
{
  "audio_data": "base64_encoded_audio"
}
```

Note: Language parameter is optional and not used - language is automatically detected!

**Response:**
```json
{
  "success": true,
  "text": "मेरे खेत में कीट लग गए हैं",
  "confidence": 0.95,
  "language": "hi",
  "processing_time": 1.23
}
```

### Voice Capabilities Endpoint

**GET** `/api/voice/capabilities`

Returns list of supported languages and whether the service is available.

## Configuration

### Environment Variables

- `DEEPGRAM_API_KEY`: Your Deepgram API key (required)

### Frontend Configuration

No configuration needed! The system automatically detects the spoken language using Deepgram Nova-2's built-in language detection. Just speak in any of the supported languages and it will work.

## Testing Voice Input

To test voice input with Indian languages:

1. **Start Recording**: Tap the microphone button (🎤)
2. **Speak**: Speak clearly in ANY supported language - no need to select language first!
3. **Stop Recording**: Tap the microphone button again
4. **Review**: The transcribed text appears in the input box in your spoken language
5. **Add More**: Tap the microphone again to add more text (works with mixed languages too!)
6. **Edit**: You can manually edit the transcribed text before sending
7. **Send**: Press Enter or click the send button

**Try it**: Speak in Hindi, then click 🎤 again and speak in English - both will be transcribed correctly!

## Accuracy Tips

For best transcription accuracy:

1. **Clear Audio**: Speak in a quiet environment
2. **Moderate Pace**: Speak at a natural, moderate pace
3. **Good Microphone**: Use a quality microphone (or phone's built-in mic works great!)
4. **Clear Speech**: Speak clearly and naturally - the AI will detect your language automatically
5. **Short Segments**: For long messages, break them into shorter voice segments (use multiple recordings)

## Troubleshooting

### No Transcription
- Check if `DEEPGRAM_API_KEY` is configured
- Verify microphone permissions are granted
- Check internet connection
- Ensure you're speaking clearly

### Poor Accuracy
- Reduce background noise
- Speak more clearly
- Try shorter voice segments (5-10 seconds)
- Make sure your microphone is working properly

### Text Being Replaced Instead of Appended
- This has been fixed in the latest version
- Clear browser cache and reload if you still experience this issue

## Language Detection ✨

The system **AUTOMATICALLY detects** the spoken language using Deepgram Nova-2's built-in detection:

1. **No Setup Required**: Just tap 🎤 and speak in any supported language
2. **Mixed Languages**: You can even use different languages in different voice inputs
3. **Instant Detection**: Language is detected in real-time during transcription
4. **High Accuracy**: Deepgram's Nova-2 model is highly accurate at detecting Indian languages

**Example Mixed Language Usage**:
- First input (Hindi): "मेरे खेत में कीट लग गए हैं"
- Second input (English): "what should I do"
- Result: "मेरे खेत में कीट लग गए हैं what should I do"

## Future Enhancements

Potential improvements for future versions:
- ✅ ~~Automatic language detection~~ (Already implemented!)
- Real-time streaming transcription (transcribe while speaking)
- Speaker diarization for multi-person conversations
- Custom vocabulary for agricultural terms (crop names, disease names, etc.)
- Dialect support within languages (regional variations)
- Voice commands (e.g., "send message", "clear text")
