# Voice Speech-to-Text Setup with Deepgram Nova-2

This guide explains how to set up and use the voice input feature powered by Deepgram Nova-2.

## Overview

The voice input system uses **Deepgram Nova-2** for Speech-to-Text (STT) transcription. When users click the microphone button and speak, their voice is transcribed to text and populated in the chat input box.

## Features

- **STT Only**: Transcribes voice to text without automatic response generation
- **Multilingual Support**: Supports 10+ languages including English, Hindi, Punjabi, Tamil, Telugu, Marathi, Bengali, Gujarati, Kannada, and Malayalam
- **Smart Formatting**: Automatic punctuation and formatting
- **Review Before Send**: Users can review and edit transcribed text before sending

## Setup Instructions

### 1. Get Deepgram API Key

1. Sign up for a free account at [Deepgram](https://console.deepgram.com/signup)
2. Navigate to the API Keys section in your dashboard
3. Create a new API key
4. Copy the API key

### 2. Configure Environment Variable

Add the Deepgram API key to your `.env` file:

```bash
DEEPGRAM_API_KEY=your_deepgram_api_key_here
```

### 3. Install Dependencies

Install the required Python package:

```bash
pip install deepgram-sdk==3.7.0
```

Or if using the requirements file:

```bash
pip install -r backend/requirements.txt
```

## API Endpoints

### POST `/api/voice/transcribe`

Transcribes audio using Deepgram Nova-2 and returns the text.

**Request:**
```json
{
  "audio_data": "base64_encoded_audio",
  "language": "en"
}
```

**Response:**
```json
{
  "success": true,
  "text": "transcribed text here",
  "confidence": 0.95,
  "language": "en",
  "processing_time": 0.5
}
```

### GET `/api/voice/capabilities`

Get voice processing capabilities and supported languages.

**Response:**
```json
{
  "available": true,
  "model": "nova-2",
  "provider": "Deepgram",
  "supported_languages": [
    {"code": "en", "name": "English", "deepgram": "en-US"},
    {"code": "hi", "name": "Hindi", "deepgram": "hi"},
    ...
  ],
  "features": ["smart_format", "punctuation", "multilingual"]
}
```

## Usage Flow

1. **User clicks microphone button** → Starts recording
2. **User speaks** → Audio is captured via MediaRecorder API
3. **User clicks again to stop** → Recording stops
4. **Audio is sent to backend** → Transcribed by Deepgram Nova-2
5. **Text appears in input box** → User can review and edit
6. **User clicks send** → Message is sent to chat as normal

## Supported Audio Formats

- WebM (preferred for web browsers)
- MP4
- WAV

The system automatically detects and uses the best format supported by the browser.

## Language Support

| Language   | Code | Deepgram Code |
|------------|------|---------------|
| English    | en   | en-US         |
| Hindi      | hi   | hi            |
| Punjabi    | pa   | pa            |
| Tamil      | ta   | ta            |
| Telugu     | te   | te            |
| Marathi    | mr   | mr            |
| Bengali    | bn   | bn            |
| Gujarati   | gu   | gu            |
| Kannada    | kn   | kn            |
| Malayalam  | ml   | ml            |

## Troubleshooting

### "Deepgram API key not configured" Error

**Solution:** Make sure the `DEEPGRAM_API_KEY` environment variable is set in your `.env` file.

### Transcription Quality Issues

1. Check microphone permissions in browser
2. Ensure good audio quality (reduce background noise)
3. Speak clearly and at a moderate pace
4. Check that the correct language is selected

### Browser Compatibility

The voice input feature requires:
- Modern browser with MediaRecorder API support
- Microphone permissions granted
- HTTPS connection (required for microphone access)

### Performance Tips

- Deepgram Nova-2 typically processes audio in < 1 second
- Audio is automatically compressed before sending
- Mobile devices use optimized settings (lower sample rate, mono audio)

## Cost Considerations

Deepgram offers:
- **Free Tier**: $200 credit for new accounts
- **Pay-as-you-go**: ~$0.0125 per minute of audio
- Nova-2 is their latest model with best accuracy

## Development Notes

### Key Files

- `backend/voice_stt_service.py` - Main STT service implementation
- `backend/server.py` - API endpoint definition
- `frontend/src/components/ChatInterface.js` - Frontend voice recording logic

### Testing

Test the voice capabilities endpoint:

```bash
curl http://localhost:8000/api/voice/capabilities
```

Test transcription with a sample audio file:

```bash
# Convert audio to base64
base64 sample.wav | tr -d '\n' > audio_base64.txt

# Send to API
curl -X POST http://localhost:8000/api/voice/transcribe \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "audio_data": "'$(cat audio_base64.txt)'",
    "language": "en"
  }'
```

## Future Enhancements

Potential improvements:
- Real-time streaming transcription (using WebSocket)
- Voice activity detection
- Automatic language detection
- Speaker diarization for multi-speaker scenarios
- Custom vocabulary for agricultural terms

## Support

For Deepgram-specific issues:
- [Deepgram Documentation](https://developers.deepgram.com/)
- [Deepgram Support](https://deepgram.com/contact-us)

For application issues:
- Check server logs for detailed error messages
- Verify API key is valid
- Test with `/api/voice/capabilities` endpoint first
