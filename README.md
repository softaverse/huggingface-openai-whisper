
# OpenAI Whisper Speech Recognition App

## Description

This Python script implements OpenAI's Whisper speech recognition model using the Hugging Face Transformers library. Its primary function is to transcribe audio files into text, with a focus on Chinese language input.

## Key Features

- Uses the "whisper-medium" model from OpenAI
- GPU acceleration support (falls back to CPU if unavailable)
- Efficient memory usage through chunk processing
- Word-level timestamp generation
- Command-line interface for easy audio file input
- Performance timing for execution analysis

## Additional Capabilities

The script includes commented code for potential integration with OpenAI's GPT-3.5 Turbo model, allowing for:

- Summarization of transcribed text
- Further natural language processing tasks

## Use Case

Ideal for quickly and accurately transcribing Chinese audio content, with room for expansion into more advanced NLP applications.
