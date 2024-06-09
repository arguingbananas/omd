import socket
import wave
import pyaudio
import whisper
import os
import argparse
import numpy as np
from datetime import datetime

# Define host and port
HOST = "localhost"
PORT = 50007

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100


def transcribe_audio_chunk(model, chunk_filename):
    result = model.transcribe(chunk_filename)
    return result["text"]


def save_transcription(transcription, file_path):
    with open(file_path, "a") as f:
        f.write(transcription + "\n")


def main(chunk_size, model_type):
    # Calculate chunk size in frames
    CHUNK = RATE * chunk_size  # Number of frames in chunk_size seconds

    # Load the Whisper model
    model = whisper.load_model(model_type)

    # Generate a dynamic filename for the transcription
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    transcription_filename = (
        f"transcriptions_{model_type}_{chunk_size}s_{timestamp}.txt"
    )

    # Set up the server
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        print(f"Server listening on {HOST}:{PORT}")

        # Wait for a connection
        conn, addr = s.accept()
        print("Connected by", addr)

        audio = pyaudio.PyAudio()

        # Open a .wav file for writing
        wf = wave.open("received_audio.wav", "wb")
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)

        buffer = bytearray()
        try:
            while True:
                data = conn.recv(CHUNK)
                if not data:
                    break
                buffer.extend(data)
                wf.writeframes(data)

                if len(buffer) >= CHUNK:
                    chunk_data = buffer[:CHUNK]
                    buffer = buffer[CHUNK:]

                    # Save the chunk to a file
                    chunk_filename = "chunk.wav"
                    with wave.open(chunk_filename, "wb") as chunk_file:
                        chunk_file.setnchannels(CHANNELS)
                        chunk_file.setsampwidth(audio.get_sample_size(FORMAT))
                        chunk_file.setframerate(RATE)
                        chunk_file.writeframes(chunk_data)

                    # Transcribe the chunk and save the result
                    transcription = transcribe_audio_chunk(model, chunk_filename)
                    save_transcription(transcription, transcription_filename)

        except Exception as e:
            print(f"An error occurred: {e}")

        finally:
            wf.close()
            conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Radio program to stream audio and transcribe using Whisper."
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=10,
        help="Chunk size in seconds for processing audio.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="base",
        choices=[
            "tiny",
            "tiny.en",
            "base",
            "base.en",
            "small",
            "small.en",
            "medium",
            "medium.en",
            "large",
            "large-v2",
            "large-v3",
        ],
        help="Whisper model type to use for transcription.",
    )
    args = parser.parse_args()

    main(args.chunk_size, args.model_type)
