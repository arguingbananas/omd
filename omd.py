import socket
import wave
import pyaudio
import os
import argparse
from datetime import datetime
from faster_whisper import WhisperModel

# Define host and port
HOST = "localhost"
PORT = 50007

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024


def transcribe_audio_chunk(model, chunk_filename, beam_size=5):
    segments, info = model.transcribe(chunk_filename, beam_size=beam_size)
    transcription = f"Detected language '{info.language}' with probability {info.language_probability}\n"
    for segment in segments:
        transcription += (
            f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}\n"
        )
    return transcription


def save_transcription(transcription, file_path):
    with open(file_path, "a") as f:
        f.write(transcription + "\n")


def main(chunk_size, model_size, device, compute_type):
    # Calculate chunk size in frames
    CHUNK_FRAMES = RATE * chunk_size  # Number of frames in chunk_size seconds

    # Load the Whisper model
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    # Generate a dynamic filename for the transcription
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    transcription_filename = (
        f"transcriptions_{model_size}_{chunk_size}s_{timestamp}.txt"
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

                if len(buffer) >= CHUNK_FRAMES:
                    chunk_data = buffer[:CHUNK_FRAMES]
                    buffer = buffer[CHUNK_FRAMES:]

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
        description="Radio program to stream audio and transcribe using faster Whisper."
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=10,
        help="Chunk size in seconds for processing audio.",
    )
    parser.add_argument(
        "--model_size",
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
        help="Faster Whisper model size to use for transcription.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run the model on (cpu or cuda).",
    )
    parser.add_argument(
        "--compute_type",
        type=str,
        default="float32",
        choices=["float32", "float16", "int8", "int8_float16"],
        help="Compute type for the model.",
    )
    args = parser.parse_args()

    main(args.chunk_size, args.model_size, args.device, args.compute_type)
