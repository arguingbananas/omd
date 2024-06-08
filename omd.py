import socket
import wave
import pyaudio
import whisper
import os
import numpy as np

# Define host and port
HOST = "localhost"
PORT = 50007

# Define chunk size in seconds
CHUNK_SIZE = 10  # seconds
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = RATE * CHUNK_SIZE  # Number of frames in CHUNK_SIZE seconds

# Load the Whisper model
model = whisper.load_model("medium")


def transcribe_audio_chunk(chunk_filename):
    result = model.transcribe(chunk_filename)
    return result["text"]


def save_transcription(transcription, file_path):
    with open(file_path, "a") as f:
        f.write(transcription + "\n")


def main():
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
                    transcription = transcribe_audio_chunk(chunk_filename)
                    save_transcription(transcription, "transcriptions.txt")

        except Exception as e:
            print(f"An error occurred: {e}")

        finally:
            wf.close()
            conn.close()


if __name__ == "__main__":
    main()
