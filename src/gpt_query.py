from dotenv import load_dotenv
from loguru import logger
import ollama
from faster_whisper import WhisperModel

from src.config import DEFAULT_MODEL, DEFAULT_POSITION, OUTPUT_FILE_NAME

SYS_PREFIX: str = "You are interviewing for a "
SYS_SUFFIX: str = """ position.
You will receive an audio transcription of the question. It may not be complete. You need to understand the question and write an answer to it.\n
"""

SHORT_INSTRUCTION: str = "Concisely respond, limiting your answer to 50 words."
LONG_INSTRUCTION: str = "Before answering, take a deep breath and think one step at a time. Believe the answer in no more than 150 words."

load_dotenv()

# Initialize Whisper model (this will download the model on first run)
whisper_model = WhisperModel("base", device="cpu", compute_type="int8")

def transcribe_audio(path_to_file: str = OUTPUT_FILE_NAME) -> str:
    """
    Transcribe audio from a file using faster-whisper model locally.

    Args:
        path_to_file (str, optional): Path to the audio file. Defaults to OUTPUT_FILE_NAME.

    Returns:
        str: The audio transcription.
    """
    logger.debug(f"Transcribing audio from: {path_to_file}...")
    
    try:
        # Transcribe audio
        segments, _ = whisper_model.transcribe(path_to_file, beam_size=5)
        
        # Combine all segments into one text
        transcript = " ".join([segment.text for segment in segments])
        
        logger.debug("Audio transcribed successfully")
        print("Transcription:", transcript)
        
        return transcript.strip()
        
    except Exception as error:
        logger.error(f"Can't transcribe audio: {error}")
        raise error

def generate_answer(
    transcript: str,
    short_answer: bool = True,
    temperature: float = 0.7,
    model: str = DEFAULT_MODEL,
    position: str = DEFAULT_POSITION,
) -> str:
    """
    Generate an answer to the question using Ollama.
    """
    # Generate system prompt
    system_prompt: str = SYS_PREFIX + position + SYS_SUFFIX
    if short_answer:
        system_prompt += SHORT_INSTRUCTION
    else:
        system_prompt += LONG_INSTRUCTION

    # Generate answer using Ollama
    try:
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": transcript}
            ],
            temperature=temperature,
        )
    except Exception as error:
        logger.error(f"Can't generate answer: {error}")
        raise error

    return response['message']['content']
