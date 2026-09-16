import os
import io
import re
import json
import base64
import audioop
import wave
import asyncio
import html
import httpx

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import Response, JSONResponse

load_dotenv()

app = FastAPI()

# =========================
# CONFIG
# =========================

VOBIZ_AUTH_ID = os.getenv("VOBIZ_AUTH_ID")
VOBIZ_AUTH_TOKEN = os.getenv("VOBIZ_AUTH_TOKEN")
VOBIZ_NUMBER = os.getenv("VOBIZ_NUMBER")
TO_NUMBER = os.getenv("TO_NUMBER", "8828821585")

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

PUBLIC_URL = os.getenv("PUBLIC_URL", "").rstrip("/")

# Female Sarvam voice
SARVAM_SPEAKER = "simran"

SARVAM_TTS_MODEL = "bulbul:v3"
DEEPGRAM_MODEL = "nova-3"
GROQ_MODEL = "openai/gpt-oss-20b"

SAMPLE_RATE = 8000

# =========================
# DEBUG LOGGING
# =========================

def log(stage, message, data=None):
    print(f"\n[{stage}] {message}")
    if data is not None:
        print(json.dumps(data, ensure_ascii=False, default=str))


# =========================
# REAL ESTATE PERSONA
# =========================

SYSTEM_PROMPT = """
You are Riya, a female real-estate lead qualification caller.

You are calling a potential property buyer/seller.

Personality:
- Female
- Indian
- From Pune
- Calm
- Warm
- Conversational
- Professional but not corporate
- Never sounds robotic
- Never gives long speeches
- Uses natural Hinglish
- Can switch between Hindi and English naturally
- Says "ji" naturally
- Does not overuse "sir/ma'am"
- Sounds like an experienced real-estate relationship manager

Your job is ONLY to collect basic real-estate requirements.

Collect:
1. Name
2. Whether they are looking to buy, sell, rent, or invest
3. Property/site location
4. Property type
5. Approximate area
6. Budget
7. Timeline
8. Any important requirement

If the caller gives partial information, don't repeat everything.
Ask only the next useful question.

Keep responses SHORT.
Usually one sentence.
Maximum two short sentences.

Speak naturally in Hinglish.

Example:
"Achha ji, aap property buy karna chah rahe hain ya investment ke liye dekh rahe hain?"

Never use:
- bullet points
- markdown
- emojis
- long explanations
- technical language

If the caller says they are not interested:
politely acknowledge and end the conversation.

If the caller asks why you are calling:
say that you are calling regarding their real-estate requirement and want to understand what kind of property they are looking for.

Do not invent property listings or prices.
"""


# =========================
# CONVERSATION
# =========================

class CallSession:

    def __init__(self, websocket):
        self.ws = websocket

        self.stream_id = None
        self.input_rate = 8000
        self.input_encoding = "audio/x-mulaw"

        self.audio = bytearray()
        self.speech_ms = 0
        self.silence_ms = 0
        self.speech_active = False
        self.playing = False

        self.messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            }
        ]

    # -------------------------
    # VOBIZ EVENTS
    # -------------------------

    async def handle(self, raw):

        data = json.loads(raw)

        event = data.get("event")

        log("VOBIZ", event)

        if event == "start":
            await self.start(data)

        elif event == "media":
            await self.media(data)

        elif event in ["playedStream", "clearedAudio"]:
            self.playing = False

    # -------------------------
    # START CALL
    # -------------------------

    async def start(self, data):

        start = data.get("start", {})

        self.stream_id = (
            data.get("streamId")
            or start.get("streamId")
        )

        media_format = start.get("mediaFormat", {})

        self.input_rate = int(
            media_format.get("sampleRate", 8000)
        )

        self.input_encoding = media_format.get(
            "encoding",
            "audio/x-mulaw"
        )

        log(
            "CALL",
            "Call started",
            {
                "stream_id": self.stream_id,
                "sample_rate": self.input_rate,
                "encoding": self.input_encoding
            }
        )

        await self.speak(
            "Namaste ji, main Riya bol rahi hoon. Aapke real estate requirement ke regarding call kiya tha. Do minute baat kar sakte hain?"
        )

    # -------------------------
    # INCOMING AUDIO
    # -------------------------

    async def media(self, data):

        media = data.get("media", {})
        payload = media.get("payload")

        if not payload:
            return

        raw = base64.b64decode(payload)

        if self.input_encoding == "audio/x-mulaw":
            pcm = audioop.ulaw2lin(raw, 2)
        else:
            pcm = raw

        chunk_ms = (
            len(pcm)
            * 1000
            / (self.input_rate * 2)
        )

        rms = audioop.rms(pcm, 2)

        # simple VAD
        if rms > 250:

            if self.playing:
                await self.clear_audio()

            self.speech_active = True
            self.speech_ms += chunk_ms
            self.silence_ms = 0

            self.audio.extend(pcm)

        elif self.speech_active:

            self.silence_ms += chunk_ms
            self.audio.extend(pcm)

        # caller stopped speaking
        if (
            self.speech_active
            and self.silence_ms >= 700
        ):

            audio = bytes(self.audio)

            speech_ms = self.speech_ms

            self.audio.clear()
            self.speech_ms = 0
            self.silence_ms = 0
            self.speech_active = False

            if speech_ms >= 200:

                asyncio.create_task(
                    self.process_audio(audio)
                )

    # -------------------------
    # PROCESS CALLER SPEECH
    # -------------------------

    async def process_audio(self, pcm):

        log(
            "PIPELINE",
            "Caller stopped speaking"
        )

        try:

            # STT
            transcript = await deepgram_stt(
                pcm,
                self.input_rate
            )

            if not transcript:
                log("STT", "Empty transcript")
                return

            log(
                "STT",
                "Transcript received",
                {"text": transcript}
            )

            self.messages.append({
                "role": "user",
                "content": transcript
            })

            # LLM
            reply = await groq_llm(
                self.messages
            )

            log(
                "LLM",
                "Reply generated",
                {"reply": reply}
            )

            if not reply:
                reply = "Ji, ek baar phir bataiye."

            self.messages.append({
                "role": "assistant",
                "content": reply
            })

            # TTS
            audio = await sarvam_tts(reply)

            log(
                "TTS",
                "Audio generated",
                {"bytes": len(audio)}
            )

            await self.play(audio)

        except Exception as e:

            log(
                "ERROR",
                "Pipeline failed",
                {"error": str(e)}
            )

            await self.speak(
                "Sorry ji, ek second. Aap phir se bataiye."
            )

    # -------------------------
    # PLAY AUDIO
    # -------------------------

    async def play(self, pcm):

        chunk_size = (
            SAMPLE_RATE * 2 * 20 // 1000
        )

        self.playing = True

        for i in range(
            0,
            len(pcm),
            chunk_size
        ):

            chunk = pcm[
                i:i + chunk_size
            ]

            message = {
                "event": "playAudio",
                "streamId": self.stream_id,
                "media": {
                    "contentType": "audio/x-l16",
                    "sampleRate": SAMPLE_RATE,
                    "payload": base64.b64encode(
                        chunk
                    ).decode()
                }
            }

            await self.ws.send_text(
                json.dumps(message)
            )

        await self.ws.send_text(
            json.dumps({
                "event": "checkpoint",
                "streamId": self.stream_id,
                "name": "tts"
            })
        )

    async def speak(self, text):

        log("AGENT", text)

        audio = await sarvam_tts(text)

        await self.play(audio)

    async def clear_audio(self):

        log("BARGE-IN", "Clearing agent audio")

        await self.ws.send_text(
            json.dumps({
                "event": "clearAudio",
                "streamId": self.stream_id
            })
        )

        self.playing = False


# =========================
# DEEPGRAM STT
# =========================

async def deepgram_stt(pcm, sample_rate):

    log(
        "DEEPGRAM",
        "Sending audio to STT"
    )

    wav = io.BytesIO()

    with wave.open(wav, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)

    url = (
        "https://api.deepgram.com/v1/listen"
        "?model=nova-3"
        "&language=multi"
        "&smart_format=true"
    )

    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type": "audio/wav"
    }

    async with httpx.AsyncClient(
        timeout=15
    ) as client:

        response = await client.post(
            url,
            headers=headers,
            content=wav.getvalue()
        )

        response.raise_for_status()

        result = response.json()

    transcript = (
        result
        .get("results", {})
        .get("channels", [{}])[0]
        .get("alternatives", [{}])[0]
        .get("transcript", "")
        .strip()
    )

    log(
        "DEEPGRAM",
        "STT complete",
        {"transcript": transcript}
    )

    return transcript


# =========================
# GROQ GPT-OSS-20B
# =========================

async def groq_llm(messages):

    log(
        "GROQ",
        "Sending conversation to GPT-OSS-20B"
    )

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": GROQ_MODEL,
        "messages": messages,
        "temperature": 0.4,
        "max_tokens": 120
    }

    async with httpx.AsyncClient(
        timeout=20
    ) as client:

        response = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload
        )

        response.raise_for_status()

        result = response.json()

    reply = (
        result["choices"][0]["message"]["content"]
        .strip()
    )

    return reply


# =========================
# SARVAM TTS
# =========================

async def sarvam_tts(text):

    log(
        "SARVAM",
        "Generating female Hinglish voice"
    )

    headers = {
        "api-subscription-key": SARVAM_API_KEY,
        "Content-Type": "application/json"
    }

    payload = {
        "text": text,
        "target_language_code": "hi-IN",
        "speaker": SARVAM_SPEAKER,
        "model": SARVAM_TTS_MODEL,
        "speech_sample_rate": 8000,
        "output_audio_codec": "linear16"
    }

    async with httpx.AsyncClient(
        timeout=20
    ) as client:

        response = await client.post(
            "https://api.sarvam.ai/text-to-speech",
            headers=headers,
            json=payload
        )

        response.raise_for_status()

        result = response.json()

    audio = base64.b64decode(
        "".join(result["audios"])
    )

    # Remove WAV header if Sarvam returned WAV.
    if audio.startswith(b"RIFF"):

        with wave.open(
            io.BytesIO(audio),
            "rb"
        ) as wav:

            audio = wav.readframes(
                wav.getnframes()
            )

    return audio


# =========================
# VOBIZ ANSWER XML
# =========================

@app.api_route(
    "/answer",
    methods=["GET", "POST"]
)
async def answer(request: Request):

    ws_url = (
        PUBLIC_URL
        .replace("https://", "wss://")
        .replace("http://", "ws://")
        + "/ws"
    )

    ws_url = html.escape(
        ws_url,
        quote=False
    )

    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Stream
        bidirectional="true"
        audioTrack="inbound"
        keepCallAlive="true"
        contentType="audio/x-mulaw;rate=8000">
        {ws_url}
    </Stream>
    <Hangup/>
</Response>
"""

    log(
        "VOBIZ",
        "Answer XML requested"
    )

    return Response(
        content=xml,
        media_type="application/xml"
    )


# =========================
# VOBIZ WEBSOCKET
# =========================

@app.websocket("/ws")
async def websocket(websocket: WebSocket):

    await websocket.accept()

    log(
        "WEBSOCKET",
        "Vobiz connected"
    )

    session = CallSession(websocket)

    try:

        async for message in websocket.iter_text():

            await session.handle(message)

    except WebSocketDisconnect:

        log(
            "WEBSOCKET",
            "Call disconnected"
        )

    except Exception as e:

        log(
            "ERROR",
            "WebSocket error",
            {"error": str(e)}
        )


# =========================
# OUTBOUND CALL
# =========================

@app.post("/call")
async def make_call():

    url = (
        f"https://api.vobiz.ai/api/v1/"
        f"Account/{VOBIZ_AUTH_ID}/Call/"
    )

    payload = {
        "from": VOBIZ_NUMBER,
        "to": TO_NUMBER,
        "answer_url": f"{PUBLIC_URL}/answer"
    }

    log(
        "VOBIZ",
        "Starting outbound call",
        payload
    )

    headers = {
        "X-Auth-ID": VOBIZ_AUTH_ID,
        "X-Auth-Token": VOBIZ_AUTH_TOKEN,
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient(
        timeout=20
    ) as client:

        response = await client.post(
            url,
            headers=headers,
            json=payload
        )

    log(
        "VOBIZ",
        "Outbound response",
        {
            "status": response.status_code,
            "body": response.text[:1000]
        }
    )

    return JSONResponse(
        status_code=response.status_code,
        content={
            "status": response.status_code,
            "response": response.json()
            if response.headers.get(
                "content-type",
                ""
            ).startswith("application/json")
            else response.text
        }
    )


# =========================
# HEALTH
# =========================

@app.get("/")
async def health():

    return {
        "status": "ok",
        "agent": "Riya",
        "to": TO_NUMBER,
        "stack": [
            "Vobiz",
            "Deepgram",
            "Groq GPT-OSS-20B",
            "Sarvam Bulbul"
        ]
    }


# =========================
# RUN
# =========================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )