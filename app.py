"""
AgriMind AI — Smart Agriculture Assistant
Flask + Gemini multimodal plant-image assessment.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
from io import BytesIO

from flask import Flask, jsonify, render_template, request, session
from google import genai
from google.genai import types
from PIL import Image, UnidentifiedImageError


APP_ENV = os.getenv("APP_ENV", "production").lower()
MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_MB", "8")) * 1024 * 1024

# Updated Gemini model
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

app = Flask(__name__)

app.config.update(
    SECRET_KEY=os.getenv("FLASK_SECRET_KEY") or secrets.token_hex(32),
    MAX_CONTENT_LENGTH=MAX_IMAGE_BYTES + 256 * 1024,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=(APP_ENV == "production"),
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agrimind")

client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None


VISION_PROMPT = """
You are AgriMind AI, an agricultural plant-health assistant.

Analyze the uploaded plant image carefully.

Return ONLY valid JSON with these keys:

{
  "is_plant": true,
  "plant": "Plant name",
  "diagnosis": "Likely condition",
  "confidence_label": "low, medium, or high",
  "observations": ["observation 1", "observation 2"],
  "treatment": "Recommended next steps",
  "prevention": "Prevention advice",
  "home_care": "Supportive home care",
  "limitations": "What cannot be confirmed"
}

Rules:
- Never invent a disease.
- If the image is not a plant, set is_plant=false.
- Do not provide fake confidence percentages.
- Do not claim to measure soil moisture or nutrients from an image.
- Avoid prescribing pesticide doses.
- Explain uncertainty clearly.
- Recommend local agricultural expert confirmation when needed.
"""


CHAT_SYSTEM_PROMPT = """
You are AgriMind AI, a smart agricultural assistant.

Help farmers, students, and gardeners with:

1. Crop information.
2. Plant diseases.
3. Plant health.
4. Irrigation guidance.
5. Soil management.
6. Fertilizers and nutrients.
7. Sustainable agriculture.
8. Gardening.
9. Crop protection.
10. Agricultural education.

LANGUAGE SUPPORT:

You support:
- English
- Hindi
- Telugu
- Tamil
- Kannada

Always respond in the language requested by the user.

If the user writes in Telugu, respond in Telugu.
If the user writes in Hindi, respond in Hindi.
If the user writes in Tamil, respond in Tamil.
If the user writes in Kannada, respond in Kannada.

Use simple language and practical examples.

Do not invent scientific sources, field trials, or pesticide registrations.
Do not prescribe chemical doses.
Do not claim that a photograph confirms a disease with certainty.
"""


def get_client():
    if client is None:
        raise RuntimeError(
            "Gemini API is not configured. "
            "Please add GEMINI_API_KEY in Render Environment."
        )

    return client


def read_and_validate_image(file_storage):

    if file_storage is None or not file_storage.filename:
        raise ValueError("Please choose an image first.")

    raw = file_storage.read(MAX_IMAGE_BYTES + 1)

    if not raw:
        raise ValueError("The uploaded image is empty.")

    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError("Image exceeds the maximum upload size.")

    try:

        with Image.open(BytesIO(raw)) as image:
            image.verify()

        with Image.open(BytesIO(raw)) as image:

            image = image.convert("RGB")
            image.thumbnail((1600, 1600))

            output = BytesIO()

            image.save(
                output,
                format="JPEG",
                quality=88,
                optimize=True
            )

            return output.getvalue(), "image/jpeg"

    except (UnidentifiedImageError, OSError, ValueError) as exc:

        raise ValueError(
            "Please upload a valid JPG, PNG, or WEBP image."
        ) from exc


def strip_fences(value):

    if value.startswith("```"):

        lines = value.splitlines()

        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        return "\n".join(lines).strip()

    return value


def parse_model_json(text):

    cleaned = strip_fences((text or "").strip())

    try:
        data = json.loads(cleaned)

    except json.JSONDecodeError as exc:

        raise RuntimeError(
            "AI returned an unreadable result. Please try again."
        ) from exc

    if not isinstance(data, dict):
        raise RuntimeError("Unexpected AI response.")

    return data


def clean_text(value, fallback):

    if not isinstance(value, str):
        return fallback

    value = value.strip()

    return value[:1200] if value else fallback


def clean_list(value, limit=5):

    if not isinstance(value, list):
        return []

    return [
        clean_text(item, "")
        for item in value
        if isinstance(item, str) and item.strip()
    ][:limit]


def assess_plant_image(image_bytes, mime_type="image/jpeg"):

    response = get_client().models.generate_content(

        model=GEMINI_MODEL,

        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(
                        text=VISION_PROMPT
                    ),
                    types.Part.from_bytes(
                        data=image_bytes,
                        mime_type=mime_type
                    ),
                ],
            )
        ],

        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            max_output_tokens=900,
            temperature=0.2,
        ),
    )

    data = parse_model_json(response.text or "")

    is_plant = bool(data.get("is_plant", False))

    confidence = str(
        data.get("confidence_label", "low")
    ).lower()

    if confidence not in {"low", "medium", "high"}:
        confidence = "low"

    diagnosis = clean_text(
        data.get("diagnosis"),
        "Unclear image"
    )

    if not is_plant:
        diagnosis = "Not a plant image"
        confidence = "low"

    return {

        "is_plant": is_plant,

        "result": diagnosis,

        "plant": clean_text(
            data.get("plant"),
            "Unknown"
        ),

        "confidence_label": confidence.capitalize(),

        "confidence": None,

        "soil_moisture": None,

        "observations": clean_list(
            data.get("observations")
        ) if is_plant else [],

        "pesticide": clean_text(
            data.get("treatment"),
            "Please consult an agricultural expert."
        ),

        "prevention": clean_text(
            data.get("prevention"),
            "Monitor your plant regularly."
        ),

        "remedy_name": "Supportive Plant Care",

        "remedy_process": clean_text(
            data.get("home_care"),
            "Keep the plant healthy and monitor symptoms."
        ),

        "limitations": clean_text(
            data.get("limitations"),
            "A photograph cannot confirm every plant condition."
        ),
    }


def save_detection(result):

    session["detection"] = {
        key: result.get(key)
        for key in (
            "is_plant",
            "result",
            "plant",
            "confidence_label",
            "observations",
            "pesticide",
            "prevention",
            "remedy_name",
            "remedy_process",
            "limitations",
        )
    }


def empty_template_context():

    return dict(
        result=None,
        confidence=None,
        confidence_label=None,
        pesticide=None,
        soil_moisture=None,
        remedy_name=None,
        remedy_process=None,
        plant=None,
        observations=[],
        prevention=None,
        limitations=None,
    )


@app.get("/")
def home():

    return render_template(
        "index.html",
        **empty_template_context()
    )


@app.post("/analyze")
def analyze():

    try:

        image_bytes, mime_type = read_and_validate_image(
            request.files.get("image")
        )

        result = assess_plant_image(
            image_bytes,
            mime_type
        )

        save_detection(result)

        return render_template(
            "index.html",
            result=result["result"],
            confidence=None,
            confidence_label=result["confidence_label"],
            pesticide=result["pesticide"],
            soil_moisture=None,
            remedy_name=result["remedy_name"],
            remedy_process=result["remedy_process"],
            plant=result["plant"],
            observations=result["observations"],
            prevention=result["prevention"],
            limitations=result["limitations"],
        )

    except ValueError as exc:

        return render_template(
            "index.html",
            error=str(exc),
            **empty_template_context()
        ), 400

    except RuntimeError as exc:

        logger.warning("Image assessment unavailable: %s", exc)

        return render_template(
            "index.html",
            error=str(exc),
            **empty_template_context()
        ), 503

    except Exception:

        logger.exception("Unexpected image assessment error")

        return render_template(
            "index.html",
            error="Image analysis failed. Please try again.",
            **empty_template_context()
        ), 500


@app.post("/camera-detect")
def camera_detect():

    try:

        image_bytes, mime_type = read_and_validate_image(
            request.files.get("image")
        )

        result = assess_plant_image(
            image_bytes,
            mime_type
        )

        save_detection(result)

        return jsonify(result)

    except ValueError as exc:

        return jsonify({"error": str(exc)}), 400

    except RuntimeError as exc:

        return jsonify({"error": str(exc)}), 503

    except Exception:

        logger.exception("Camera assessment failed")

        return jsonify({
            "error": "Camera analysis failed."
        }), 500


@app.post("/chat")
def chat():

    data = request.get_json(silent=True) or {}

    user_message = data.get("message", "")

    if not isinstance(user_message, str):
        return jsonify({"error": "Message must be text."}), 400

    user_message = user_message.strip()

    if not user_message:
        return jsonify({"error": "Please enter a message."}), 400

    if len(user_message) > 4000:

        return jsonify({
            "error": "Message is too long."
        }), 400

    try:

        history = session.get("conversation", [])

        contents = []

        for item in history[-10:]:

            role = item.get("role")
            text = item.get("content", "")

            if role in {"user", "model"} and isinstance(text, str):

                contents.append(
                    types.Content(
                        role=role,
                        parts=[
                            types.Part.from_text(text=text)
                        ]
                    )
                )

        detection = session.get("detection")

        context = ""

        if detection:

            context = (
                "\n\nLATEST PLANT ASSESSMENT:\n"
                + json.dumps(
                    detection,
                    ensure_ascii=False
                )
            )

        contents.append(

            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(
                        text=(
                            context
                            + "\n\nUSER QUESTION:\n"
                            + user_message
                        )
                    )
                ],
            )
        )

        response = get_client().models.generate_content(

            model=GEMINI_MODEL,

            contents=contents,

            config=types.GenerateContentConfig(
                system_instruction=CHAT_SYSTEM_PROMPT,
                max_output_tokens=1000,
                temperature=0.5,
            ),
        )

        answer = (response.text or "").strip()

        if not answer:

            answer = "I couldn't generate a response. Please try again."

        history.extend([

            {
                "role": "user",
                "content": user_message
            },

            {
                "role": "model",
                "content": answer
            },

        ])

        session["conversation"] = history[-10:]

        return jsonify({
            "answer": answer
        })

    except RuntimeError as exc:

        return jsonify({
            "error": str(exc)
        }), 503

    except Exception:

        logger.exception("Gemini chat request failed")

        return jsonify({
            "error": "AI assistant could not respond. Please try again."
        }), 500


@app.post("/clear-chat")
def clear_chat():

    session.pop("conversation", None)

    return jsonify({
        "success": True
    })


@app.get("/health")
def health():

    return jsonify({

        "status": "ok",

        "service": "AgriMind AI",

        "vision_ai_configured": client is not None,

        "model": GEMINI_MODEL,

    })


@app.errorhandler(413)
def request_too_large(_error):

    return jsonify({

        "error": (
            f"Upload exceeds the "
            f"{MAX_IMAGE_BYTES // (1024 * 1024)} MB limit."
        )

    }), 413


if __name__ == "__main__":

    app.run(

        host="0.0.0.0",

        port=int(os.getenv("PORT", "5000")),

        debug=(APP_ENV == "development"),

        )
