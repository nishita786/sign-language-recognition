"""Local website that spells words from a small ASL alphabet.

Run from the project root:

    python src/app.py

Then open http://127.0.0.1:7860
"""

import sys
from collections import deque
from pathlib import Path

import gradio as gr
import mediapipe as mp
import numpy as np
import tensorflow as tf

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from asl_letters import MODEL_PATH, hand_features, load_letters

STABLE_COUNT = 3
CONFIDENCE = 0.7


def new_state():
    return {
        "recent": deque(maxlen=STABLE_COUNT),
        "letter": "...",
        "word": "",
        "missing": 0,
    }


print("Loading model...")
MODEL = tf.keras.models.load_model(MODEL_PATH)
LETTERS = load_letters()
HANDS = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)
print(f"Model ready. Letters: {', '.join(LETTERS)}")


def recognize_frame(image, state):
    if not isinstance(state, dict) or "recent" not in state:
        state = new_state()

    if image is None:
        return word_markup(state), state

    frame = np.ascontiguousarray(image)
    if frame.ndim == 2:
        frame = np.stack([frame, frame, frame], axis=-1)
    rgb = frame[:, :, :3]
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    detected = HANDS.process(rgb).multi_hand_landmarks or []
    if not detected:
        state["missing"] += 1
        if state["missing"] >= 8:
            state["letter"] = "..."
            state["recent"].clear()
        return word_markup(state), state

    state["missing"] = 0
    features = hand_features(detected[0])[None, :]
    probabilities = MODEL.predict(features, verbose=0)[0]
    choice = int(np.argmax(probabilities))
    if float(probabilities[choice]) >= CONFIDENCE:
        state["recent"].append(LETTERS[choice])
        if (
            len(state["recent"]) == STABLE_COUNT
            and len(set(state["recent"])) == 1
            and state["letter"] != LETTERS[choice]
        ):
            state["letter"] = LETTERS[choice]
            state["word"] += LETTERS[choice]
            print(state["word"], flush=True)

    return word_markup(state), state


def clear_word(state):
    if not isinstance(state, dict) or "recent" not in state:
        state = new_state()
    state["word"] = ""
    state["letter"] = "..."
    state["recent"].clear()
    state["missing"] = 0
    print("", flush=True)
    return word_markup(state), state


def escape(text):
    return (
        (text or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def word_markup(state):
    word = escape(state["word"]) or escape(state["letter"] or "...")
    return f"<div class='sign-word'>{word}</div>"


CSS = """
.sign-word {
    font-size: 96px;
    font-weight: 700;
    text-align: center;
    color: #1f8f4e;
    min-height: 140px;
    line-height: 1.1;
    letter-spacing: 0.08em;
}
"""


with gr.Blocks(title="ASL Words") as demo:
    gr.Markdown("# ASL Words")
    gr.Markdown(
        "Hold a letter still. It is added to the word as soon as it appears. "
        "Lower your hand before a repeated letter, such as the second L in BALL. "
        "Letters: **A, B, C, D, F, I, L, V, W, Y**."
    )
    state = gr.State(new_state())
    with gr.Row():
        camera = gr.Image(
            sources=["webcam"],
            streaming=True,
            type="numpy",
            label="Camera",
            webcam_options=gr.WebcamOptions(mirror=True),
        )
        with gr.Column():
            shown = gr.HTML(value=word_markup(new_state()))
            clear = gr.Button("Clear word")
    camera.stream(
        recognize_frame,
        inputs=[camera, state],
        outputs=[shown, state],
    )
    clear.click(clear_word, inputs=[state], outputs=[shown, state])


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        css=CSS,
    )
