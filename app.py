import os
import json
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="TriageBuddy", page_icon="🩺", layout="centered")

st.markdown("""
<style>
.stApp { background: linear-gradient(180deg, #f0f7ff 0%, #ffffff 40%); }
h1 { color: #0b5394; }
.care-badge { padding: 6px 14px; border-radius: 20px; font-weight: 700; color: white; display:inline-block; }
.badge-emergency { background:#d32f2f; }
.badge-urgent-care { background:#f57c00; }
.badge-doctor-soon { background:#fbc02d; color:black; }
.badge-pharmacist { background:#7cb342; }
.badge-self-care { background:#2e7d32; }
.card { background:white; border:1px solid #e3e8ef; border-radius:12px; padding:16px; margin-bottom:12px; box-shadow: 0 2px 6px rgba(0,0,0,0.05); }
</style>
""", unsafe_allow_html=True)

PROVIDER = os.getenv("AI_PROVIDER", "groq")  # groq | ollama | openai | featherless | mock
BASE_URLS = {
    "groq": os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
    "openai": "https://api.openai.com/v1",
    "featherless": os.getenv("FEATHERLESS_BASE_URL", "https://api.featherless.ai/v1"),
    "ollama": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
}
MODELS = {
    "groq": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
    "openai": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    "featherless": os.getenv("FEATHERLESS_MODEL", "zai-org/GLM-4.5-Air"),
    "ollama": os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
}
API_KEYS = {
    "groq": os.getenv("GROQ_API_KEY", ""),
    "openai": os.getenv("OPENAI_API_KEY", ""),
    "featherless": os.getenv("FEATHERLESS_API_KEY", ""),
    "ollama": os.getenv("OLLAMA_API_KEY", "ollama"),
}

SYSTEM_PROMPT = """You are TriageBuddy, an AI health intake assistant. You are NOT a doctor and must never claim to diagnose.

Your job:
1. Ask EXACTLY ONE short question per turn, max one sentence. Never ask more than one thing at a time. Never repeat a question you already asked. Progress through intake topics one turn at a time: age range → main symptom → how long → severity → red flags (chest pain, breathing trouble, fainting, severe bleeding) → fever → meds/allergies → other symptoms. Vary your phrasing; do not copy earlier questions.
2. After 6-10 exchanges with the patient (not counting your questions), or immediately if the patient reports an emergency red flag, output the final triage report.

When you decide it's time to wrap up, respond with a JSON object wrapped in ```json fences containing exactly these keys:
{
  "summary": "1-2 sentence plain-English summary of the situation",
  "care_level": one of "self-care", "pharmacist", "doctor-soon", "urgent-care", "emergency",
  "reasons": ["bullet reason 1", "..."],
  "red_flags": ["red flag text or empty"],
  "self_care_tips": ["tip 1", "..."]
}
Do not include any other text when outputting the JSON.

If the patient's answers suggest an emergency, always set care_level to "emergency" and put "Call emergency services now" in red_flags.
"""

DISCLAIMER = "⚠️ TriageBuddy is an AI intake helper, not a medical professional. Always seek real care for anything serious."

SAMPLE_SCENARIOS = [
    "I've had chest pain and shortness of breath for 20 minutes",
    "My 7-year-old has had a fever of 39°C for 3 days",
    "I've had a headache and mild nausea since yesterday morning",
    "I cut my finger pretty deep cooking, it won't stop bleeding",
]


def get_client():
    from openai import OpenAI
    return OpenAI(base_url=BASE_URLS[PROVIDER], api_key=API_KEYS[PROVIDER])


def ask_llm(messages):
    if PROVIDER == "mock":
        n_user = sum(1 for m in messages if m["role"] == "user")
        convo = " ".join(m["content"].lower() for m in messages if m["role"] == "user")
        emergency = any(w in convo for w in ["chest pain", "shortness of breath", "can't breathe", "breathing trouble", "fainting", "unconscious", "bleeding"])
        if n_user <= 1:
            return "Sorry to hear that. How old are you (age range is fine), and how long have you had these symptoms?"
        if n_user == 2:
            return "Got it. On a scale of 1-10, how bad is it? Any fever or chills?"
        if emergency:
            return '```json\n{"summary": "Possible cardiac or respiratory emergency — needs immediate evaluation.", "care_level": "emergency", "reasons": ["Patient reported chest pain / breathing trouble"], "red_flags": ["Call emergency services now"], "self_care_tips": ["Do not drive yourself", "Chew aspirin only if advised"]}\n```'
        return '```json\n{"summary": "Mock triage: mild symptoms reported, likely viral.", "care_level": "doctor-soon", "reasons": ["Symptoms persist", "Severity reported"], "red_flags": [], "self_care_tips": ["Rest", "Hydrate", "See a doctor if it worsens"]}\n```'
    client = get_client()
    resp = client.chat.completions.create(
        model=MODELS[PROVIDER],
        messages=messages,
        temperature=0.3,
    )
    text = resp.choices[0].message.content
    if "```json" not in text:
        # Guardrail: if the model dumps a wall of questions, force a single short question.
        n_questions = text.count("?")
        n_lines = text.count("\n")
        if n_questions > 1 or n_lines > 4:
            retry = client.chat.completions.create(
                model=MODELS[PROVIDER],
                messages=messages + [{"role": "user", "content": "[system] Please respond with exactly ONE short question, one sentence."}],
                temperature=0.2,
            )
            text = retry.choices[0].message.content
    return text


def try_parse_final(text):
    if "```json" not in text:
        return None
    try:
        block = text.split("```json", 1)[1].split("```", 1)[0].strip()
        return json.loads(block)
    except Exception:
        return None


def render_scale_chart(question_text):
    """If the assistant asked a 1-10 scale question, show a symptom-specific scale."""
    import re
    m = re.search(r"scale of 1[-–]10[^.]*?([a-z]+)", question_text, re.I)
    if not m or "scale of 1" not in question_text.lower():
        m = re.search(r"([a-z ]{3,20}) (?:on|of) a scale|scale of 1[-–]10.*?(pain|nausea|vomiting|dizziness|fatigue|headache|weakness|anxiety|fever|itching|shortness)", question_text, re.I)
    symptom = None
    keywords = ["pain", "nausea", "vomiting", "dizziness", "fatigue", "headache", "weakness", "anxiety", "fever", "itching", "shortness"]
    lower = question_text.lower()
    focus = lower
    for anchor in ["rate your", "rate the", "rate how", "rating for", "level of"]:
        if anchor in lower:
            focus = lower.split(anchor, 1)[1]
            break
    for k in keywords:
        if k in focus:
            symptom = k
            break
    if symptom is None:
        for k in keywords:
            if k in lower:
                symptom = k
                break
    if "scale of 1" not in lower and "1-10" not in lower and "1–10" not in lower:
        return
    label = symptom.capitalize() if symptom else "Symptom"
    descriptors = {
        "pain": ["barely noticeable", "very mild, occasional", "mild, nagging", "noticeable, distracting", "moderate, affects activity", "uncomfortable often", "strong, hard to focus", "very strong", "severe, keeps you up", "worst imaginable"],
        "nausea": ["no nausea at all", "faint hint", "slight queasiness", "noticeable queasiness", "moderate, upset stomach", "clearly nauseous", "hard to eat", "want to vomit", "almost vomiting", "vomiting constantly"],
        "vomiting": ["no vomiting", "almost none", "1x today", "a couple times", "a few times", "keeps coming back", "frequently", "barely stopped", "continuous", "unable to keep anything down"],
        "dizziness": ["steady", "slight light-headed", "occasional wobble", "noticeable spin", "frequent lightheadedness", "hard to walk straight", "often unsteady", "very unsteady", "almost fainting", "can't stay on feet"],
        "fatigue": ["full energy", "slightly tired", "a bit drained", "low energy", "moderately tired", "want to nap", "hard to stay awake", "very exhausted", "need to lie down", "completely exhausted"],
        "headache": ["none", "tiny ache", "mild pressure", "noticeable", "moderate throbbing", "steady pain", "strong pain", "very strong", "severe pounding", "worst ever"],
        "weakness": ["none", "slight", "minor", "noticeable", "moderate", "heavy limbs", "hard to move", "very weak", "barely able to stand", "can't move"],
        "anxiety": ["calm", "slightly on edge", "a bit worried", "noticeable worry", "moderate anxiety", "often keyed up", "hard to relax", "very anxious", "near panic", "panic level"],
        "fever": ["none", "98.8°F / slight", "99.5°F", "100.0°F", "100.4°F", "101.0°F", "101.8°F", "102.5°F", "103.5°F", "104°F+"],
        "itching": ["none", "barely noticeable", "occasional", "mild", "moderate", "frequent", "strong urge to scratch", "very itchy", "severe", "unbearable"],
        "shortness": ["breathing easy", "slightly winded", "noticeable", "out of breath on stairs", "moderate", "harder to talk", "need to rest", "very labored", "gasping", "can't breathe"],
    }
    desc = descriptors.get(symptom, ["very mild", "slight", "a bit more", "mild-moderate", "moderate", "clearly noticeable", "strong", "very strong", "severe", "worst imaginable"])
    colors = ["#2e7d32", "#43a047", "#7cb342", "#c0ca33", "#fdd835", "#fbc02d", "#fb8c00", "#f4511e", "#e53935", "#b71c1c"]
    html = f'<div class="card"><b>How would you rate your {label.lower()}?</b><div style="display:flex;gap:4px;margin-top:8px;">'
    for i, c in enumerate(colors, start=1):
        html += (
            f'<div style="flex:1;min-width:0;">'
            f'<div style="background:{c};color:white;text-align:center;padding:10px 0;border-radius:6px;font-weight:700;">{i}</div>'
            f'<div style="font-size:10px;color:#555;text-align:center;margin-top:4px;line-height:1.2;">{desc[i-1]}</div>'
            f'</div>'
        )
    html += '</div><div style="color:#555;font-size:13px;margin-top:10px;">Tap a number below or reply with a number from <b>1</b> to <b>10</b>.</div></div>'
    st.markdown(html, unsafe_allow_html=True)
    cols = st.columns(10)
    for i in range(1, 11):
        if cols[i - 1].button(str(i), key=f"scale_{label}_{i}_{hash(question_text)}"):
            st.session_state.scale_pick = str(i)
            st.rerun()


st.title("🩺 TriageBuddy")
st.caption(DISCLAIMER)

big_mode = st.toggle("♿ Big-button mode (for people in severe pain who can't type)", value=False)

with st.sidebar:
    st.header("Demo scenarios")
    st.caption("Click one to pre-fill your opening message:")
    for i, s in enumerate(SAMPLE_SCENARIOS):
        if st.button(s, key=f"scenario_{i}"):
            st.session_state.prefill = s
    if st.button("🔄 Reset conversation"):
        st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT},
                                     {"role": "assistant", "content": "Hi, I'm TriageBuddy 🩺. I'll ask a few quick questions to understand how you're feeling, then give you a clear care recommendation. What's going on today?"}]
        st.session_state.final_report = None
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT},
                                 {"role": "assistant", "content": "Hi, I'm TriageBuddy 🩺. I'll ask a few quick questions to understand how you're feeling, then give you a clear care recommendation. What's going on today?"}]
    st.session_state.final_report = None

# Render history (skip the system message)
for m in st.session_state.messages[1:]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
    if m["role"] == "assistant":
        render_scale_chart(m["content"])

if st.session_state.final_report:
    report = st.session_state.final_report
    level = report['care_level'].replace(' ', '-')
    st.markdown(f'<span class="care-badge badge-{level}">{report["care_level"].upper()}</span>', unsafe_allow_html=True)
    st.markdown(f'<div class="card"><b>Summary</b><br>{report["summary"]}</div>', unsafe_allow_html=True)
    st.subheader("Why")
    for r in report.get("reasons", []):
        st.markdown(f"- {r}")
    if report.get("red_flags"):
        st.error("**⚠️ Red flags**")
        for f in report["red_flags"]:
            st.markdown(f"- {f}")
    tips = report.get("self_care_tips", [])
    if tips:
        st.subheader("Self-care suggestions")
        for t in tips:
            st.markdown(f"- {t}")
    if st.button("Start over"):
        st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT},
                                     {"role": "assistant", "content": "Hi, I'm TriageBuddy 🩺. I'll ask a few quick questions to understand how you're feeling, then give you a clear care recommendation. What's going on today?"}]
        st.session_state.final_report = None
        st.rerun()
else:
    if big_mode:
        st.markdown("#### Tap an answer:")
        quick = ["Yes", "No", "Not sure", "Just a little", "A lot", "I'm in severe pain", "I can't breathe", "Call emergency services", "Fever", "No fever", "I want to vomit", "I'm dizzy"]
        cols = st.columns(3)
        for i, qopt in enumerate(quick):
            if cols[i % 3].button(qopt, key=f"quick_{i}", use_container_width=True):
                st.session_state.scale_pick = qopt
                st.rerun()
    prompt = st.chat_input("Describe how you're feeling...")
    if "scale_pick" in st.session_state:
        prompt = st.session_state.pop("scale_pick")
    if "prefill" in st.session_state:
        prompt = st.session_state.pop("prefill")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        reply = ask_llm(st.session_state.messages)
        final = try_parse_final(reply)
        if final:
            st.session_state.final_report = final
            with st.chat_message("assistant"):
                st.markdown("Here's your intake summary:")
            st.rerun()
        else:
            st.session_state.messages.append({"role": "assistant", "content": reply})
            with st.chat_message("assistant"):
                st.markdown(reply)
            render_scale_chart(reply)
