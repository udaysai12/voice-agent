import os
try:
    import site
    for sp in site.getsitepackages():
        cublas_dir = os.path.join(sp, "nvidia", "cublas", "bin")
        if os.path.exists(cublas_dir):
            try:
                os.add_dll_directory(cublas_dir)
            except Exception:
                pass
            os.environ["PATH"] = cublas_dir + os.pathsep + os.environ.get("PATH", "")
except Exception:
    pass
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

import argparse
import datetime
import json
import re
import subprocess
import sys
import threading
import time
import uuid
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel, decode_audio
from nlu_parser import (
    clean_company_name,
    clean_door_no,
    clean_street,
    clean_email,
    clean_category,
    clean_location,
    clean_person_name,
    clean_phone,
    clean_pincode,
    is_skip,
    is_user_asking_question,
    denoise_audio,
    detect_language_choice,
    translate_indic_to_english,
)

STORE_FIELDS = [
    {
        "key": "companyName",
        "label": "Company / Store Name",
        "label_te": "Store లేదా Company Name",
        "question": "What is your company or store name?",
        "question_en": "What is your company or store name?",
        "question_hi": "आपकी दुकान या कंपनी का नाम क्या है?",
        "question_te": "మీ store లేదా company name ఏంటి?",
        "required": True,
        "hint": "Store name cheppandi (e.g. Brostide, Fresh Mart, కిరాణా, బట్టల షాప్)",
    },
    {
        "key": "personName",
        "label": "Contact Person Name",
        "label_te": "Contact Person Name",
        "question": "Who is the contact person name?",
        "question_en": "Who is the contact person name?",
        "question_hi": "संपर्क व्यक्ति का नाम क्या है?",
        "question_te": "కాంటాక్ట్ person name ఏంటి?",
        "required": True,
        "hint": "Mee name cheppandi (e.g. Uday Kumar, Rahul, ఉదయ్)",
    },
    {
        "key": "phone",
        "label": "Phone Number",
        "label_te": "Mobile Number",
        "question": "What is your contact phone number?",
        "question_en": "What is your contact phone number?",
        "question_hi": "आपका मोबाइल या फ़ोन नंबर क्या है?",
        "question_te": "మీ 10 digits mobile number చెప్పండి.",
        "required": True,
        "hint": "10 digits mobile number (e.g., 9840123456)",
    },
    {
        "key": "email",
        "label": "Email Address",
        "label_te": "Email Address",
        "question": "What is your email address?",
        "question_en": "What is your email address?",
        "question_hi": "आपका ईमेल पता क्या है?",
        "question_te": "మీ email address ఏంటి? లేకపోతే skip అనండి.",
        "required": False,
        "hint": "e.g., store@gmail.com (లేదా skip anali)",
    },
    {
        "key": "storeCategory",
        "label": "Store Category",
        "label_te": "Store Category",
        "question": "What is your store category?",
        "question_en": "What is your store category?",
        "question_hi": "आपकी दुकान किस श्रेणी की है? (जैसे कपड़े, किराना, दवा)",
        "question_te": "మీ store category ఏంటి? లైక్ clothing, grocery, medical, electronics?",
        "required": True,
        "is_list": True,
        "hint": "Clothing, Grocery, Medical, Electronics, etc.",
    },
    {
        "key": "doorNo",
        "label": "Door / Shop Number",
        "label_te": "Shop లేదా Door Number",
        "question": "What is your shop or door number? Or say skip.",
        "question_en": "What is your shop or door number? Or say skip.",
        "question_hi": "आपकी दुकान या मकान का नंबर क्या है? या छोड़ दो बोलें।",
        "question_te": "మీ shop లేదా door number ఎంత? లేకపోతే skip అనండి.",
        "required": False,
        "hint": "e.g., Shop 4B, 1-2-3 (లేదా skip anali)",
    },
    {
        "key": "street",
        "label": "Street Name",
        "label_te": "Street లేదా Road Name",
        "question": "What street is your store on? Or say skip.",
        "question_en": "What street is your store on? Or say skip.",
        "question_hi": "आपकी दुकान किस गली या रोड पर है? या छोड़ दो बोलें।",
        "question_te": "మీ store ఏ street లేదా road లో ఉంది? లేకపోతే skip అనండి.",
        "required": False,
        "hint": "e.g., Main Road, MG Road, Gandhi Road (లేదా skip anali)",
    },
    {
        "key": "area",
        "label": "Area / Locality",
        "label_te": "Area లేదా Locality",
        "question": "Which area or locality is it located in? Or say skip.",
        "question_en": "Which area or locality is it located in? Or say skip.",
        "question_hi": "यह किस इलाके या एरिया में स्थित है? या छोड़ दो बोलें।",
        "question_te": "మీ store ఏ area లో ఉంది? లైక్ Madhapur, Gajuwaka? లేకపోతే skip అనండి.",
        "required": False,
        "hint": "e.g., Gajuwaka, Madhapur, Jubilee Hills (లేదా skip anali)",
    },
    {
        "key": "town",
        "label": "Town / Village",
        "label_te": "Town లేదా Village",
        "question": "Which town is it in? Or say skip.",
        "question_en": "Which town is it in? Or say skip.",
        "question_hi": "यह किस कस्बे या गाँव में है? या छोड़ दो बोलें।",
        "question_te": "మీ town లేదా village name ఏంటి? లేకపోతే skip అనండి.",
        "required": False,
        "hint": "Town లేదా village name (లేదా skip anali)",
    },
    {
        "key": "city",
        "label": "City",
        "label_te": "City Name",
        "question": "Which city is your store in?",
        "question_en": "Which city is your store in?",
        "question_hi": "आपकी दुकान किस शहर में है?",
        "question_te": "మీ store ఏ city లో ఉంది? లైక్ Vizag, Hyderabad?",
        "required": True,
        "hint": "Visakhapatnam, Hyderabad, Vijayawada, etc.",
    },
    {
        "key": "state",
        "label": "State",
        "label_te": "State",
        "question": "Which state are you located in?",
        "question_en": "Which state are you located in?",
        "question_hi": "आप किस राज्य में स्थित हैं?",
        "question_te": "ఏ state లో ఉన్నారు? Andhra Pradesh లేదా Telangana?",
        "required": True,
        "hint": "Andhra Pradesh, Telangana, etc.",
    },
    {
        "key": "pincode",
        "label": "Pincode / Zip Code",
        "label_te": "Pincode",
        "question": "What is your pincode or zip code?",
        "question_en": "What is your pincode or zip code?",
        "question_hi": "आपका पिन कोड क्या है?",
        "question_te": "మీ area 6 digits pincode ఎంత?",
        "required": True,
        "hint": "6-digit pincode (e.g., 530016, 500081)",
    },
    {
        "key": "usedReferalCode",
        "label": "Referral Code",
        "label_te": "Referral Code",
        "question": "Do you have any referral code? Or say skip.",
        "question_en": "Do you have any referral code? Or say skip.",
        "question_hi": "क्या आपके पास कोई रेफ़रल कोड है? या छोड़ दो बोलें।",
        "question_te": "దగ్గర ఏదైనా referral code ఉందా? లేకపోతే skip అనండి.",
        "required": False,
        "hint": "Referral code లేదా skip anali (Optional)",
    },
]

WHISPER_PROMPT = (
    "Indian store onboarding in English, Hindi, and Telugu. "
    "Brostide, Rahul Sharma, Uday Kumar, Visakhapatnam, Hyderabad, Andhra Pradesh, Telangana, "
    "दुकान, स्टोर, नाम, फ़ोन नंबर, कपड़े, किराना, విలేజ్, పిన్‌కోడ్, బట్టలు, కిరాణా, మందులు, "
    "email at gmail dot com, pincode 530016, phone number 9840123456"
)

FIELD_PROMPTS = {
    "companyName": "Company name, store name, shop name, Brostide, Fresh Mart, Textiles, Enterprises, दुकान का नाम, किराना, దుకాణం పేరు, బట్టల దుకాణం",
    "personName": "Contact person name, Rahul Sharma, Uday Kumar, Suresh, Priya, नाम, संपर्क, పేరు, రాహుల్, ఉదయ్, राहुल, సురేష్",
    "phone": "Phone number, mobile number, 10 digits, call me on 9840123456, 9876543210, मोबाइल नंबर, ఫోన్ నంబర్, ek do teen, okati rendu",
    "email": "Email address, name at gmail dot com, name at yahoo dot com, contact mail, ईमेल पता, ఇమెయిల్",
    "storeCategory": "Store category, clothing, grocery, fashion, electronics, pharmacy, retail, कपड़े, किराना, दवा, బట్టలు, కిరాణా, ఎలక్ట్రానిక్స్, మందులు",
    "doorNo": "Door number, flat number, building number, plot number, 1-2-3, 402, मकान नंबर, షాప్ నంబర్, డోర్ నంబర్",
    "street": "Street name, road, MG Road, main road, bazaar, 5th avenue, lane, गली, रोड, బజార్, మెయిన్ రోడ్డు",
    "area": "Area, locality, neighborhood, colony, Gajuwaka, T Nagar, Madhapur, इलाका, ప్రాంతం, గాజువాక, गाजुवाका",
    "town": "Town, village, district, mandal, कस्बा, गाँव, గ్రామం, మండలం, skip, छोड़ दो, వద్దు",
    "city": "City name, Visakhapatnam, Vizag, Hyderabad, Vijayawada, Chennai, Bengaluru, Mumbai, Delhi, विशाखापट्टनम, हैदराबाद, విజయవాడ, విశాఖపట్నం",
    "state": "State name, Andhra Pradesh, Telangana, Tamil Nadu, Karnataka, Maharashtra, आंध्र प्रदेश, ఆంధ్రప్రదేశ్, తెలంగాణ",
    "pincode": "Pincode, postal code, zip code, 6 digits, 530016, 500081, 520001, पिन कोड, పిన్‌కోడ్, six digits, छह अंक",
    "usedReferalCode": "Referral code, coupon code, skip, none, छोड़ दो, వద్దు, లేదు",
}


def speak_windows(text):
    """Speaks text using Windows built-in speech synthesizer."""
    if not text:
        return
    clean_text = re.sub(r'["\']', " ", text)
    ps_cmd = f'Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; $s.Speak("{clean_text}")'
    try:
        subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_cmd],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass


def speak_voice(text, lang="auto"):
    """
    Speaks text in Telugu, Hindi, or English.
    Uses gTTS + sounddevice for pristine native Telugu and Hindi speech.
    Falls back to Windows SAPI speech for English.
    """
    if not text:
        return

    is_te = any(0x0C00 <= ord(c) <= 0x0C7F for c in text) or lang == "te"
    is_hi = any(0x0900 <= ord(c) <= 0x097F for c in text) or lang == "hi"

    if is_te or is_hi:
        target_lang = "te" if is_te else "hi"
        try:
            import io
            import av
            from gtts import gTTS

            buf = io.BytesIO()
            tts = gTTS(text=text, lang=target_lang, timeout=4.0)
            tts.write_to_fp(buf)
            buf.seek(0)
            container = av.open(buf)
            frames = [f.to_ndarray() for f in container.decode(audio=0)]
            if frames:
                audio_data = np.concatenate(frames, axis=1)
                if audio_data.ndim == 2:
                    audio_data = audio_data.T
                sr = container.streams.audio[0].sample_rate
                sd.play(audio_data, samplerate=sr)
                sd.wait()
                return
        except Exception:
            pass

    speak_windows(text)


def record_audio(sample_rate=16000, max_seconds=12, silence_break_ms=1500):
    """
    Records audio with automatic speech break/pause detection.
    Stops recording automatically when user words complete and a ~900ms pause is detected.
    """
    print("\n[Listening... Speak your answer (Auto-detecting break)]:", end="", flush=True)

    recorded_chunks = []
    stop_event = threading.Event()
    state = {
        "has_spoken": False,
        "silence_chunks": 0,
        "noise_floor": 0.005,
    }

    chunks_for_break = max(int(silence_break_ms / 100), 7)

    def callback(indata, frames, time_info, status):
        if not stop_event.is_set():
            recorded_chunks.append(indata.copy())
            rms = float(np.sqrt(np.mean(indata**2)))

            # Continuous noise floor tracking (rejects fan / AC hum)
            if len(recorded_chunks) <= 5:
                state["noise_floor"] = min(state["noise_floor"], rms)
            else:
                if rms < state["noise_floor"]:
                    state["noise_floor"] = state["noise_floor"] * 0.85 + rms * 0.15
                else:
                    state["noise_floor"] = state["noise_floor"] * 0.995 + rms * 0.005

            speech_threshold = max(state["noise_floor"] * 2.2, 0.012)
            silence_threshold = max(state["noise_floor"] * 1.3, 0.005)

            if rms > speech_threshold:
                state["has_spoken"] = True
                state["silence_chunks"] = 0
            elif state["has_spoken"] and rms < silence_threshold:
                state["silence_chunks"] += 1
                if state["silence_chunks"] >= chunks_for_break:
                    stop_event.set()

    stream = sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        callback=callback,
    )

    with stream:
        def wait_for_enter():
            try:
                input()
                stop_event.set()
            except Exception:
                pass

        key_thread = threading.Thread(target=wait_for_enter, daemon=True)
        key_thread.start()

        start_time = time.time()
        while time.time() - start_time < max_seconds and not stop_event.is_set():
            sd.sleep(100)

        stop_event.set()

    if not recorded_chunks:
        return np.zeros(0, dtype=np.float32)

    audio = np.concatenate(recorded_chunks, axis=0).flatten()

    # Multi-stage spectral noise gate, FFT bandpass filter & normalization
    audio = denoise_audio(audio, sample_rate=sample_rate)

    return audio


def ensure_english_val(val):
    if val is None:
        return None
    if isinstance(val, str):
        v = val.strip()
        if any(('ఀ' <= c <= '౿') or ('ऀ' <= c <= 'ॿ') for c in v):
            v = translate_indic_to_english(v)
        return v
    if isinstance(val, list):
        return [ensure_english_val(x) for x in val if x]
    return val


def build_final_json(data, backend_id=None):
    if not backend_id:
        backend_id = "comp_" + uuid.uuid4().hex[:10]

    categories = data.get("storeCategory", [])
    if isinstance(categories, str):
        categories = [c.strip() for c in categories.replace(";", ",").split(",") if c.strip()]
    cleaned_cats = ensure_english_val(categories)

    comp_name = ensure_english_val(data.get("companyName", ""))
    if comp_name and comp_name.lower() in ("skip", "none", "null"):
        comp_name = ""

    ref_code = ensure_english_val(data.get("usedReferalCode", None))
    if ref_code and str(ref_code).strip().upper() in ("SKIP", "NONE", "NULL", "NA", "N/A", "NO", "NAHIN", "LEDU", "VADDU", "NAHI", "CHHOD DO"):
        ref_code = None

    return {
        "id": backend_id,
        "companyId": backend_id,
        "companyName": comp_name,
        "personName": ensure_english_val(data.get("personName", "")),
        "phone": ensure_english_val(data.get("phone", "")),
        "email": ensure_english_val(data.get("email", "")),
        "logoUrl": data.get("logoUrl", None),
        "storeCategory": cleaned_cats,
        "doorNo": ensure_english_val(data.get("doorNo", None)),
        "street": ensure_english_val(data.get("street", None)),
        "area": ensure_english_val(data.get("area", None)),
        "town": ensure_english_val(data.get("town", "")),
        "city": ensure_english_val(data.get("city", "")),
        "state": ensure_english_val(data.get("state", "")),
        "pincode": ensure_english_val(data.get("pincode", "")),
        "usedReferalCode": ref_code,
        "createdAt": data.get("createdAt", None),
        "productCount": data.get("productCount", 0),
        "isPhoneMandatoryForBrowsing": data.get("isPhoneMandatoryForBrowsing", False),
        "isStoreLoginMandatory": data.get("isStoreLoginMandatory", False),
    }


def parse_and_validate(field, user_text, collected_data):
    """
    Intelligently parses user speech like a human agent.
    Returns: (parsed_value, error_message, auto_detected_state)
    """
    t = str(user_text).strip()

    # Check if user is asking a conversational question
    if is_user_asking_question(t):
        clarification = (
            f"I am asking for your {field['key']} ({field.get('hint', '')}) so we can complete your store registration. "
            f"You can tell me the details or say 'skip'."
        )
        return None, clarification, None

    # Check if user wants to skip
    if is_skip(t):
        if not field["required"]:
            return None, None, None
        return None, f"'{field['key']}' is a required field. {field['question']}", None

    key = field["key"]

    if key == "companyName":
        val = clean_company_name(t)
        return (val if val else None), None, None

    if key == "personName":
        val, warning = clean_person_name(t)
        if warning == "phone_detected":
            return None, "That sounded like a phone number. What is your contact person's actual name?", None
        return (val if val else None), None, None

    if key == "phone":
        digits = clean_phone(t)
        if len(digits) < 6:
            return None, "Please provide a valid phone number (at least 10 digits).", None
        return digits, None, None

    if key == "email":
        val = clean_email(t)
        return val, None, None

    if key == "storeCategory":
        cats = clean_category(t)
        return cats, None, None

    if key == "doorNo":
        val = clean_door_no(t)
        return val, None, None

    if key == "street":
        val = clean_street(t)
        return val, None, None

    if key in ("city", "state", "town", "area"):
        current_city = collected_data.get("city", None)
        val, auto_state = clean_location(t, key, current_city=current_city)
        if key == "town" and val is None:
            val = ""
        return val, None, auto_state

    if key == "pincode":
        val = clean_pincode(t)
        return val, None, None

    if key == "usedReferalCode":
        if is_skip(t):
            return None, None, None
        eng_ref = translate_indic_to_english(t).strip().upper()
        if is_skip(eng_ref.lower()) or eng_ref in ("NONE", "SKIP", "NA", "N/A", "NO", "NIL", "NULL", "NAHIN", "LEDU", "VADDU", "NAHI", "CHHOD DO"):
            return None, None, None
        return eng_ref if eng_ref else None, None, None

    return translate_indic_to_english(t.strip(". ,")), None, None


def run_voice_form(model, args):
    collected_data = {}
    current_idx = 0
    total_fields = len(STORE_FIELDS)

    print("=" * 65)
    print("      SMART STORE ONBOARDING VOICE BOT (Faster-Whisper)")
    print("=" * 65)
    print("The Voice Bot understands conversational sentences and Indian locations.")
    print("Commands: 'skip' for optional fields, 'show json' to preview, 'exit' to quit.\n")

    # Language Selection Step
    if args.language == "auto":
        print("\n" + "=" * 65)
        print("          LANGUAGE SELECTION / भाषा का चयन / భాష ఎంపిక")
        print("=" * 65)
        lang_prompt = (
            "Please choose your preferred language:\n"
            "  [1] English\n"
            "  [2] Hindi (हिंदी)\n"
            "  [3] Telugu (తెలుగు)\n"
            "Speak your language choice or type 1, 2, 3:"
        )
        print(f"They told: {lang_prompt}")
        if args.tts:
            speak_windows("Please select your language: 1 for English, 2 for Hindi, or 3 for Telugu.")

        while True:
            if args.text_only:
                raw_choice = input("I told (Type 1, 2, 3 or English/Hindi/Telugu): ").strip()
            else:
                print("[Press ENTER to speak your language | or type 1, 2, 3]:", end=" ")
                c = input().strip()
                if c:
                    raw_choice = c
                else:
                    audio = record_audio(sample_rate=16000, max_seconds=5)
                    if len(audio) < 1600 or np.max(np.abs(audio)) < 0.005:
                        print("[No speech detected. Please speak or type 1, 2, or 3.]")
                        continue
                    segments, info = model.transcribe(
                        audio,
                        beam_size=5,
                        language=None,
                        initial_prompt="English, Hindi, Telugu, हिंदी, తెలుగు, 1, 2, 3, ek, do, teen, okati, rendu, moodu",
                        temperature=0.0
                    )
                    raw_choice = " ".join(s.text.strip() for s in segments).strip()
                    print(f"I told: {raw_choice}")

            chosen = detect_language_choice(raw_choice)
            if chosen in ("en", "hi", "te"):
                args.language = chosen
                break
            else:
                retry_msg = "Please choose a valid language: 1 for English, 2 for Hindi, or 3 for Telugu."
                print(f"They told: {retry_msg}")
                if args.tts:
                    speak_voice(retry_msg, lang="en")

    lang_labels = {"en": "English", "hi": "हिंदी (Hindi)", "te": "తెలుగు (Telugu)"}
    print(f"\n[Language Selected: {lang_labels.get(args.language, args.language)}]")

    if args.language == "hi":
        greeting = "नमस्ते! मैं आपकी दुकान पंजीकरण सहायक हूँ। आइए आपकी दुकान का प्रोफ़ाइल सेट करते हैं।"
    elif args.language == "te":
        greeting = "నమస్కారం! నేను మీ store registration assistant ని. మీ store profile ని set up చేద్దాం."
    else:
        greeting = "Hello! I am your store registration assistant. Let's set up your store profile."

    print(f"They told: {greeting}")
    if args.tts:
        speak_voice(greeting, lang=args.language)

    while current_idx < total_fields:
        field = STORE_FIELDS[current_idx]
        key = field["key"]

        # Check if state was already auto-detected from city
        if key == "state" and collected_data.get("state"):
            auto_st = collected_data["state"]
            msg = f"Since your city is {collected_data.get('city')}, I have set your state as {auto_st}."
            print(f"\nThey told: {msg}")
            if args.tts:
                speak_voice(msg, lang=args.language)
            current_idx += 1
            continue

        question = field.get(f"question_{args.language}") or field.get("question_en") or field["question"]
        print(f"\n[{current_idx + 1}/{total_fields}] They told: {question}")
        if args.tts:
            speak_voice(question, lang=args.language)

        user_text = ""
        if args.text_only:
            user_text = input("I told (Type here): ").strip()
        else:
            print("[Press ENTER to speak | or type response directly and press Enter]:", end=" ")
            choice = input().strip()
            if choice:
                user_text = choice
            else:
                audio = record_audio(sample_rate=16000, max_seconds=10)
                if len(audio) < 1600 or np.max(np.abs(audio)) < 0.005:
                    print("[No speech detected. Please speak closer to the mic.]")
                    continue

                whisper_lang = None if args.language in ("auto", "none", "") else args.language
                field_prompt = FIELD_PROMPTS.get(key, WHISPER_PROMPT)
                segments, info = model.transcribe(
                    audio,
                    beam_size=5,
                    best_of=5,
                    task="transcribe",
                    language=whisper_lang,
                    vad_filter=True,
                    vad_parameters=dict(
                        threshold=0.45,
                        min_speech_duration_ms=180,
                        min_silence_duration_ms=500,
                        speech_pad_ms=300,
                    ),
                    no_speech_threshold=0.55,
                    log_prob_threshold=-1.0,
                    compression_ratio_threshold=2.4,
                    initial_prompt=field_prompt,
                    condition_on_previous_text=False,
                    temperature=[0.0, 0.2],
                )
                user_text = " ".join(s.text.strip() for s in segments).strip()
                if whisper_lang is None and hasattr(info, 'language'):
                    prob = f" ({info.language_probability*100:.0f}%)" if hasattr(info, 'language_probability') else ""
                    print(f"[Detected Language: {info.language}{prob}]")

        if not user_text:
            print("[Empty response. Let's try again.]")
            continue

        print(f"I told: {user_text}")

        cmd = user_text.lower().strip()
        if cmd in ("exit", "quit", "q"):
            print("\nExiting voice bot registration. Goodbye!")
            return

        if cmd in ("show json", "json", "status", "preview"):
            preview_json = build_final_json(collected_data, backend_id="PREVIEW_ID")
            print("\n--- Current JSON (In Progress) ---")
            print(json.dumps(preview_json, indent=2))
            continue

        parsed_val, error_msg, auto_state = parse_and_validate(field, user_text, collected_data)

        if error_msg:
            print(f"They told: {error_msg}")
            if args.tts:
                speak_voice(error_msg, lang=args.language)
            continue

        collected_data[key] = parsed_val
        if auto_state and not collected_data.get("state"):
            collected_data["state"] = auto_state

        current_idx += 1
        lbl = field.get("label_te", field.get("label", key))
        if args.language == "te":
            ack = f"Okay, {lbl} save అయ్యింది."
        elif args.language == "hi":
            ack = f"ठीक है, {key} दर्ज हुआ।"
        else:
            ack = f"Got it. Saved {key}: {parsed_val}"
        print(f"They told: {ack}")

    # Complete!
    backend_id = "comp_" + uuid.uuid4().hex[:10]
    final_json = build_final_json(collected_data, backend_id=backend_id)

    if args.language == "te":
        completion_msg = "Congratulations! మీ store details అన్నీ successfully save అయ్యాయి. మీ store profile JSON ready:"
    elif args.language == "hi":
        completion_msg = "बधाई हो! आपकी दुकान के सभी विवरण सफलतापूर्वक एकत्र कर लिए गए हैं:"
    else:
        completion_msg = "Awesome! All fields have been collected. Here is your complete store profile JSON payload:"
    print("\n" + "=" * 65)
    print(f"They told: {completion_msg}")
    print("=" * 65)

    print("\n```json")
    print(json.dumps(final_json, indent=2))
    print("```\n")

    out_file = "store_profile.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(final_json, f, indent=2)
    print(f"Full JSON successfully saved to: {out_file}")

    if args.tts:
        speak_voice(completion_msg, lang=args.language)


def main():
    parser = argparse.ArgumentParser(description="Store Onboarding Voice Bot")
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="base",
        help="Whisper model: tiny, base (instant default), small (higher accuracy), medium, large-v3",
    )
    parser.add_argument("--device", "-d", type=str, default="cpu", help="Device: cpu, cuda (default: cpu)")
    parser.add_argument("--compute_type", "-c", type=str, default="int8", help="Compute type: int8, float32, float16 (default: int8)")
    parser.add_argument(
        "--language",
        "-l",
        type=str,
        default="auto",
        choices=["auto", "en", "hi", "te"],
        help="Language: auto (auto-detect Hindi/Telugu/English), en, hi, te (default: auto)",
    )
    parser.add_argument("--tts", action="store_true", default=False, help="Enable text-to-speech voice replies")
    parser.add_argument("--text-only", action="store_true", default=False, help="Text input mode without mic")
    parser.add_argument("--demo", action="store_true", default=False, help="Run an automated demo to generate JSON")

    args = parser.parse_args()

    print(f"Loading Faster-Whisper '{args.model}' on {args.device} ({args.compute_type})...")
    try:
        model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type, cpu_threads=4)
    except RuntimeError as _e:
        if "mkl_malloc" in str(_e).lower() and args.model != "tiny":
            print(f"[Warning] MKL memory allocation error on '{args.model}'. Falling back to 'tiny'...")
            model = WhisperModel("tiny", device=args.device, compute_type=args.compute_type, cpu_threads=2)
        else:
            raise

    if args.demo:
        print("\n--- Running Conversational Speech Simulation ---")
        conversational_inputs = {
            "companyName": "My store name is Brostide",
            "personName": "Rahul Sharma",
            "phone": "Call me on 9840123456",
            "email": "odanesizetadred at gmail dot com",
            "storeCategory": "We sell clothing and fashion",
            "doorNo": "Door number 1 dash 2 dash 3",
            "street": "It is on Google umpalim street",
            "area": "Vishaka Patnam area",
            "town": "skip",
            "city": "Vishaka Patnam",
            "state": "Andhra Pradesh",
            "pincode": "pincode is 530016",
            "usedReferalCode": "none",
        }
        collected = {}
        for field in STORE_FIELDS:
            inp = conversational_inputs.get(field["key"], "skip")
            print(f"They told: {field['question']}")
            print(f"I told (User said): \"{inp}\"")
            val, _, auto_st = parse_and_validate(field, inp, collected)
            collected[field["key"]] = val
            if auto_st:
                collected["state"] = auto_st
            print(f"They told (Bot extracted): {val}\n")

        backend_id = "comp_" + uuid.uuid4().hex[:10]
        final_json = build_final_json(collected, backend_id=backend_id)

        print("=" * 65)
        print("They told: All fields collected! Here is the clean JSON:")
        print("=" * 65)
        print(json.dumps(final_json, indent=2))
        return

    run_voice_form(model, args)


if __name__ == "__main__":
    main()
