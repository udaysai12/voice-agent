import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

import base64
import datetime
import hashlib
import http.server
import io
import json
import re
import socketserver
import threading
import time
import urllib.parse
import uuid
import numpy as np
from faster_whisper import WhisperModel, decode_audio

def safe_decode_audio(buf):
    """Safely decodes audio bytes or file buffer into float32 numpy array.
    Returns an empty array instead of raising PyAV EOFError or InvalidDataError."""
    try:
        if isinstance(buf, (bytes, bytearray)):
            if len(buf) < 1000:
                return np.array([], dtype=np.float32)
            buf = io.BytesIO(buf)
        return decode_audio(buf)
    except Exception:
        return np.array([], dtype=np.float32)
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
    is_whisper_hallucination,
    translate_indic_to_english,
)

PORT = 8000

WHISPER_PROMPT = (
    "Indian store onboarding in English, Hindi, and Telugu. "
    "Brostide, Rahul Sharma, Uday Kumar, Visakhapatnam, Hyderabad, Andhra Pradesh, Telangana, "
    "दुकान, स्टोर, नाम, फ़ोन नंबर, कपड़े, किराना, విలేజ్, పిన్‌కోడ్, బట్టలు, కిరాణా, మందులు, "
    "email at gmail dot com, pincode 530016, phone number 9840123456"
)

FIELD_PROMPTS = {
    "companyName": {
        "en": "Company name, store name, shop name, Brostide, Fresh Mart, Textiles, Enterprises, Fashion, Retail",
        "hi": "दुकान का नाम, कंपनी, किराना, क्लॉथ्स, स्टोर, एंटरप्राइजेज, वस्त्र",
        "te": "Store name, company name, shop name, Brostide, Fresh Mart, కిరాణా, బట్టల షాప్, టెక్స్‌టైల్స్, ఎంటర్‌ప్రైజెస్, స్టోర్ నేమ్",
    },
    "personName": {
        "en": "Contact person name, Rahul Sharma, Uday Kumar, Suresh, Priya, Anand, Rajesh",
        "hi": "संपर्क व्यक्ति का नाम, राहुल शर्मा, उदय कुमार, सुरेश, प्रिया, आनंद, राजेश, मेरा नाम",
        "te": "Contact person name, నా పేరు, person name, Rahul Sharma, Uday Kumar, రాహుల్, ఉదయ్ కుమార్, సురేష్",
    },
    "phone": {
        "en": "Phone number, mobile number, 10 digits, call 9840123456, 9876543210, 9949012345",
        "hi": "मोबाइल नंबर, फोन नंबर, 10 अंक, नौ आठ सात छह पांच चार तीन दो एक शून्य, 9876543210",
        "te": "Mobile number, phone number, 10 digits, నా నంబర్, పది అంకెలు, 9840123456, 9876543210, కాల్ చేయండి",
    },
    "email": {
        "en": "Email address, contact email, user at gmail dot com, name at yahoo dot com",
        "hi": "ईमेल पता, नाम एट जीमेल डॉट कॉम, कॉन्टैक्ट ईमेल",
        "te": "Email address, mail id, store at gmail dot com, skip, లేదు",
    },
    "storeCategory": {
        "en": "Clothing, Grocery, Pharmacy, Electronics, Footwear, Fashion, Supermarket, Restaurant, Retail",
        "hi": "कपड़े, किराना, दवा, इलेक्ट्रॉनिक्स, जूते, परिधान, मेडिकल, जनरल स्टोर",
        "te": "Store category, clothing, grocery, pharmacy, medical, electronics, supermarket, బట్టలు, కిరాణా, మందులు",
    },
    "doorNo": {
        "en": "Door number, flat number, building number, plot number, shop number, 1-2-3, 402, skip, roadside",
        "hi": "मकान नंबर, दुकान नंबर, प्लॉट नंबर, 1-2-3, 402, छोड़ दो, रोडसाइड, पटरी",
        "te": "Shop number, door number, flat number, skip, వద్దు, 1-2-3, 402, రోడ్డు పక్కన",
    },
    "street": {
        "en": "Street name, road, MG Road, Main Road, Market Street, Bazaar, 5th Avenue, Lane",
        "hi": "सड़क, गली, रोड, एमजी रोड, मेन रोड, बाजार, मार्केट",
        "te": "Street name, road name, Main Road, MG Road, Bazaar, కాలనీ, వీధి, రోడ్డు, skip",
    },
    "area": {
        "en": "Area, locality, neighborhood, colony, Gajuwaka, T Nagar, Madhapur, Indiranagar",
        "hi": "इलाका, कॉलोनी, नगर, गाजुवाका, टी नगर, माधपुर, क्षेत्र",
        "te": "Area, locality, neighborhood, Madhapur, Gajuwaka, T Nagar, కాలనీ, ఏరియా, ప్రాంతం, skip",
    },
    "town": {
        "en": "Town, village, mandal, skip, none",
        "hi": "कस्बा, गाँव, मंडल, छोड़ दो",
        "te": "Town, village name, పట్టణం, గ్రామం, skip, లేదు, వద్దు",
    },
    "city": {
        "en": "City name, Visakhapatnam, Vizag, Hyderabad, Vijayawada, Guntur, Chennai, Bengaluru, Mumbai, Delhi",
        "hi": "शहर का नाम, विशाखापट्टनम, हैदराबाद, विजयवाड़ा, गुंटूर, दिल्ली, मुंबई, जयपुर",
        "te": "City name, Visakhapatnam, Vizag, Hyderabad, Vijayawada, Guntur, తిరుపతి, సిటీ",
    },
    "state": {
        "en": "State name, Andhra Pradesh, Telangana, Tamil Nadu, Karnataka, Maharashtra, Rajasthan, Uttar Pradesh",
        "hi": "राज्य का नाम, आंध्र प्रदेश, तेलंगाना, राजस्थान, उत्तर प्रदेश, महाराष्ट्र, कर्नाटक",
        "te": "State name, Andhra Pradesh, Telangana, ఆంధ్రప్రదేశ్, తెలంగాణ, స్టేట్",
    },
    "pincode": {
        "en": "Pincode, postal code, zip code, 6 digits, 530016, 500081, 520001",
        "hi": "पिन कोड, 6 अंक, 530016, 500081, 520001, छह अंक",
        "te": "Pincode, postal code, 6 digits, ఆరు అంకెలు, 530016, 500081, పిన్‌కోడ్",
    },
    "usedReferalCode": {
        "en": "Referral code, coupon code, skip, none, no code, don't have",
        "hi": "रेफ़रल कोड, कूपन कोड, छोड़ दो, नहीं है, कोई कोड नहीं",
        "te": "Referral code, coupon code, skip, లేదు, వద్దు, no code",
    },
}


def get_field_prompt(field_key, lang="en"):
    prompts = FIELD_PROMPTS.get(field_key)
    if isinstance(prompts, dict):
        return prompts.get(lang, prompts.get("en", WHISPER_PROMPT))
    return prompts or WHISPER_PROMPT

def load_best_whisper_model():
    """Dynamically loads the best Faster-Whisper model, prioritizing RTX 2050 GPU (CUDA float16) with 'small'."""
    # Register NVIDIA CUDA cuBLAS runtime DLL directory
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

    # 1. Priority: RTX 2050 GPU (CUDA float16) with high-accuracy 'small' model
    try:
        print("Loading high-accuracy Faster-Whisper 'small' on RTX 2050 GPU (CUDA float16)...")
        m = WhisperModel("small", device="cuda", compute_type="float16")
        # Warmup GPU kernels
        m.transcribe(np.zeros(16000, dtype=np.float32), beam_size=1)
        print("[AI Engine Ready] Faster-Whisper 'small' active on NVIDIA GeForce RTX 2050 GPU (Maximum Perfection)!")
        return m, "cuda", "small"
    except Exception as e_gpu:
        print(f"[Notice] GPU init ({e_gpu}). Falling back to CPU...")

    # 2. Fallback: CPU int8 with 'small'
    try:
        print("Loading Faster-Whisper 'small' on CPU int8...")
        m = WhisperModel("small", device="cpu", compute_type="int8", cpu_threads=4)
        print("[AI Engine Ready] Faster-Whisper 'small' active on CPU int8!")
        return m, "cpu", "small"
    except Exception as e_small:
        print(f"[Notice] CPU small fallback ({e_small}). Using 'base'...")

    # 3. Fallback: CPU int8 with 'base'
    try:
        m = WhisperModel("base", device="cpu", compute_type="int8", cpu_threads=4)
        print("[AI Engine Ready] Faster-Whisper 'base' active on CPU int8.")
        return m, "cpu", "base"
    except Exception:
        m = WhisperModel("tiny", device="cpu", compute_type="int8", cpu_threads=2)
        print("[AI Engine Ready] Faster-Whisper 'tiny' active on CPU.")
        return m, "cpu", "tiny"

model, MODEL_DEVICE, MODEL_SIZE = load_best_whisper_model()

STORE_FIELDS = [
    {
        "key": "companyName",
        "label": "Company / Store Name",
        "label_hi": "दुकान का नाम",
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
        "label_hi": "संपर्क व्यक्ति का नाम",
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
        "label_hi": "फ़ोन नंबर",
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
        "label_hi": "ईमेल पता",
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
        "label_hi": "स्टोर श्रेणी",
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
        "label_hi": "दुकान या मकान नंबर",
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
        "label_hi": "गली या रोड",
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
        "label_hi": "इलाका",
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
        "label_hi": "गाँव या कस्बा",
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
        "label_hi": "शहर",
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
        "label_hi": "राज्य",
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
        "label_hi": "पिन कोड",
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
        "label_hi": "रेफ़रल कोड",
        "label_te": "Referral Code",
        "question": "Do you have any referral code? Or say skip.",
        "question_en": "Do you have any referral code? Or say skip.",
        "question_hi": "क्या आपके पास कोई रेफ़रल कोड है? या छोड़ दो बोलें।",
        "question_te": "దగ్గర ఏదైనా referral code ఉందా? లేకపోతే skip అనండి.",
        "required": False,
        "hint": "Referral code లేదా skip anali (Optional)",
    },
]

TTS_CACHE = {}
TTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tts_cache")
os.makedirs(TTS_DIR, exist_ok=True)


def get_tts_audio_bytes(text, lang="te"):
    """Returns MP3 audio bytes for text in Telugu, Hindi, or English using fast disk/memory cache."""
    global TTS_CACHE
    clean_text = text.strip()
    if not clean_text:
        return b""

    gtts_lang = "te" if lang == "te" else ("hi" if lang == "hi" else "en")
    cache_key = (clean_text, gtts_lang)
    if cache_key in TTS_CACHE:
        return TTS_CACHE[cache_key]

    # Instant check in pre-cached disk directory
    key_str = f"{gtts_lang}:{clean_text}"
    fn = hashlib.md5(key_str.encode("utf-8")).hexdigest() + ".mp3"
    disk_path = os.path.join(TTS_DIR, fn)
    if os.path.exists(disk_path) and os.path.getsize(disk_path) > 100:
        try:
            with open(disk_path, "rb") as f:
                data = f.read()
            TTS_CACHE[cache_key] = data
            return data
        except Exception:
            pass

    try:
        from gtts import gTTS
        tts = gTTS(text=clean_text, lang=gtts_lang, timeout=4.0)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        data = buf.getvalue()
        if data:
            TTS_CACHE[cache_key] = data
            try:
                with open(disk_path, "wb") as f:
                    f.write(data)
            except Exception:
                pass
            return data
    except Exception as e:
        print(f"[TTS Error] failed to generate audio for '{clean_text[:30]}': {e}")
    return None


def prewarm_tts_cache():
    """Loads all pre-cached MP3 audio files into memory on startup for 0ms retrieval."""
    time.sleep(0.05)
    items = [
        ("Welcome! Please choose your preferred language: English, Hindi, or Telugu.", "en"),
        ("कृपया अपनी भाषा चुनें: हिंदी, तेलुगु, या अंग्रेज़ी।", "hi"),
        ("దయచేసి మీ language ని select చేసుకోండి: English, Hindi, లేదా Telugu.", "te"),
        ("Language set to English! Let's get your store registered. What is your company or store name?", "en"),
        ("भाषा हिंदी चुनी गई है! आइए आपकी दुकान का पंजीकरण शुरू करते हैं। आपकी दुकान या कंपनी का नाम क्या है?", "hi"),
        ("Telugu language select అయింది! Store register చేద్దాం. మీ store లేదా company name ఏంటి?", "te"),
        ("I could not hear any speech. Let's try that question again.", "en"),
        ("मुझे कोई आवाज़ सुनाई नहीं दी। आइए फिर से कोशिश करें।", "hi"),
        ("నాకు voice వినిపించలేదు. Please మళ్ళీ ఒకసారి చెప్పండి.", "te"),
        ("Super! All your store details have been collected successfully.", "en"),
        ("बधाई हो! आपकी दुकान के सभी विवरण सफलतापूर्वक एकत्र कर लिए गए हैं।", "hi"),
        ("Super! మీ store details అన్నీ successfully save అయిపోయాయి. Store profile ready!", "te"),
    ]
    for f in STORE_FIELDS:
        q_te = f.get("question_te")
        if q_te:
            items.append((q_te, "te"))
            items.append((f"Done! {q_te}", "te"))
        q_hi = f.get("question_hi")
        if q_hi:
            items.append((q_hi, "hi"))
            items.append((f"ठीक है! {q_hi}", "hi"))
        q_en = f.get("question_en", f.get("question"))
        if q_en:
            items.append((q_en, "en"))
            items.append((f"Done! {q_en}", "en"))

    for txt, l in items:
        try:
            get_tts_audio_bytes(txt, l)
        except Exception:
            pass
    print(f"[TTS Engine] Pre-warmed {len(TTS_CACHE)} speech audio assets in memory (Zero Latency Ready).")


threading.Thread(target=prewarm_tts_cache, daemon=True).start()

SESSION = {
    "current_index": 0,
    "backend_id": "comp_" + uuid.uuid4().hex[:10],
    "data": {},
    "is_completed": False,
    "language": None,
    "language_chosen": False,
}


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


def build_store_json(data, backend_id):
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


def parse_and_validate(field, user_text, collected_data, lang="en"):
    t = str(user_text).strip()

    if is_user_asking_question(t):
        if lang == "te":
            clarification = (
                f"మీ store registration కోసం మీ {field.get('label_te', field['label'])} ({field.get('hint', '')}) అడుగుతున్నాను. "
                f"Please మీ answer చెప్పండి లేదా skip అనండి."
            )
        elif lang == "hi":
            clarification = (
                f"आपकी दुकान के पंजीकरण के लिए {field.get('label_hi', field['label'])} ({field.get('hint', '')}) की आवश्यकता है। "
                f"कृपया अपना उत्तर बताएं या छोड़ दो बोलें।"
            )
        else:
            clarification = (
                f"I am asking for your {field['label']} ({field.get('hint', '')}) to register your store. "
                f"You can speak your answer or say 'skip' if optional."
            )
        return None, clarification, None

    if is_skip(t):
        if not field["required"]:
            return None, None, None
        req_msg = {
            "te": f"'{field.get('label_te', field['label'])}' mandatory field. {field.get('question_te', field['question'])}",
            "hi": f"'{field.get('label_hi', field['label'])}' आवश्यक है। {field.get('question_hi', field['question'])}",
        }.get(lang, f"'{field['label']}' is required. {field['question']}")
        return None, req_msg, None

    key = field["key"]

    if key == "companyName":
        val = clean_company_name(t)
        if not val or val.lower() in ("skip", "none"):
            req_msg = {
                "te": "Please మీ store లేదా company name ని clear గా చెప్పండి.",
                "hi": "कृपया अपनी दुकान का नाम स्पष्ट रूप से बताएं।",
            }.get(lang, "Please provide your valid store or company name.")
            return None, req_msg, None
        return val, None, None

    if key == "personName":
        val, warning = clean_person_name(t)
        if warning == "phone_detected":
            warn_msg = {
                "te": "అది phone number లా ఉంది. Contact person name ఏమిటి?",
                "hi": "वह एक फोन नंबर जैसा सुनाई दिया। आपके संपर्क व्यक्ति का नाम क्या है?",
            }.get(lang, "That sounded like a phone number. What is your contact person's actual name?")
            return None, warn_msg, None
        return (val if val else None), None, None

    if key == "phone":
        digits = clean_phone(t)
        if len(digits) != 10:
            err_msg = {
                "te": "Please మీ 10 digits mobile number ని clear గా చెప్పండి.",
                "hi": "कृपया अपना 10 अंकों का मोबाइल नंबर स्पष्ट बताएं।",
            }.get(lang, "Please provide your complete 10-digit mobile number (e.g., 9840123456).")
            return None, err_msg, None
        return digits, None, None

    if key == "email":
        val = clean_email(t)
        return val, None, None

    if key == "storeCategory":
        cats = clean_category(t)
        return cats, None, None

    if key == "doorNo":
        val = clean_door_no(t)
        return val if val else None, None, None

    if key == "street":
        val = clean_street(t)
        return val if val else None, None, None

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
        eng_ref = translate_indic_to_english(t).strip().upper()
        if is_skip(eng_ref.lower()) or eng_ref in ("NONE", "SKIP", "NA", "N/A", "NO", "NIL", "NULL", "NAHIN", "LEDU", "VADDU", "NAHI", "CHHOD DO"):
            return None, None, None
        return eng_ref if eng_ref else None, None, None

    # Fallback: translate any Indic text to English
    return translate_indic_to_english(t).strip(". ,"), None, None


def get_current_field_info(lang="auto"):
    global SESSION
    idx = SESSION["current_index"]
    if idx < len(STORE_FIELDS):
        f = STORE_FIELDS[idx]
        q = f.get(f"question_{lang}") or f.get("question_en") or f["question"]
        return {
            "index": idx + 1,
            "total": len(STORE_FIELDS),
            "key": f["key"],
            "label": f["label"],
            "question": q,
            "question_en": f.get("question_en", f["question"]),
            "question_hi": f.get("question_hi", f["question"]),
            "question_te": f.get("question_te", f["question"]),
            "required": f["required"],
            "hint": f["hint"],
        }
    return None


def build_language_selection_response(detected_choice, user_text=""):
    """Unified handler for initial language selection across /api/transcribe, /api/set_language, and /api/answer."""
    global SESSION
    SESSION["language"] = detected_choice
    SESSION["language_chosen"] = True
    if detected_choice == "hi":
        ack = "भाषा हिंदी चुनी गई है! आइए आपकी दुकान का पंजीकरण शुरू करते हैं। आपकी दुकान या कंपनी का नाम क्या है?"
    elif detected_choice == "te":
        ack = "Telugu language select అయింది! Store register చేద్దాం. మీ store లేదా company name ఏంటి?"
    else:
        ack = "Language set to English! Let's get your store registered. What is your company or store name?"

    resp = {
        "status": "success",
        "language": detected_choice,
        "user_text": user_text,
        "bot_text": ack,
        "ack_text": ack,
        "json": build_store_json(SESSION["data"], SESSION["backend_id"]),
        "progress": 0,
        "current_field": get_current_field_info(detected_choice),
        "detected_language": detected_choice,
        "detected_prob": 1.0,
        "language_selected": detected_choice,
    }
    audio_b = get_tts_audio_bytes(ack, lang=detected_choice)
    if audio_b:
        resp["tts_audio_base64"] = base64.b64encode(audio_b).decode("ascii")
    return resp


def process_user_turn(user_text, lang="auto", target_key=None):
    global SESSION
    idx = SESSION["current_index"]
    backend_id = SESSION["backend_id"]

    if SESSION["is_completed"]:
        done_msg = {
            "hi": "आपकी दुकान का पंजीकरण पहले ही पूरा हो चुका है! दराज में पूरा JSON देखें।",
            "te": "మీ store registration ఆల్రెడీ complete అయింది! Full JSON ని preview లో చూడవచ్చు.",
        }.get(lang, "Your store registration is already complete! Review the full JSON in the drawer.")
        return {
            "user_text": user_text,
            "bot_text": done_msg,
            "current_field": None,
            "is_completed": True,
            "json": build_store_json(SESSION["data"], backend_id),
            "progress": 100,
        }

    # If target_key is explicitly for a past field, update that field only
    if target_key and idx < len(STORE_FIELDS) and target_key != STORE_FIELDS[idx]["key"]:
        target_fld = next((f for f in STORE_FIELDS if f["key"] == target_key), None)
        if target_fld:
            parsed_val, error_msg, auto_state = parse_and_validate(target_fld, user_text, SESSION["data"], lang=lang)
            if not error_msg:
                SESSION["data"][target_key] = parsed_val
                if auto_state and not SESSION["data"].get("state"):
                    SESSION["data"]["state"] = auto_state
            return {
                "user_text": user_text,
                "bot_text": None,
                "current_field": get_current_field_info(lang),
                "is_completed": False,
                "json": build_store_json(SESSION["data"], backend_id),
                "progress": int((idx / len(STORE_FIELDS)) * 100),
                "field_key": target_key,
                "extracted_value": ensure_english_val(parsed_val),
            }

    field = STORE_FIELDS[idx]
    parsed_val, error_msg, auto_state = parse_and_validate(field, user_text, SESSION["data"], lang=lang)

    if error_msg:
        return {
            "user_text": user_text,
            "bot_text": error_msg,
            "current_field": get_current_field_info(lang),
            "is_completed": False,
            "json": build_store_json(SESSION["data"], backend_id),
            "progress": int((idx / len(STORE_FIELDS)) * 100),
            "field_key": field["key"],
            "extracted_value": None,
        }

    SESSION["data"][field["key"]] = parsed_val
    if auto_state and not SESSION["data"].get("state"):
        SESSION["data"]["state"] = auto_state

    SESSION["current_index"] += 1
    new_idx = SESSION["current_index"]

    # Auto-fill state if next and already known
    if new_idx < len(STORE_FIELDS) and STORE_FIELDS[new_idx]["key"] == "state" and SESSION["data"].get("state"):
        auto_st = SESSION["data"]["state"]
        SESSION["current_index"] += 1
        new_idx = SESSION["current_index"]

    if new_idx >= len(STORE_FIELDS):
        SESSION["is_completed"] = True
        final_json = build_store_json(SESSION["data"], backend_id)
        if lang == "hi":
            bot_text = "बधाई हो! आपकी दुकान के सभी विवरण सफलतापूर्वक एकत्र कर लिए गए हैं।"
        elif lang == "te":
            bot_text = "Super! మీ store details అన్నీ successfully save అయిపోయాయి. Store profile ready!"
        else:
            bot_text = "Super! All your store details have been collected successfully."
        return {
            "user_text": user_text,
            "bot_text": bot_text,
            "current_field": None,
            "is_completed": True,
            "json": final_json,
            "progress": 100,
            "field_key": field["key"],
            "extracted_value": ensure_english_val(parsed_val),
        }

    next_field = STORE_FIELDS[new_idx]
    next_q = next_field.get(f"question_{lang}") or next_field.get("question_en") or next_field["question"]
    if lang == "hi":
        bot_text = f"ठीक है! {next_q}"
    elif lang == "te":
        bot_text = f"Done! {next_q}"
    else:
        bot_text = f"Done! {next_q}"

    return {
        "user_text": user_text,
        "bot_text": bot_text,
        "current_field": get_current_field_info(lang),
        "is_completed": False,
        "json": build_store_json(SESSION["data"], backend_id),
        "progress": int((new_idx / len(STORE_FIELDS)) * 100),
        "field_key": field["key"],
        "extracted_value": ensure_english_val(parsed_val),
    }


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Autonomous Voice AI - Store Onboarding</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-dark: #070608;
      --copper: #c27847;
      --copper-glow: rgba(194, 120, 71, 0.45);
      --amber: #f59e0b;
      --amber-glow: rgba(245, 158, 11, 0.4);
      --card-glass: rgba(18, 14, 24, 0.7);
      --border-glass: rgba(255, 255, 255, 0.08);
      --text: #f8fafc;
      --text-muted: #94a3b8;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Plus Jakarta Sans', sans-serif; }

    body {
      background-color: var(--bg-dark);
      color: var(--text);
      min-height: 100vh;
      overflow-x: hidden;
      display: flex;
      flex-direction: column;
      position: relative;
    }

    /* Ambient background wave ribbons canvas */
    #meshCanvas {
      position: fixed;
      top: 0;
      left: 0;
      width: 100vw;
      height: 100vh;
      z-index: 0;
      pointer-events: none;
      opacity: 0.55;
    }

    /* Top Navigation */
    nav {
      position: relative;
      z-index: 10;
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 24px 40px;
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
    }

    .brand-logo {
      width: 32px;
      height: 32px;
      border-radius: 50%;
      background: linear-gradient(135deg, var(--copper), #8b4513);
      display: flex;
      align-items: center;
      justify-content: center;
      box-shadow: 0 0 16px var(--copper-glow);
    }

    .brand-title {
      font-size: 17px;
      font-weight: 700;
      letter-spacing: -0.3px;
      color: #fff;
    }

    .nav-actions {
      display: flex;
      align-items: center;
      gap: 12px;
    }

    .glass-btn {
      background: rgba(255, 255, 255, 0.05);
      backdrop-filter: blur(12px);
      border: 1px solid var(--border-glass);
      color: #fff;
      padding: 10px 18px;
      border-radius: 30px;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 8px;
      transition: all 0.25s ease;
    }
    .glass-btn:hover {
      background: rgba(255, 255, 255, 0.12);
      border-color: rgba(194, 120, 71, 0.4);
      transform: translateY(-1px);
    }

    .lang-selector {
      display: inline-flex;
      background: rgba(255, 255, 255, 0.05);
      backdrop-filter: blur(12px);
      border: 1px solid var(--border-glass);
      border-radius: 30px;
      padding: 3px;
      gap: 3px;
    }

    .lang-pill {
      background: transparent;
      border: none;
      color: var(--text-muted);
      font-size: 12.5px;
      font-weight: 600;
      padding: 6px 13px;
      border-radius: 20px;
      cursor: pointer;
      transition: all 0.2s ease;
      display: inline-flex;
      align-items: center;
      gap: 5px;
    }
    .lang-pill:hover {
      color: #fff;
      background: rgba(255, 255, 255, 0.08);
    }
    .lang-pill.active {
      background: linear-gradient(135deg, rgba(194, 120, 71, 0.45), rgba(245, 158, 11, 0.35));
      border: 1px solid rgba(245, 158, 11, 0.6);
      color: #fff;
      box-shadow: 0 0 14px rgba(245, 158, 11, 0.25);
    }

    .detected-lang-pill {
      font-size: 11px;
      font-weight: 700;
      padding: 3px 10px;
      border-radius: 12px;
      background: rgba(16, 185, 129, 0.15);
      color: #34d399;
      border: 1px solid rgba(16, 185, 129, 0.35);
      display: inline-flex;
      align-items: center;
      gap: 4px;
      letter-spacing: 0.3px;
    }

    /* Language Selection Modal */
    .lang-modal-backdrop {
      position: fixed;
      top: 0;
      left: 0;
      width: 100vw;
      height: 100vh;
      background: rgba(7, 6, 8, 0.82);
      backdrop-filter: blur(16px);
      z-index: 1000;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 20px;
      opacity: 1;
      visibility: visible;
      transition: opacity 0.35s ease, visibility 0.35s ease;
    }
    .lang-modal-backdrop.hidden {
      opacity: 0;
      visibility: hidden;
      pointer-events: none;
    }
    .lang-modal-card {
      background: rgba(18, 14, 24, 0.95);
      border: 1px solid rgba(194, 120, 71, 0.45);
      box-shadow: 0 25px 70px rgba(0, 0, 0, 0.8), 0 0 50px rgba(194, 120, 71, 0.25);
      border-radius: 28px;
      max-width: 620px;
      width: 100%;
      padding: 36px 32px;
      text-align: center;
      position: relative;
      animation: modalSlideUp 0.35s cubic-bezier(0.16, 1, 0.3, 1);
    }
    @keyframes modalSlideUp {
      from { transform: translateY(24px) scale(0.96); opacity: 0; }
      to { transform: translateY(0) scale(1); opacity: 1; }
    }
    .lang-modal-badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 5px 16px;
      border-radius: 20px;
      background: rgba(245, 158, 11, 0.14);
      border: 1px solid rgba(245, 158, 11, 0.4);
      color: var(--amber);
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 1px;
      margin-bottom: 14px;
    }
    .lang-modal-title {
      font-size: 26px;
      font-weight: 700;
      color: #fff;
      margin-bottom: 8px;
    }
    .lang-modal-close {
      position: absolute;
      top: 18px;
      right: 20px;
      background: rgba(255, 255, 255, 0.08);
      border: 1px solid rgba(255, 255, 255, 0.15);
      color: #cbd5e1;
      width: 32px;
      height: 32px;
      border-radius: 50%;
      cursor: pointer;
      font-size: 14px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      transition: all 0.2s;
    }
    .lang-modal-close:hover {
      background: rgba(255, 255, 255, 0.2);
      color: #fff;
    }
    .lang-autoplay-notice {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 10px;
      background: linear-gradient(135deg, rgba(245, 158, 11, 0.25), rgba(194, 120, 71, 0.35));
      border: 1px solid rgba(245, 158, 11, 0.6);
      color: #fde68a;
      padding: 10px 20px;
      border-radius: 30px;
      font-size: 13.5px;
      font-weight: 700;
      cursor: pointer;
      margin-bottom: 18px;
      transition: all 0.25s ease;
      animation: pulseGlow 1.4s infinite alternate;
    }
    .lang-autoplay-notice:hover {
      transform: scale(1.03);
      background: linear-gradient(135deg, rgba(245, 158, 11, 0.4), rgba(194, 120, 71, 0.5));
      box-shadow: 0 0 25px rgba(245, 158, 11, 0.5);
    }
    .lang-modal-sub {
      color: var(--text-muted);
      font-size: 13.5px;
      margin-bottom: 26px;
      line-height: 1.6;
    }
    .lang-grid {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 14px;
      margin-bottom: 22px;
    }
    .lang-card-btn {
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid rgba(255, 255, 255, 0.1);
      border-radius: 18px;
      padding: 22px 14px;
      cursor: pointer;
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 8px;
      transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
      color: #fff;
      position: relative;
    }
    .lang-card-btn:hover {
      background: rgba(194, 120, 71, 0.18);
      border-color: rgba(245, 158, 11, 0.6);
      transform: translateY(-3px);
      box-shadow: 0 10px 25px rgba(245, 158, 11, 0.25);
    }
    .lang-card-btn.active {
      background: linear-gradient(135deg, rgba(194, 120, 71, 0.35), rgba(245, 158, 11, 0.3));
      border-color: #f59e0b;
      box-shadow: 0 0 25px rgba(245, 158, 11, 0.4);
    }
    .lang-card-flag {
      font-size: 34px;
      line-height: 1;
    }
    .lang-card-name {
      font-size: 17px;
      font-weight: 700;
      letter-spacing: -0.2px;
    }
    .lang-card-desc {
      font-size: 11.5px;
      color: var(--text-muted);
    }
    .lang-voice-hint {
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      font-size: 12.5px;
      color: #38bdf8;
      background: rgba(56, 189, 248, 0.08);
      border: 1px dashed rgba(56, 189, 248, 0.3);
      padding: 12px 16px;
      border-radius: 14px;
    }

    /* Hero Section */
    .hero-container {
      position: relative;
      z-index: 10;
      flex: 1;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      text-align: center;
      padding: 20px 24px 150px;
      max-width: 900px;
      margin: 0 auto;
      width: 100%;
    }

    .hero-eyebrow {
      font-size: 14px;
      font-weight: 600;
      color: var(--copper);
      letter-spacing: 1.5px;
      text-transform: uppercase;
      margin-bottom: 8px;
    }

    .hero-title {
      font-size: 48px;
      font-weight: 700;
      letter-spacing: -1.2px;
      line-height: 1.15;
      margin-bottom: 12px;
      background: linear-gradient(180deg, #ffffff 40%, rgba(255, 255, 255, 0.65) 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }

    .hero-subtitle {
      font-size: 16px;
      color: var(--text-muted);
      max-width: 540px;
      margin-bottom: 32px;
      line-height: 1.5;
    }

    /* Central Holographic Audio Orb */
    .orb-stage {
      position: relative;
      width: 260px;
      height: 260px;
      display: flex;
      align-items: center;
      justify-content: center;
      margin-bottom: 36px;
      cursor: pointer;
    }

    .orb-canvas {
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      border-radius: 50%;
    }

    /* Outer glowing torus ring */
    .orb-halo {
      position: absolute;
      width: 250px;
      height: 250px;
      border-radius: 50%;
      border: 1px solid rgba(194, 120, 71, 0.3);
      box-shadow: 0 0 40px var(--copper-glow), inset 0 0 30px rgba(194, 120, 71, 0.2);
      pointer-events: none;
      transition: all 0.5s ease;
    }

    .orb-stage.speaking .orb-halo {
      border-color: rgba(245, 158, 11, 0.7);
      box-shadow: 0 0 60px var(--amber-glow), inset 0 0 40px rgba(245, 158, 11, 0.3);
      animation: breathe 2s infinite alternate;
    }

    .orb-stage.listening .orb-halo {
      border-color: rgba(6, 182, 212, 0.7);
      box-shadow: 0 0 60px rgba(6, 182, 212, 0.4), inset 0 0 40px rgba(6, 182, 212, 0.3);
      animation: pulseListening 1.4s infinite;
    }

    @keyframes breathe {
      0% { transform: scale(0.96); opacity: 0.8; }
      100% { transform: scale(1.04); opacity: 1; }
    }

    @keyframes pulseListening {
      0% { box-shadow: 0 0 30px rgba(6, 182, 212, 0.3); }
      50% { box-shadow: 0 0 70px rgba(6, 182, 212, 0.7); }
      100% { box-shadow: 0 0 30px rgba(6, 182, 212, 0.3); }
    }

    /* Capsule Action Button ("Talk to AI") */
    .capsule-btn {
      background: linear-gradient(135deg, rgba(194, 120, 71, 0.25), rgba(80, 40, 20, 0.5));
      border: 1px solid rgba(194, 120, 71, 0.6);
      backdrop-filter: blur(16px);
      color: #fff;
      padding: 14px 32px;
      border-radius: 40px;
      font-size: 15px;
      font-weight: 600;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 12px;
      box-shadow: 0 8px 30px rgba(194, 120, 71, 0.25);
      transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
    }
    .capsule-btn:hover {
      transform: translateY(-2px);
      box-shadow: 0 12px 40px rgba(194, 120, 71, 0.4);
      background: linear-gradient(135deg, rgba(194, 120, 71, 0.4), rgba(80, 40, 20, 0.7));
    }
    .capsule-btn.active {
      border-color: #ef4444;
      background: rgba(239, 68, 68, 0.2);
      box-shadow: 0 0 25px rgba(239, 68, 68, 0.4);
    }

    .orb-controls-row {
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 14px;
      flex-wrap: wrap;
    }

    .center-next-btn {
      background: linear-gradient(135deg, rgba(194, 120, 71, 0.9), rgba(245, 158, 11, 0.95));
      border: 1px solid rgba(245, 158, 11, 0.7);
      backdrop-filter: blur(16px);
      color: #fff;
      padding: 13px 26px;
      border-radius: 40px;
      font-size: 14.5px;
      font-weight: 700;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 8px;
      box-shadow: 0 8px 24px rgba(245, 158, 11, 0.35);
      transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
      user-select: none;
    }
    .center-next-btn:hover {
      transform: translateY(-2px);
      box-shadow: 0 12px 32px rgba(245, 158, 11, 0.55);
      background: linear-gradient(135deg, #c27847, #f59e0b);
    }
    .center-next-btn:active {
      transform: translateY(0);
    }
    .center-next-btn.ready-pulse {
      animation: centerNextGlow 1.2s infinite alternate;
    }
    @keyframes centerNextGlow {
      0% { box-shadow: 0 0 12px rgba(245, 158, 11, 0.4); }
      100% { box-shadow: 0 0 32px rgba(245, 158, 11, 0.95); transform: translateY(-2px) scale(1.04); }
    }

    .audio-icon-bars {
      display: flex;
      align-items: center;
      gap: 3px;
      height: 16px;
    }
    .audio-icon-bars span {
      width: 3px;
      height: 100%;
      background: #fff;
      border-radius: 2px;
      animation: barJump 1s infinite ease-in-out;
    }
    .audio-icon-bars span:nth-child(1) { animation-delay: 0.1s; height: 8px; }
    .audio-icon-bars span:nth-child(2) { animation-delay: 0.3s; height: 16px; }
    .audio-icon-bars span:nth-child(3) { animation-delay: 0.2s; height: 12px; }

    @keyframes barJump {
      0%, 100% { transform: scaleY(0.4); }
      50% { transform: scaleY(1); }
    }

    /* Dialogue Subtitle Card */
    .subtitle-card {
      margin-top: 28px;
      max-width: 650px;
      width: 100%;
      background: rgba(18, 14, 24, 0.6);
      backdrop-filter: blur(16px);
      border: 1px solid var(--border-glass);
      border-radius: 18px;
      padding: 18px 24px;
      display: flex;
      flex-direction: column;
      gap: 8px;
      transition: all 0.3s;
    }

    .dialogue-ai {
      font-size: 15px;
      font-weight: 600;
      color: #f1f5f9;
      display: flex;
      align-items: flex-start;
      gap: 8px;
      text-align: left;
    }

    .dialogue-user {
      font-size: 13.5px;
      color: #38bdf8;
      display: flex;
      align-items: flex-start;
      gap: 8px;
      text-align: left;
      min-height: 20px;
    }

    .field-tag {
      font-size: 10px;
      font-weight: 700;
      padding: 2px 8px;
      border-radius: 6px;
      text-transform: uppercase;
      margin-left: 6px;
    }
    .tag-req { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.3); }
    .tag-opt { background: rgba(6, 182, 212, 0.2); color: #38bdf8; border: 1px solid rgba(6, 182, 212, 0.3); }

    .status-badge {
      margin-top: 10px;
      font-size: 12px;
      color: var(--copper);
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
    }

    .footer-note {
      position: relative;
      z-index: 10;
      text-align: center;
      padding: 20px;
      font-size: 13px;
      color: rgba(255, 255, 255, 0.4);
      letter-spacing: 0.3px;
    }

    /* Floating Bottom Live Subtitle & Correction Dock */
    .bottom-dock {
      position: fixed;
      bottom: 22px;
      left: 50%;
      transform: translateX(-50%);
      width: calc(100% - 36px);
      max-width: 860px;
      background: rgba(14, 11, 20, 0.88);
      backdrop-filter: blur(24px);
      -webkit-backdrop-filter: blur(24px);
      border: 1px solid rgba(194, 120, 71, 0.38);
      border-radius: 20px;
      box-shadow: 0 16px 50px rgba(0, 0, 0, 0.75), 0 0 35px rgba(194, 120, 71, 0.22);
      z-index: 95;
      padding: 14px 20px;
      display: flex;
      flex-direction: column;
      gap: 10px;
      transition: all 0.35s cubic-bezier(0.16, 1, 0.3, 1);
    }
    .bottom-dock.listening {
      border-color: rgba(6, 182, 212, 0.7);
      box-shadow: 0 16px 50px rgba(0, 0, 0, 0.75), 0 0 45px rgba(6, 182, 212, 0.35);
    }

    .dock-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      flex-wrap: wrap;
    }

    .dock-status-left {
      display: flex;
      align-items: center;
      gap: 10px;
      font-size: 12px;
      font-weight: 600;
      letter-spacing: 0.5px;
    }

    .live-mic-dot {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background: #64748b;
      transition: all 0.3s;
    }
    .live-mic-dot.active {
      background: #06b6d4;
      box-shadow: 0 0 14px #06b6d4;
      animation: pulseMic 1s infinite alternate;
    }
    @keyframes pulseMic {
      0% { transform: scale(0.9); opacity: 0.7; }
      100% { transform: scale(1.25); opacity: 1; }
    }

    .dock-label {
      color: var(--copper);
      font-weight: 700;
      text-transform: uppercase;
      font-size: 11px;
    }

    .dock-field-hint {
      color: #cbd5e1;
      background: rgba(255, 255, 255, 0.08);
      padding: 3px 10px;
      border-radius: 8px;
      font-size: 11.5px;
      font-family: 'JetBrains Mono', monospace;
      border: 1px solid rgba(255, 255, 255, 0.08);
    }

    .dock-actions-right {
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .dock-action-btn {
      background: rgba(255, 255, 255, 0.07);
      border: 1px solid rgba(255, 255, 255, 0.12);
      color: #f1f5f9;
      font-size: 12px;
      font-weight: 600;
      padding: 6px 14px;
      border-radius: 10px;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      transition: all 0.2s;
    }
    .dock-action-btn:hover {
      background: rgba(255, 255, 255, 0.15);
      transform: translateY(-1px);
    }
    .dock-action-btn.next-btn {
      background: linear-gradient(135deg, rgba(194, 120, 71, 0.95), rgba(245, 158, 11, 0.95));
      border: 1px solid rgba(245, 158, 11, 0.7);
      color: #fff;
      font-weight: 700;
      padding: 7px 18px;
      box-shadow: 0 4px 14px rgba(245, 158, 11, 0.35);
      transition: all 0.25s ease;
    }
    .dock-action-btn.next-btn:hover {
      background: linear-gradient(135deg, #c27847, #f59e0b);
      box-shadow: 0 6px 20px rgba(245, 158, 11, 0.55);
      transform: translateY(-2px);
    }
    .dock-action-btn.next-btn.pulse {
      animation: nextBtnGlow 1.2s infinite alternate;
    }
    @keyframes nextBtnGlow {
      0% { box-shadow: 0 0 10px rgba(245, 158, 11, 0.4); }
      100% { box-shadow: 0 0 24px rgba(245, 158, 11, 0.85); transform: translateY(-1px) scale(1.04); }
    }
    .dock-action-btn.edit-btn {
      border-color: rgba(245, 158, 11, 0.5);
      background: rgba(245, 158, 11, 0.14);
      color: #fde68a;
    }
    .dock-action-btn.edit-btn:hover {
      background: rgba(245, 158, 11, 0.28);
    }
    .dock-action-btn.auto-toggle-btn {
      background: rgba(255, 255, 255, 0.05);
      border-color: rgba(255, 255, 255, 0.15);
      color: #94a3b8;
    }
    .dock-action-btn.auto-toggle-btn.active {
      background: rgba(16, 185, 129, 0.18);
      border-color: rgba(16, 185, 129, 0.5);
      color: #34d399;
    }

    .dock-subtitle-row {
      display: flex;
      align-items: center;
      min-height: 40px;
      background: rgba(0, 0, 0, 0.5);
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 12px;
      padding: 8px 16px;
    }

    .subtitle-content {
      display: flex;
      align-items: center;
      flex-wrap: wrap;
      gap: 8px;
      width: 100%;
      font-size: 14.5px;
      line-height: 1.5;
    }

    .subtitle-prefix {
      color: #38bdf8;
      font-weight: 700;
      font-size: 12px;
      letter-spacing: 0.5px;
      text-transform: uppercase;
      flex-shrink: 0;
    }

    .mic-vu-meter {
      width: 76px;
      height: 9px;
      background: rgba(255, 255, 255, 0.08);
      border-radius: 6px;
      overflow: hidden;
      border: 1px solid rgba(255, 255, 255, 0.14);
      display: inline-flex;
      align-items: center;
      margin-left: 6px;
    }
    .mic-vu-bar {
      height: 100%;
      width: 0%;
      background: linear-gradient(90deg, #10b981 0%, #f59e0b 75%, #ef4444 100%);
      border-radius: 6px;
      transition: width 0.08s ease-out;
    }
    .dock-pitch-badge {
      font-size: 11px;
      font-weight: 700;
      padding: 2px 8px;
      border-radius: 8px;
      background: rgba(16, 185, 129, 0.2);
      color: #34d399;
      border: 1px solid rgba(16, 185, 129, 0.45);
      font-family: 'JetBrains Mono', monospace;
      display: inline-flex;
      align-items: center;
      gap: 4px;
      margin-left: 6px;
      transition: all 0.2s ease;
    }
    .dock-pitch-badge.high {
      background: rgba(245, 158, 11, 0.25);
      color: #fbbf24;
      border-color: rgba(245, 158, 11, 0.6);
      animation: pulseGlow 0.8s infinite alternate;
    }

    .live-words-container {
      display: inline-flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 6px;
      margin: 0 4px;
    }
    .live-word-pill {
      display: inline-flex;
      align-items: center;
      padding: 3px 9px;
      border-radius: 6px;
      font-size: 13.5px;
      font-weight: 600;
      color: #38bdf8;
      background: rgba(56, 189, 248, 0.12);
      border: 1px solid rgba(56, 189, 248, 0.35);
      animation: wordPop 0.15s cubic-bezier(0.175, 0.885, 0.32, 1.275);
      transition: all 0.2s ease;
    }
    .live-word-pill.active {
      color: #f59e0b;
      background: rgba(245, 158, 11, 0.2);
      border-color: rgba(245, 158, 11, 0.6);
      box-shadow: 0 0 12px rgba(245, 158, 11, 0.4);
    }
    .live-word-pill.committed {
      color: #34d399;
      background: rgba(16, 185, 129, 0.18);
      border-color: rgba(16, 185, 129, 0.5);
      box-shadow: 0 0 10px rgba(16, 185, 129, 0.3);
    }
    @keyframes wordPop {
      0% { transform: scale(0.7); opacity: 0; }
      100% { transform: scale(1); opacity: 1; }
    }
    .speed-metric-badge {
      display: inline-flex;
      align-items: center;
      gap: 4px;
      padding: 2px 8px;
      border-radius: 6px;
      font-size: 11.5px;
      font-family: 'JetBrains Mono', monospace;
      font-weight: 700;
      color: #34d399;
      background: rgba(16, 185, 129, 0.15);
      border: 1px solid rgba(16, 185, 129, 0.4);
      margin-left: auto;
      animation: pulseGlow 1s ease-in-out infinite alternate;
    }
    @keyframes pulseGlow {
      0% { opacity: 0.8; }
      100% { opacity: 1; transform: scale(1.03); }
    }

    .live-subtitles {
      color: #f8fafc;
      font-weight: 500;
      word-break: break-word;
      transition: color 0.2s;
    }
    .live-subtitles.interim {
      color: #94a3b8;
      font-style: italic;
    }
    .live-subtitles .highlight-clean {
      color: #34d399;
      font-weight: 700;
      background: rgba(16, 185, 129, 0.18);
      padding: 2px 8px;
      border-radius: 6px;
      border: 1px solid rgba(16, 185, 129, 0.35);
      margin-left: 4px;
    }

    .typing-cursor {
      display: inline-block;
      width: 7px;
      height: 15px;
      background: #06b6d4;
      margin-left: 4px;
      animation: blinkCursor 0.8s infinite;
      vertical-align: middle;
    }
    @keyframes blinkCursor {
      0%, 100% { opacity: 0; }
      50% { opacity: 1; }
    }

    /* Inline Bottom Correction Bar */
    .dock-correction-box {
      display: flex;
      gap: 8px;
      align-items: center;
      background: rgba(18, 14, 24, 0.95);
      border: 1px solid rgba(245, 158, 11, 0.4);
      border-radius: 12px;
      padding: 8px 12px;
      animation: fadeIn 0.25s ease-out;
    }
    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(4px); }
      to { opacity: 1; transform: translateY(0); }
    }

    .correction-input {
      flex: 1;
      background: #040306;
      border: 1px solid rgba(255, 255, 255, 0.12);
      border-radius: 8px;
      color: #f8fafc;
      font-size: 14px;
      padding: 8px 14px;
      outline: none;
      font-family: inherit;
      transition: border-color 0.2s;
    }
    .correction-input:focus {
      border-color: #f59e0b;
      box-shadow: 0 0 10px rgba(245, 158, 11, 0.3);
    }

    .correction-apply-btn {
      background: linear-gradient(135deg, #f59e0b, #d97706);
      color: #000;
      font-weight: 700;
      border: none;
      border-radius: 8px;
      padding: 8px 16px;
      cursor: pointer;
      font-size: 13px;
      transition: all 0.2s;
    }
    .correction-apply-btn:hover {
      transform: translateY(-1px);
      box-shadow: 0 4px 15px rgba(245, 158, 11, 0.4);
    }

    .correction-cancel-btn {
      background: rgba(255, 255, 255, 0.08);
      color: #94a3b8;
      border: none;
      border-radius: 8px;
      padding: 8px 12px;
      cursor: pointer;
      font-size: 14px;
    }
    .correction-cancel-btn:hover {
      background: rgba(255, 255, 255, 0.16);
      color: #fff;
    }

    /* Live JSON Drawer (Slide-Over) */
    .drawer-edit-panel {
      margin-top: 14px;
      margin-bottom: 14px;
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid rgba(194, 120, 71, 0.3);
      border-radius: 14px;
      padding: 12px 14px;
    }
    .drawer-edit-title {
      font-size: 11.5px;
      font-weight: 700;
      color: var(--amber);
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }
    .drawer-select, .dock-field-select {
      background: #0d0a12;
      border: 1px solid rgba(255, 255, 255, 0.15);
      color: #fff;
      font-size: 13px;
      padding: 8px 12px;
      border-radius: 8px;
      outline: none;
      width: 100%;
      cursor: pointer;
    }
    .dock-field-select {
      max-width: 170px;
      flex-shrink: 0;
    }
    .drawer-input {
      background: #040306;
      border: 1px solid rgba(255, 255, 255, 0.15);
      color: #fff;
      font-size: 13px;
      padding: 8px 12px;
      border-radius: 8px;
      flex: 1;
      outline: none;
    }
    .drawer-input:focus {
      border-color: var(--amber);
    }
    .drawer-save-btn {
      background: linear-gradient(135deg, #c27847, #f59e0b);
      border: none;
      color: #fff;
      font-weight: 700;
      font-size: 12px;
      padding: 8px 16px;
      border-radius: 8px;
      cursor: pointer;
      transition: all 0.2s;
      white-space: nowrap;
    }
    .drawer-save-btn:hover {
      transform: translateY(-1px);
      box-shadow: 0 4px 12px rgba(245, 158, 11, 0.4);
    }

    .drawer-overlay {
      position: fixed;
      top: 0;
      left: 0;
      width: 100vw;
      height: 100vh;
      background: rgba(0, 0, 0, 0.6);
      backdrop-filter: blur(8px);
      z-index: 99;
      opacity: 0;
      pointer-events: none;
      transition: opacity 0.3s ease;
    }
    .drawer-overlay.open {
      opacity: 1;
      pointer-events: auto;
    }

    .drawer {
      position: fixed;
      top: 0;
      right: 0;
      width: 480px;
      max-width: 90vw;
      height: 100vh;
      background: rgba(13, 10, 18, 0.95);
      backdrop-filter: blur(24px);
      border-left: 1px solid var(--border-glass);
      z-index: 100;
      transform: translateX(100%);
      transition: transform 0.35s cubic-bezier(0.16, 1, 0.3, 1);
      display: flex;
      flex-direction: column;
      padding: 24px;
      gap: 16px;
    }
    .drawer.open {
      transform: translateX(0);
    }

    .drawer-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    pre.json-box {
      flex: 1;
      background: #040306;
      border: 1px solid var(--border-glass);
      border-radius: 14px;
      padding: 16px;
      overflow: auto;
      font-family: 'JetBrains Mono', monospace;
      font-size: 12px;
      color: #38bdf8;
      line-height: 1.45;
    }
  </style>
</head>
<body>
  <!-- Ambient Ribbon Wave Canvas -->
  <canvas id="meshCanvas"></canvas>

  <nav>
    <div class="brand">
      <div class="brand-logo">🎙️</div>
      <span class="brand-title">Store Onboarding AI</span>
    </div>
    <div class="nav-actions">
      <div class="lang-selector" id="langSelector" title="Select Speech & UI Language">
        <button class="lang-pill" data-lang="en" onclick="selectInitialLanguage('en')">English</button>
        <button class="lang-pill" data-lang="hi" onclick="selectInitialLanguage('hi')">हिंदी</button>
        <button class="lang-pill" data-lang="te" onclick="selectInitialLanguage('te')">తెలుగు (Tenglish)</button>
      </div>
      <a href="/docs" target="_blank" class="glass-btn" style="text-decoration:none; display:inline-flex; align-items:center; gap:5px;" title="Open Interactive Swagger API Documentation">📜 Swagger API</a>
      <button class="glass-btn" id="toggleAutoNextBtn" onclick="toggleAutoNext()" title="Toggle Auto-Next vs Manual Next confirmation">
        <span id="autoNextIcon">⏸️</span> <span id="autoNextLabel">Auto-Next: OFF</span>
      </button>
      <button class="glass-btn" id="voiceSpeedBtn" onclick="toggleVoiceSpeed()" title="Toggle AI Voice Speed (1.0x, 1.25x, 1.5x)">
        <span id="voiceSpeedIcon">⚡</span> <span id="voiceSpeedLabel">Speed: 1.25x</span>
      </button>
      <button class="glass-btn" onclick="openLanguageModal(true)" title="Change language">
        <span>🌐</span> Language
      </button>
      <button class="glass-btn" onclick="toggleDrawer()">
        <span>✨</span> View Live JSON (<span id="filledCount">0</span>/13)
      </button>
      <button class="glass-btn" onclick="resetForm()">
        <span>🔄</span> Reset
      </button>
    </div>
  </nav>

  <main class="hero-container">
    <div class="hero-eyebrow">Autonomous Voice Agent</div>
    <h1 class="hero-title">Experience the future<br>of store onboarding.</h1>
    <p class="hero-subtitle">Automate registration seamlessly with your voice. The AI speaks, listens, and fills your store profile automatically.</p>

    <!-- Holographic Audio Orb Stage -->
    <div class="orb-stage" id="orbStage" onclick="handleOrbClick()">
      <div class="orb-halo"></div>
      <canvas class="orb-canvas" id="orbCanvas" width="260" height="260"></canvas>
    </div>

    <!-- Main Capsule Controls Row -->
    <div class="orb-controls-row">
      <button class="capsule-btn" id="talkBtn" onclick="handleOrbClick()">
        <div class="audio-icon-bars" id="audioBars">
          <span></span><span></span><span></span>
        </div>
        <span id="btnLabel">Talk to AI</span>
      </button>

      <button class="center-next-btn" id="centerNextBtn" onclick="handleNextClick()" title="Next Question: click when your input is completed (or press Enter)">
        <span>Next</span>
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M5 12h14M12 5l7 7-7 7"/></svg>
      </button>
    </div>

    <!-- Real-time Dialogue & Question Subtitle -->
    <div class="subtitle-card" id="subtitleCard">
      <div class="dialogue-ai">
        <span style="color: #a78bfa; flex-shrink: 0;">🤖 They told:</span>
        <span id="aiDialogueText">Welcome! Please choose your preferred language: English, Hindi, or Telugu.</span>
        <span id="fieldBadge" class="field-tag tag-req" style="display:none;">REQUIRED</span>
      </div>
      <div class="dialogue-user">
        <span style="color: #38bdf8; flex-shrink: 0;">👤 I told:</span>
        <span id="userDialogueText">Waiting for your voice or language selection...</span>
      </div>
      <div class="status-badge" id="botStatus">
        <span id="statusDot" style="width: 8px; height: 8px; border-radius: 50%; background: #f59e0b;"></span>
        <span id="statusText">Choose your language to begin</span>
        <span id="langDetectedBadge" class="detected-lang-pill" style="display:none;"></span>
      </div>
    </div>
  </main>

  <footer class="footer-note">
    Powered by Faster-Whisper Base &bull; Visakhapatnam, Andhra Pradesh context ready &bull; Experience what the future feels like.
  </footer>

  <!-- Interactive Language Selection Modal -->
  <div class="lang-modal-backdrop" id="langModalBackdrop">
    <div class="lang-modal-card">
      <button class="lang-modal-close" id="langModalCloseBtn" onclick="closeLanguageModal()" title="Close" style="display: none;">✕</button>
      <div class="lang-modal-badge">🌐 Language Preference</div>
      <h2 class="lang-modal-title">Choose Your Language</h2>
      <p class="lang-modal-sub">
        अपनी पसंदीदा भाषा चुनें &bull; దయచేసి మీ భాషను ఎంచుకోండి<br>
        Select how you want to speak and listen with the Store Onboarding Assistant.
      </p>

      <div class="lang-autoplay-notice" id="autoplayNotice" onclick="triggerUserGestureGreeting(event)" style="display: none;">
        <span>🔊 <strong>Click to hear AI Greeting & Speak</strong> 🎙️</span>
      </div>

      <div class="lang-grid">
        <button class="lang-card-btn" id="langCardEn" onclick="selectInitialLanguage('en')">
          <span class="lang-card-flag">🇬🇧</span>
          <span class="lang-card-name">English</span>
          <span class="lang-card-desc">Standard Business</span>
        </button>
        <button class="lang-card-btn" id="langCardHi" onclick="selectInitialLanguage('hi')">
          <span class="lang-card-flag">🇮🇳</span>
          <span class="lang-card-name">हिंदी</span>
          <span class="lang-card-desc">Hindi Assistant</span>
        </button>
        <button class="lang-card-btn" id="langCardTe" onclick="selectInitialLanguage('te')">
          <span class="lang-card-flag">🇮🇳</span>
          <span class="lang-card-name">తెలుగు (Tenglish)</span>
          <span class="lang-card-desc">Modern Telugu + English</span>
        </button>
      </div>

      <div class="lang-voice-hint" id="langVoiceHint">
        <span>🎙️</span>
        <span id="langVoiceHintText">Tap any language above or speak: <strong>"Telugu"</strong>, <strong>"Hindi"</strong>, or <strong>"English"</strong></span>
      </div>
    </div>
  </div>

  <!-- Fixed Floating Bottom Live Subtitle & Correction Dock -->
  <div class="bottom-dock" id="bottomDock">
    <div class="dock-header">
      <div class="dock-status-left">
        <div class="live-mic-dot" id="dockMicDot"></div>
        <span class="dock-label" id="dockStatusLabel">LIVE SUBTITLES</span>
        <span class="dock-field-hint" id="dockFieldHint">Language Selection</span>
        <div class="mic-vu-meter" id="micVuMeter" title="Live Microphone Audio Level">
          <div class="mic-vu-bar" id="micVuBar"></div>
        </div>
        <span id="dockPitchBadge" class="dock-pitch-badge" style="display:none;" title="Live Detected Voice Pitch">🎵 High Pitch</span>
      </div>
      <div class="dock-actions-right">
        <button class="dock-action-btn auto-toggle-btn" id="dockAutoNextBtn" onclick="toggleAutoNext()" title="Toggle Auto-Next vs Manual Next Button">
          <span id="dockAutoNextIcon">⏸️</span> <span id="dockAutoNextLabel">Auto-Next: OFF</span>
        </button>
        <button class="dock-action-btn next-btn" id="dockNextBtn" onclick="handleNextClick()" title="Store to JSON & ask next question (or press Enter)">
          <span>Next</span> <span style="font-size: 14px;">➔</span>
        </button>
        <button class="dock-action-btn edit-btn" id="btnToggleCorrection" onclick="toggleCorrectionMode()" title="Correct or edit what was captured">
          <span>✏️</span> <span id="correctBtnText">Correction</span>
        </button>
        <button class="dock-action-btn" onclick="retryVoice()" title="Re-speak current question">
          <span>🔄</span> Redo
        </button>
        <button class="dock-action-btn" id="dockSkipBtn" onclick="skipField()" title="Skip field">
          <span>⏭️</span> Skip
        </button>
      </div>
    </div>

    <!-- Live Speech Subtitles & Real-time Words Display -->
    <div class="dock-subtitle-row" id="dockSubtitleRow">
      <div class="subtitle-content">
        <span class="subtitle-prefix" id="subtitlePrefix">👤 You:</span>
        <div id="liveWordsContainer" class="live-words-container"></div>
        <span class="live-subtitles" id="liveSubtitles">Speak or select: English, Hindi, or Telugu / अपनी भाषा चुनें / మీ భాషను ఎంచుకోండి</span>
        <span class="typing-cursor" id="typingCursor" style="display: none;"></span>
        <span id="speedMetricBadge" class="speed-metric-badge" style="display: none;">⚡ 8ms</span>
      </div>
    </div>

    <!-- Inline Bottom Correction Bar -->
    <div class="dock-correction-box" id="dockCorrectionBox" style="display: none;">
      <select id="correctionFieldSelect" class="dock-field-select" onchange="onCorrectionFieldSelectChange()" title="Pick which field to correct">
        <!-- Auto-populated -->
      </select>
      <input type="text" id="correctionInput" class="correction-input" placeholder="Type or edit your answer..." />
      <button class="correction-apply-btn" onclick="submitCorrection()">✓ Save & Next ➔</button>
      <button class="correction-cancel-btn" onclick="toggleCorrectionMode(false)">✕</button>
    </div>
  </div>

  <!-- Slide-Over Drawer for Live JSON -->
  <div class="drawer-overlay" id="drawerOverlay" onclick="toggleDrawer()"></div>
  <aside class="drawer" id="drawer">
    <div class="drawer-header">
      <div>
        <h3 style="font-size: 16px; font-weight: 700;">Live Store Profile JSON</h3>
        <p style="font-size: 12px; color: var(--text-muted); margin-top: 4px;">Fills automatically as you speak</p>
      </div>
      <button class="glass-btn" onclick="toggleDrawer()">✕ Close</button>
    </div>

    <div style="display: flex; gap: 8px;">
      <button class="glass-btn" style="flex: 1;" onclick="copyJSON()">📋 Copy JSON</button>
      <button class="glass-btn" onclick="resetForm()">Reset</button>
    </div>

    <!-- Quick Correct Any JSON Field Panel -->
    <div class="drawer-edit-panel">
      <div class="drawer-edit-title">✏️ Correct Any JSON Field:</div>
      <div style="margin-top: 8px;">
        <select id="drawerFieldSelect" class="drawer-select" onchange="onDrawerFieldChange()">
          <option value="companyName">1. Company / Store Name</option>
          <option value="personName">2. Contact Person Name</option>
          <option value="phone">3. Phone Number</option>
          <option value="email">4. Email Address</option>
          <option value="storeCategory">5. Store Category</option>
          <option value="doorNo">6. Door / Shop Number</option>
          <option value="street">7. Street Name</option>
          <option value="area">8. Area / Locality</option>
          <option value="town">9. Town / Village</option>
          <option value="city">10. City</option>
          <option value="state">11. State</option>
          <option value="pincode">12. Pincode</option>
          <option value="usedReferalCode">13. Referral Code</option>
        </select>
      </div>
      <div style="display: flex; gap: 8px; margin-top: 8px;">
        <input type="text" id="drawerFieldInput" class="drawer-input" placeholder="Type correct value..." />
        <button class="drawer-save-btn" onclick="saveDrawerFieldEdit()">✓ Save to JSON</button>
      </div>
    </div>

    <pre class="json-box" id="jsonBox">Loading...</pre>
  </aside>

  <script>
    // Audio & State
    let isSessionRunning = false;
    let isAiSpeaking = false;
    let isUserListening = false;
    let currentField = null;

    let mediaRecorder = null;
    let audioChunks = [];
    let audioCtx = null;
    let analyser = null;
    let dataArray = new Uint8Array(256);
    let silenceTimer = null;
    let speechBreakTimer = null;
    let hasSpoken = false;
    let ambientNoiseFloor = 10;
    let highPitchNoiseFloor = 8;
    let lastDetectedPitch = 0;
    let noiseCalibrationSamples = 0;

    const orbStage = document.getElementById('orbStage');
    const talkBtn = document.getElementById('talkBtn');
    const btnLabel = document.getElementById('btnLabel');
    const aiDialogueText = document.getElementById('aiDialogueText');
    const userDialogueText = document.getElementById('userDialogueText');
    const fieldBadge = document.getElementById('fieldBadge');
    const statusText = document.getElementById('statusText');
    const statusDot = document.getElementById('statusDot');
    const jsonBox = document.getElementById('jsonBox');
    const filledCount = document.getElementById('filledCount');
    const drawer = document.getElementById('drawer');
    const drawerOverlay = document.getElementById('drawerOverlay');

    // Bottom Live Dock & Subtitle elements
    const bottomDock = document.getElementById('bottomDock');
    const dockMicDot = document.getElementById('dockMicDot');
    const dockStatusLabel = document.getElementById('dockStatusLabel');
    const dockFieldHint = document.getElementById('dockFieldHint');
    const liveSubtitles = document.getElementById('liveSubtitles');
    const typingCursor = document.getElementById('typingCursor');
    const dockCorrectionBox = document.getElementById('dockCorrectionBox');
    const correctionInput = document.getElementById('correctionInput');
    const btnToggleCorrection = document.getElementById('btnToggleCorrection');
    const correctBtnText = document.getElementById('correctBtnText');
    const dockSkipBtn = document.getElementById('dockSkipBtn');
    const langDetectedBadge = document.getElementById('langDetectedBadge');

    let currentLang = 'auto'; // 'auto' | 'en' | 'hi' | 'te'
    let hasSelectedLanguage = false;
    let autoProceedTimer = null;
    let liveRecognition = null;
    let isCorrectionMode = false;
    let autoNextEnabled = true;
    let currentVoiceSpeed = 1.25; // 1.25x Fast speech speed

    function toggleVoiceSpeed() {
      if (currentVoiceSpeed === 1.25) currentVoiceSpeed = 1.5;
      else if (currentVoiceSpeed === 1.5) currentVoiceSpeed = 1.0;
      else currentVoiceSpeed = 1.25;

      const label = document.getElementById('voiceSpeedLabel');
      if (label) label.innerText = `Speed: ${currentVoiceSpeed}x`;
      const btn = document.getElementById('voiceSpeedBtn');
      if (btn) {
        if (currentVoiceSpeed > 1.0) btn.classList.add('active');
        else btn.classList.remove('active');
      }
      const speedBadge = document.getElementById('speedMetricBadge');
      if (speedBadge) {
        speedBadge.innerText = `⚡ ${currentVoiceSpeed}x Speed`;
        speedBadge.style.display = 'inline-flex';
      }
    }

    function toggleAutoNext(forceState) {
      if (forceState !== undefined) {
        autoNextEnabled = forceState;
      } else {
        autoNextEnabled = !autoNextEnabled;
      }
      const btn = document.getElementById('dockAutoNextBtn');
      const navBtn = document.getElementById('toggleAutoNextBtn');
      const label = autoNextEnabled ? "Auto-Next: ON" : "Auto-Next: OFF";
      const icon = autoNextEnabled ? "⚡" : "⏸️";
      if (btn) {
        btn.innerHTML = `<span>${icon}</span> <span>${label}</span>`;
        if (autoNextEnabled) btn.classList.add('active');
        else btn.classList.remove('active');
      }
      if (navBtn) {
        navBtn.innerHTML = `<span>${icon}</span> <span>${label}</span>`;
      }
      if (!autoNextEnabled && autoProceedTimer) {
        clearTimeout(autoProceedTimer);
        autoProceedTimer = null;
        statusText.innerText = "Manual Next mode active. Click 'Next' (or press Enter) to store to JSON.";
        statusDot.style.background = "#10b981";
      }
    }

    function openLanguageModal(isManualOpen = false) {
      const modal = document.getElementById('langModalBackdrop');
      const closeBtn = document.getElementById('langModalCloseBtn');
      if (modal) modal.classList.remove('hidden');
      if (closeBtn) {
        closeBtn.style.display = (isManualOpen && hasSelectedLanguage) ? 'inline-flex' : 'none';
      }
    }

    function closeLanguageModal() {
      const modal = document.getElementById('langModalBackdrop');
      if (modal) modal.classList.add('hidden');
    }

    function showAutoplayBadge() {
      const notice = document.getElementById('autoplayNotice');
      if (notice) notice.style.display = 'inline-flex';
      const onFirstTap = (e) => {
        if (e.target && e.target.closest('.lang-card-btn')) return;
        triggerUserGestureGreeting(e);
        window.removeEventListener('pointerdown', onFirstTap);
        window.removeEventListener('click', onFirstTap);
      };
      window.addEventListener('pointerdown', onFirstTap, { once: true });
      window.addEventListener('click', onFirstTap, { once: true });
    }

    function hideAutoplayBadge() {
      const notice = document.getElementById('autoplayNotice');
      if (notice) notice.style.display = 'none';
    }

    function triggerUserGestureGreeting(e) {
      if (e) e.stopPropagation();
      hideAutoplayBadge();
      if (!hasSelectedLanguage) {
        playGreetingWithAutoplayFallback("Welcome! Please choose your preferred language: English, Hindi, or Telugu.");
      }
    }

    function playGreetingWithAutoplayFallback(promptText) {
      stopAllSpeech();
      setAiSpeakingState(true);

      const ttsUrl = `/api/tts?text=${encodeURIComponent(promptText)}&lang=en`;
      const audio = new Audio(ttsUrl);
      audio.playbackRate = currentVoiceSpeed;
      currentTtsAudio = audio;

      let finished = false;
      const done = () => {
        if (finished) return;
        finished = true;
        setAiSpeakingState(false);
        currentTtsAudio = null;
        if (!hasSelectedLanguage) {
          setTimeout(autoEnableUserMicrophone, 200);
        }
      };

      audio.onended = done;
      audio.onerror = () => {
        fallbackWebSpeech(promptText, 'en-IN', done);
      };

      const playPromise = audio.play();
      if (playPromise !== undefined) {
        playPromise.then(() => {
          hideAutoplayBadge();
        }).catch(err => {
          console.log("Autoplay waiting for user gesture:", err);
          setAiSpeakingState(false);
          showAutoplayBadge();
        });
      }
    }

    function initiateInitialLanguageSelection() {
      if (hasSelectedLanguage) return;
      openLanguageModal(false);

      const promptText = "Welcome! Please choose your preferred language: English, Hindi, or Telugu.";
      aiDialogueText.innerText = promptText;
      liveSubtitles.innerText = "Welcome! Speak or select: English, Hindi, or Telugu / अपनी भाषा चुनें / మీ భాషను ఎంచుకోండి";
      dockFieldHint.innerText = "Language Selection";
      statusText.innerText = "🌐 Please choose your language to begin...";
      statusDot.style.background = "#f59e0b";

      playGreetingWithAutoplayFallback(promptText);
    }

    function selectInitialLanguage(lang) {
      currentLang = lang;
      hasSelectedLanguage = true;
      hideAutoplayBadge();
      closeLanguageModal();
      setLanguage(lang);

      try { localStorage.setItem('preferredLang', lang); } catch (e) {}

      let ack = "Language set to English! Let's get your store registered. What is your company or store name?";
      if (lang === 'hi') {
        ack = "भाषा हिंदी चुनी गई है! आइए आपकी दुकान का पंजीकरण शुरू करते हैं। आपकी दुकान या कंपनी का नाम क्या है?";
      } else if (lang === 'te') {
        ack = "Telugu language select అయింది! Store register చేద్దాం. మీ store లేదా company name ఏంటి?";
      }

      isSessionRunning = true;
      talkBtn.classList.add('active');
      btnLabel.innerText = "AI Active (Listening...)";

      setAiSpeakingState(true);
      aiDialogueText.innerText = ack;
      liveSubtitles.innerText = `Language set to ${lang.toUpperCase()}`;

      fetch('/api/set_language', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ lang: lang })
      })
      .then(r => r.json())
      .then(data => {
        if (data.current_field) {
          currentField = data.current_field;
          updateFieldDisplay(currentField);
        }
        if (data.tts_audio_base64) {
          playBotSpeech(data, () => {
            setAiSpeakingState(false);
            autoEnableUserMicrophone();
          });
        } else {
          speakText(ack, () => {
            setAiSpeakingState(false);
            autoEnableUserMicrophone();
          });
        }
      })
      .catch(err => {
        speakText(ack, () => {
          setAiSpeakingState(false);
          autoEnableUserMicrophone();
        });
      });
    }

    function setLanguage(lang) {
      currentLang = lang;
      document.querySelectorAll('.lang-pill').forEach(btn => {
        if (btn.getAttribute('data-lang') === lang) {
          btn.classList.add('active');
        } else {
          btn.classList.remove('active');
        }
      });

      if (lang !== 'auto') {
        const langNames = { hi: "हिंदी (Hindi)", te: "తెలుగు (Tenglish)", en: "English" };
        langDetectedBadge.innerText = `🌐 ${langNames[lang] || lang.toUpperCase()}`;
        langDetectedBadge.style.display = 'inline-flex';
      } else {
        langDetectedBadge.style.display = 'none';
      }

      if (liveRecognition) {
        // Recognition language optimization:
        // en-IN recognizes Indian English words, Indian store names, numbers, and Tenglish loanwords with 99% accuracy!
        if (currentLang === 'hi') {
          liveRecognition.lang = 'hi-IN';
        } else if (currentLang === 'te') {
          liveRecognition.lang = (window.isPureTeluguScript) ? 'te-IN' : 'en-IN';
        } else {
          liveRecognition.lang = 'en-IN';
        }
      }

      if (currentField) {
        let q = currentField.question_en || currentField.question;
        if (lang === 'hi') q = currentField.question_hi || q;
        else if (lang === 'te') q = currentField.question_te || q;
        else if (lang === 'en') q = currentField.question_en || q;

        currentField.question = q;
        aiDialogueText.innerText = q;
        liveSubtitles.innerText = `AI asked: "${q}" — waiting for speech...`;
        correctionInput.placeholder = `Type ${currentField.label} (${lang.toUpperCase()})...`;
      }
    }

    // Canvas Background Waves
    const meshCanvas = document.getElementById('meshCanvas');
    const mCtx = meshCanvas.getContext('2d');
    function resizeMesh() {
      meshCanvas.width = window.innerWidth;
      meshCanvas.height = window.innerHeight;
    }
    window.addEventListener('resize', resizeMesh);
    resizeMesh();

    let waveOffset = 0;
    function renderMeshBackground() {
      mCtx.clearRect(0, 0, meshCanvas.width, meshCanvas.height);
      waveOffset += 0.008;

      const lines = 18;
      const cy = meshCanvas.height * 0.48;

      for (let i = 0; i < lines; i++) {
        mCtx.beginPath();
        const factor = i / lines;
        mCtx.strokeStyle = `rgba(194, 120, 71, ${0.08 + factor * 0.15})`;
        mCtx.lineWidth = 1;

        for (let x = 0; x < meshCanvas.width; x += 20) {
          const y = cy + Math.sin(x * 0.003 + waveOffset + factor * 2) * (60 + factor * 80)
                       + Math.cos(x * 0.001 - waveOffset) * 40;
          if (x === 0) mCtx.moveTo(x, y);
          else mCtx.lineTo(x, y);
        }
        mCtx.stroke();
      }
      requestAnimationFrame(renderMeshBackground);
    }
    renderMeshBackground();

    // Central Holographic Orb Drawing
    const orbCanvas = document.getElementById('orbCanvas');
    const oCtx = orbCanvas.getContext('2d');
    let orbRotation = 0;

    function renderOrb() {
      oCtx.clearRect(0, 0, orbCanvas.width, orbCanvas.height);
      orbRotation += 0.02;

      const cx = 130;
      const cy = 130;

      let volumeBoost = 0;
      if (analyser && dataArray instanceof Uint8Array && isUserListening) {
        try {
          analyser.getByteFrequencyData(dataArray);
          let highSum = 0;
          let count = 0;
          // Focus on high pitch vocal bins (180 Hz to 3800 Hz)
          const binHz = audioCtx ? audioCtx.sampleRate / analyser.fftSize : 93.75;
          const start = Math.max(1, Math.round(180 / binHz));
          const end = Math.min(dataArray.length - 1, Math.round(3800 / binHz));
          for (let i = start; i <= end; i++) {
            highSum += dataArray[i];
            count++;
          }
          const highAvg = count > 0 ? (highSum / count) : 0;
          volumeBoost = highAvg * 0.55;
        } catch (e) {}
      }

      const baseRadius = 75 + volumeBoost;

      // Draw multi-layered glowing plasma rings
      for (let ring = 0; ring < 3; ring++) {
        oCtx.beginPath();
        const r = baseRadius + ring * 12;
        const grad = oCtx.createLinearGradient(0, 0, 260, 260);

        if (isAiSpeaking) {
          grad.addColorStop(0, '#f59e0b');
          grad.addColorStop(0.5, '#c27847');
          grad.addColorStop(1, '#ef4444');
        } else if (isUserListening) {
          grad.addColorStop(0, '#06b6d4');
          grad.addColorStop(0.5, '#3b82f6');
          grad.addColorStop(1, '#10b981');
        } else {
          grad.addColorStop(0, '#c27847');
          grad.addColorStop(0.5, '#8b4513');
          grad.addColorStop(1, '#502814');
        }

        oCtx.strokeStyle = grad;
        oCtx.lineWidth = 2.5;

        for (let a = 0; a < Math.PI * 2; a += 0.1) {
          const wobble = Math.sin(a * 4 + orbRotation + ring) * (4 + volumeBoost * 0.2);
          const x = cx + Math.cos(a) * (r + wobble);
          const y = cy + Math.sin(a) * (r + wobble);
          if (a === 0) oCtx.moveTo(x, y);
          else oCtx.lineTo(x, y);
        }
        oCtx.closePath();
        oCtx.stroke();
      }

      // Inner Core Glow
      const coreGrad = oCtx.createRadialGradient(cx, cy, 10, cx, cy, 70);
      if (isAiSpeaking) {
        coreGrad.addColorStop(0, 'rgba(245, 158, 11, 0.45)');
        coreGrad.addColorStop(1, 'rgba(0, 0, 0, 0)');
      } else if (isUserListening) {
        coreGrad.addColorStop(0, 'rgba(6, 182, 212, 0.45)');
        coreGrad.addColorStop(1, 'rgba(0, 0, 0, 0)');
      } else {
        coreGrad.addColorStop(0, 'rgba(194, 120, 71, 0.25)');
        coreGrad.addColorStop(1, 'rgba(0, 0, 0, 0)');
      }
      oCtx.fillStyle = coreGrad;
      oCtx.beginPath();
      oCtx.arc(cx, cy, 70, 0, Math.PI * 2);
      oCtx.fill();

      requestAnimationFrame(renderOrb);
    }
    renderOrb();

    // Load initial JSON & Session State
    fetch('/api/get_state')
      .then(r => r.json())
      .then(data => {
        updateJSONView(data.json);
        currentField = data.current_field;
        updateFieldDisplay(currentField);

        if (data.language_chosen && data.language && data.language !== 'auto') {
          hasSelectedLanguage = true;
          currentLang = data.language;
          setLanguage(data.language);
          closeLanguageModal();
        } else {
          // Inital load: immediately prompt for language!
          initiateInitialLanguageSelection();
        }
      })
      .catch(err => {
        console.warn("get_state error, initiating initial language prompt:", err);
        initiateInitialLanguageSelection();
      });

    // Main Interaction: Click to start or toggle
    function handleOrbClick() {
      if (!isSessionRunning) {
        startAutonomousSession();
      } else if (isAiSpeaking) {
        // User clicked while AI was speaking: stop speech and listen immediately
        stopAllSpeech();
        setAiSpeakingState(false);
        autoEnableUserMicrophone();
      } else if (isUserListening) {
        // Manual override: user clicked orb to finish immediately
        finishUserVoiceTurn();
      }
    }

    // Step 1: Start the session & auto-ask question
    function startAutonomousSession() {
      isSessionRunning = true;
      talkBtn.classList.add('active');
      btnLabel.innerText = "AI Active (Listening...)";
      if (!hasSelectedLanguage) {
        initiateInitialLanguageSelection();
      } else {
        askQuestionAutomatically();
      }
    }

    let latestLiveTranscript = '';
    let currentStoreJson = {};

    // Live Web Speech Recognition for Real-time Subtitles
    function startLiveSpeech() {
      const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
      if (!SpeechRec) return;

      stopLiveSpeech();

      try {
        liveRecognition = new SpeechRec();
        liveRecognition.continuous = true;
        liveRecognition.interimResults = true;
        // Recognition language optimization:
        // en-IN recognizes Indian English words, Indian store names, numbers, and Tenglish loanwords with 99% accuracy!
        if (currentLang === 'hi') {
          liveRecognition.lang = 'hi-IN';
        } else if (currentLang === 'te') {
          liveRecognition.lang = (window.isPureTeluguScript) ? 'te-IN' : 'en-IN';
        } else {
          liveRecognition.lang = 'en-IN';
        }

        liveRecognition.onresult = (e) => {
          let interim = '';
          let final = '';
          for (let i = e.resultIndex; i < e.results.length; ++i) {
            if (e.results[i].isFinal) final += e.results[i][0].transcript;
            else interim += e.results[i][0].transcript;
          }
          const spoken = (final || interim).trim();
          if (!spoken) return;

          hasSpoken = true;
          latestLiveTranscript = spoken;
          renderLiveWords(spoken, false);
          liveSubtitles.innerText = '';
          typingCursor.style.display = 'inline-block';
          userDialogueText.innerText = spoken;
          dockMicDot.classList.add('active');
          bottomDock.classList.add('listening');
          correctionInput.value = spoken;
          triggerNextPulse(true);

          const curKey = currentField ? currentField.key : '';
          const spokenLower = spoken.toLowerCase().trim();

          // 1. Language Selection Voice Trigger (instant ms save)
          if (!hasSelectedLanguage) {
            if (spokenLower.includes('telugu') || spokenLower.includes('tenglish') || spokenLower.includes('తెలుగు')) {
              selectInitialLanguage('te');
              return;
            }
            if (spokenLower.includes('hindi') || spokenLower.includes('हिंदी')) {
              selectInitialLanguage('hi');
              return;
            }
            if (spokenLower.includes('english') || spokenLower.includes('ఆంగ్ల') || spokenLower.includes('अंग्रेज़ी') || spokenLower.includes('angrezi')) {
              selectInitialLanguage('en');
              return;
            }
          }

          // 2. Skip Voice Trigger (instant ms save)
          const isSkipPhrase = /^(skip|skip chey|skip cheyyi|skip cheseyandi|em ledu|em ledhu|emi ledhu|ledandi|avsaram ledu|avsaram ledhu|ippudu ledu|ippudu vaddu|daggara ledu|oddhu|vaddandi|chhod do|nahi chahiye|leave it|next)$/i.test(spokenLower) ||
                               spokenLower.startsWith('skip') || spokenLower.startsWith('em ledu') || spokenLower.startsWith('em ledhu');
          if (isSkipPhrase && currentField && !currentField.required) {
            saveVoiceAnswerFast("skip");
            return;
          }

          // 3. 10-digit Phone Number Instant Trigger
          if (curKey === 'phone') {
            const digitsOnly = spoken.replace(/\D/g, '');
            if (digitsOnly.length === 10) {
              saveVoiceAnswerFast(digitsOnly);
              return;
            }
          }

          // 4. 6-digit Pincode Instant Trigger
          if (curKey === 'pincode') {
            const digitsOnly = spoken.replace(/\D/g, '');
            if (digitsOnly.length === 6) {
              saveVoiceAnswerFast(digitsOnly);
              return;
            }
          }

          // 5. Intelligent Pause Wait: 650ms on final speech, 900ms on interim (never cuts off user mid-thought)
          if (speechBreakTimer) clearTimeout(speechBreakTimer);
          const pauseWait = final ? 650 : 900;
          speechBreakTimer = setTimeout(() => {
            if (isUserListening && hasSpoken && spoken) {
              saveVoiceAnswerFast(spoken);
            }
          }, pauseWait);
        };

        liveRecognition.onerror = (e) => {
          if (e.error !== 'no-speech') {
            console.warn("[SpeechRec Event]:", e.error);
          }
        };

        liveRecognition.onend = () => {
          // Self-healing: if browser ends recognition while user is still listening, restart immediately!
          if (isUserListening && !isAiSpeaking) {
            setTimeout(() => {
              if (isUserListening && !isAiSpeaking) {
                try { liveRecognition.start(); } catch (err) {}
              }
            }, 80);
          }
        };

        liveRecognition.start();
      } catch (err) {
        console.warn("SpeechRec start caught:", err);
      }
    }

    function stopLiveSpeech() {
      if (liveRecognition) {
        try {
          liveRecognition.onend = null;
          liveRecognition.onerror = null;
          liveRecognition.abort(); // Immediately releases hardware capture without delay
        } catch (err) {}
        liveRecognition = null;
      }
      typingCursor.style.display = 'none';
      dockMicDot.classList.remove('active');
      bottomDock.classList.remove('listening');
      updatePitchBadge(0, false);
    }

    // Step 2: Auto-Ask Question using TTS
    function askQuestionAutomatically() {
      triggerNextPulse(false);
      if (autoProceedTimer) {
        clearTimeout(autoProceedTimer);
        autoProceedTimer = null;
      }
      toggleCorrectionMode(false);

      if (!currentField) {
        setAiSpeakingState(true);
        let finalMsg = "Congratulations! All store onboarding details have been completed.";
        if (currentLang === 'hi') finalMsg = "बधाई हो! आपकी दुकान के सभी विवरण सफलतापूर्वक एकत्र कर लिए गए हैं।";
        else if (currentLang === 'te') finalMsg = "అభినందనలు! మీ స్టోర్ వివరాలన్నీ విజయవంతంగా సేకరించబడ్డాయి.";

        aiDialogueText.innerText = finalMsg;
        dockFieldHint.innerText = "Completed (13/13)";
        liveSubtitles.innerText = "All fields collected. Profile JSON is ready!";
        dockSkipBtn.style.display = 'none';

        speakText(finalMsg, () => {
          setAiSpeakingState(false);
          statusText.innerText = "Onboarding completed!";
          statusDot.style.background = "#10b981";
        });
        return;
      }

      let q = currentField.question_en || currentField.question;
      if (currentLang === 'hi') q = currentField.question_hi || q;
      else if (currentLang === 'te') q = currentField.question_te || q;
      else if (currentLang === 'en') q = currentField.question_en || q;
      currentField.question = q;

      setAiSpeakingState(true);
      aiDialogueText.innerText = q;
      dockFieldHint.innerText = `${currentField.label} (${currentField.index}/${currentField.total})`;
      dockSkipBtn.style.display = currentField.required ? 'none' : 'inline-flex';
      liveSubtitles.innerText = `AI asked: "${q}" — waiting for speech...`;
      liveSubtitles.className = 'live-subtitles';
      typingCursor.style.display = 'none';
      correctionInput.placeholder = `Type ${currentField.label} (or correct)...`;
      updateFieldDisplay(currentField);

      statusText.innerText = "AI is asking question...";
      statusDot.style.background = "#f59e0b";

      speakText(q, () => {
        // TTS Finished -> Auto-enable microphone instantly!
        setAiSpeakingState(false);
        autoEnableUserMicrophone();
      });
    }

    let persistentMicStream = null;
    let persistentRecordStream = null;
    let voiceListenStartTime = 0;

    // Step 3: Auto-Enable Microphone with Zero-Latency Stream Reuse & Hardware AEC
    async function autoEnableUserMicrophone() {
      // 1. Prevent mic from opening if AI is actively speaking
      if (isAiSpeaking) return;

      // 2. Stop any lingering previous instances
      stopLiveSpeech();
      if (mediaRecorder && mediaRecorder.state === 'recording') {
        try {
          mediaRecorder.ondataavailable = null;
          mediaRecorder.onstop = null;
          mediaRecorder.stop();
        } catch (e) {}
      }

      // 3. Check if persistent hardware stream is already active
      const isStreamAlive = persistentMicStream &&
        persistentMicStream.active &&
        persistentMicStream.getAudioTracks().some(t => t.readyState === 'live');

      if (!isStreamAlive) {
        try {
          persistentMicStream = await navigator.mediaDevices.getUserMedia({
            audio: {
              channelCount: 1,
              echoCancellation: true,
              noiseSuppression: true,
              autoGainControl: true,
              googEchoCancellation: true,
              googAutoGainControl: true,
              googNoiseSuppression: true,
              googHighpassFilter: true,
            }
          });
        } catch (permErr) {
          console.warn("Microphone permission required for voice bot:", permErr);
          statusText.innerText = "⚠️ Microphone permission required. Click Orb to allow.";
          statusDot.style.background = "#ef4444";
          isSessionRunning = false;
          talkBtn.classList.remove('active');
          return;
        }

        try {
          audioCtx = new (window.AudioContext || window.webkitAudioContext)();
          const source = audioCtx.createMediaStreamSource(persistentMicStream);

          analyser = audioCtx.createAnalyser();
          analyser.fftSize = 512; // 256 fine frequency bins (~93.75 Hz resolution)
          analyser.smoothingTimeConstant = 0.3; // Rapid reaction to voice onsets
          dataArray = new Uint8Array(analyser.frequencyBinCount);

          source.connect(analyser);
          persistentRecordStream = persistentMicStream;
        } catch (audioInitErr) {
          console.warn("AudioContext processing warning:", audioInitErr);
          persistentRecordStream = persistentMicStream;
        }
      }

      if (audioCtx && audioCtx.state === 'suspended') {
        try { await audioCtx.resume(); } catch (e) {}
      }

      try {
        const streamToRecord = persistentRecordStream || persistentMicStream;
        mediaRecorder = new MediaRecorder(streamToRecord);
        audioChunks = [];

        mediaRecorder.ondataavailable = e => { if (e.data.size > 0) audioChunks.push(e.data); };
        mediaRecorder.onstop = uploadUserVoice;
        mediaRecorder.start(250); // Emit audio buffers continuously every 250ms

        setUserListeningState(true);
        hasSpoken = false;
        latestLiveTranscript = '';
        voiceListenStartTime = Date.now();
        noiseCalibrationSamples = 0;
        if (speechBreakTimer) clearTimeout(speechBreakTimer);
        speechBreakTimer = null;
        if (silenceTimer) clearTimeout(silenceTimer);
        silenceTimer = null;

        userDialogueText.innerText = "Listening... Speak your answer now.";
        liveSubtitles.innerText = "Listening to your voice... Speak now.";
        liveSubtitles.className = 'live-subtitles interim';
        typingCursor.style.display = 'inline-block';
        statusText.innerText = "🎙️ Listening... Speak your answer now.";
        statusDot.style.background = "#10b981";
        dockMicDot.classList.add('active');
        bottomDock.classList.add('listening');

        // Start live real-time speech subtitles stream
        startLiveSpeech();

        // VAD loop for auto-silence detect
        monitorVoiceActivity();

      } catch (err) {
        console.error("Recording setup error:", err);
      }
    }

    function updatePitchBadge(pitchHz, isHigh) {
      const badge = document.getElementById('dockPitchBadge');
      if (!badge) return;
      if (pitchHz && pitchHz >= 100) {
        badge.style.display = 'inline-flex';
        if (isHigh) {
          badge.className = 'dock-pitch-badge high';
          badge.innerHTML = `<span>🎯</span> <span>High Pitch: ${pitchHz} Hz</span>`;
        } else {
          badge.className = 'dock-pitch-badge';
          badge.innerHTML = `<span>🎵</span> <span>${pitchHz} Hz</span>`;
        }
      } else {
        badge.style.display = 'none';
      }
    }

    // Step 4: Voice Activity Detection (Detect High Pitch First & Auto Silence Completion)
    function monitorVoiceActivity() {
      if (!isUserListening || !analyser || !(dataArray instanceof Uint8Array)) return;

      try {
        analyser.getByteFrequencyData(dataArray);
      } catch (e) {
        return;
      }

      const sampleRate = audioCtx ? audioCtx.sampleRate : 48000;
      const binHz = sampleRate / analyser.fftSize; // ~93.75 Hz per bin

      // 1. HIGH PITCH VOICE FIRST ANALYSIS:
      // Focus on vocal pitch & higher harmonics (~180 Hz to ~3800 Hz)
      const startHighBin = Math.max(1, Math.round(180 / binHz));
      const endHighBin = Math.min(dataArray.length - 1, Math.round(3800 / binHz));

      let highPitchSum = 0;
      let highPitchCount = 0;
      let peakVal = 0;
      let peakBin = 0;

      for (let i = startHighBin; i <= endHighBin; i++) {
        const val = dataArray[i];
        highPitchSum += val;
        highPitchCount++;
        if (val > peakVal) {
          peakVal = val;
          peakBin = i;
        }
      }

      const highPitchAvg = highPitchCount > 0 ? (highPitchSum / highPitchCount) : 0;
      const detectedPitchHz = Math.round(peakBin * binHz);
      const isHighPitch = (detectedPitchHz >= 180 && detectedPitchHz <= 850 && peakVal >= 15);

      // 2. LOW PITCH / NATURAL VOICE ANALYSIS (~85 Hz to ~180 Hz: deep voice, soft words)
      const startLowBin = Math.max(1, Math.round(85 / binHz));
      const endLowBin = Math.min(dataArray.length - 1, Math.round(180 / binHz));
      let lowPitchSum = 0;
      let lowPitchCount = 0;
      let lowPeakVal = 0;
      for (let i = startLowBin; i <= endLowBin; i++) {
        const val = dataArray[i];
        lowPitchSum += val;
        lowPitchCount++;
        if (val > lowPeakVal) lowPeakVal = val;
      }
      const lowPitchAvg = lowPitchCount > 0 ? (lowPitchSum / lowPitchCount) : 0;

      // 3. Full voice band (85 Hz to 3800 Hz)
      const fullVoiceAvg = (highPitchSum + lowPitchSum) / (highPitchCount + lowPitchCount || 1);

      // Echo immunity grace period: in first 250ms after mic opens, only calibrate noise, do not trigger false voice from speaker echo
      const timeSinceListen = Date.now() - voiceListenStartTime;
      if (timeSinceListen < 250) {
        ambientNoiseFloor = Math.min(ambientNoiseFloor, fullVoiceAvg);
        highPitchNoiseFloor = Math.min(highPitchNoiseFloor, highPitchAvg);
        if (isUserListening) requestAnimationFrame(monitorVoiceActivity);
        return;
      }

      // Dynamic Noise Floor Tracking
      if (noiseCalibrationSamples < 20) {
        ambientNoiseFloor = Math.min(ambientNoiseFloor, fullVoiceAvg);
        highPitchNoiseFloor = Math.min(highPitchNoiseFloor, highPitchAvg);
        noiseCalibrationSamples++;
      } else {
        highPitchNoiseFloor = (highPitchAvg < highPitchNoiseFloor)
          ? (highPitchNoiseFloor * 0.90 + highPitchAvg * 0.10)
          : (highPitchNoiseFloor * 0.995 + highPitchAvg * 0.005);

        ambientNoiseFloor = (fullVoiceAvg < ambientNoiseFloor)
          ? (ambientNoiseFloor * 0.90 + fullVoiceAvg * 0.10)
          : (ambientNoiseFloor * 0.995 + fullVoiceAvg * 0.005);
      }

      const highPitchDelta = highPitchAvg - highPitchNoiseFloor;
      const fullVoiceDelta = fullVoiceAvg - ambientNoiseFloor;

      // Sensitive voice triggers (picks up high pitch instantly, and also soft/low voice!)
      const isHighPitchTrigger = (highPitchDelta >= 2.2 && peakVal >= 14) || (isHighPitch && highPitchDelta >= 1.6);
      const isLowVoiceTrigger = (fullVoiceDelta >= 1.8 && (peakVal >= 12 || lowPeakVal >= 14)) || (lowPitchAvg > ambientNoiseFloor + 2.0);

      if (isHighPitchTrigger) {
        // High pitch voice takes input FIRST!
        hasSpoken = true;
        lastDetectedPitch = detectedPitchHz;
        if (silenceTimer) {
          clearTimeout(silenceTimer);
          silenceTimer = null;
        }
        statusText.innerText = `🎯 High pitch voice detected (${detectedPitchHz} Hz)! Listening...`;
        statusDot.style.background = "#10b981";
        updatePitchBadge(detectedPitchHz, true);
      } else if (isLowVoiceTrigger) {
        // Low pitch / soft voice takes input reliably!
        hasSpoken = true;
        lastDetectedPitch = detectedPitchHz;
        if (silenceTimer) {
          clearTimeout(silenceTimer);
          silenceTimer = null;
        }
        statusText.innerText = `🎙️ Voice detected (${detectedPitchHz || 120} Hz)... Listening.`;
        statusDot.style.background = "#06b6d4";
        updatePitchBadge(detectedPitchHz || 120, false);
      } else if (hasSpoken) {
        // User has spoken: trigger save when a natural pause occurs
        if (highPitchDelta < 1.4 && fullVoiceDelta < 1.5) {
          if (!silenceTimer) {
            // If live recognition has words, save in 650ms. If Whisper audio fallback, wait 1200ms so user isn't cut off!
            const silenceCutoff = latestLiveTranscript ? 650 : 1200;
            statusText.innerText = latestLiveTranscript ? "⚡ Voice heard... fast-saving answer." : "🎙️ Voice heard... finalizing answer.";
            statusDot.style.background = "#f59e0b";
            silenceTimer = setTimeout(() => {
              if (isUserListening) {
                finishUserVoiceTurn();
              }
            }, silenceCutoff);
          }
          updatePitchBadge(0, false);
        } else {
          // Sustained speech keeps recording alive
          if (silenceTimer) {
            clearTimeout(silenceTimer);
            silenceTimer = null;
          }
        }
      }

      // Animate microphone VU meter
      const vuBar = document.getElementById('micVuBar');
      if (vuBar) {
        const effectiveDelta = Math.max(highPitchDelta * 7.0, fullVoiceDelta * 6.0);
        const levelPct = Math.min(100, Math.max(0, effectiveDelta));
        vuBar.style.width = isUserListening ? `${Math.round(levelPct)}%` : '0%';
      }

      if (isUserListening) {
        requestAnimationFrame(monitorVoiceActivity);
      }
    }

    // Step 5: Finish recording & upload (retains persistent mic hardware for instant next turn)
    function playBotSpeech(data, onComplete) {
      if (data && data.tts_audio_base64) {
        stopAllSpeech();
        setAiSpeakingState(true);
        const audio = new Audio("data:audio/mp3;base64," + data.tts_audio_base64);
        audio.playbackRate = currentVoiceSpeed;
        currentTtsAudio = audio;
        let finished = false;
        const done = () => {
          if (finished) return;
          finished = true;
          setAiSpeakingState(false);
          currentTtsAudio = null;
          if (onComplete) {
            // 250ms echo settling delay before auto-enabling microphone
            setTimeout(onComplete, 250);
          }
        };

        const estimatedMs = Math.max(2000, (((data.bot_text || '').length * 85) / currentVoiceSpeed) + 3000);
        const watchdog = setTimeout(() => {
          if (!finished) {
            console.warn("[TTS Watchdog] Proceeding to microphone.");
            done();
          }
        }, estimatedMs);

        audio.onended = () => {
          clearTimeout(watchdog);
          done();
        };
        audio.onerror = () => {
          clearTimeout(watchdog);
          speakText(data.bot_text, onComplete);
        };
        audio.play().catch(() => {
          clearTimeout(watchdog);
          speakText(data.bot_text, onComplete);
        });
      } else if (data && data.bot_text) {
        speakText(data.bot_text, onComplete);
      } else if (onComplete) {
        setTimeout(onComplete, 200);
      }
    }

    function finishUserVoiceTurn() {
      if (silenceTimer) clearTimeout(silenceTimer);
      silenceTimer = null;
      if (speechBreakTimer) clearTimeout(speechBreakTimer);
      speechBreakTimer = null;

      // INSTANT FAST SAVE: If live recognition already has text, save immediately via JSON!
      if (latestLiveTranscript && latestLiveTranscript.trim()) {
        const textToSave = latestLiveTranscript.trim();
        latestLiveTranscript = '';
        saveVoiceAnswerFast(textToSave);
        return;
      }

      stopLiveSpeech();

      if (mediaRecorder && isUserListening) {
        setUserListeningState(false);
        try { mediaRecorder.stop(); } catch (e) {}
        // NOTE: Keep persistentMicStream & audioCtx warm — 0ms hardware delay on next question!

        statusText.innerText = "⚡ Fast-processing answer...";
        statusDot.style.background = "#a78bfa";
        liveSubtitles.innerText = "Fast-processing answer...";
      }
    }

    function renderLiveWords(spoken, isCommitted = false) {
      const container = document.getElementById('liveWordsContainer');
      if (!container) return;
      if (!spoken) {
        container.innerHTML = '';
        return;
      }
      const words = spoken.trim().split(/\s+/).filter(Boolean);
      if (!words.length) {
        container.innerHTML = '';
        return;
      }
      container.innerHTML = words.map((w, idx) => {
        const isLast = idx === words.length - 1;
        const cls = isCommitted ? 'live-word-pill committed' : (isLast ? 'live-word-pill active' : 'live-word-pill');
        const safe = w.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
        return `<span class="${cls}">${safe}</span>`;
      }).join(' ');
    }

    let isSavingAnswer = false;

    async function saveVoiceAnswerFast(spokenText) {
      if (isSavingAnswer || !spokenText) return;
      isSavingAnswer = true;

      const tStart = performance.now();
      if (silenceTimer) clearTimeout(silenceTimer);
      silenceTimer = null;
      if (speechBreakTimer) clearTimeout(speechBreakTimer);
      speechBreakTimer = null;

      stopLiveSpeech();
      setUserListeningState(false);
      try {
        if (mediaRecorder && mediaRecorder.state === 'recording') {
          mediaRecorder.ondataavailable = null;
          mediaRecorder.onstop = null;
          mediaRecorder.stop();
        }
      } catch (e) {}

      renderLiveWords(spokenText, true);
      statusText.innerText = "⚡ Saving in milliseconds...";
      statusDot.style.background = "#f59e0b";

      try {
        const res = await fetch('/api/answer', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            key: currentField ? currentField.key : null,
            answer: spokenText,
            lang: currentLang
          })
        });
        const data = await res.json();
        const elapsed = Math.round(performance.now() - tStart);
        handleServerResponse(data, spokenText, elapsed);
      } catch (err) {
        console.error("Fast voice save error:", err);
        statusText.innerText = "Error saving answer: " + err;
        setTimeout(autoEnableUserMicrophone, 300);
      } finally {
        isSavingAnswer = false;
      }
    }

    function handleServerResponse(data, spokenText, elapsedMs = null) {
      if (data.detected_language) {
        const langNames = { hi: "हिंदी (Hindi)", te: "తెలుగు (Tenglish)", en: "English" };
        const name = langNames[data.detected_language] || data.detected_language.toUpperCase();
        const prob = data.detected_prob ? ` (${Math.round(data.detected_prob * 100)}%)` : '';
        langDetectedBadge.innerText = `🌐 ${name}${prob}`;
        langDetectedBadge.style.display = 'inline-flex';
      }

      if (elapsedMs !== null) {
        const speedBadge = document.getElementById('speedMetricBadge');
        if (speedBadge) {
          speedBadge.innerText = `⚡ ${elapsedMs}ms`;
          speedBadge.style.display = 'inline-flex';
        }
      }

      if (data.language_selected || data.language_switched) {
        const lang = data.language_selected || data.language_switched;
        if (data.language_selected) {
          hasSelectedLanguage = true;
          closeLanguageModal();
        }
        currentLang = lang;
        setLanguage(lang);
        aiDialogueText.innerText = data.bot_text;
        setAiSpeakingState(true);
        playBotSpeech(data, () => {
          setAiSpeakingState(false);
          autoEnableUserMicrophone();
        });
        return;
      }

      if (data.language_chosen === false) {
        aiDialogueText.innerText = data.bot_text;
        setAiSpeakingState(true);
        playBotSpeech(data, () => {
          setAiSpeakingState(false);
          autoEnableUserMicrophone();
        });
        return;
      }

      if (data.user_text || spokenText) {
        const userSpoken = data.user_text || spokenText;
        userDialogueText.innerText = userSpoken;
        const isErr = !!data.error_msg;
        const isSkipped = data.extracted_value === null && !isErr;
        let cleanVal = '';
        if (isSkipped) {
          cleanVal = '';
        } else if (data.extracted_value !== null && data.extracted_value !== undefined) {
          cleanVal = Array.isArray(data.extracted_value) ? data.extracted_value.join(', ') : String(data.extracted_value);
        } else if (!isErr) {
          cleanVal = userSpoken;
        }

        renderLiveWords(userSpoken, true);

        if (isErr) {
          liveSubtitles.className = 'live-subtitles';
          liveSubtitles.innerHTML = `⚠️ <span style="color:#f87171;">${data.error_msg}</span>`;
          statusText.innerText = data.error_msg;
          statusDot.style.background = "#ef4444";
          correctionInput.value = '';
          aiDialogueText.innerText = data.error_msg;
          setAiSpeakingState(true);
          playBotSpeech(data, () => {
            setAiSpeakingState(false);
            autoEnableUserMicrophone();
          });
        } else {
          const msInfo = elapsedMs !== null ? `<span class="speed-metric-badge">⚡ ${elapsedMs}ms</span> ` : '';
          liveSubtitles.className = 'live-subtitles';
          if (isSkipped) {
            liveSubtitles.innerHTML = `${msInfo}🗣️ "<strong>${userSpoken}</strong>" ➔ <span class="highlight-clean">(Skipped / None)</span>`;
            statusText.innerText = elapsedMs !== null ? `✓ Skipped in ${elapsedMs}ms! Asking next question...` : "✓ Optional field skipped.";
          } else {
            liveSubtitles.innerHTML = `${msInfo}🗣️ "<strong>${userSpoken}</strong>" ➔ Value: <span class="highlight-clean">${cleanVal || '(Recorded)'}</span>`;
            toggleCorrectionMode(false);
            statusText.innerText = elapsedMs !== null ? `✓ Saved in ${elapsedMs}ms! Next question...` : "✓ Recorded! Asking next question...";
          }
          correctionInput.value = '';
          statusDot.style.background = "#10b981";

          if (data.json) updateJSONView(data.json);
          currentField = data.current_field;
          updateFieldDisplay(currentField);

          if (data.is_completed) {
            setAiSpeakingState(true);
            aiDialogueText.innerText = data.bot_text;
            statusText.innerText = "Store onboarding completed successfully!";
            statusDot.style.background = "#10b981";
            playBotSpeech(data, () => setAiSpeakingState(false));
          } else if (data.bot_text) {
            aiDialogueText.innerText = data.bot_text;
            dockFieldHint.innerText = currentField ? `${currentField.label} (${currentField.index}/${currentField.total})` : "Done";
            dockSkipBtn.style.display = (currentField && !currentField.required) ? 'inline-flex' : 'none';
            setAiSpeakingState(true);
            playBotSpeech(data, () => {
              setAiSpeakingState(false);
              autoEnableUserMicrophone();
            });
          }
        }
      } else {
        userDialogueText.innerText = "(No clear voice detected. Retrying...)";
        liveSubtitles.innerText = "No voice heard. Please speak clearly.";
        statusText.innerText = "No voice heard. Retrying question...";
        statusDot.style.background = "#ef4444";
        if (data.bot_text) {
          aiDialogueText.innerText = data.bot_text;
          setAiSpeakingState(true);
          playBotSpeech(data, () => {
            setAiSpeakingState(false);
            autoEnableUserMicrophone();
          });
        } else {
          autoEnableUserMicrophone();
        }
      }
    }

    // Step 6: Send audio to Faster-Whisper backend (fallback)
    async function uploadUserVoice() {
      const totalBytes = audioChunks.reduce((acc, c) => acc + (c.size || 0), 0);
      if (totalBytes < 400 && !latestLiveTranscript) {
        userDialogueText.innerText = "(No clear voice detected. Retrying...)";
        liveSubtitles.innerText = "No voice heard. Please speak clearly.";
        statusText.innerText = "No voice heard. Retrying question...";
        statusDot.style.background = "#f59e0b";
        setTimeout(autoEnableUserMicrophone, 300);
        return;
      }
      const blob = new Blob(audioChunks, { type: 'audio/webm' });
      try {
        const res = await fetch('/api/transcribe', {
          method: 'POST',
          headers: {
            'X-Language': currentLang,
            'X-Client-Transcript': encodeURIComponent(latestLiveTranscript || '')
          },
          body: blob
        });
        const data = await res.json();
        handleServerResponse(data, data.user_text || latestLiveTranscript);
      } catch (err) {
        statusText.innerText = "Error: " + err;
      }
    }

    // Bottom Correction & Interactive Helpers
    function toggleCorrectionMode(forceState) {
      if (autoProceedTimer) {
        clearTimeout(autoProceedTimer);
        autoProceedTimer = null;
      }
      if (forceState !== undefined) {
        isCorrectionMode = forceState;
      } else {
        isCorrectionMode = !isCorrectionMode;
      }

      if (isCorrectionMode) {
        refreshCorrectionFieldOptions();
        dockCorrectionBox.style.display = 'flex';
        correctBtnText.innerText = "Close";
        btnToggleCorrection.classList.add('active');
        setTimeout(() => correctionInput.focus(), 50);
      } else {
        dockCorrectionBox.style.display = 'none';
        correctBtnText.innerText = "Correction";
        btnToggleCorrection.classList.remove('active');
      }
    }

    function refreshCorrectionFieldOptions() {
      const sel = document.getElementById('correctionFieldSelect');
      if (!sel) return;
      sel.innerHTML = '';

      if (currentField) {
        const opt = document.createElement('option');
        opt.value = currentField.key;
        opt.innerText = `Current: ${currentField.label}`;
        sel.appendChild(opt);
      }

      const allFields = [
        { key: "companyName", label: "Company / Store Name" },
        { key: "personName", label: "Contact Person Name" },
        { key: "phone", label: "Phone Number" },
        { key: "email", label: "Email Address" },
        { key: "storeCategory", label: "Store Category" },
        { key: "doorNo", label: "Door / Shop Number" },
        { key: "street", label: "Street Name" },
        { key: "area", label: "Area / Locality" },
        { key: "town", label: "Town / Village" },
        { key: "city", label: "City" },
        { key: "state", label: "State" },
        { key: "pincode", label: "Pincode" },
        { key: "usedReferalCode", label: "Referral Code" },
      ];

      for (const f of allFields) {
        if (!currentField || f.key !== currentField.key) {
          const opt = document.createElement('option');
          opt.value = f.key;
          const val = currentStoreJson[f.key];
          const hasVal = (val !== "" && val !== null && val !== undefined && !(Array.isArray(val) && val.length === 0));
          opt.innerText = `${f.label}${hasVal ? ' ✓' : ''}`;
          sel.appendChild(opt);
        }
      }

      // ALWAYS default to currentField.key
      if (currentField) {
        sel.value = currentField.key;
      }
      const btn = document.querySelector('.correction-apply-btn');
      if (btn) btn.innerText = "✓ Save & Next ➔";
    }

    function onCorrectionFieldSelectChange() {
      const sel = document.getElementById('correctionFieldSelect');
      const btn = document.querySelector('.correction-apply-btn');
      if (!sel) return;
      const key = sel.value;
      if (currentField && key === currentField.key) {
        if (btn) btn.innerText = "✓ Save & Next ➔";
        const curVal = currentStoreJson[key];
        correctionInput.value = (curVal !== null && curVal !== undefined) ? (Array.isArray(curVal) ? curVal.join(', ') : String(curVal)) : '';
      } else {
        if (btn) btn.innerText = "✓ Update JSON";
        const val = currentStoreJson[key];
        correctionInput.value = (val !== null && val !== undefined) ? (Array.isArray(val) ? val.join(', ') : String(val)) : '';
      }
    }

    function onDrawerFieldChange() {
      const select = document.getElementById('drawerFieldSelect');
      const input = document.getElementById('drawerFieldInput');
      if (!select || !input) return;
      const key = select.value;
      const currentVal = currentStoreJson[key];
      input.value = (currentVal !== null && currentVal !== undefined) ? (Array.isArray(currentVal) ? currentVal.join(', ') : String(currentVal)) : '';
    }

    async function saveDrawerFieldEdit() {
      const select = document.getElementById('drawerFieldSelect');
      const input = document.getElementById('drawerFieldInput');
      if (!select || !input) return;
      const key = select.value;
      const val = input.value.trim();

      statusText.innerText = `Updating ${key} in JSON...`;
      statusDot.style.background = "#f59e0b";

      try {
        const res = await fetch('/api/correct_field', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ key: key, value: val })
        });
        const data = await res.json();
        if (data.json) updateJSONView(data.json);
        statusText.innerText = `Updated ${key} successfully!`;
        statusDot.style.background = "#10b981";
        liveSubtitles.className = 'live-subtitles';
        liveSubtitles.innerHTML = `✏️ JSON Updated: <strong>${key}</strong> ➔ <span class="highlight-clean">${val || '(Cleared)'}</span>`;
      } catch (err) {
        statusText.innerText = "Error updating JSON: " + err;
      }
    }

    async function submitCorrection() {
      const sel = document.getElementById('correctionFieldSelect');
      const targetKey = (sel && sel.value) ? sel.value : (currentField ? currentField.key : null);
      const isCurrent = currentField && targetKey === currentField.key;

      if (isCurrent || !targetKey) {
        handleNextClick();
        return;
      }

      // User explicitly picked a past field in the dropdown
      const val = correctionInput.value.trim();
      statusText.innerText = `Updating ${targetKey} in JSON...`;
      statusDot.style.background = "#f59e0b";

      try {
        const res = await fetch('/api/correct_field', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ key: targetKey, value: val })
        });
        const data = await res.json();
        if (data.json) updateJSONView(data.json);
        liveSubtitles.className = 'live-subtitles';
        liveSubtitles.innerHTML = `✏️ Updated <strong>${targetKey}</strong> in JSON to: <span class="highlight-clean">${val || '(Cleared)'}</span>`;
        statusText.innerText = `Updated ${targetKey} in JSON!`;
        statusDot.style.background = "#10b981";

        // Reset dropdown back to currentField
        refreshCorrectionFieldOptions();
        if (currentField) {
          const curVal = currentStoreJson[currentField.key];
          correctionInput.value = (curVal !== null && curVal !== undefined) ? (Array.isArray(curVal) ? curVal.join(', ') : String(curVal)) : '';
        }
      } catch (err) {
        statusText.innerText = "Error: " + err;
      }
    }

    async function skipField() {
      if (autoProceedTimer) clearTimeout(autoProceedTimer);
      toggleCorrectionMode(false);
      correctionInput.value = "";
      statusText.innerText = "✓ Skipping optional field...";
      statusDot.style.background = "#10b981";

      try {
        const res = await fetch('/api/answer', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({ key: currentField ? currentField.key : null, answer: "skip", lang: currentLang })
        });
        const data = await res.json();
        handleServerResponse(data, "skip");
      } catch (err) {
        console.warn("skip error:", err);
      }
    }

    function triggerNextPulse(active) {
      const centerNext = document.getElementById('centerNextBtn');
      const dockNext = document.getElementById('dockNextBtn');
      if (active) {
        if (centerNext) centerNext.classList.add('ready-pulse');
        if (dockNext) dockNext.classList.add('pulse');
      } else {
        if (centerNext) centerNext.classList.remove('ready-pulse');
        if (dockNext) dockNext.classList.remove('pulse');
      }
    }

    async function handleNextClick() {
      if (autoProceedTimer) {
        clearTimeout(autoProceedTimer);
        autoProceedTimer = null;
      }

      // Tactile visual feedback animation
      const centerNext = document.getElementById('centerNextBtn');
      const dockNext = document.getElementById('dockNextBtn');
      if (centerNext) {
        centerNext.style.transform = 'scale(0.92)';
        setTimeout(() => centerNext.style.transform = '', 160);
      }
      if (dockNext) {
        dockNext.style.transform = 'scale(0.92)';
        setTimeout(() => dockNext.style.transform = '', 160);
      }

      // 1. If user is currently in active listening/recording turn
      if (isUserListening) {
        if (latestLiveTranscript) {
          userDialogueText.innerText = latestLiveTranscript;
        }
        statusText.innerText = "Next clicked: finalizing speech...";
        statusDot.style.background = "#f59e0b";
        finishUserVoiceTurn();
        return;
      }

      // 2. If AI is speaking the question, stop TTS and immediately listen/advance
      if (isAiSpeaking) {
        stopAllSpeech();
        setAiSpeakingState(false);
        autoEnableUserMicrophone();
        return;
      }

      // 3. If session hasn't started yet, start it
      if (!isSessionRunning) {
        handleOrbClick();
        return;
      }

      if (!currentField) {
        return;
      }

      // 4. Value to save into current field ONLY
      const val = correctionInput.value.trim();

      // If optional and blank, skip it
      if (!val && !currentField.required) {
        skipField();
        return;
      }

      // If required and no input provided yet
      if (!val && (!hasSpoken || !userDialogueText.innerText.trim())) {
        statusText.innerText = `Please answer '${currentField.label}' to proceed.`;
        statusDot.style.background = "#ef4444";
        autoEnableUserMicrophone();
        return;
      }

      statusText.innerText = `Saving ${currentField.label} to JSON...`;
      statusDot.style.background = "#f59e0b";
      triggerNextPulse(false);

      try {
        const res = await fetch('/api/answer', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({ key: currentField.key, answer: val, lang: currentLang })
        });
        const data = await res.json();
        handleServerResponse(data, val);
      } catch (err) {
        statusText.innerText = "Error saving answer: " + err;
        statusDot.style.background = "#ef4444";
      }
    }

    correctionInput.addEventListener('input', (e) => {
      if (autoProceedTimer) {
        clearTimeout(autoProceedTimer);
        autoProceedTimer = null;
      }
      statusText.innerText = "✏️ Editing correction... Click Next or press Enter to save to JSON.";
      statusDot.style.background = "#f59e0b";
      triggerNextPulse(!!e.target.value.trim());
    });

    correctionInput.addEventListener('focus', () => {
      if (autoProceedTimer) {
        clearTimeout(autoProceedTimer);
        autoProceedTimer = null;
        statusText.innerText = "✏️ Editing correction... Click Next or press Enter to save to JSON.";
        statusDot.style.background = "#f59e0b";
      }
    });

    window.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && document.activeElement !== correctionInput) {
        e.preventDefault();
        handleNextClick();
      }
    });

    correctionInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        submitCorrection();
      } else if (e.key === 'Escape') {
        toggleCorrectionMode(false);
      }
    });

    function setAiSpeakingState(speaking) {
      isAiSpeaking = speaking;
      if (speaking) {
        orbStage.className = "orb-stage speaking";
      } else {
        orbStage.className = "orb-stage";
      }
    }

    function setUserListeningState(listening) {
      isUserListening = listening;
      if (listening) {
        orbStage.className = "orb-stage listening";
      } else {
        orbStage.className = "orb-stage";
      }
    }

    function updateFieldDisplay(field) {
      if (!field) {
        fieldBadge.style.display = "none";
        return;
      }
      fieldBadge.style.display = "inline-block";
      if (field.required) {
        fieldBadge.className = "field-tag tag-req";
        fieldBadge.innerText = "REQUIRED";
      } else {
        fieldBadge.className = "field-tag tag-opt";
        fieldBadge.innerText = "OPTIONAL (or say skip)";
      }
    }

    function updateJSONView(jsonObj) {
      currentStoreJson = jsonObj;
      jsonBox.innerText = JSON.stringify(jsonObj, null, 2);
      // Count filled fields
      let count = 0;
      for (const [k, v] of Object.entries(jsonObj)) {
        if (v !== "" && v !== null && !(Array.isArray(v) && v.length === 0)) {
          count++;
        }
      }
      filledCount.innerText = count;
      onDrawerFieldChange();
      refreshCorrectionFieldOptions();
    }

    let currentTtsAudio = null;

    function stopAllSpeech() {
      if (currentTtsAudio) {
        try {
          currentTtsAudio.pause();
          currentTtsAudio.currentTime = 0;
        } catch (e) {}
        currentTtsAudio = null;
      }
      if ('speechSynthesis' in window) {
        try { window.speechSynthesis.cancel(); } catch (e) {}
      }
    }

    function speakText(text, onComplete) {
      if (!text || !text.trim()) {
        if (onComplete) onComplete();
        return;
      }

      stopAllSpeech();
      setAiSpeakingState(true);

      const isTelugu = currentLang === 'te' || /[\u0C00-\u0C7F]/.test(text);
      const isHindi = currentLang === 'hi' || /[\u0900-\u097F]/.test(text);
      const langCode = isTelugu ? 'te' : (isHindi ? 'hi' : 'en');
      const fallbackLangTag = isTelugu ? 'te-IN' : (isHindi ? 'hi-IN' : 'en-IN');

      const ttsUrl = `/api/tts?text=${encodeURIComponent(text)}&lang=${langCode}`;
      const audio = new Audio(ttsUrl);
      audio.playbackRate = currentVoiceSpeed;
      currentTtsAudio = audio;

      let finished = false;
      const done = () => {
        if (finished) return;
        finished = true;
        setAiSpeakingState(false);
        currentTtsAudio = null;
        if (onComplete) {
          // 250ms echo settling delay before opening microphone
          setTimeout(onComplete, 250);
        }
      };

      const estimatedMs = Math.max(2000, ((text.length * 85) / currentVoiceSpeed) + 3000);
      const watchdog = setTimeout(() => {
        if (!finished) {
          console.warn("[TTS Watchdog] Audio timeout, proceeding to mic.");
          done();
        }
      }, estimatedMs);

      audio.onended = () => {
        clearTimeout(watchdog);
        done();
      };
      audio.onerror = (e) => {
        console.warn("TTS audio fetch error, trying Web Speech fallback:", e);
        clearTimeout(watchdog);
        fallbackWebSpeech(text, fallbackLangTag, onComplete);
      };
      audio.play().catch(e => {
        console.warn("TTS audio play blocked, trying Web Speech fallback:", e);
        clearTimeout(watchdog);
        fallbackWebSpeech(text, fallbackLangTag, onComplete);
      });
    }

    function fallbackWebSpeech(text, targetLangCode, onComplete) {
      if ('speechSynthesis' in window) {
        try { window.speechSynthesis.cancel(); } catch (e) {}
        const u = new SpeechSynthesisUtterance(text);
        window.activeUtterance = u; // Prevents Chrome garbage collection dropping onend!
        u.rate = currentVoiceSpeed;
        u.pitch = 1.0;
        u.lang = targetLangCode;

        const voices = window.speechSynthesis.getVoices();
        let targetVoice = null;
        if (targetLangCode.startsWith('te')) {
          targetVoice = voices.find(v => v.lang.startsWith('te') || v.name.toLowerCase().includes('telugu'));
        } else if (targetLangCode.startsWith('hi')) {
          targetVoice = voices.find(v => v.lang.startsWith('hi') || v.name.toLowerCase().includes('hindi'));
        } else {
          targetVoice = voices.find(v => v.lang.startsWith('en-IN') || v.lang.startsWith('en-US') || v.lang.startsWith('en'));
        }
        if (targetVoice) u.voice = targetVoice;

        let ended = false;
        const done = () => {
          if (ended) return;
          ended = true;
          window.activeUtterance = null;
          setAiSpeakingState(false);
          if (onComplete) {
            setTimeout(onComplete, 250);
          }
        };

        u.onend = done;
        u.onerror = done;

        // Chrome speech synthesis watchdog
        const estimatedMs = Math.max(2000, ((text.length * 90) / currentVoiceSpeed) + 2000);
        setTimeout(done, estimatedMs);

        try {
          window.speechSynthesis.speak(u);
        } catch (e) {
          done();
        }
      } else {
        setAiSpeakingState(false);
        if (onComplete) setTimeout(onComplete, 250);
      }
    }

    function toggleDrawer() {
      drawer.classList.toggle('open');
      drawerOverlay.classList.toggle('open');
    }

    function copyJSON() {
      navigator.clipboard.writeText(jsonBox.innerText);
      alert("JSON copied to clipboard!");
    }

    async function resetForm() {
      hasSelectedLanguage = false;
      try { localStorage.removeItem('preferredLang'); } catch (e) {}
      if (persistentMicStream) {
        try { persistentMicStream.getTracks().forEach(t => t.stop()); } catch (e) {}
        persistentMicStream = null;
      }
      if (audioCtx) {
        try { audioCtx.close(); } catch (e) {}
        audioCtx = null;
      }
      await fetch('/api/reset', { method: 'POST' });
      location.reload();
    }
  </script>
</body>
</html>
"""



# ==============================================================================
# OpenAPI 3.0.3 Specification & Swagger UI
# ==============================================================================
OPENAPI_SPEC = {
    "openapi": "3.0.3",
    "info": {
        "title": "Autonomous Multilingual Store Onboarding Voice Bot API",
        "description": "Production Faster-Whisper + Chrome Web Speech API Voice Bot backend for onboarding Indian stores in English, Hindi, and Telugu. Automatically extracts, validates, and transliterates conversational speech into structured store JSON.",
        "version": "1.0.0",
        "contact": {
            "name": "Store Onboarding AI Support"
        }
    },
    "servers": [
        {
            "url": "http://localhost:8000",
            "description": "Local Development Server"
        }
    ],
    "tags": [
        {"name": "Session", "description": "Manage onboarding state, languages, and session reset"},
        {"name": "Voice & Transcription", "description": "Speech-to-Text transcription and Text-to-Speech synthesis"},
        {"name": "Survey & Correction", "description": "Answer questions, validate inputs, and update store JSON fields"}
    ],
    "paths": {
        "/api/get_state": {
            "get": {
                "tags": ["Session"],
                "summary": "Get Current Session State",
                "description": "Returns current onboarding progress, active question, language settings, and live store profile JSON.",
                "parameters": [
                    {
                        "name": "lang",
                        "in": "query",
                        "required": False,
                        "description": "Language code for question text ('en', 'hi', 'te')",
                        "schema": {"type": "string", "enum": ["en", "hi", "te"], "default": "en"}
                    }
                ],
                "responses": {
                    "200": {
                        "description": "Current onboarding state",
                        "content": {
                            "application/json": {
                                "example": {
                                    "json": {
                                        "id": "comp_b4b298d66d",
                                        "companyId": "comp_b4b298d66d",
                                        "companyName": "Royal Clothes",
                                        "personName": "Ramesh Kumar",
                                        "phone": "9876543210",
                                        "email": "ramesh@gmail.com",
                                        "logoUrl": None,
                                        "storeCategory": ["Clothing"],
                                        "doorNo": "402",
                                        "street": "MG Road",
                                        "area": "Gajuwaka",
                                        "town": "",
                                        "city": "Visakhapatnam",
                                        "state": "Andhra Pradesh",
                                        "pincode": "530016",
                                        "usedReferalCode": None,
                                        "createdAt": None,
                                        "productCount": 0,
                                        "isPhoneMandatoryForBrowsing": False,
                                        "isStoreLoginMandatory": False
                                    },
                                    "progress": 38,
                                    "current_field": {
                                        "index": 5,
                                        "total": 13,
                                        "key": "doorNo",
                                        "label": "Door / Shop Number",
                                        "question": "What is your door, shop, or flat number? (Optional: say 'skip' if none)",
                                        "required": False,
                                        "hint": "e.g., 402, Shop 4, Plot 12"
                                    },
                                    "language": "te",
                                    "language_chosen": True
                                }
                            }
                        }
                    }
                }
            }
        },
        "/api/transcribe": {
            "post": {
                "tags": ["Voice & Transcription"],
                "summary": "Transcribe Voice Audio with Dual-Engine Fusion",
                "description": "Uploads recorded user speech (WebM or WAV), runs Faster-Whisper with 5-beam search, fuses with Chrome Web Speech subtitle if available, removes conversational filler/prefixes, and returns clean English extraction.",
                "parameters": [
                    {
                        "name": "X-Language",
                        "in": "header",
                        "required": False,
                        "description": "Selected interview language ('en', 'hi', 'te')",
                        "schema": {"type": "string", "enum": ["en", "hi", "te"], "default": "en"}
                    },
                    {
                        "name": "X-Client-Transcript",
                        "in": "header",
                        "required": False,
                        "description": "URL-encoded real-time transcript from Chrome Web Speech API for dual-engine accuracy cross-verification",
                        "schema": {"type": "string"}
                    }
                ],
                "requestBody": {
                    "description": "Raw binary audio recorded from user microphone (audio/webm or audio/wav)",
                    "required": True,
                    "content": {
                        "audio/webm": {
                            "schema": {"type": "string", "format": "binary"}
                        },
                        "audio/wav": {
                            "schema": {"type": "string", "format": "binary"}
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "Transcribed text and English extracted value",
                        "content": {
                            "application/json": {
                                "examples": {
                                    "teluguName": {
                                        "summary": "Telugu Name Answer",
                                        "value": {
                                            "status": "ready_to_save",
                                            "user_text": "నా పేరు రమేష్ కుమార్",
                                            "extracted_value": "Ramesh Kumar",
                                            "error_msg": None,
                                            "bot_text": None,
                                            "field_key": "personName",
                                            "detected_language": "te",
                                            "detected_prob": 1.0,
                                            "json": {
                                                "companyName": "Royal Clothes",
                                                "personName": "",
                                                "phone": ""
                                            }
                                        }
                                    },
                                    "hindiShop": {
                                        "summary": "Hindi Shop Name Answer",
                                        "value": {
                                            "status": "ready_to_save",
                                            "user_text": "मेरी दुकान रॉयल क्लॉथ्स है",
                                            "extracted_value": "Royal Clothes",
                                            "error_msg": None,
                                            "bot_text": None,
                                            "field_key": "companyName",
                                            "detected_language": "hi",
                                            "detected_prob": 1.0
                                        }
                                    },
                                    "spokenDigits": {
                                        "summary": "Spoken 10-Digit Mobile",
                                        "value": {
                                            "status": "ready_to_save",
                                            "user_text": "తొమ్మిది ఎనిమిది ఏడు ఆరు ఐదు నాలుగు మూడు రెండు ఒకటి సున్నా",
                                            "extracted_value": "9876543210",
                                            "error_msg": None,
                                            "field_key": "phone"
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "/api/answer": {
            "post": {
                "tags": ["Survey & Correction"],
                "summary": "Submit Field Answer & Advance to Next Question",
                "description": "Stores user answer for the current field into live JSON, advances the question index, and returns the next question with TTS audio prompts.",
                "parameters": [
                    {
                        "name": "X-Language",
                        "in": "header",
                        "required": False,
                        "description": "Active language ('en', 'hi', 'te')",
                        "schema": {"type": "string", "enum": ["en", "hi", "te"], "default": "en"}
                    }
                ],
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "key": {"type": "string", "description": "Field key being answered", "example": "personName"},
                                    "answer": {"type": "string", "description": "User spoken or typed answer", "example": "Ramesh Kumar"},
                                    "lang": {"type": "string", "description": "Language code", "example": "en"}
                                },
                                "required": ["answer"]
                            },
                            "examples": {
                                "savePersonName": {
                                    "summary": "Save Person Name",
                                    "value": {
                                        "key": "personName",
                                        "answer": "Ramesh Kumar",
                                        "lang": "en"
                                    }
                                },
                                "skipOptionalField": {
                                    "summary": "Skip Optional Field (e.g. Referral Code)",
                                    "value": {
                                        "key": "usedReferalCode",
                                        "answer": "skip",
                                        "lang": "te"
                                    }
                                }
                            }
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "Field saved and next question returned",
                        "content": {
                            "application/json": {
                                "example": {
                                    "user_text": "Ramesh Kumar",
                                    "bot_text": "Great, saved Contact Person Name. What is your 10-digit mobile number?",
                                    "current_field": {
                                        "index": 3,
                                        "total": 13,
                                        "key": "phone",
                                        "label": "Phone Number",
                                        "question": "What is your 10-digit mobile number?",
                                        "required": True
                                    },
                                    "progress": 23,
                                    "field_key": "personName",
                                    "extracted_value": "Ramesh Kumar",
                                    "is_completed": False,
                                    "json": {
                                        "id": "comp_b4b298d66d",
                                        "companyName": "Royal Clothes",
                                        "personName": "Ramesh Kumar"
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "/api/correct_field": {
            "post": {
                "tags": ["Survey & Correction"],
                "summary": "Directly Correct or Edit Any JSON Field",
                "description": "Edits any past or current field directly in the store profile without disrupting current question order.",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "key": {"type": "string", "description": "Store JSON field key to edit", "example": "phone"},
                                    "value": {"type": "string", "description": "New updated value", "example": "9876543210"}
                                },
                                "required": ["key", "value"]
                            },
                            "examples": {
                                "updatePhone": {
                                    "summary": "Update Phone Number",
                                    "value": {"key": "phone", "value": "9876543210"}
                                },
                                "updateCategory": {
                                    "summary": "Update Store Category",
                                    "value": {"key": "storeCategory", "value": "Clothing, Fashion"}
                                }
                            }
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "Field updated successfully",
                        "content": {
                            "application/json": {
                                "example": {
                                    "status": "updated",
                                    "progress": 30,
                                    "json": {
                                        "phone": "9876543210",
                                        "storeCategory": ["Clothing", "Fashion"]
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "/api/tts": {
            "get": {
                "tags": ["Voice & Transcription"],
                "summary": "Get Speech Audio (Text-to-Speech MP3)",
                "description": "Returns pre-warmed or dynamically synthesized MP3 speech audio in Telugu, Hindi, or English.",
                "parameters": [
                    {
                        "name": "text",
                        "in": "query",
                        "required": True,
                        "description": "Text to synthesize into speech",
                        "schema": {"type": "string"},
                        "example": "మీ దుకాణం లేదా కంపెనీ పేరు ఏమిటి?"
                    },
                    {
                        "name": "lang",
                        "in": "query",
                        "required": True,
                        "description": "Language code ('te' for Telugu, 'hi' for Hindi, 'en' for English)",
                        "schema": {"type": "string", "enum": ["te", "hi", "en"], "default": "te"},
                        "example": "te"
                    }
                ],
                "responses": {
                    "200": {
                        "description": "Synthesized MP3 audio stream",
                        "content": {
                            "audio/mpeg": {
                                "schema": {"type": "string", "format": "binary"}
                            }
                        }
                    },
                    "400": {"description": "Missing required text or lang parameter"}
                }
            }
        },
        "/api/set_language": {
            "post": {
                "tags": ["Session"],
                "summary": "Set or Switch Active Language",
                "description": "Switches the interview language between English, Hindi, and Telugu.",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "lang": {"type": "string", "enum": ["en", "hi", "te"], "example": "te"}
                                },
                                "required": ["lang"]
                            }
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "Language switched successfully",
                        "content": {
                            "application/json": {
                                "example": {
                                    "status": "success",
                                    "language": "te",
                                    "current_field": {
                                        "key": "companyName",
                                        "question": "మీ దుకాణం లేదా కంపెనీ పేరు ఏమిటి?"
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "/api/reset": {
            "post": {
                "tags": ["Session"],
                "summary": "Reset Onboarding Session",
                "description": "Resets all collected data, generates a new store UUID, and restarts onboarding from Question 1.",
                "responses": {
                    "200": {
                        "description": "Session reset",
                        "content": {
                            "application/json": {
                                "example": {
                                    "status": "reset",
                                    "progress": 0,
                                    "json": {
                                        "id": "comp_a1b2c3d4e5",
                                        "companyId": "comp_a1b2c3d4e5",
                                        "companyName": "",
                                        "personName": "",
                                        "phone": ""
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

SWAGGER_UI_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Store Onboarding Voice Bot - Swagger API Docs</title>
  <link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@5.11.0/swagger-ui.css" />
  <link rel="icon" type="image/png" href="https://unpkg.com/swagger-ui-dist@5.11.0/favicon-32x32.png" />
  <style>
    body {
      margin: 0;
      background: #0b0f19;
      color: #e2e8f0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    .top-banner {
      background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #31104b 100%);
      border-bottom: 1px solid rgba(255,255,255,0.12);
      padding: 16px 28px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    }
    .top-banner h1 {
      margin: 0;
      font-size: 20px;
      color: #f8fafc;
      display: flex;
      align-items: center;
      gap: 12px;
      font-weight: 700;
    }
    .top-banner .badge {
      font-size: 11px;
      background: #f59e0b;
      color: #000;
      font-weight: 800;
      padding: 3px 8px;
      border-radius: 6px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }
    .top-links {
      display: flex;
      gap: 12px;
      align-items: center;
    }
    .top-links a {
      color: #f8fafc;
      text-decoration: none;
      font-size: 13px;
      font-weight: 600;
      background: rgba(255, 255, 255, 0.08);
      padding: 8px 16px;
      border-radius: 8px;
      border: 1px solid rgba(255, 255, 255, 0.15);
      transition: all 0.2s;
    }
    .top-links a:hover {
      background: #f59e0b;
      color: #0f172a;
      border-color: #f59e0b;
    }
    .swagger-ui .topbar { display: none; }
    .swagger-ui {
      background: #0b0f19;
      padding: 20px 24px 60px;
      max-width: 1280px;
      margin: 0 auto;
    }
    .swagger-ui .info .title { color: #f8fafc; }
    .swagger-ui .info p, .swagger-ui .info li { color: #cbd5e1; }
    .swagger-ui .scheme-container { background: #111827; box-shadow: none; border-bottom: 1px solid #1f2937; border-radius: 8px; margin-bottom: 20px; }
    .swagger-ui .opblock .opblock-summary-operation-id, .swagger-ui .opblock .opblock-summary-path, .swagger-ui .opblock .opblock-summary-description { color: #f1f5f9; }
    .swagger-ui .opblock-tag { color: #f8fafc; border-bottom: 1px solid #334155; }
    .swagger-ui .opblock-tag small { color: #94a3b8; }
    .swagger-ui .opblock .opblock-section-header { background: #1e293b; color: #f8fafc; }
    .swagger-ui .tab li button.tablinks { color: #cbd5e1; }
    .swagger-ui table thead tr td, .swagger-ui table thead tr th { color: #f1f5f9; border-bottom: 1px solid #334155; }
    .swagger-ui .response-col_status { color: #34d399; }
    .swagger-ui select, .swagger-ui input[type=text] { background: #1e293b; color: #f8fafc; border: 1px solid #475569; }
    .swagger-ui .btn { background: #3b82f6; color: #ffffff; border-radius: 6px; }
    .swagger-ui .btn:hover { background: #2563eb; }
    .swagger-ui .parameters-col_name { color: #f8fafc; }
    .swagger-ui .parameter__name { color: #38bdf8; font-weight: 600; }
    .swagger-ui .parameter__type { color: #94a3b8; }
    .swagger-ui .prop-format { color: #a78bfa; }
  </style>
</head>
<body>
  <div class="top-banner">
    <h1>
      <span>🎙️ Store Onboarding Voice Bot</span>
      <span class="badge">OpenAPI 3.0</span>
    </h1>
    <div class="top-links">
      <a href="/" target="_blank">🌐 Live Voice Bot</a>
      <a href="/openapi.json" target="_blank">📄 /openapi.json</a>
    </div>
  </div>
  <div id="swagger-ui"></div>
  <script src="https://unpkg.com/swagger-ui-dist@5.11.0/swagger-ui-bundle.js"></script>
  <script src="https://unpkg.com/swagger-ui-dist@5.11.0/swagger-ui-standalone-preset.js"></script>
  <script>
    window.onload = () => {
      window.ui = SwaggerUIBundle({
        url: '/openapi.json',
        dom_id: '#swagger-ui',
        presets: [
          SwaggerUIBundle.presets.apis,
          SwaggerUIStandalonePreset
        ],
        layout: "BaseLayout",
        deepLinking: true,
        showExtensions: true,
        showCommonExtensions: true,
        defaultModelsExpandDepth: 2,
        defaultModelExpandDepth: 2,
        displayRequestDuration: true,
        docExpansion: "list"
      });
    };
  </script>
</body>
</html>
"""


class AutonomousStoreBotHandler(http.server.SimpleHTTPRequestHandler):
    def handle(self):
        try:
            super().handle()
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError, OSError):
            pass

    def handle_one_request(self):
        try:
            super().handle_one_request()
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError, OSError):
            pass

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS, PUT, DELETE")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Language, X-Client-Transcript, Authorization, Accept, Origin, User-Agent")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Max-Age", "86400")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def send_html_response(self, html_str, status=200):
        try:
            body = html_str.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError, OSError):
            pass

    def send_json_response(self, data, status=200):
        try:
            body = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError, OSError):
            pass

    def do_GET(self):
        try:
            clean_path = self.path.split("?")[0]
            if clean_path in ("/", "/index.html"):
                self.send_html_response(HTML_PAGE)
            elif clean_path in ("/docs", "/swagger", "/docs/", "/swagger/"):
                self.send_html_response(SWAGGER_UI_HTML)
            elif clean_path in ("/openapi.json", "/swagger.json", "/api/openapi.json"):
                self.send_json_response(OPENAPI_SPEC)
            elif clean_path == "/api/get_state":
                global SESSION
                lang = "auto"
                if "?" in self.path:
                    params = urllib.parse.parse_qs(self.path.split("?", 1)[1])
                    lang = params.get("lang", ["auto"])[0]
                curr_json = build_store_json(SESSION["data"], SESSION["backend_id"])
                progress = int((SESSION["current_index"] / len(STORE_FIELDS)) * 100)
                res = {
                    "json": curr_json,
                    "progress": progress,
                    "current_field": get_current_field_info(lang),
                    "language": SESSION.get("language", "auto"),
                    "language_chosen": SESSION.get("language_chosen", False),
                }
                self.send_json_response(res)
            elif clean_path == "/api/tts":
                parsed = urllib.parse.urlparse(self.path)
                params = urllib.parse.parse_qs(parsed.query)
                text = params.get("text", [""])[0].strip()
                lang = params.get("lang", ["te"])[0].strip().lower()
                if not text:
                    self.send_response(400)
                    self.end_headers()
                    return

                audio_data = get_tts_audio_bytes(text, lang)
                if audio_data:
                    self.send_response(200)
                    self.send_header("Content-Type", "audio/mpeg")
                    self.send_header("Content-Length", str(len(audio_data)))
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.send_header("Cache-Control", "public, max-age=86400")
                    self.end_headers()
                    try:
                        self.wfile.write(audio_data)
                    except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError, OSError):
                        pass
                else:
                    self.send_response(500)
                    self.end_headers()
            elif clean_path == "/favicon.ico":
                self.send_response(204)
                self.end_headers()
            else:
                self.send_response(404)
                self.end_headers()
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError, OSError):
            pass

    def do_POST(self):
        global SESSION
        clean_path = self.path.split("?")[0]
        length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(length) if length > 0 else b""
        client_lang = self.headers.get("X-Language", "auto").strip().lower()
        whisper_lang = None if client_lang in ("auto", "", "none") else client_lang
        client_transcript = urllib.parse.unquote(self.headers.get("X-Client-Transcript", "")).strip()

        if clean_path == "/api/transcribe":
            try:
                curr_idx = SESSION["current_index"]
                curr_field_dict = STORE_FIELDS[curr_idx] if curr_idx < len(STORE_FIELDS) else None
                curr_field = curr_field_dict["key"] if curr_field_dict else ""
                target_lang = whisper_lang if whisper_lang in ("hi", "te", "en") else (SESSION.get("language") if SESSION.get("language") in ("hi", "te", "en") else None)
                effective_lang = target_lang or (client_lang if client_lang in ("hi", "te", "en") else "en")

                c_clean = "" if is_whisper_hallucination(client_transcript) else client_transcript.strip()
                user_text = ""
                detected_lang = effective_lang
                lang_prob = 1.0

                # FAST PATH (<5ms): If browser Web Speech transcript is already valid, bypass heavy Whisper!
                if c_clean:
                    if not SESSION.get("language_chosen"):
                        ch = detect_language_choice(c_clean)
                        if ch:
                            user_text = c_clean
                            detected_lang = ch
                    elif curr_field_dict:
                        if is_skip(c_clean) and not curr_field_dict.get("required"):
                            user_text = c_clean
                        else:
                            c_val, c_err, _ = parse_and_validate(curr_field_dict, c_clean, SESSION["data"], lang=effective_lang)
                            if (c_val is not None or is_skip(c_clean)) and not c_err:
                                user_text = c_clean

                # FALLBACK PATH (Whisper greedy beam_size=1): Only when Web Speech is absent or invalid
                if not user_text and post_data and len(post_data) > 1000:
                    audio_arr = safe_decode_audio(post_data)
                    if len(audio_arr) >= 1600 and np.max(np.abs(audio_arr)) >= 0.01:
                        audio_arr = denoise_audio(audio_arr, sample_rate=16000)
                        field_prompt = get_field_prompt(curr_field, target_lang or "en")
                        try:
                            segments, info = model.transcribe(
                                audio_arr,
                                beam_size=1,
                                best_of=1,
                                task="transcribe",
                                language=target_lang,
                                vad_filter=True,
                                vad_parameters=dict(
                                    threshold=0.45,
                                    min_speech_duration_ms=150,
                                    min_silence_duration_ms=300,
                                    speech_pad_ms=200,
                                ),
                                initial_prompt=field_prompt,
                                condition_on_previous_text=False,
                                temperature=0.0,
                            )
                            whisper_text = " ".join(s.text.strip() for s in segments).strip()
                            w_clean = "" if is_whisper_hallucination(whisper_text) else whisper_text.strip()
                            if w_clean:
                                user_text = w_clean
                                if hasattr(info, "language"):
                                    detected_lang = info.language
                                    lang_prob = float(info.language_probability) if hasattr(info, "language_probability") else 1.0
                        except Exception as w_err:
                            print(f"[Whisper Transcribe Error] {w_err}")
                    elif c_clean:
                        user_text = c_clean

                if not user_text:
                    no_voice_msg = {
                        "hi": "मुझे कोई आवाज़ सुनाई नहीं दी। आइए फिर से कोशिश करें।",
                        "te": "నాకు voice వినిపించలేదు. Please మళ్ళీ ఒకసారి చెప్పండి.",
                    }.get(effective_lang, "I could not hear any speech. Let's try that question again.")
                    response = {
                        "user_text": "",
                        "bot_text": no_voice_msg,
                        "json": build_store_json(SESSION["data"], SESSION["backend_id"]),
                        "progress": int((SESSION["current_index"] / len(STORE_FIELDS)) * 100),
                        "current_field": get_current_field_info(effective_lang),
                        "detected_language": detected_lang,
                        "detected_prob": lang_prob,
                    }
                else:
                    detected_choice = detect_language_choice(user_text)
                    if not SESSION.get("language_chosen"):
                        if detected_choice in ("en", "hi", "te"):
                            effective_lang = detected_choice
                            response = build_language_selection_response(detected_choice, user_text)
                        else:
                            rep = {
                                "hi": "कृपया अपनी भाषा चुनें: हिंदी, तेलुगु, या अंग्रेज़ी।",
                                "te": "దయచేసి మీ language ని select చేసుకోండి: English, Hindi, లేదా Telugu.",
                            }.get(effective_lang, "Please choose your language: English, Hindi, or Telugu.")
                            response = {
                                "user_text": user_text,
                                "bot_text": rep,
                                "json": build_store_json(SESSION["data"], SESSION["backend_id"]),
                                "progress": 0,
                                "current_field": get_current_field_info(effective_lang),
                                "detected_language": detected_lang,
                                "detected_prob": lang_prob,
                                "language_chosen": False,
                            }
                    else:
                        is_switch_req = detected_choice and any(
                            kw in user_text.lower() for kw in ("language", "switch", "change", "speak", "talk", "in", "bhasha", "bhasa", "भाषा", "భాష")
                        )
                        if is_switch_req:
                            SESSION["language"] = detected_choice
                            effective_lang = detected_choice
                            curr_q = STORE_FIELDS[SESSION["current_index"]].get(f"question_{detected_choice}") or STORE_FIELDS[SESSION["current_index"]]["question"]
                            if detected_choice == "hi":
                                ack = f"भाषा बदलकर हिंदी कर दी गई है। {curr_q}"
                            elif detected_choice == "te":
                                ack = f"Language ని Telugu కి switch చేశాను. {curr_q}"
                            else:
                                ack = f"Switched language to English. {curr_q}"
                            response = {
                                "user_text": user_text,
                                "bot_text": ack,
                                "json": build_store_json(SESSION["data"], SESSION["backend_id"]),
                                "progress": int((SESSION["current_index"] / len(STORE_FIELDS)) * 100),
                                "current_field": get_current_field_info(detected_choice),
                                "detected_language": detected_choice,
                                "detected_prob": 1.0,
                                "language_switched": detected_choice,
                            }
                        else:
                            response = process_user_turn(user_text, lang=effective_lang)
                            response["detected_language"] = detected_lang
                            response["detected_prob"] = lang_prob

                # Embed pre-warmed audio directly for zero-latency instant playback
                bot_txt = response.get("bot_text")
                if bot_txt and not response.get("tts_audio_base64"):
                    audio_b = get_tts_audio_bytes(bot_txt, lang=effective_lang)
                    if audio_b:
                        response["tts_audio_base64"] = base64.b64encode(audio_b).decode("ascii")

            except Exception as e:
                print(f"[Transcribe Warning Handled] {e}")
                friendly_msg = {
                    "te": "నాకు voice వినిపించలేదు. Please మళ్ళీ ఒకసారి చెప్పండి.",
                    "hi": "मुझे कोई आवाज़ सुनाई नहीं दी। आइए फिर से कोशिश करें।",
                }.get(effective_lang, "I could not hear any speech. Let's try that question again.")
                response = {
                    "user_text": "",
                    "bot_text": friendly_msg,
                    "json": build_store_json(SESSION["data"], SESSION["backend_id"]),
                    "progress": int((SESSION["current_index"] / len(STORE_FIELDS)) * 100),
                    "current_field": get_current_field_info(effective_lang),
                    "detected_language": detected_lang,
                    "detected_prob": lang_prob,
                }
                audio_b = get_tts_audio_bytes(friendly_msg, lang=effective_lang)
                if audio_b:
                    response["tts_audio_base64"] = base64.b64encode(audio_b).decode("ascii")

            self.send_json_response(response)

        elif clean_path == "/api/set_language":
            try:
                payload = json.loads(post_data.decode("utf-8"))
                ch_lang = payload.get("lang", "en").lower()
                if ch_lang not in ("en", "hi", "te"):
                    ch_lang = "en"
                response = build_language_selection_response(ch_lang)
            except Exception as e:
                response = {"error": str(e)}
            self.send_json_response(response)

        elif clean_path in ("/api/answer", "/api/save_field"):
            try:
                payload = json.loads(post_data.decode("utf-8"))
                user_text = payload.get("answer", "")
                ans_lang = payload.get("lang") or client_lang
                target_key = payload.get("key")

                if not SESSION.get("language_chosen"):
                    detected_choice = detect_language_choice(user_text)
                    if detected_choice in ("en", "hi", "te"):
                        response = build_language_selection_response(detected_choice, user_text)
                    else:
                        rep = {
                            "hi": "कृपया अपनी भाषा चुनें: हिंदी, तेलुगु, या अंग्रेज़ी।",
                            "te": "దయచేసి మీ language ని select చేసుకోండి: English, Hindi, లేదా Telugu.",
                        }.get(ans_lang, "Please choose your language: English, Hindi, or Telugu.")
                        response = {
                            "user_text": user_text,
                            "bot_text": rep,
                            "json": build_store_json(SESSION["data"], SESSION["backend_id"]),
                            "progress": 0,
                            "current_field": get_current_field_info(ans_lang),
                            "detected_language": ans_lang,
                            "detected_prob": 1.0,
                            "language_chosen": False,
                        }
                else:
                    response = process_user_turn(user_text, lang=ans_lang, target_key=target_key)

                bot_txt = response.get("bot_text")
                if bot_txt and not response.get("tts_audio_base64"):
                    audio_b = get_tts_audio_bytes(bot_txt, lang=ans_lang)
                    if audio_b:
                        response["tts_audio_base64"] = base64.b64encode(audio_b).decode("ascii")
            except Exception as e:
                response = {"error": str(e)}
            self.send_json_response(response)

        elif clean_path == "/api/correct_field":
            try:
                payload = json.loads(post_data.decode("utf-8"))
                field_key = payload.get("key")
                raw_value = payload.get("value", "")
                target_field = next((f for f in STORE_FIELDS if f["key"] == field_key), None)
                if target_field:
                    raw_str = str(raw_value).strip()
                    if target_field.get("required") and (is_skip(raw_str) or not raw_str or raw_str.lower() in ("skip", "none", "null")):
                        # Cannot save skip or blank to a required field
                        pass
                    else:
                        parsed_val, err, auto_state = parse_and_validate(target_field, raw_str, SESSION["data"], lang=client_lang)
                        if parsed_val is not None:
                            SESSION["data"][field_key] = parsed_val
                        elif is_skip(raw_str):
                            SESSION["data"][field_key] = None if field_key in ("doorNo", "street", "area", "usedReferalCode") else ""
                        else:
                            eng = translate_indic_to_english(raw_str).strip()
                            if not (eng.lower() in ("skip", "none", "null") and target_field.get("required")):
                                SESSION["data"][field_key] = eng
                        if auto_state and not SESSION["data"].get("state"):
                            SESSION["data"]["state"] = auto_state

                response = {
                    "status": "updated",
                    "json": build_store_json(SESSION["data"], SESSION["backend_id"]),
                    "current_field": get_current_field_info(client_lang),
                    "progress": int((SESSION["current_index"] / len(STORE_FIELDS)) * 100),
                }
            except Exception as e:
                response = {"error": str(e)}
            self.send_json_response(response)

        elif clean_path == "/api/reset":
            SESSION = {
                "current_index": 0,
                "backend_id": "comp_" + uuid.uuid4().hex[:10],
                "data": {},
                "is_completed": False,
                "language": None,
                "language_chosen": False,
            }
            self.send_json_response({"status": "reset"})
        else:
            self.send_response(404)
            self.end_headers()


class ThreadingTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    daemon_threads = True
    allow_reuse_address = True

    def handle_error(self, request, client_address):
        exc_type, exc_val, _ = sys.exc_info()
        if exc_type in (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
            return
        if isinstance(exc_val, OSError) and getattr(exc_val, "winerror", None) in (10053, 10054, 10038):
            return
        super().handle_error(request, client_address)


def run_server():
    with ThreadingTCPServer(("", PORT), AutonomousStoreBotHandler) as httpd:
        print(f"\nAutonomous Voice AI Server running at: http://localhost:{PORT}")
        print("Open http://localhost:8000 in your browser to experience the future of store onboarding!\n")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server.")


if __name__ == "__main__":
    run_server()
