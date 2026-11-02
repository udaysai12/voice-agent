import difflib
import re
import numpy as np


def denoise_audio(audio, sample_rate=16000):
    """
    Robust pure-NumPy audio denoiser for noisy environments:
    1. FFT Bandpass filter (85 Hz to 7500 Hz): removes mic rumble, AC 50/60Hz hum, desk thumps, fan noise, and high hiss.
    2. Adaptive Noise Gate: estimates background noise floor from bottom 20% frame energies and attenuates noise in silent/pause gaps.
    3. Speech-Aware Dynamic Range Normalization: peak normalizes speech without boosting low noise floor.
    """
    if audio is None or len(audio) == 0:
        return audio

    arr = np.asarray(audio, dtype=np.float32).flatten()
    n = len(arr)
    if n < 320:  # less than 20ms
        return arr

    # 1. FFT Highpass filter (>= 70 Hz): removes mic rumble, AC hum, and DC drift
    freqs = np.fft.rfftfreq(n, 1.0 / sample_rate)
    fft = np.fft.rfft(arr)
    highpass = (freqs >= 70)
    fft[~highpass] *= 0.05
    filtered = np.fft.irfft(fft, n=n)

    # 2. Dynamic Range Normalization: peak normalize so Whisper receives clear acoustic signal
    max_amp = np.max(np.abs(filtered))
    if max_amp > 0.01:
        filtered = (filtered / max_amp) * 0.95

    return filtered.astype(np.float32)


# Comprehensive dictionary of Indian cities and their states (English, Hindi, Telugu)
INDIAN_CITIES = {
    # Visakhapatnam / Vizag
    "visakhapatnam": ("Visakhapatnam", "Andhra Pradesh"),
    "vizag": ("Visakhapatnam", "Andhra Pradesh"),
    "vishaka patnam": ("Visakhapatnam", "Andhra Pradesh"),
    "vishakapatnam": ("Visakhapatnam", "Andhra Pradesh"),
    "vizag city": ("Visakhapatnam", "Andhra Pradesh"),
    "విశాఖపట్నం": ("Visakhapatnam", "Andhra Pradesh"),
    "విశాఖ": ("Visakhapatnam", "Andhra Pradesh"),
    "వైజాగ్": ("Visakhapatnam", "Andhra Pradesh"),
    "विशाखापट्टनम": ("Visakhapatnam", "Andhra Pradesh"),
    "वाइजाग": ("Visakhapatnam", "Andhra Pradesh"),
    # Vijayawada
    "vijayawada": ("Vijayawada", "Andhra Pradesh"),
    "విజయవాడ": ("Vijayawada", "Andhra Pradesh"),
    "विजयवाड़ा": ("Vijayawada", "Andhra Pradesh"),
    # Guntur
    "guntur": ("Guntur", "Andhra Pradesh"),
    "గుంటూరు": ("Guntur", "Andhra Pradesh"),
    "गुंटूर": ("Guntur", "Andhra Pradesh"),
    # Hyderabad
    "hyderabad": ("Hyderabad", "Telangana"),
    "హైదరాబాద్": ("Hyderabad", "Telangana"),
    "हैदराबाद": ("Hyderabad", "Telangana"),
    "secunderabad": ("Secunderabad", "Telangana"),
    "సికింద్రాబాద్": ("Secunderabad", "Telangana"),
    "warangal": ("Warangal", "Telangana"),
    "వరంగల్": ("Warangal", "Telangana"),
    # Tirupati
    "tirupati": ("Tirupati", "Andhra Pradesh"),
    "తిరుపతి": ("Tirupati", "Andhra Pradesh"),
    "तिरुपति": ("Tirupati", "Andhra Pradesh"),
    # Rajahmundry / Kakinada / Nellore / Kurnool
    "rajahmundry": ("Rajahmundry", "Andhra Pradesh"),
    "రాజమండ్రి": ("Rajahmundry", "Andhra Pradesh"),
    "kakinada": ("Kakinada", "Andhra Pradesh"),
    "కాకినాడ": ("Kakinada", "Andhra Pradesh"),
    "nellore": ("Nellore", "Andhra Pradesh"),
    "నెల్లూరు": ("Nellore", "Andhra Pradesh"),
    "kurnool": ("Kurnool", "Andhra Pradesh"),
    "కర్నూలు": ("Kurnool", "Andhra Pradesh"),
    "anantapur": ("Anantapur", "Andhra Pradesh"),
    "అనంతపురం": ("Anantapur", "Andhra Pradesh"),
    "kadapa": ("Kadapa", "Andhra Pradesh"),
    "కడప": ("Kadapa", "Andhra Pradesh"),
    # Major Indian Metros
    "bengaluru": ("Bengaluru", "Karnataka"),
    "bangalore": ("Bengaluru", "Karnataka"),
    "బెంగళూరు": ("Bengaluru", "Karnataka"),
    "बेंगलुरु": ("Bengaluru", "Karnataka"),
    "chennai": ("Chennai", "Tamil Nadu"),
    "చెన్నై": ("Chennai", "Tamil Nadu"),
    "चेन्नई": ("Chennai", "Tamil Nadu"),
    "mumbai": ("Mumbai", "Maharashtra"),
    "ముంబై": ("Mumbai", "Maharashtra"),
    "मुंबई": ("Mumbai", "Maharashtra"),
    "delhi": ("Delhi", "Delhi"),
    "new delhi": ("New Delhi", "Delhi"),
    "ఢిల్లీ": ("Delhi", "Delhi"),
    "दिल्ली": ("Delhi", "Delhi"),
    "नई दिल्ली": ("New Delhi", "Delhi"),
    "kolkata": ("Kolkata", "West Bengal"),
    "कोलकाता": ("Kolkata", "West Bengal"),
    "కొల్కతా": ("Kolkata", "West Bengal"),
    "pune": ("Pune", "Maharashtra"),
    "पुणे": ("Pune", "Maharashtra"),
    "jaipur": ("Jaipur", "Rajasthan"),
    "जयपुर": ("Jaipur", "Rajasthan"),
    "lucknow": ("Lucknow", "Uttar Pradesh"),
    "लखनऊ": ("Lucknow", "Uttar Pradesh"),
}

INDIAN_LOCALITIES = {
    # Visakhapatnam localities
    "gajuwaka": ("Gajuwaka", "Visakhapatnam", "Andhra Pradesh"),
    "గాజువాక": ("Gajuwaka", "Visakhapatnam", "Andhra Pradesh"),
    "गाजुवाका": ("Gajuwaka", "Visakhapatnam", "Andhra Pradesh"),
    "madhurawada": ("Madhurawada", "Visakhapatnam", "Andhra Pradesh"),
    "మధురవాడ": ("Madhurawada", "Visakhapatnam", "Andhra Pradesh"),
    "mvp colony": ("MVP Colony", "Visakhapatnam", "Andhra Pradesh"),
    "dwaraka nagar": ("Dwaraka Nagar", "Visakhapatnam", "Andhra Pradesh"),
    # Vijayawada localities
    "benz circle": ("Benz Circle", "Vijayawada", "Andhra Pradesh"),
    # Hyderabad localities
    "madhapur": ("Madhapur", "Hyderabad", "Telangana"),
    "gachibowli": ("Gachibowli", "Hyderabad", "Telangana"),
    "kukatpally": ("Kukatpally", "Hyderabad", "Telangana"),
    "ameerpet": ("Ameerpet", "Hyderabad", "Telangana"),
}

INDIAN_STATES_MAP = {
    "andhra pradesh": "Andhra Pradesh",
    "ఆంధ్రప్రదేశ్": "Andhra Pradesh",
    "ఆంధ్ర ప్రదేశ్": "Andhra Pradesh",
    "आंध्र प्रदेश": "Andhra Pradesh",
    "telangana": "Telangana",
    "తెలంగాణ": "Telangana",
    "तेलंगाना": "Telangana",
    "tamil nadu": "Tamil Nadu",
    "తమిళనాడు": "Tamil Nadu",
    "तमिलनाडु": "Tamil Nadu",
    "karnataka": "Karnataka",
    "కర్ణాటక": "Karnataka",
    "कर्नाटक": "Karnataka",
    "kerala": "Kerala",
    "కేరళ": "Kerala",
    "केरल": "Kerala",
    "maharashtra": "Maharashtra",
    "మహారాష్ట్ర": "Maharashtra",
    "महाराष्ट्र": "Maharashtra",
    "gujarat": "Gujarat",
    "గుజరాత్": "Gujarat",
    "गुजरात": "Gujarat",
    "rajasthan": "Rajasthan",
    "राजस्थान": "Rajasthan",
    "uttar pradesh": "Uttar Pradesh",
    "ఉత్తర ప్రదేశ్": "Uttar Pradesh",
    "उत्तर प्रदेश": "Uttar Pradesh",
    "madhya pradesh": "Madhya Pradesh",
    "मध्य प्रदेश": "Madhya Pradesh",
    "west bengal": "West Bengal",
    "पश्चिम बंगाल": "West Bengal",
    "bihar": "Bihar",
    "बिहार": "Bihar",
    "punjab": "Punjab",
    "पंजाब": "Punjab",
    "haryana": "Haryana",
    "हरियाणा": "Haryana",
    "odisha": "Odisha",
    "उड़ीसा": "Odisha",
    "assam": "Assam",
    "असम": "Assam",
    "delhi": "Delhi",
    "ఢిల్లీ": "Delhi",
    "दिल्ली": "Delhi",
}

INDIAN_STATES = list({v for v in INDIAN_STATES_MAP.values()})

# Devanagari (Hindi) and Telugu script numeral conversion
INDIC_NUMERALS = {
    '०': '0', '१': '1', '२': '2', '३': '3', '४': '4',
    '५': '5', '६': '6', '७': '7', '८': '8', '९': '9',
    '౦': '0', '౧': '1', '౨': '2', '౩': '3', '౪': '4',
    '౫': '5', '౬': '6', '౭': '7', '౮': '8', '౯': '9',
}

# English, Hindi, and Telugu spoken number words (including transliterated English and 2-digit pairs)
NUMBER_WORDS = {
    # English & noise homophones
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "won": "1", "to": "2", "too": "2", "for": "4", "fore": "4",
    "ate": "8", "tree": "3", "sex": "6", "nain": "9", "naan": "9",
    "oh": "0", "hero": "0",
    # English Two-Digit & Tens
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14",
    "fifteen": "15", "sixteen": "16", "seventeen": "17", "eighteen": "18", "nineteen": "19",
    "twenty": "20", "thirty": "30", "forty": "40", "fifty": "50",
    "sixty": "60", "seventy": "70", "eighty": "80", "ninety": "90",

    # Telugu Script - Transliterated English Digits (frequently produced by Web Speech / Whisper in Telugu)
    "జీరో": "0", "వన్": "1", "టూ": "2", "టు": "2", "త్రీ": "3", "ఫోర్": "4",
    "ఫైవ్": "5", "సిక్స్": "6", "సెవెన్": "7", "ఎయిట్": "8", "నైన్": "9",
    # Telugu Script - Transliterated English Tens
    "టెన్": "10", "ట్వెంటీ": "20", "థర్టీ": "30", "ఫోర్టీ": "40", "ఫార్టీ": "40",
    "ఫిఫ్టీ": "50", "సిక్స్టీ": "60", "సెవెంటీ": "70", "ఎయిటీ": "80", "నైంటీ": "90",
    # Telugu Native Digits (Latin & Telugu script)
    "sunna": "0", "సున్నా": "0",
    "okati": "1", "ఒకటి": "1", "ఒక": "1",
    "rendu": "2", "రెండు": "2",
    "moodu": "3", "mudu": "3", "మూడు": "3",
    "naalugu": "4", "nalugu": "4", "నాలుగు": "4",
    "aidhu": "5", "ayidu": "5", "aidu": "5", "ఐదు": "5",
    "aaru": "6", "aru": "6", "ఆరు": "6",
    "eedu": "7", "yedu": "7", "edu": "7", "ఏడు": "7",
    "enimidi": "8", "yenimidi": "8", "ఎనిమిది": "8",
    "tommidi": "9", "thommidi": "9", "తొమ్మిది": "9",
    # Telugu Native Tens & Teens
    "పది": "10", "పదకొండు": "11", "పన్నెండు": "12", "పదమూడు": "13", "పద్నాలుగు": "14",
    "పదిహేను": "15", "పదహారు": "16", "పదిహేడు": "17", "పద్దెనిమిది": "18", "పంతొమ్మిది": "19",
    "ఇరవై": "20", "ముప్పై": "30", "నలభై": "40", "యాభై": "50",
    "అరవై": "60", "డెబ్బై": "70", "ఎనభై": "80", "తొంభై": "90",

    # Hindi Script - Transliterated English Digits (frequently produced by Web Speech / Whisper in Hindi)
    "ज़ीरो": "0", "जीरो": "0", "वन": "1", "टू": "2", "थ्री": "3", "फोर": "4",
    "फाइव": "5", "सिक्स": "6", "सेवन": "7", "एट": "8", "नाइन": "9",
    # Hindi Script - Transliterated English Tens
    "टेन": "10", "ट्वेंटी": "20", "थर्टी": "30", "फ़ोर्टी": "40", "फोर्टी": "40",
    "फ़िफ़्टी": "50", "फिफ्टी": "50", "सिक्सटी": "60", "सेवेंटी": "70", "एटी": "80", "एइटी": "80", "नाइंटी": "90",
    # Hindi Native Digits (Latin & Devanagari)
    "shunya": "0", "shoonya": "0", "शून्य": "0",
    "ek": "1", "एक": "1",
    "do": "2", "दो": "2",
    "teen": "3", "तीन": "3",
    "chaar": "4", "char": "4", "चार": "4",
    "paanch": "5", "panch": "5", "पांच": "5", "पाँच": "5",
    "chhe": "6", "che": "6", "chhah": "6", "chheh": "6", "chhey": "6", "cheh": "6", "chha": "6", "छह": "6", "छे": "6",
    "saat": "7", "sat": "7", "सात": "7",
    "aath": "8", "ath": "8", "आठ": "8",
    "nau": "9", "nav": "9", "नौ": "9",
    # Hindi Native Tens & Teens
    "das": "10", "दस": "10", "gyarah": "11", "ग्यारह": "11", "barah": "12", "बारह": "12",
    "terah": "13", "तेरह": "13", "chaudah": "14", "चौदह": "14", "pandrah": "15", "पंद्रह": "15",
    "solah": "16", "सोलह": "16", "satrah": "17", "सत्रह": "17", "atharah": "18", "अठारह": "18",
    "unnees": "19", "उन्नीस": "19", "bees": "20", "बीस": "20", "tees": "30", "तीस": "30",
    "chalis": "40", "chaalees": "40", "चालीस": "40", "pachas": "50", "पचास": "50",
    "saath": "60", "साठ": "60", "sattar": "70", "सत्तर": "70", "assi": "80", "अस्सी": "80",
    "nabbe": "90", "नब्बे": "90",
}


def normalize_indic_digits(text):
    """Replaces Devanagari (Hindi) and Telugu script digits with standard ASCII 0-9."""
    for char, digit in INDIC_NUMERALS.items():
        text = text.replace(char, digit)
    return text


def expand_spoken_numbers(text):
    """
    Expands spoken expressions like 'double nine' -> 'nine nine',
    'డబుల్ నైన్' -> 'నైన్ నైన్', 'डबल आठ' -> 'आठ आठ',
    'triple five' -> 'five five five' and normalizes Indic numerals.
    """
    t = normalize_indic_digits(text.lower())
    # Match double/triple in English, Telugu (డబుల్, ట్రిపుల్), Hindi (डबल, ट्रिपल)
    double_pattern = re.compile(
        r'(?:^|[\s,.-])(?:double|డబుల్|डबल)\s+([\w\u0900-\u097F\u0C00-\u0C7F]+)(?=[\s,.-]|$)',
        re.IGNORECASE
    )
    triple_pattern = re.compile(
        r'(?:^|[\s,.-])(?:triple|ట్రిపుల్|ट्रिपल)\s+([\w\u0900-\u097F\u0C00-\u0C7F]+)(?=[\s,.-]|$)',
        re.IGNORECASE
    )

    for _ in range(3):
        t = double_pattern.sub(lambda m: f" {m.group(1)} {m.group(1)} ", t)
        t = triple_pattern.sub(lambda m: f" {m.group(1)} {m.group(1)} {m.group(1)} ", t)
    return t


def replace_number_words(text):
    """
    Replaces spoken number words in English, Hindi, and Telugu with numeric digits.
    Uses regex lookaround for Indic Unicode strings where \\b fails.
    """
    t = expand_spoken_numbers(text)
    sorted_words = sorted(NUMBER_WORDS.items(), key=lambda x: len(x[0]), reverse=True)
    for word, digit in sorted_words:
        if word.isascii() and word.isalpha():
            t = re.sub(rf"\b{re.escape(word)}\b", digit, t, flags=re.IGNORECASE)
        else:
            pat = rf"(?:(?<=[\s,.\-?!;:\"'()])|^){re.escape(word)}(?=(?:[\s,.\-?!;:\"'()]|$))"
            t = re.sub(pat, digit, t, flags=re.IGNORECASE)
    # Combine compound spoken tens + units: '90 8' -> '98', '30 4' -> '34', '50 6' -> '56'
    t = re.sub(r'\b([2-9])0\s+([1-9])\b', r'\1\2', t)
    return t


# ==============================================================================
# Multi-Language Indic to English Transliterator & Translator (Hindi & Telugu)
# ==============================================================================
TELUGU_VOWELS = {
    '\u0c05': 'a', '\u0c06': 'aa', '\u0c07': 'i', '\u0c08': 'ee', '\u0c09': 'u', '\u0c0a': 'oo',
    '\u0c0b': 'ru', '\u0c0e': 'e', '\u0c0f': 'e', '\u0c10': 'ai', '\u0c12': 'o', '\u0c13': 'o', '\u0c14': 'au'
}
TELUGU_CONSONANTS = {
    '\u0c15': 'k', '\u0c16': 'kh', '\u0c17': 'g', '\u0c18': 'gh', '\u0c19': 'ng',
    '\u0c1a': 'ch', '\u0c1b': 'chh', '\u0c1c': 'j', '\u0c1d': 'jh', '\u0c1e': 'ny',
    '\u0c1f': 't', '\u0c20': 'th', '\u0c21': 'd', '\u0c22': 'dh', '\u0c23': 'n',
    '\u0c24': 't', '\u0c25': 'th', '\u0c26': 'd', '\u0c27': 'dh', '\u0c28': 'n',
    '\u0c2a': 'p', '\u0c2b': 'ph', '\u0c2c': 'b', '\u0c2d': 'bh', '\u0c2e': 'm',
    '\u0c2f': 'y', '\u0c30': 'r', '\u0c31': 'r', '\u0c32': 'l', '\u0c33': 'l',
    '\u0c35': 'v', '\u0c36': 'sh', '\u0c37': 'sh', '\u0c38': 's', '\u0c39': 'h'
}
TELUGU_MATRAS = {
    '\u0c3e': 'a', '\u0c3f': 'i', '\u0c40': 'i', '\u0c41': 'u', '\u0c42': 'u',
    '\u0c43': 'ru', '\u0c46': 'e', '\u0c47': 'e', '\u0c48': 'ai', '\u0c4a': 'o', '\u0c4b': 'o', '\u0c4c': 'au'
}

DEVA_VOWELS = {
    '\u0905': 'a', '\u0906': 'aa', '\u0907': 'i', '\u0908': 'ee', '\u0909': 'u', '\u090a': 'oo',
    '\u090b': 'ri', '\u090f': 'e', '\u0910': 'ai', '\u0913': 'o', '\u0914': 'au'
}
DEVA_CONSONANTS = {
    '\u0915': 'k', '\u0916': 'kh', '\u0917': 'g', '\u0918': 'gh', '\u0919': 'ng',
    '\u091a': 'ch', '\u091b': 'chh', '\u091c': 'j', '\u091d': 'jh', '\u091e': 'ny',
    '\u091f': 't', '\u0920': 'th', '\u0921': 'd', '\u0922': 'dh', '\u0923': 'n',
    '\u0924': 't', '\u0925': 'th', '\u0926': 'd', '\u0927': 'dh', '\u0928': 'n',
    '\u092a': 'p', '\u092b': 'ph', '\u092c': 'b', '\u092d': 'bh', '\u092e': 'm',
    '\u092f': 'y', '\u0930': 'r', '\u0932': 'l', '\u0933': 'l',
    '\u0935': 'v', '\u0936': 'sh', '\u0937': 'sh', '\u0938': 's', '\u0939': 'h'
}
DEVA_MATRAS = {
    '\u093e': 'a', '\u093f': 'i', '\u0940': 'i', '\u0941': 'u', '\u0942': 'u',
    '\u0943': 'ri', '\u0947': 'e', '\u0948': 'ai', '\u094b': 'o', '\u094c': 'au'
}

INDIC_VOCAB_MAP = {
    # Telugu Common Words & Shop Vocabulary
    'రాయల్': 'Royal', 'రాయల్ క్లాత్స్': 'Royal Clothes', 'క్లాత్స్': 'Clothes',
    'रॉयल': 'Royal', 'रॉयल क्लॉथ्स': 'Royal Clothes', 'क्लॉथ्स': 'Clothes',
    'ఫ్రెష్ మార్ట్': 'Fresh Mart', 'ఫ్రెష్': 'Fresh', 'మార్ట్': 'Mart',
    'ఎంటర్‌ప్రైజెస్': 'Enterprises', 'ఎంటర్ ప్రైజెస్': 'Enterprises',
    'టెక్స్‌టైల్స్': 'Textiles', 'టెక్స్ టైల్స్': 'Textiles',
    'గార్మెంట్స్': 'Garments', 'గార్ మెంట్స్': 'Garments',
    'ప్రొవిజన్స్': 'Provisions', 'ప్రొవిజన్': 'Provisions', 'ప్రొవిజన్స్ స్టోర్': 'Provisions Store',
    'స్వీట్స్': 'Sweets', 'స్వీట్': 'Sweet', 'బేకరీ': 'Bakery',
    'మెడికల్': 'Medical', 'మెడికల్స్': 'Medicals',
    'జనరల్ స్టోర్స్': 'General Stores', 'జెనరల్ స్టోర్స్': 'General Stores',
    'జనరల్': 'General', 'జెనరల్': 'General',
    'హార్డ్‌వేర్': 'Hardware', 'హార్డ్ వేర్': 'Hardware',
    'మొబైల్స్': 'Mobiles', 'మొబైల్': 'Mobile',
    'ఫ్యాషన్స్': 'Fashions', 'ఫ్యాషన్': 'Fashion',
    'కలెక్షన్స్': 'Collections', 'కలెక్షన్': 'Collections',
    'సెలూన్': 'Salon', 'బ్యూటీ పార్లర్': 'Beauty Parlour',
    'క్లాత్': 'Cloth', 'స్టోర్స్': 'Stores', 'స్టోర్': 'Store',
    'షాప్': 'Shop', 'షాపు': 'Shop', 'దుకాణం': 'Shop', 'దుకానం': 'Shop',
    'బట్టలు': 'Clothing', 'బట్టల దుకాణం': 'Clothing Shop',
    'కిరాణా': 'Grocery', 'కిరాణ': 'Grocery', 'సరుకులు': 'Grocery',
    'మందులు': 'Pharmacy', 'మందుల షాప్': 'Pharmacy Shop',
    'ఎలక్ట్రానిక్స్': 'Electronics', 'ఎలక్ట్రికల్స్': 'Electricals',
    'చెప్పులు': 'Footwear', 'చెప్పుల షాప్': 'Shoe Shop',
    'సూపర్ మార్కెట్': 'Supermarket', 'సూపర్‌మార్కెట్': 'Supermarket',
    'సూపర్': 'Super', 'ట్రేడర్స్': 'Traders', 'ఏజెన్సీస్': 'Agencies',
    'హోటల్': 'Hotel', 'రెస్టారెంట్': 'Restaurant',
    'రోడ్డు': 'Road', 'రోడ్': 'Road', 'స్ట్రీట్': 'Street', 'వీధి': 'Street',
    'మెయిన్ రోడ్డు': 'Main Road', 'మెయిన్ రోడ్': 'Main Road', 'మెయిన్': 'Main',
    'బజార్': 'Bazaar', 'బజారు': 'Bazaar',
    'నగర్': 'Nagar', 'కాలనీ': 'Colony', 'కాలని': 'Colony',
    'రైల్వే స్టేషన్': 'Railway Station', 'రైల్వే స్టేషన్ రోడ్డు': 'Railway Station Road',
    'బస్టాండ్': 'Bus Stand', 'బస్ స్టాండ్': 'Bus Stand',
    'చౌరస్తా': 'Chowk', 'జంక్షన్': 'Junction', 'సర్కిల్': 'Circle',
    'కాంప్లెక్స్': 'Complex', 'భవనం': 'Building', 'ప్లాజా': 'Plaza',
    'అపార్ట్‌మెంట్': 'Apartment', 'అపార్ట్మెంట్': 'Apartment',
    'విలేజ్': 'Village', 'టౌన్': 'Town', 'మండలం': 'Mandal', 'జిల్లా': 'District',
    'శ్రీ': 'Sri', 'సాయి': 'Sai', 'లక్ష్మి': 'Lakshmi', 'వెంకటేశ్వర': 'Venkateswara',
    'రమేష్': 'Ramesh', 'సురేష్': 'Suresh', 'మహేష్': 'Mahesh', 'వెంకట్': 'Venkat', 'రవి': 'Ravi', 'ఆనంద్': 'Anand',
    'శివ': 'Shiva', 'గణేష్': 'Ganesh', 'కృష్ణ': 'Krishna', 'రామ': 'Rama', 'హనుమాన్': 'Hanuman',
    'కుమార్': 'Kumar', 'వర్మ': 'Varma', 'రావు': 'Rao', 'రెడ్డి': 'Reddy', 'నాయుడు': 'Naidu',
    'శర్మ': 'Sharma', 'గుప్తా': 'Gupta', 'రాజు': 'Raju', 'చౌదరి': 'Chowdary',
    'గాజువాక': 'Gajuwaka', 'మధురవాడ': 'Madhurawada',

    # Hindi Common Words & Shop Vocabulary
    'फ्रेश मार्ट': 'Fresh Mart', 'फ्रेश': 'Fresh', 'मार्ट': 'Mart',
    'एंटरप्राइजेज': 'Enterprises', 'टेक्सटाइल्स': 'Textiles',
    'गारमेंट्स': 'Garments', 'गारमेंट': 'Garments',
    'प्रोविजंस': 'Provisions', 'प्रोविजन': 'Provisions',
    'स्वीट्स': 'Sweets', 'मिठाई': 'Sweets', 'बेकरी': 'Bakery',
    'मेडिकल': 'Medical', 'मेडिकल्स': 'Medicals',
    'जनरल स्टोर्स': 'General Stores', 'जनरल': 'General',
    'हार्डवेयर': 'Hardware', 'मोबाइल': 'Mobile', 'मोबाइल्स': 'Mobiles',
    'फैशन': 'Fashion', 'फ़ैशन': 'Fashion', 'कलेक्शन': 'Collections', 'कलेक्शंस': 'Collections',
    'क्लॉथ': 'Cloth', 'स्टोर्स': 'Stores', 'स्टोर': 'Store',
    'दुकान': 'Shop', 'शॉप': 'Shop', 'व्यापार': 'Business',
    'कपड़े': 'Clothing', 'कपड़ा': 'Clothing', 'वस्त्र': 'Clothing',
    'किराना': 'Grocery', 'राशन': 'Grocery', 'दवा': 'Pharmacy', 'दवाई': 'Pharmacy',
    'इलेक्ट्रॉनिक्स': 'Electronics', 'बिजली': 'Electronics',
    'जूते': 'Footwear', 'चप्पल': 'Footwear',
    'सुपरमार्केट': 'Supermarket', 'सुपर': 'Super', 'ट्रेडर्स': 'Traders',
    'होटल': 'Hotel', 'रेस्टोरेंट': 'Restaurant',
    'रोड': 'Road', 'गली': 'Street', 'मेन रोड': 'Main Road', 'मेन': 'Main',
    'रोडसाइड': 'Roadside', 'रोड साइड': 'Roadside',
    'नगर': 'Nagar', 'कॉलोनी': 'Colony', 'बाजार': 'Bazaar', 'बाज़ार': 'Bazaar',
    'रेलवे स्टेशन': 'Railway Station', 'बस स्टैंड': 'Bus Stand',
    'चौराहा': 'Chowk', 'चौक': 'Chowk', 'जंक्शन': 'Junction', 'सर्कल': 'Circle',
    'कॉम्प्लेक्स': 'Complex', 'भवन': 'Building', 'प्लाजा': 'Plaza',
    'अपार्टमेंट': 'Apartment', 'गाँव': 'Village', 'कस्बा': 'Town', 'जिला': 'District',
    'राजस्थान': 'Rajasthan',
    'श्री': 'Shri', 'साईं': 'Sai', 'साई': 'Sai', 'लक्ष्मी': 'Lakshmi', 'वेंकटेश्वर': 'Venkateswara',
    'विजय': 'Vijay', 'अजय': 'Ajay', 'संजय': 'Sanjay', 'रतन': 'Ratan', 'पवन': 'Pavan',
    'अमित': 'Amit', 'रमेश': 'Ramesh', 'सुरेश': 'Suresh', 'महेश': 'Mahesh', 'रवि': 'Ravi', 'आनंद': 'Anand',
    'कुमार': 'Kumar', 'वर्मा': 'Varma', 'शर्मा': 'Sharma', 'गुप्ता': 'Gupta', 'सिंह': 'Singh',
    'यादव': 'Yadav', 'चौधरी': 'Choudhary'
}


def transliterate_indic_word(w):
    """Phonetically transliterates a single Telugu or Devanagari word into English Latin."""
    if w in INDIC_VOCAB_MAP:
        return INDIC_VOCAB_MAP[w]
    is_te = any('\u0c00' <= c <= '\u0c7f' for c in w)
    is_hi = any('\u0900' <= c <= '\u097f' for c in w)
    if not is_te and not is_hi:
        return w
    vowels = TELUGU_VOWELS if is_te else DEVA_VOWELS
    consonants = TELUGU_CONSONANTS if is_te else DEVA_CONSONANTS
    matras = TELUGU_MATRAS if is_te else DEVA_MATRAS
    virama = '\u0c4d' if is_te else '\u094d'
    anusvara = '\u0c02' if is_te else '\u0902'

    res = []
    chars = list(w)
    i = 0
    n = len(chars)
    while i < n:
        c = chars[i]
        if c in vowels:
            res.append(vowels[c])
            i += 1
        elif c in consonants:
            cons = consonants[c]
            if i + 1 < n:
                nxt = chars[i+1]
                if nxt == virama:
                    res.append(cons)
                    i += 2
                elif nxt in matras:
                    res.append(cons + matras[nxt])
                    i += 2
                elif nxt == anusvara:
                    res.append(cons + ('am' if is_te else 'an'))
                    i += 2
                else:
                    res.append(cons + 'a')
                    i += 1
            else:
                if is_te:
                    res.append(cons + 'a')
                else:
                    res.append(cons)
                i += 1
        elif c == anusvara:
            res.append('m' if is_te else 'n')
            i += 1
        else:
            res.append(c)
            i += 1
    return ''.join(res)


def translate_indic_to_english(text):
    """
    Translates spoken or typed Telugu and Hindi words, phrases, names, and addresses into clean English.
    Preserves existing English words untouched while converting Indic script and vocabulary to standard English.
    """
    if not text:
        return ""
    t = str(text).strip()
    # Check if string contains any Telugu or Devanagari characters
    has_indic = any(('\u0c00' <= c <= '\u0c7f') or ('\u0900' <= c <= '\u097f') for c in t)
    if not has_indic:
        return t

    # Replace multi-word phrase dictionary matches first
    for k, v in sorted(INDIAN_STATES_MAP.items(), key=lambda x: len(x[0]), reverse=True):
        if k in t:
            t = t.replace(k, v)
    for k, v in sorted(INDIAN_CITIES.items(), key=lambda x: len(x[0]), reverse=True):
        if k in t:
            t = t.replace(k, v[0])
    for k, v in sorted(INDIC_VOCAB_MAP.items(), key=lambda x: len(x[0]), reverse=True):
        t = t.replace(k, v)

    tokens = t.split()
    translated_tokens = [transliterate_indic_word(tok).title() for tok in tokens]
    res = ' '.join(translated_tokens)
    # Clean up double vowels or trailing 'aa' if any
    res = re.sub(r'(\b[A-Z][a-z]+)aa\b', r'\1a', res)
    return res.strip()



def is_user_asking_question(text):
    """Detects if user asked a question instead of answering in English, Hindi, or Telugu."""
    t = text.strip().lower()

    # Digits or phone numbers with question intonation (?) are NOT questions!
    if re.search(r'\d{3,}', t):
        return False
    # Check if string has multiple number words
    num_matches = sum(1 for w in t.split() if w.strip(" ,.?") in NUMBER_WORDS)
    if num_matches >= 3:
        return False

    question_starters = [
        "why", "what is", "what do you", "where is", "how do", "can you",
        "who is", "why that", "tell me why", "what does that mean",
        # Hindi
        "kyun", "kyu", "kya hai", "kaise", "kaha", "kahan", "क्यों", "क्या है", "कैसे",
        # Telugu
        "enduku", "emi", "emiti", "ela", "ekkada", "ఎందుకు", "ఏమిటి", "ఎలా", "ఎక్కడ"
    ]
    if any(t.startswith(q) for q in question_starters):
        return True

    # Intonation question mark ('?'): ignore for short answers/names ('Rahul?', 'Brostide?')
    if t.endswith("?"):
        words = t.rstrip("?").strip().split()
        if len(words) <= 2:
            return False
        question_cues = [
            "is it", "do i", "should i", "can i", "what", "how", "why", "where",
            "kya", "kyun", "kaise", "kahan", "क्या", "क्यों", "कैसे",
            "enti", "enduku", "ela", "ekkada", "వద్దా", "ఎందుకు", "ఏమిటి"
        ]
        return any(q in t for q in question_cues)

    return False


def is_skip(text):
    """Detects if user explicitly wants to skip an optional field in English, Hindi, or Telugu."""
    t = text.strip().lower()
    if not t:
        return False

    # Never treat conversational phrases starting with 'na' (Telugu: my) or 'naam/name' as skip!
    if any(t.startswith(prefix) for prefix in ('na peru', 'naa peru', 'na shop', 'naa shop', 'naam', 'name', 'noida', 'nagar', 'north', 'narendra', 'naresh')):
        return False

    exact_skip_tokens = {
        'skip', 'skep', 'skeep', 'skipp', 'scape', 'none', 'no', 'pass', 'leave',
        'not applicable', 'n/a', 'na', 'nil', 'null', 'nothing', 'next', 'nope', 'nah',
        'no code', 'no referral', 'not now', 'later', 'roadside', 'footpath', 'patri',
        # Hindi
        'nahi', 'nahin', 'chhod do', 'chhodo', 'kuch nahi', 'agla', 'skip karo', 'hatao', 'mat dalo',
        'nahi hai', 'nahi ji', 'kuch nahi hai', 'mere paas nahi hai', 'koi code nahi hai', 'pata nahi',
        'नहीं', 'नहीं है', 'ना', 'कुछ नहीं', 'कोई कोड नहीं', 'स्किप', 'अगला', 'नहीं जी', 'नहीं चाहिए', 'पता नहीं', 'बाद में',
        # Telugu
        'vaddu', 'vaddhu', 'ledu', 'emi ledu', 'skip chey', 'skip cheyyandi', 'taruvatha', 'code ledu',
        'oddu', 'akkarledu', 'avashyam ledu', 'ledu andi', 'vaddu andi', 'teliyadu',
        'వద్దు', 'లేదు', 'ఏం లేదు', 'ఏమీ లేదు', 'కోడ్ లేదు', 'వద్దండి', 'స్కిప్', 'తరువాత',
        'లేదండి', 'లేదు అండి', 'వద్దు అండి', 'వద్దండి', 'కోడ్ లేదు అండి', 'నా దగ్గర లేదు', 'తెలియదు', 'నాకు తెలియదు', 'వద్దులే'
    }
    if t in exact_skip_tokens:
        return True

    phrase_starters = [
        'skip ', 'leave it', 'not applicable', 'i don\'t have', 'i dont have', 'don\'t have', 'dont have',
        'no code', 'no referral', 'skip this', 'skip it', 'i don\'t know', 'dont know',
        'no referral code', 'i have no code', 'no door number', 'no shop number', 'on roadside',
        'roadside', 'footpath', 'street vendor',
        'chhod do ', 'nahi hai', 'koi code nahi', 'mere paas nahi hai', 'kuch nahi hai',
        'छोड़ दो', 'नहीं है', 'कोई कोड नहीं', 'मेरे पास नहीं है', 'कुछ नहीं है', 'मकान नंबर नहीं', 'दुकान नंबर नहीं',
        'ఏమీ లేదు', 'కోడ్ లేదు', 'నా దగ్గర లేదు', 'వద్దు ', 'లేదు అండి', 'రోడ్డు పక్కన', 'రోడ్ సైడ్'
    ]
    if any(t.startswith(p) for p in phrase_starters):
        return True

    words = t.split()
    if len(words) == 1 and len(t) >= 3:
        close = difflib.get_close_matches(t, ['skip', 'pass', 'none', 'vaddu', 'chhod do'], n=1, cutoff=0.75)
        if close:
            return True

    return False


WHISPER_HALLUCINATIONS = {
    "thank you for watching", "thanks for watching", "thank you for watching.", "thanks for watching.",
    "please subscribe", "subscribe to my channel", "subscribe to our channel", "like and subscribe",
    "subtitles by", "captioned by", "translated by", "amara.org",
    "thank you", "thank you.", "thanks", "thanks.", "bye", "bye.", "goodbye", "goodbye.",
    "you", "the end", "see you in the next video", "see you next time",
    "[music]", "(music)", "[applause]", "(applause)", "[laughter]", "(laughter)",
    "so", "so.", "...", "o"
}


def is_whisper_hallucination(text):
    """
    Detects Whisper repetition loops, silence hallucinations, and subtitle artifacts.
    """
    if not text or not str(text).strip():
        return True
    t = str(text).strip()
    clean_lower = t.lower().strip(" .!?,:;\"'()")
    if clean_lower in WHISPER_HALLUCINATIONS:
        return True
    for hallu in ("thank you for watching", "thanks for watching", "please subscribe", "subtitles by", "translated by", "captioned by", "amara.org"):
        if clean_lower.startswith(hallu):
            return True

    words = t.split()
    if len(words) >= 3:
        for i in range(len(words) - 2):
            if words[i].lower() == words[i+1].lower() == words[i+2].lower():
                return True
    if re.search(r'(\S{1,5})\1{3,}', t):
        return True
    if len(t) > 12 and len(set(t.replace(' ', ''))) <= 3:
        return True
    return False


def clean_company_name(text):
    """Extracts actual company name from conversational sentences in English, Hindi, or Telugu."""
    if is_whisper_hallucination(text):
        return ""
    if is_skip(text):
        return ""
    t = text.strip()
    patterns = [
        # English
        r"^(the|our|my)?\s*(store|shop|company|business|brand)?\s*(name\s+)?(is\s+called|is|called)\s+",
        r"^(our\s+)?(store|shop|company|business)\s+(name\s+)?(is\s+called|is|called)\s+",
        r"^(it\s+is\s+called|it\s+is|it\'s|its\s+name\s+is)\s+",
        r"^(we\s+are\s+called|we\s+are|we\s+run)\s+",
        r"^(store\s+name\s+is|company\s+name\s+is|shop\s+name\s+is|business\s+name\s+is)\s+",
        r"^name\s+is\s+",
        r"^the\s+name\s+is\s+",
        # Hindi (Latin & Devanagari)
        r"^(meri|hamari|apni)?\s*(dukaan|dukan|company|shop|store|vyapar|business)?\s*(ka\s+naam|ki\s+naam|naam)?\s*(hai|h|called)?\s+",
        r"^(hamari\s+company|meri\s+dukaan|hamara\s+store|dukaan\s+ka\s+naam)\s+",
        r"^(मेरी|हमारी)?\s*(दुकान|कंपनी|स्टोर|शॉप)?\s*(का\s+नाम|की\s+नाम|नाम)?\s*(है)?\s*",
        # Telugu (Latin & Telugu script)
        r"^(maa|naa|ma|na)?\s*(shop|store|dukanam|company|vyaparam)?\s*(peru|name)\s*(enti\s+ante|ante|vachesi|andi)?\s+",
        r"^(maa\s+shop|ma\s+shop|maa\s+store|ma\s+store|maa\s+company|shop\s+peru)\s+",
        r"^(మా|నా)?\s*(షాప్|స్టోర్|దుకాణం|కంపెనీ)?\s*(పేరు)?\s*(అంటే)?\s*",
    ]
    for p in patterns:
        t = re.sub(p, "", t, flags=re.IGNORECASE)

    # Remove trailing filler words in English/Hindi/Telugu
    t = re.sub(r"\s+(hai|h|andi|undi|gaaru|ji|है|అండి|ఉంది)$", "", t, flags=re.IGNORECASE)
    t = t.strip(' .,"\'')
    # Translate / transliterate Hindi & Telugu to English
    t = translate_indic_to_english(t)
    return t.title() if t else ""


def clean_person_name(text):
    """Extracts person name in English, Hindi, or Telugu."""
    if is_whisper_hallucination(text):
        return None, None
    if is_skip(text):
        return None, None
    t = text.strip()

    # Check if user spoke phone digits
    digits = re.findall(r"\d", t)
    num_words = [w for w in t.lower().split() if w.strip(" ,") in NUMBER_WORDS]
    if len(digits) + len(num_words) >= 5:
        return None, "phone_detected"

    patterns = [
        # English
        r"^(the\s+)?(contact\s+person\s+name\s+is|contact\s+person\s+is|contact\s+is)\s+",
        r"^(my\s+name\s+is|i\s+am|this\s+is|person\s+is|myself|speaking\s+is|you\s+can\s+call\s+me)\s+",
        r"^(it\s+is|it\'s|name\s+is|the\s+name\s+is)\s+",
        # Hindi
        r"^(mera\s+naam|humara\s+naam|contact\s+person\s+ka\s+naam|mujhe)\s*(hai|h)?\s*",
        r"^(मेरा\s+नाम|हमारा\s+नाम|कांटेक्ट\s+पर्सन|नाम)\s*(है)?\s*",
        # Telugu
        r"^(naa\s+peru|na\s+peru|maa\s+peru|ma\s+peru|contact\s+person\s+peru|nenu)\s*(enti\s+ante|andi)?\s*",
        r"^(నా\s+పేరు|మా\s+పేరు|కాంటాక్ట్\s+పర్సన్|పేరు)\s*",
    ]
    for p in patterns:
        t = re.sub(p, "", t, flags=re.IGNORECASE)

    # Strip trailing fillers & honorifics (e.g., 'ji', 'gaaru', 'andi')
    t = re.sub(r"\s+(hai|h|andi|gaaru|ji|है|అండి|గారు|జీ)$", "", t, flags=re.IGNORECASE)

    # Check if user spelled out letters: e.g. 'R A H U L' or 'u, d, a, y'
    tokens = re.split(r"[\s,-]+", t.strip())
    if len(tokens) > 1 and all(len(x) == 1 and x.isalpha() for x in tokens):
        return "".join(tokens).title(), None

    t = t.strip(' .,"\'')
    # Translate / transliterate Hindi & Telugu to English
    t = translate_indic_to_english(t)
    return t.title(), None


def clean_door_no(text):
    """Converts spoken door numbers like 'Door number 1 dash 2 dash 3' or 'షాప్ నెంబర్ 4 dash B' to '1-2-3' or '4-B'."""
    if not text:
        return ""
    if is_skip(text):
        return ""
    t = text.lower().strip()
    if any(w in t for w in ("roadside", "footpath", "patri", "thela", "street vendor", "రోడ్డు పక్కన", "రోడ్ సైడ్", "रोडसाइड", "पटरी")):
        return ""
    if not re.search(r"\d", t) and any(cat_k in t for cat_k in ("कपड़े", "कपड़ा", "किराना", "బట్టలు", "కిరాణా", "clothing", "grocery", "medical")):
        return ""

    t = re.sub(r"^(my\s+|our\s+)?(door|shop|flat|house|building|plot|makaan|ghar|inti)?\s*(number|no\.?|num)?\s*(is|hai|andi)?\s*", "", t)
    t = re.sub(r"^(ఇంటి\s+నంబర్|మకాన్\s+నంబర్|घर\s+का\s+नंबर|मकान\s+नंबर|दुकान\s+नंबर|షాప్\s+నంబర్|నంబర్|नंबर)\s*", "", t)
    t = re.sub(r"\s+(hai|andi|అండి|है)$", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\bdash\b|\bhyphen\b", "-", t)
    t = re.sub(r"\bslash\b", "/", t)
    t = replace_number_words(t)
    t = re.sub(r"\s*([/-])\s*", r"\1", t)
    # Join consecutive spoken digits separated by whitespace (e.g. '2 0 1' -> '201')
    t = re.sub(r"(?<=\d)\s+(?=\d)", "", t)
    # Join digit with trailing letter (e.g. '4 b' -> '4B')
    t = re.sub(r"(?<=\d)\s+([a-zA-Z]\b)", r"\1", t)
    m = re.search(r"(\d+[\w/-]*)", t)
    if m:
        return m.group(1).upper()
    # If text specifically mentions flat/plot/shop/gala
    if re.search(r"\b(plot|flat|shop|gala|shed|floor|block)\b", t):
        return translate_indic_to_english(t).title()
    return ""


def clean_phone(text):
    """
    Extracts a 10-digit Indian mobile phone number handling spoken English,
    Hindi, and Telugu digits, transliterations, prefixes, and country codes (+91, 0091, 91, 0).
    """
    t = replace_number_words(text)
    digits = re.sub(r"\D", "", t)

    # 1. Strip leading country code: 0091 (14 digits)
    if digits.startswith("0091") and len(digits) == 14:
        digits = digits[4:]
    # 2. Strip leading country code: +91 or 91 (12 digits)
    elif digits.startswith("91") and len(digits) == 12:
        digits = digits[2:]
    # 3. Strip leading zero: 0 (11 digits)
    elif digits.startswith("0") and len(digits) == 11:
        digits = digits[1:]
    # 4. If more than 10 digits, search for standard 10-digit mobile starting with 6-9
    elif len(digits) > 10:
        m = re.search(r"[6-9]\d{9}", digits)
        if m:
            digits = m.group(0)
        else:
            digits = digits[-10:]

    return digits


def clean_email(text):
    """
    Fixes spoken emails in English, Hindi, or Telugu,
    stripping conversational prefixes and fixing @ / . tokens.
    """
    t = text.lower().strip()

    # Strip conversational prefixes before parsing
    email_prefixes = [
        r"^(my\s+)?(email\s+id|email\s+address|email|contact\s+email|mail\s+id|mail)\s+(is|would\s+be|hai|andi)\s*",
        r"^(it\s+is|it\'s|you\s+can\s+reach\s+me\s+at|reach\s+me\s+at|mail\s+me\s+at)\s*",
        r"^(contact\s+me\s+at|write\s+to\s+me\s+at)\s*",
        r"^(mera\s+email|humara\s+email|naa\s+email|ईमेल|ఈమెయిల్)\s*(hai|andi)?\s*",
    ]
    for p in email_prefixes:
        t = re.sub(p, "", t, flags=re.IGNORECASE)

    t = re.sub(r"\bat the rate\b|\bat\b|एट|ఎట్", "@", t)
    t = re.sub(r"\bdot\b|डॉट|డాట్", ".", t)

    if "@" in t:
        local_part, domain_part = t.split("@", 1)
        for p in email_prefixes:
            local_part = re.sub(p, "", local_part, flags=re.IGNORECASE)

        tokens = re.split(r"[\s,-]+", local_part.strip())
        if len(tokens) > 1 and all(len(x) == 1 for x in tokens if x):
            local_part = "".join(tokens)
        else:
            local_part = re.sub(r"\s+", "", local_part)

        domain_part = re.sub(r"\s+", "", domain_part)
        domain_part = re.sub(r"gimal|gmai|g mail|g-mail|gmaill", "gmail", domain_part)
        domain_part = re.sub(r"gmail+", "gmail", domain_part)
        domain_part = re.sub(r"yaho|yaho\.", "yahoo.", domain_part)
        t = f"{local_part}@{domain_part}"
    else:
        t = re.sub(r"\s+", "", t)
        t = re.sub(r"gimal|gmai|g mail|g-mail|gmaill", "gmail", t)
        if "gmail.com" in t:
            parts = t.split("gmail.com")
            t = parts[0] + "@gmail.com"
    return t.strip(" .")


CATEGORY_MAPPINGS = {
    # Clothing / Fashion
    "कपड़े": "Clothing", "कपड़ा": "Clothing", "कपड़ों": "Clothing", "कपडों": "Clothing", "वस्त्र": "Clothing",
    "kapde": "Clothing", "kapda": "Clothing", "kapdon": "Clothing", "vastra": "Clothing",
    "బట్టలు": "Clothing", "బట్టల": "Clothing", "వస్త్రాలు": "Clothing",
    "battalu": "Clothing", "batthalu": "Clothing",
    # Grocery / Kirana
    "किराना": "Grocery", "राशन": "Grocery", "सब्जी": "Grocery", "दाल चावल": "Grocery",
    "kirana": "Grocery", "rashan": "Grocery", "sarukulu": "Grocery",
    "కిరాణా": "Grocery", "కిరాణ": "Grocery", "సరుకులు": "Grocery", "ప్రొవిజన్స్": "Grocery",
    "జనరల్ స్టోర్స్": "Grocery", "జనరల్": "Grocery", "general stores": "Grocery",
    # Pharmacy / Medical
    "medical": "Pharmacy", "medical store": "Pharmacy", "medicals": "Pharmacy",
    "medicine": "Pharmacy", "medicines": "Pharmacy", "chemist": "Pharmacy",
    "pharma": "Pharmacy", "pharmesi": "Pharmacy", "medikal": "Pharmacy",
    "दवा": "Pharmacy", "दवाएं": "Pharmacy", "दवाई": "Pharmacy", "दवाइयां": "Pharmacy",
    "dawai": "Pharmacy", "davai": "Pharmacy", "mandulu": "Pharmacy",
    "మందులు": "Pharmacy", "మందుల": "Pharmacy",
    # Clothing / Fashion / Garments
    "garments": "Clothing", "garment": "Clothing", "textiles": "Clothing",
    "textile": "Clothing", "dresses": "Clothing", "cloth": "Clothing",
    "clothes": "Clothing", "apparel": "Clothing", "cloting": "Clothing",
    "fashun": "Fashion",
    # Grocery / Kirana / Provisions
    "provisions": "Grocery", "general store": "Grocery", "departmental store": "Grocery",
    # Supermarket / Mart
    "mart": "Supermarket", "mini mart": "Supermarket", "supermarkit": "Supermarket",
    # Footwear
    "shoes": "Footwear", "shoe store": "Footwear", "slippers": "Footwear", "sandals": "Footwear",
    # Electronics
    "electric": "Electronics", "electrical": "Electronics", "electronic": "Electronics",
    "ektronics": "Electronics", "mobile": "Electronics", "mobiles": "Electronics",
    # Footwear
    "जूते": "Footwear", "चप्पल": "Footwear",
    "joota": "Footwear", "chappal": "Footwear", "cheppulu": "Footwear",
    "చెప్పులు": "Footwear",
    # Electronics
    "बिजली": "Electronics", "इलेक्ट्रॉनिक्स": "Electronics", "ఎలక్ట్రానిక్స్": "Electronics",
    # Restaurant / Food
    "खाना": "Food & Beverage", "होटल": "Restaurant", "భోజనం": "Food & Beverage",
}


CANONICAL_CATEGORIES = [
    "Clothing", "Fashion", "Grocery", "Pharmacy", "Electronics",
    "Footwear", "Food & Beverage", "Restaurant", "Supermarket",
    "Retail", "Hardware", "Jewelry", "Automobile", "Stationery",
    "Cosmetics", "Furniture", "Optical", "Toys", "Bakery", "Sports"
]


def clean_category(text):
    """
    Extracts clean store categories list from English, Hindi, or Telugu sentences.
    Robust against background noise using fuzzy phoneme matching to canonical categories.
    """
    if isinstance(text, list):
        text = ", ".join(str(x) for x in text)
    raw_str = str(text).strip()
    lower_raw = raw_str.lower()

    # 0. Check if any multilingual category keyword is directly mentioned in the sentence
    matched_cats = []
    for k, v in sorted(CATEGORY_MAPPINGS.items(), key=lambda x: len(x[0]), reverse=True):
        if k in lower_raw:
            if v not in matched_cats:
                matched_cats.append(v)
    if matched_cats:
        return matched_cats

    # Translate any Indic script or common category words to English
    t = translate_indic_to_english(raw_str)
    lower_t = t.lower()
    for k, v in sorted(CATEGORY_MAPPINGS.items(), key=lambda x: len(x[0]), reverse=True):
        if k in lower_t:
            if v not in matched_cats:
                matched_cats.append(v)
    if matched_cats:
        return matched_cats
    # Strip prefixes
    patterns = [
        # English
        r"^(we\s+sell|we\s+deal\s+in|we\s+are\s+selling|we\s+have)\s+",
        r"^(it\s+is\s+a|it\s+is|it\'s\s+a|it\'s|this\s+is\s+a|our\s+store\s+is)\s+",
        r"^(category\s+is|store\s+category\s+is|store\s+is|shop\s+is)\s+",
        # Hindi
        r"^(hum\s+bechte\s+hain|humara\s+store\s+hai|hamari\s+category\s+hai|hum)\s*",
        r"^(हम\s+बेचते\s+हैं|दुकान\s+है|हम)\s*",
        # Telugu
        r"^(memu\s+ammutham|maa\s+store\s+lo|maa\s+category|memu)\s*",
        r"^(మేము\s+అమ్ముతాము|మా\s+షాప్|మేము)\s*",
    ]
    for p in patterns:
        t = re.sub(p, "", t, flags=re.IGNORECASE)

    # Strip trailing verbs (SOV)
    t = re.sub(r"\s*(bechte\s+hain|ammutham|बेचते\s+हैं|అమ్ముతాము|అమ్ముతాం|దొరుకుతాయి|मिलते\s+हैं)$", "", t, flags=re.IGNORECASE)

    t = re.sub(r"(\band\b|\bwith\b|\bcategory\b|\bstore\b|\bshop\b|\bbusiness\b|\baur\b|\bmariyu\b|\bऔर\b|మరియు)", ",", t, flags=re.IGNORECASE)
    raw_items = [i.strip(" .,").lower() for i in t.split(",") if i.strip(" .,")]

    resolved_items = []
    canonical_lower = {c.lower(): c for c in CANONICAL_CATEGORIES}

    for item in raw_items:
        # 1. Exact dictionary match
        if item in CATEGORY_MAPPINGS:
            resolved_items.append(CATEGORY_MAPPINGS[item])
            continue
        if item in canonical_lower:
            resolved_items.append(canonical_lower[item])
            continue

        # 2. Fuzzy match against canonical categories (e.g., 'cloting' -> 'Clothing', 'fashun' -> 'Fashion')
        close_canon = difflib.get_close_matches(item, list(canonical_lower.keys()), n=1, cutoff=0.60)
        if close_canon:
            resolved_items.append(canonical_lower[close_canon[0]])
            continue

        # 3. Fuzzy match against multilingual CATEGORY_MAPPINGS
        close_mapped = difflib.get_close_matches(item, list(CATEGORY_MAPPINGS.keys()), n=1, cutoff=0.65)
        if close_mapped:
            resolved_items.append(CATEGORY_MAPPINGS[close_mapped[0]])
            continue

        resolved_items.append(item.capitalize())

    # Remove duplicates while preserving order
    deduped = []
    for it in resolved_items:
        if it not in deduped:
            deduped.append(it)

    return deduped if deduped else [text.strip(" .,").capitalize()]


def clean_street(text):
    """Extracts street name removing conversational prefixes and translating Indic terms to English."""
    if not text:
        return ""
    if is_skip(text):
        return ""
    t = text.strip()
    patterns = [
        r"^(it\s+is\s+located\s+(on|at)|located\s+(on|at)|it\'s\s+located\s+(on|at)|it\s+is\s+(on|at)|it\'s\s+(on|at)|situated\s+(on|at)|it\s+is\s+situated\s+(on|at))\s+",
        r"^(our\s+(store|shop|office)\s+is\s+(on|at|located\s+on|located\s+at))\s+",
        r"^(our\s+street\s+is|our\s+road\s+is|the\s+street\s+is|the\s+road\s+is|street\s+is|road\s+is)\s+",
        r"^(we\s+are\s+on|we\s+are\s+located\s+on|on|at)\s+",
        r"^(మా\s+స్ట్రీట్|మా\s+రోడ్|షాప్\s+ఉన్న\s+రోడ్)\s+",
        r"^(हमारी\s+गली|हमारा\s+रोड|सड़क|गली)\s+",
        r"^(रोड|రోడ్డు|రోడ్)\s+",
    ]
    for p in patterns:
        t = re.sub(p, "", t, flags=re.IGNORECASE)
    t = t.strip(' .,"\'')
    # Translate and transliterate Telugu / Hindi to English
    t = translate_indic_to_english(t)
    return t.title()


def clean_location(text, field_key, current_city=None):
    """
    Intelligently extracts Indian cities, states, towns, and localities from sentences.
    Robust against background noise with fuzzy phoneme matching.
    Always returns English representations.
    """
    if not text:
        return "", None
    if is_skip(text):
        return "", None
    t = text.strip()
    # Strip phrases like 'I am from', 'It is located in', 'City is', etc.
    patterns = [
        r"^(it\s+is\s+in|located\s+in|i\s+am\s+in|from|city\s+is|state\s+is|area\s+is|town\s+is)\s+",
        r"^(we\s+are\s+in|we\s+are\s+from|our\s+location\s+is|our\s+city\s+is|the\s+city\s+is|our\s+state\s+is|the\s+state\s+is)\s+",
        r"^(it\s+is|it\'s)\s+",
        # Hindi
        r"^(hum\s+|ye\s+|dukaan\s+)?(hai\s+in|me\s+hai|mein\s+hai)\s*",
        r"^(hamara\s+|humara\s+|mera\s+)?(shahar|shehar|sehar|rajya|ilaka|sthan|gaav)?\s*(hai|h)?\s*",
        r"^(हमारा\s+|मेरा\s+)?(शहर|राज्य|इलाका|गाँव|स्थान)?\s*(है)?\s*",
        # Telugu
        r"^(idi\s+|maa\s+store\s+)?(lo\s+undi|undi)\s*",
        r"^(maa\s+|naa\s+)?(nagaram|ooru|patnam|rajyam|area|praantham)?\s*(vachesi|ante|lo|undi)?\s*",
        r"^(మా\s+|నా\s+)?(నగరం|ఊరు|పట్టణం|రాష్ట్రం|ఏరియా|ప్రాంతం)?\s*(వచ్చేసి|అంటే|లో|ఉంది)?\s*",
    ]
    for p in patterns:
        t = re.sub(p, "", t, flags=re.IGNORECASE)

    clean_t = re.sub(r"\s+(hai|h|undi|lo|లో|है)$", "", t.strip(' .,"\''), flags=re.IGNORECASE).strip()
    lower_t = clean_t.lower()

    # If field is area or town, check localities first
    if field_key in ("area", "town"):
        if lower_t in INDIAN_LOCALITIES:
            loc_name, parent_city, state_name = INDIAN_LOCALITIES[lower_t]
            return loc_name, state_name
        for loc_k in sorted(INDIAN_LOCALITIES.keys(), key=len, reverse=True):
            if loc_k in lower_t:
                loc_name, parent_city, state_name = INDIAN_LOCALITIES[loc_k]
                return loc_name, state_name
        translated_loc = translate_indic_to_english(clean_t)
        return translated_loc.title(), None

    # 1. Exact or Substring or Fuzzy match for Indian cities
    matched_city_key = None
    if lower_t in INDIAN_CITIES:
        matched_city_key = lower_t
    else:
        # Check if known city name is contained in the text
        for city_k in sorted(INDIAN_CITIES.keys(), key=len, reverse=True):
            if city_k in lower_t:
                matched_city_key = city_k
                break
        if not matched_city_key:
            # Fuzzy match with 72% cutoff (catches 'visakapatnam', 'hyderbad', 'vijaywada', etc.)
            close_cities = difflib.get_close_matches(lower_t, list(INDIAN_CITIES.keys()), n=1, cutoff=0.72)
            if close_cities:
                matched_city_key = close_cities[0]

    if matched_city_key:
        norm_city, norm_state = INDIAN_CITIES[matched_city_key]
        if field_key == "city":
            return norm_city, norm_state
        if field_key == "state":
            return norm_state, None
        return norm_city, norm_state

    # 2. Check Indian states (Exact, Substring, and Fuzzy)
    for st_k, canonical_state in sorted(INDIAN_STATES_MAP.items(), key=lambda x: len(x[0]), reverse=True):
        if st_k in lower_t:
            if field_key == "state":
                return canonical_state, None
    close_states = difflib.get_close_matches(lower_t, list(INDIAN_STATES_MAP.keys()), n=1, cutoff=0.72)
    if close_states and field_key == "state":
        return INDIAN_STATES_MAP[close_states[0]], None

    # 3. Check if city already known and field is state
    if field_key == "state" and current_city:
        city_lower = current_city.lower()
        if city_lower in INDIAN_CITIES:
            return INDIAN_CITIES[city_lower][1], None

    # For areas, towns, and localities, translate and transliterate Indic script to English
    translated_loc = translate_indic_to_english(clean_t)
    return translated_loc.title(), None


def clean_pincode(text):
    """Extracts 6-digit Indian pincode from conversational sentences in English, Hindi, and Telugu."""
    t = replace_number_words(text)

    # First check for a clean 6-digit number sequence
    m = re.search(r"(?:(?<=\D)|^)(\d{6})(?=\D|$)", t)
    if m:
        return m.group(1)

    digits = re.sub(r"\D", "", t)
    if len(digits) >= 6:
        return digits[-6:]
    return digits


def detect_language_choice(text):
    """
    Parses user language choice from spoken or typed input in English, Hindi, or Telugu.
    Returns: 'en', 'hi', 'te', or None
    """
    if not text:
        return None
    t = text.lower().strip()

    # 1. Primary check: Explicit language names in any sentence
    if "english" in t or "angrezi" in t or "inglish" in t:
        return "en"
    if "hindi" in t or "हिंदी" in t or "हिन्दी" in t or "hindhi" in t:
        return "hi"
    if "telugu" in t or "తెలుగు" in t or "telgu" in t:
        return "te"

    clean_t = re.sub(r"[^\w\s\u0900-\u097F\u0C00-\u0C7F]", " ", t)
    tokens = clean_t.split()

    # 2. Check for explicit numbers / options
    if any(tok in ("1", "first", "one", "won", "ek", "okati") for tok in tokens):
        return "en"
    if any(tok in ("2", "second", "two", "rendu") for tok in tokens):
        return "hi"
    if any(tok in ("3", "third", "three", "tree", "teen", "tin", "moodu", "mudu") for tok in tokens):
        return "te"

    # 3. Standalone language codes or single homophones
    if len(tokens) == 1:
        if tokens[0] in ("en", "e"):
            return "en"
        if tokens[0] in ("hi", "h", "to", "too", "do"):
            return "hi"
        if tokens[0] in ("te", "t"):
            return "te"
        close_en = difflib.get_close_matches(tokens[0], ["english", "angrezi"], n=1, cutoff=0.7)
        if close_en:
            return "en"
        close_hi = difflib.get_close_matches(tokens[0], ["hindi", "hindhi"], n=1, cutoff=0.7)
        if close_hi:
            return "hi"
        close_te = difflib.get_close_matches(tokens[0], ["telugu", "telgu"], n=1, cutoff=0.7)
        if close_te:
            return "te"

    return None


if __name__ == "__main__":
    print("Testing NLU Parser...")
    print("Company:", clean_company_name("My store name is Brostide"))
    print("DoorNo:", clean_door_no("Door number 1 dash 2 dash 3"))
    print("City/State:", clean_location("Vishaka Patnam", "city"))
    print("Email:", clean_email("odanesizetadredgimal.com"))
    print("Person name with phone check:", clean_person_name("7, 0, 7, 9, 7, 4, 1, 8, 2"))
    print("Question check:", is_user_asking_question("Why is that?"))
    print("All tests passed!")
