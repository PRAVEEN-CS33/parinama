WHISPER_LANGUAGES ={
    "english": "en",
    "chinese": "zh",
    "german": "de",
    "spanish": "es",
    "russian": "ru",
    "korean": "ko",
    "french": "fr",
    "japanese": "ja",
    "portuguese": "pt",
    "turkish": "tr",
    "polish": "pl",
    "catalan": "ca",
    "dutch": "nl",
    "arabic": "ar",
    "swedish": "sv",
    "italian": "it",
    "indonesian": "id",
    "hindi": "hi",
    "finnish": "fi",
    "vietnamese": "vi",
    "hebrew": "he",
    "ukrainian": "uk",
    "greek": "el",
    "malay": "ms",
    "czech": "cs",
    "romanian": "ro",
    "danish": "da",
    "hungarian": "hu",
    "tamil": "ta",
    "norwegian": "no",
    "thai": "th",
    "urdu": "ur",
    "croatian": "hr",
    "bulgarian": "bg",
    "lithuanian": "lt",
    "latin": "la",
    "maori": "mi",
    "malayalam": "ml",
    "welsh": "cy",
    "slovak": "sk",
    "telugu": "te",
    "persian": "fa",
    "latvian": "lv",
    "bengali": "bn",
    "serbian": "sr",
    "azerbaijani": "az",
    "slovenian": "sl",
    "kannada": "kn",
    "estonian": "et",
    "macedonian": "mk",
    "breton": "br",
    "basque": "eu",
    "icelandic": "is",
    "armenian": "hy",
    "nepali": "ne",
    "mongolian": "mn",
    "bosnian": "bs",
    "kazakh": "kk",
    "albanian": "sq",
    "swahili": "sw",
    "galician": "gl",
    "marathi": "mr",
    "punjabi": "pa",
    "sinhala": "si",
    "khmer": "km",
    "shona": "sn",
    "yoruba": "yo",
    "somali": "so",
    "afrikaans": "af",
    "occitan": "oc",
    "georgian": "ka",
    "belarusian": "be",
    "tajik": "tg",
    "sindhi": "sd",
    "gujarati": "gu",
    "amharic": "am",
    "yiddish": "yi",
    "lao": "lo",
    "uzbek": "uz",
    "faroese": "fo",
    "haitian creole": "ht",
    "pashto": "ps",
    "turkmen": "tk",
    "nynorsk": "nn",
    "maltese": "mt",
    "sanskrit": "sa",
    "luxembourgish": "lb",
    "myanmar": "my",
    "tibetan": "bo",
    "tagalog": "tl",
    "malagasy": "mg",
    "assamese": "as",
    "tatar": "tt",
    "hawaiian": "haw",
    "lingala": "ln",
    "hausa": "ha",
    "bashkir": "ba",
    "javanese": "jw",
    "sundanese": "su",
    "cantonese": "yue"
}

# ================================
# Customization Constants for Indic Parler-TTS
# ================================

# Available Voices per Language (Recommended included)
VOICES = {
    "Assamese": ["Amit", "Sita", "Poonam", "Rakesh"],
    "Bengali": ["Arjun", "Aditi", "Tapan", "Rashmi", "Arnav", "Riya"],
    "Bodo": ["Bikram", "Maya", "Kalpana"],
    "Dogri": ["Karan"],
    "English": ["Thoma", "Mary", "Swapna", "Dinesh", "Meera", "Jatin", "Aakash", "Sneha", "Kabir", "Tisha", "Chingkhei", "Thoiba", "Priya", "Tarun", "Gauri", "Nisha", "Raghav", "Kavya", "Ravi", "Vikas", "Riya"],
    "Gujarati": ["Yash", "Neha"],
    "Hindi": ["Rohit", "Divya", "Aman", "Rani"],
    "Kannada": ["Suresh", "Anu", "Chetan", "Vidya"],
    "Malayalam": ["Anjali", "Anju", "Harish"],
    "Manipuri": ["Laishram", "Ranjit"],
    "Marathi": ["Sanjay", "Sunita", "Nikhil", "Radha", "Varun", "Isha"],
    "Nepali": ["Amrita"],
    "Odia": ["Manas", "Debjani"],
    "Punjabi": ["Divjot", "Gurpreet"],
    "Sanskrit": ["Aryan"],
    "Tamil": ["Kavitha", "Jaya"],
    "Telugu": ["Prakash", "Lalitha", "Kiran"],
    "Sindhi": ["Generic-Speaker"],  # placeholder
}

# Emotions officially supported in 10 languages
EMOTIONS = [
    "Command", "Anger", "Narration", "Conversation", "Disgust", 
    "Fear", "Happy", "Neutral", "Proper Noun", "News", "Sad", "Surprise"
]

# Pitch levels
PITCH = ["very low pitch", "low pitch", "balanced pitch", "high pitch", "very high pitch"]

# Speaking rate
RATES = ["very slow pace", "slow pace", "moderate pace", "fast pace", "very fast pace"]

# Expressivity
EXPRESSIVITY = ["monotone", "slightly expressive", "expressive", "highly expressive"]

# Background noise
BACKGROUND = ["no background noise", "slightly noisy", "very noisy audio"]

# Reverberation / distance
REVERB = ["very close recording", "close recording", "medium distance", "distant recording"]

# Voice quality
QUALITY = ["basic voice quality", "clear voice", "refined high-quality audio"]