"""
Uzbek Language Support for Voice Assistant
===========================================
Intent classification, medication name recognition in Uzbek/Russian
"""

from typing import Dict, List, Tuple, Optional
import re
import logging

logger = logging.getLogger(__name__)


class UzbekNLUEngine:
    """Natural Language Understanding for Uzbek language."""
    
    # Intent patterns in Uzbek
    INTENT_PATTERNS_UZ = {
        "find_medication": [
            r"(.*?)\s*(dori|tablet|kapsul|davo).*?(top|qidiring|izla|ber|kerak)",
            r"(.*?)\s*(dori|tablet|kapsul)\s+(bor\s*mi|topasizmi)",
            r"(aspirin|paracetamol|ibuprofen|analgin|citramon|no[-\s]*shpa)\s*(bor\s*mi|kerakmi)",
            r"bosh\s*og'rig'i.*?dori",
            r"shamollash.*?dori",
            r"isitma.*?dori",
        ],
        "find_pharmacy": [
            r"yaqin.*?dorixona",
            r"dorixona.*?(qayerda|qaerda|manzil)",
            r"eng\s*yaqin\s*dorixona",
            r"ochiq.*?dorixona",
            r"24\s*soat.*?dorixona",
        ],
        "check_interaction": [
            r"(.*?)\s*va\s*(.*?)\s*(birgalikda|birga).*?(ichsa|qabul|olsa|bo'ladimi)",
            r"oʻzaro\s*taʼsir",
            r"(.*?)\s*(.*?)\s*ichish\s*mumkinmi",
            r"dorilar.*?mos\s*keladimi",
        ],
        "check_price": [
            r"(.*?)\s*(narx|narxi|qancha)",
            r"(.*?)\s*arzon.*?dorixona",
            r"narx.*?solishtir",
        ],
        "get_info": [
            r"(.*?)\s*(haqida|toʻgʻrisida)\s*(maʻlumot|axborot)",
            r"(.*?)\s*(nima|nimaga)\s*ishlatiladi",
            r"(.*?)\s*taʼsiri",
            r"(.*?)\s*qanday\s*ichish\s*kerak",
        ],
        "scan_medication": [
            r"surat.*?ol",
            r"skaner",
            r"rasm.*?yuklash",
        ],
        "greet": [
            r"^(salom|assalomu\s*alaykum|zdravstvuy)",
            r"^(qalesan|qalaysiz|yaxshimisiz)",
        ],
        "thank": [
            r"(rahmat|tashakkur|spasibo)",
        ],
        "help": [
            r"(yordam|help|pomosh)",
            r"nima.*?qila\s*olasiz",
        ]
    }
    
    # Russian patterns
    INTENT_PATTERNS_RU = {
        "find_medication": [
            r"(najti|ishu|gde)\s*(lekarstvo|tabletka|preparat)",
            r"(aspirin|paracetamol|citramon|analgin)\s*(est|nado|kupit)",
        ],
        "find_pharmacy": [
            r"(blizhayshaya|gde)\s*apteka",
            r"apteka.*?(adres|rabotaet)",
        ],
        "check_interaction": [
            r"(mozhno|sovmestimost)\s*(prinimat|priem)",
            r"vzaimodeystvie\s*preparatov",
        ],
        "check_price": [
            r"(cena|skolko\s*stoit)",
            r"sravnit\s*cen",
        ],
    }
    
    # Common medication names in Uzbek/Russian
    MEDICATION_ALIASES = {
        "bosh_ogrigi": ["aspirin", "paracetamol", "citramon", "tempalgin"],
        "shamollash": ["coldrex", "fervex", "teraflu", "coldakt"],
        "isitma": ["paracetamol", "ibuprofen", "nurofen"],
        "yurak": ["validol", "korvalol", "valocordin"],
        "oshqozon": ["omez", "omeprazol", "gaviscon", "maalox"],
        "allergiya": ["suprastin", "loratadin", "cetrin"],
        "nafas": ["lazolvan", "bromhexin", "acc"],
        "boqishda": ["sinekod", "stoptusin"],
        "burun": ["galazolin", "nazol", "nazivin"],
        "vitamin": ["undevit", "aevit", "vitamin_c"],
    }
    
    # Symptom to medication mapping (Uzbek)
    SYMPTOM_TO_MEDICATION = {
        "bosh_ogrigi": "Bosh og'rig'i uchun: Aspirin, Paracetamol, Citramon, Ibuprofen",
        "shamollash": "Shamollash uchun: Coldrex, Fervex, Teraflu, Rinza",
        "isitma": "Isitma uchun: Paracetamol, Ibuprofen, Nurofen",
        "qorin_ogrigi": "Qorin og'rig'i uchun: No-Shpa, Drotaverin, Smekta",
        "allergiya": "Allergiya uchun: Suprastin, Loratadin, Cetrin, Zodak",
        "yo'tal": "Yo'tal uchun: Lazolvan, Bromhexin, ACC, Sinekod",
        "oshqozon": "Oshqozon uchun: Omez, Omeprazol, Gaviscon, Maalox",
    }
    
    def __init__(self):
        """Initialize Uzbek NLU engine."""
        self.compiled_patterns_uz = self._compile_patterns(self.INTENT_PATTERNS_UZ)
        self.compiled_patterns_ru = self._compile_patterns(self.INTENT_PATTERNS_RU)
    
    def _compile_patterns(self, patterns: Dict[str, List[str]]) -> Dict:
        """Compile regex patterns for faster matching."""
        compiled = {}
        for intent, pattern_list in patterns.items():
            compiled[intent] = [re.compile(pattern, re.IGNORECASE) for pattern in pattern_list]
        return compiled
    
    def detect_language(self, text: str) -> str:
        """Detect language: uz, ru, or en."""
        # Uzbek-specific characters
        uz_chars = r"[oʻgʻshchʼʻ]"
        # Russian-specific characters
        ru_chars = r"[абвгдеёжзийклмнопрстуфхцчшщъыьэюя]"
        
        text_lower = text.lower()
        
        # Count Cyrillic characters
        cyrillic_count = len(re.findall(ru_chars, text_lower))
        # Count Uzbek-specific characters
        uzbek_count = len(re.findall(uz_chars, text_lower))
        
        # Common Uzbek words
        uzbek_words = ["dori", "dorixona", "tablet", "narx", "bor", "kerak", "qancha"]
        uzbek_word_count = sum(1 for word in uzbek_words if word in text_lower)
        
        # Common Russian words
        russian_words = ["apteka", "lekarstvo", "cena", "kupit", "est", "nado"]
        russian_word_count = sum(1 for word in russian_words if word in text_lower)
        
        if uzbek_count > 0 or uzbek_word_count >= 2:
            return "uz"
        elif cyrillic_count > len(text) * 0.3 or russian_word_count >= 2:
            return "ru"
        else:
            return "en"
    
    def classify_intent(self, text: str, language: str = "auto") -> Dict:
        """
        Classify user intent from text.
        
        Returns:
            {
                "intent": str,
                "confidence": float,
                "entities": Dict,
                "language": str
            }
        """
        if language == "auto":
            language = self.detect_language(text)
        
        # Select appropriate patterns
        if language == "uz":
            patterns = self.compiled_patterns_uz
        elif language == "ru":
            patterns = self.compiled_patterns_ru
        else:
            patterns = self.compiled_patterns_uz  # Default to Uzbek
        
        # Try to match patterns
        best_match = None
        best_confidence = 0.0
        matched_intent = "unknown"
        entities = {}
        
        for intent, pattern_list in patterns.items():
            for pattern in pattern_list:
                match = pattern.search(text)
                if match:
                    confidence = 0.9  # High confidence for pattern match
                    if confidence > best_confidence:
                        best_confidence = confidence
                        matched_intent = intent
                        best_match = match
                        
                        # Extract entities from groups
                        if match.groups():
                            entities = self._extract_entities(match, intent, text)
        
        # Fallback: keyword-based matching
        if best_confidence == 0.0:
            matched_intent, best_confidence = self._keyword_fallback(text, language)
        
        return {
            "intent": matched_intent,
            "confidence": best_confidence,
            "entities": entities,
            "language": language
        }
    
    def _extract_entities(self, match, intent: str, text: str) -> Dict:
        """Extract entities from regex match."""
        entities = {}
        
        if intent == "find_medication":
            # Try to extract medication name
            groups = [g for g in match.groups() if g]
            if groups:
                entities["medication_name"] = groups[0].strip()
        
        elif intent == "check_interaction":
            # Extract two medication names
            groups = [g for g in match.groups() if g and g not in ["va", "bilan", "i"]]
            if len(groups) >= 2:
                entities["medication1"] = groups[0].strip()
                entities["medication2"] = groups[1].strip()
        
        elif intent == "check_price":
            # Extract medication name
            groups = [g for g in match.groups() if g and g not in ["narx", "narxi", "qancha"]]
            if groups:
                entities["medication_name"] = groups[0].strip()
        
        return entities
    
    def _keyword_fallback(self, text: str, language: str) -> Tuple[str, float]:
        """Fallback keyword-based classification."""
        text_lower = text.lower()
        
        keywords = {
            "find_medication": ["dori", "tablet", "lekarstvo", "medication"],
            "find_pharmacy": ["dorixona", "apteka", "pharmacy"],
            "check_interaction": ["birgalikda", "sovmestimost", "interaction"],
            "check_price": ["narx", "cena", "price"],
            "help": ["yordam", "help", "pomosh"],
        }
        
        best_intent = "unknown"
        max_matches = 0
        
        for intent, words in keywords.items():
            matches = sum(1 for word in words if word in text_lower)
            if matches > max_matches:
                max_matches = matches
                best_intent = intent
        
        confidence = 0.6 if max_matches > 0 else 0.3
        return best_intent, confidence
    
    def get_response_template(self, intent: str, language: str = "uz") -> str:
        """Get response template for intent in specified language."""
        templates = {
            "uz": {
                "find_medication": "'{medication}' dorini qidiryapman...",
                "find_pharmacy": "Yaqin atrofdagi dorixonalarni topyapman...",
                "check_interaction": "'{med1}' va '{med2}' o'zaro ta'sirini tekshiryapman...",
                "check_price": "'{medication}' narxlarini solishtiryapman...",
                "greet": "Assalomu alaykum! Men sizga qanday yordam bera olaman?",
                "thank": "Marhamat! Yana savollaringiz bo'lsa so'rang.",
                "help": "Men quyidagilarni qila olaman:\n• Dori qidirish\n• Dorixona topish\n• Narx solishtirish\n• O'zaro ta'sir tekshirish",
                "unknown": "Kechirasiz, tushunmadim. Boshqa so'z bilan yozing.",
            },
            "ru": {
                "find_medication": "Ищу лекарство '{medication}'...",
                "find_pharmacy": "Ищу ближайшие аптеки...",
                "check_interaction": "Проверяю взаимодействие '{med1}' и '{med2}'...",
                "check_price": "Сравниваю цены на '{medication}'...",
                "greet": "Здравствуйте! Чем могу помочь?",
                "thank": "Пожалуйста! Обращайтесь ещё.",
                "help": "Я могу:\n• Найти лекарство\n• Найти аптеку\n• Сравнить цены\n• Проверить совместимость",
                "unknown": "Извините, не понял. Попробуйте иначе.",
            }
        }
        
        lang_templates = templates.get(language, templates["uz"])
        return lang_templates.get(intent, lang_templates["unknown"])
    
    def suggest_medication_by_symptom(self, text: str) -> Optional[str]:
        """Suggest medication based on symptom description."""
        text_lower = text.lower()
        
        for symptom, suggestion in self.SYMPTOM_TO_MEDICATION.items():
            symptom_keywords = symptom.split("_")
            if all(keyword in text_lower for keyword in symptom_keywords):
                return suggestion
        
        return None


def get_uzbek_nlu_engine() -> UzbekNLUEngine:
    """Get singleton instance."""
    return UzbekNLUEngine()
