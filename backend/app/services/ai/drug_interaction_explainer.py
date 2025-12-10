"""
Drug Interaction Service - Enhanced
====================================
Uzbek language support and better explanations for drug interactions
"""

from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class DrugInteractionExplainer:
    """Provides detailed explanations of drug interactions in Uzbek."""
    
    SEVERITY_UZ = {
        "none": {
            "name": "Oʻzaro taʼsir yo'q",
            "emoji": "✅",
            "color": "green",
            "description": "Bu dorilarni birgalikda qabul qilish xavfsiz"
        },
        "mild": {
            "name": "Engil oʻzaro taʼsir",
            "emoji": "⚠️",
            "color": "yellow",
            "description": "Kamdan-kam hollarda yengil noqu layliklar yuzaga kelishi mumkin",
            "action": "Doktor nazoratida qabul qilish tavsiya etiladi"
        },
        "moderate": {
            "name": "O'rtacha oʻzaro taʼsir",
            "emoji": "🟠",
            "color": "orange",
            "description": "Dorilarning samaradorligi yoki xavfsizligi o'zgarishi mumkin",
            "action": "Shifokor bilan maslahatlashish MAJBURIY. Dozani o'zgartirish kerak bo'lishi mumkin"
        },
        "severe": {
            "name": "Jiddiy oʻzaro taʼsir",
            "emoji": "⛔",
            "color": "red",
            "description": "Sog'liq uchun xavfli oqibatlar yuzaga kelishi mumkin",
            "action": "TEZDA shifokorga murojaat qiling! Bu dorilarni birgalikda qabul qilmang"
        },
        "fatal": {
            "name": "Hayot uchun xavfli",
            "emoji": "🚨",
            "color": "darkred",
            "description": "Juda xavfli! Hayot uchun xavf tug'dirishi mumkin",
            "action": "DARHOL tez yordam chaqiring! Bu dorilarni hech qachon birgalikda qabul qilmang!"
        }
    }
    
    INTERACTION_TYPES_UZ = {
        "pharmacodynamic": {
            "name": "Farmakodinamik oʻzaro taʼsir",
            "description": "Dorilar tanada bir xil tizimlarga taʼsir qiladi",
            "examples": [
                "Ikki dori ham bosimni pasaytiradi - haddan tashqari pasayishi mumkin",
                "Ikkalasi ham uyqu keltiradimi - haddan tashqari mudirlik",
                "Ikki og'riq qoldiruvchi - oshqozon yarasi xavfi"
            ]
        },
        "pharmacokinetic": {
            "name": "Farmakokinetik oʻzaro taʼsir",
            "description": "Bir dori ikkinchisining so'rilishi, tarqalishi yoki chiqarilishiga taʼsir qiladi",
            "examples": [
                "Bir dori ikkinchisining jigar orqali parchalanishini sekinlashtiradi",
                "Bir dori ikkinchisining qonga so'rilishini bloklaydi",
                "Bir dori ikkinchisining buyrak orqali chiqishini kamaytiradi"
            ]
        },
        "absorption": {
            "name": "So'rilish jarayoniga taʼsir",
            "description": "Bir dori ikkinchisining oshqozondan so'rilishiga halaqit beradi"
        },
        "metabolism": {
            "name": "Metabolizm (parchalanish)ga taʼsir",
            "description": "Jigar fermentlari orqali dorilarning parchalanishiga taʼsir"
        },
        "excretion": {
            "name": "Chiqarilishga taʼsir",
            "description": "Buyrak orqali chiqarish jarayoniga taʼsir qiladi"
        },
        "protein_binding": {
            "name": "Oqsil bilan bog'lanish",
            "description": "Qon oqsillariga bog'lanish uchun raqobatlashadi"
        },
        "enzyme_inhibition": {
            "name": "Ferment inhibitsiyasi",
            "description": "Bir dori jigar fermentlarini bloklaydi, ikkinchisining konsentratsiyasi oshadi"
        },
        "enzyme_induction": {
            "name": "Ferment induksiyasi",
            "description": "Bir dori jigar fermentlarini faollashtirad, ikkinchisining samaradorligi kamayadi"
        }
    }
    
    COMMON_SYMPTOMS_UZ = {
        "cardiovascular": [
            "Yurak urishi sekinlashishi yoki tezlashishi",
            "Bosh aylanishi, hushidan ketish",
            "Qon bosimining keskin o'zgarishi",
            "Ko'krak qafasida og'riq"
        ],
        "cns": [
            "Ortiqcha mudirlik, uyquchanlik",
            "Bosh og'rig'i, bosh aylanishi",
            "Aqliy faoliyatning pasayishi",
            "Titroq, muskul zaifligimi"
        ],
        "gastrointestinal": [
            "Oshqozon og'rig'i, ko'ngil aynishi",
            "Diareyamiyo ich ketish",
            "Oshqozon yarasi xavfi",
            "Jigar faoliyatining buzilishi"
        ],
        "renal": [
            "Buyrak faoliyatining buzilishi",
            "Siydik chiqarish muammolari",
            "Siydik tarkibidagi o'zgarishlar"
        ],
        "metabolic": [
            "Qon shakar darajasining o'zgarishi",
            "Elektrolitlar disbalansimi",
            "Vazn o'zgarishi"
        ],
        "allergic": [
            "Terida toshmalar",
            "Qichishish, allergiya",
            "Nafas olish qiyinlashishi",
            "Yuz, til, tomoqning shishishi (xavfli!)"
        ]
    }
    
    MONITORING_UZ = {
        "blood_pressure": "Qon bosimini kunlik o'lchang",
        "heart_rate": "Yurak urishini kuzating",
        "blood_glucose": "Qon shakarini tekshiring",
        "liver_function": "Jigar analizlarini tekshiring",
        "kidney_function": "Buyrak analizlarini tekshiring",
        "inr": "Qon ivishini tekshiring (warfarin uchun)",
        "symptoms": "O'zingizni yomon his qilsangiz darhol shifokorga murojaat qiling"
    }
    
    def get_severity_info(self, severity: str) -> Dict:
        """Get severity information in Uzbek."""
        return self.SEVERITY_UZ.get(severity, self.SEVERITY_UZ["none"])
    
    def get_interaction_type_info(self, interaction_type: str) -> Dict:
        """Get interaction type explanation in Uzbek."""
        return self.INTERACTION_TYPES_UZ.get(interaction_type, {
            "name": "Noma'lum turi",
            "description": "Oʻzaro taʼsir turi aniqlanmagan"
        })
    
    def generate_patient_explanation(
        self,
        drug1_name: str,
        drug2_name: str,
        severity: str,
        interaction_type: Optional[str] = None,
        mechanism: Optional[str] = None
    ) -> str:
        """Generate patient-friendly explanation in Uzbek."""
        severity_info = self.get_severity_info(severity)
        
        explanation = f"""
🔍 **{drug1_name}** va **{drug2_name}** o'rtasidagi o'zaro ta'sir

{severity_info['emoji']} **Xavf darajasi:** {severity_info['name']}

📝 **Nima bo'ladi:**
{severity_info['description']}
"""
        
        if severity != "none":
            explanation += f"\n\n⚡ **Nimami qilish kerak:**\n{severity_info.get('action', '')}"
            
            if interaction_type:
                type_info = self.get_interaction_type_info(interaction_type)
                explanation += f"\n\n🔬 **O'zaro ta'sir turi:**\n{type_info['name']} - {type_info['description']}"
            
            if mechanism:
                explanation += f"\n\n⚙️ **Qanday ishlaydi:**\n{mechanism}"
        
        return explanation
    
    def get_monitoring_recommendations(self, severity: str, affected_systems: List[str]) -> List[str]:
        """Get monitoring recommendations in Uzbek."""
        recommendations = []
        
        if severity in ["moderate", "severe", "fatal"]:
            recommendations.append(self.MONITORING_UZ["symptoms"])
        
        for system in affected_systems:
            if system == "cardiovascular":
                recommendations.extend([
                    self.MONITORING_UZ["blood_pressure"],
                    self.MONITORING_UZ["heart_rate"]
                ])
            elif system == "metabolic":
                recommendations.append(self.MONITORING_UZ["blood_glucose"])
            elif system == "hepatic":
                recommendations.append(self.MONITORING_UZ["liver_function"])
            elif system == "renal":
                recommendations.append(self.MONITORING_UZ["kidney_function"])
        
        return list(set(recommendations))
    
    def get_warning_symptoms(self, affected_systems: List[str]) -> List[str]:
        """Get warning symptoms to watch for in Uzbek."""
        symptoms = []
        
        for system in affected_systems:
            if system in self.COMMON_SYMPTOMS_UZ:
                symptoms.extend(self.COMMON_SYMPTOMS_UZ[system])
        
        return symptoms


def get_drug_interaction_explainer() -> DrugInteractionExplainer:
    """Get singleton instance."""
    return DrugInteractionExplainer()
