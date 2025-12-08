"""
DailyMed Integration Service
============================
National Library of Medicine (NLM) medication database integration

DailyMed provides:
- FDA-approved drug labels (SPL documents)
- NDC codes (National Drug Code)
- Drug images and packaging photos
- Active ingredients
- Imprint codes
- Safety warnings and recalls
- Drug interactions
- Dosage information
"""

from typing import Optional, List, Dict, Any
import httpx
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class DailyMedService:
    """
    Service for accessing DailyMed (NLM) drug database.
    
    Official API: https://dailymed.nlm.nih.gov/dailymed/services/v2/
    """
    
    BASE_URL = "https://dailymed.nlm.nih.gov/dailymed/services/v2"
    
    def __init__(self):
        """Initialize DailyMed service."""
        self.client = httpx.AsyncClient(
            timeout=30.0,
            headers={
                "Accept": "application/json",
                "User-Agent": "Sentinel-RX/1.0"
            }
        )
        self._cache = {}  # Simple in-memory cache
        self._cache_ttl = timedelta(hours=24)
    
    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()
    
    async def search_by_ndc(self, ndc_code: str) -> Optional[Dict[str, Any]]:
        """
        Search medication by NDC code.
        
        Args:
            ndc_code: National Drug Code (e.g., "0002-3227-30")
            
        Returns:
            Medication information with label data
        """
        cache_key = f"ndc_{ndc_code}"
        
        if cache_key in self._cache:
            cached_time, data = self._cache[cache_key]
            if datetime.utcnow() - cached_time < self._cache_ttl:
                logger.info(f"Cache hit for NDC {ndc_code}")
                return data
        
        try:
            # Step 1: Get SPL documents by NDC
            url = f"{self.BASE_URL}/spls.json"
            params = {"ndc": ndc_code}
            
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if not data.get("data"):
                logger.warning(f"No medication found for NDC {ndc_code}")
                return None
            
            # Get first match (most recent)
            spl = data["data"][0]
            setid = spl.get("setid")
            
            # Step 2: Get full medication details
            medication_data = await self.get_medication_by_setid(setid)
            
            # Cache result
            self._cache[cache_key] = (datetime.utcnow(), medication_data)
            
            return medication_data
            
        except Exception as e:
            logger.error(f"DailyMed NDC search failed: {e}", exc_info=True)
            return None
    
    async def get_medication_by_setid(self, setid: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed medication information by SET ID.
        
        Args:
            setid: DailyMed SET ID (unique identifier)
            
        Returns:
            Complete medication data
        """
        cache_key = f"setid_{setid}"
        
        if cache_key in self._cache:
            cached_time, data = self._cache[cache_key]
            if datetime.utcnow() - cached_time < self._cache_ttl:
                return data
        
        try:
            # Get SPL document
            url = f"{self.BASE_URL}/spls/{setid}.json"
            response = await self.client.get(url)
            response.raise_for_status()
            
            spl_data = response.json()
            
            # Get NDC codes
            ndcs = await self._get_ndcs(setid)
            
            # Get packaging info
            packaging = await self._get_packaging(setid)
            
            # Get media (images)
            media = await self._get_media(setid)
            
            # Parse and structure data
            medication_data = self._parse_spl_data(
                spl_data,
                ndcs,
                packaging,
                media
            )
            
            # Cache result
            self._cache[cache_key] = (datetime.utcnow(), medication_data)
            
            return medication_data
            
        except Exception as e:
            logger.error(f"DailyMed SETID lookup failed: {e}", exc_info=True)
            return None
    
    async def search_by_name(
        self,
        drug_name: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search medications by name.
        
        Args:
            drug_name: Medication name
            limit: Maximum results
            
        Returns:
            List of matching medications
        """
        try:
            url = f"{self.BASE_URL}/spls.json"
            params = {
                "drug_name": drug_name,
                "page_size": limit
            }
            
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            medications = []
            for spl in data.get("data", [])[:limit]:
                med = {
                    "setid": spl.get("setid"),
                    "title": spl.get("title"),
                    "published_date": spl.get("published_date"),
                    "route": spl.get("route", []),
                    "marketing_category": spl.get("marketing_category"),
                }
                medications.append(med)
            
            return medications
            
        except Exception as e:
            logger.error(f"DailyMed name search failed: {e}", exc_info=True)
            return []
    
    async def get_drug_image(self, setid: str) -> Optional[str]:
        """
        Get drug packaging image URL.
        
        Args:
            setid: DailyMed SET ID
            
        Returns:
            Image URL or None
        """
        media = await self._get_media(setid)
        
        if media and len(media) > 0:
            # Return first image
            return f"https://dailymed.nlm.nih.gov/dailymed/image.cfm?name={media[0]['name']}&setid={setid}"
        
        return None
    
    async def check_recalls(self, setid: str) -> Dict[str, Any]:
        """
        Check if medication has any FDA recalls.
        
        Args:
            setid: DailyMed SET ID
            
        Returns:
            Recall information
        """
        # Note: DailyMed doesn't have direct recall API
        # This would need integration with FDA FAERS/Recalls API
        # For now, return placeholder
        
        return {
            "has_recall": False,
            "recall_date": None,
            "reason": None,
            "source": "FDA Safety Recalls",
            "link": "https://www.fda.gov/safety/recalls-market-withdrawals-safety-alerts"
        }
    
    async def get_pill_imprint_info(self, setid: str) -> Optional[Dict[str, Any]]:
        """
        Extract pill imprint and physical characteristics from label.
        
        Args:
            setid: DailyMed SET ID
            
        Returns:
            Pill characteristics (imprint, shape, color, size)
        """
        try:
            # Get packaging info which contains pill descriptions
            packaging = await self._get_packaging(setid)
            
            if not packaging:
                return None
            
            # Parse packaging descriptions for pill characteristics
            pill_info = {
                "imprint": None,
                "shape": None,
                "color": None,
                "size": None,
                "score": None  # Has dividing line
            }
            
            # Look through packaging descriptions
            for pkg in packaging:
                description = pkg.get("description", "").lower()
                
                # Extract characteristics (basic parsing)
                if "round" in description:
                    pill_info["shape"] = "round"
                elif "oval" in description:
                    pill_info["shape"] = "oval"
                elif "capsule" in description:
                    pill_info["shape"] = "capsule"
                elif "oblong" in description:
                    pill_info["shape"] = "oblong"
                
                # Color detection
                colors = ["white", "blue", "red", "yellow", "green", "pink", "orange", "brown"]
                for color in colors:
                    if color in description:
                        pill_info["color"] = color
                        break
                
                # Imprint/marking
                if "imprint" in description or "debossed" in description:
                    # Try to extract imprint code (would need better parsing)
                    parts = description.split()
                    for i, part in enumerate(parts):
                        if part in ["imprint", "debossed", "engraved"] and i + 1 < len(parts):
                            pill_info["imprint"] = parts[i + 1].strip('":,.')
            
            return pill_info
            
        except Exception as e:
            logger.error(f"Failed to extract pill imprint info: {e}")
            return None
    
    async def _get_ndcs(self, setid: str) -> List[str]:
        """Get all NDC codes for a medication."""
        try:
            url = f"{self.BASE_URL}/spls/{setid}/ndcs.json"
            response = await self.client.get(url)
            response.raise_for_status()
            
            data = response.json()
            return [ndc.get("ndc") for ndc in data.get("data", [])]
            
        except Exception:
            return []
    
    async def _get_packaging(self, setid: str) -> List[Dict]:
        """Get packaging descriptions."""
        try:
            url = f"{self.BASE_URL}/spls/{setid}/packaging.json"
            response = await self.client.get(url)
            response.raise_for_status()
            
            data = response.json()
            return data.get("data", [])
            
        except Exception:
            return []
    
    async def _get_media(self, setid: str) -> List[Dict]:
        """Get media files (images)."""
        try:
            url = f"{self.BASE_URL}/spls/{setid}/media.json"
            response = await self.client.get(url)
            response.raise_for_status()
            
            data = response.json()
            return data.get("data", [])
            
        except Exception:
            return []
    
    def _parse_spl_data(
        self,
        spl_data: Dict,
        ndcs: List[str],
        packaging: List[Dict],
        media: List[Dict]
    ) -> Dict[str, Any]:
        """Parse SPL document into structured medication data."""
        
        data = spl_data.get("data", {})
        
        # Extract basic info
        medication = {
            "setid": data.get("setid"),
            "title": data.get("title"),
            "generic_name": data.get("generic_drug", [{}])[0].get("name") if data.get("generic_drug") else None,
            "brand_name": data.get("brand_name"),
            "published_date": data.get("published_date"),
            "effective_time": data.get("effective_time"),
            
            # Identification
            "ndc_codes": ndcs,
            "rxcui": data.get("rxcui"),
            "spl_version": data.get("spl_version"),
            
            # Classification
            "dosage_form": data.get("dosage_form", []),
            "route": data.get("route", []),
            "strength": data.get("strength"),
            
            # Ingredients
            "active_ingredients": self._extract_ingredients(data.get("openfda", {})),
            
            # Manufacturer
            "manufacturer": data.get("manufacturer_name"),
            "labeler": data.get("labeler_name"),
            
            # Regulatory
            "dea_schedule": data.get("dea_schedule_code"),
            "marketing_category": data.get("marketing_category"),
            "application_number": data.get("application_number"),
            
            # Safety
            "boxed_warning": data.get("boxed_warning"),
            "warnings": data.get("warnings"),
            "adverse_reactions": data.get("adverse_reactions"),
            "contraindications": data.get("contraindications"),
            
            # Usage
            "indications_and_usage": data.get("indications_and_usage"),
            "dosage_and_administration": data.get("dosage_and_administration"),
            
            # Packaging
            "packaging": packaging,
            
            # Media
            "images": [
                f"https://dailymed.nlm.nih.gov/dailymed/image.cfm?name={m['name']}&setid={data.get('setid')}"
                for m in media
            ],
            
            # Links
            "dailymed_url": f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={data.get('setid')}",
            "label_pdf": f"https://dailymed.nlm.nih.gov/dailymed/downloadpdffile.cfm?setId={data.get('setid')}",
        }
        
        return medication
    
    def _extract_ingredients(self, openfda: Dict) -> List[Dict[str, str]]:
        """Extract active ingredients from OpenFDA data."""
        
        ingredients = []
        
        if "substance_name" in openfda and "active_numerator_strength" in openfda:
            names = openfda.get("substance_name", [])
            strengths = openfda.get("active_numerator_strength", [])
            units = openfda.get("active_numerator_unit", [])
            
            for i, name in enumerate(names):
                ingredient = {
                    "name": name,
                    "strength": strengths[i] if i < len(strengths) else None,
                    "unit": units[i] if i < len(units) else None
                }
                ingredients.append(ingredient)
        
        return ingredients


# Utility functions for integration

async def enrich_medication_with_dailymed(
    medication_dict: Dict[str, Any],
    ndc_code: Optional[str] = None
) -> Dict[str, Any]:
    """
    Enrich existing medication data with DailyMed information.
    
    Args:
        medication_dict: Existing medication data
        ndc_code: Optional NDC code for lookup
        
    Returns:
        Enriched medication data
    """
    service = DailyMedService()
    
    try:
        dailymed_data = None
        
        # Try to find in DailyMed
        if ndc_code:
            dailymed_data = await service.search_by_ndc(ndc_code)
        elif medication_dict.get("name"):
            results = await service.search_by_name(medication_dict["name"], limit=1)
            if results:
                dailymed_data = await service.get_medication_by_setid(results[0]["setid"])
        
        # Merge data
        if dailymed_data:
            # Add DailyMed fields
            medication_dict["dailymed_setid"] = dailymed_data.get("setid")
            medication_dict["dailymed_url"] = dailymed_data.get("dailymed_url")
            medication_dict["official_label_pdf"] = dailymed_data.get("label_pdf")
            
            # Enrich with FDA data
            if not medication_dict.get("warnings"):
                medication_dict["warnings"] = dailymed_data.get("warnings")
            
            if not medication_dict.get("images"):
                medication_dict["images"] = dailymed_data.get("images", [])
            
            # Add pill characteristics
            pill_info = await service.get_pill_imprint_info(dailymed_data["setid"])
            if pill_info:
                if not medication_dict.get("imprint_code"):
                    medication_dict["imprint_code"] = pill_info.get("imprint")
                if not medication_dict.get("shape"):
                    medication_dict["shape"] = pill_info.get("shape")
                if not medication_dict.get("color_primary"):
                    medication_dict["color_primary"] = pill_info.get("color")
        
        return medication_dict
        
    finally:
        await service.close()
