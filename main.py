"""
Riya — AI Real Estate Sales Executive MVP
Single-file implementation containing FastAPI app, models, adapters, logic engines, and telephony webhooks.
"""

import os
import re
import json
import uuid
import time
import asyncio
import logging
from enum import Enum
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Request, Header, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, EmailStr, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict
import httpx

# =====================================================================
# SECTION 1: CONFIGURATION & LOGGING SETUP
# =====================================================================

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")
    
    APP_ENV: str = "development"
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = int(os.getenv("PORT", 8000))  # Correctly reads Render's dynamic PORT
    LOG_LEVEL: str = "INFO"
    
    # LLM
    LLM_API_KEY: str = ""
    LLM_BASE_URL: str = "https://api.openai.com/v1"
    LLM_MODEL: str = "gpt-oss-20b"
    
    # STT
    DEEPGRAM_API_KEY: str = ""
    DEEPGRAM_MODEL: str = "nova-2"
    DEEPGRAM_LANGUAGE: str = "hi-Latn"
    
    # TTS
    SARVAM_API_KEY: str = ""
    SARVAM_TTS_MODEL: str = "bulbul:v1"
    SARVAM_TTS_VOICE: str = "meera"
    SARVAM_TTS_LANGUAGE: str = "hi-IN"
    
    # Telephony
    TELEPHONY_PROVIDER: str = "vobiz"
    TELEPHONY_API_KEY: str = ""
    TELEPHONY_BASE_URL: str = "https://api.vobiz.ai/v1"
    TELEPHONY_PHONE_NUMBER: str = "+918065354620"
    TELEPHONY_WEBHOOK_SECRET: str = "mock-secret"
    
    # CRM Google Sheets
    GOOGLE_SHEETS_ENABLED: bool = False
    GOOGLE_SHEETS_SPREADSHEET_ID: str = ""
    GOOGLE_SHEETS_WORKSHEET_NAME: str = "Leads"
    GOOGLE_SERVICE_ACCOUNT_JSON: str = "{}"
    
    # Timeouts
    STT_ENDPOINTING_MS: int = 500
    CUSTOMER_SILENCE_TIMEOUT_MS: int = 7000
    AGENT_RESPONSE_TIMEOUT_MS: int = 15000
    CALL_IDLE_TIMEOUT_MS: int = 30000

settings = Settings()

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] [%(name)s] - %(message)s"
)
logger = logging.getLogger("RiyaAI")

# =====================================================================
# SECTION 2: DOMAIN MODELS & SCHEMAS
# =====================================================================

class SupportedLanguage(str, Enum):
    HINGLISH = "hinglish"
    HINDI = "hindi"
    MARATHI = "marathi"
    ENGLISH = "english"
    UNKNOWN = "unknown"

class ConversationState(str, Enum):
    GREETING = "GREETING"
    DISCOVERY = "DISCOVERY"
    PROPERTY_PRESENTATION = "PROPERTY_PRESENTATION"
    OBJECTION_HANDLING = "OBJECTION_HANDLING"
    VISIT_PROPOSAL = "VISIT_PROPOSAL"
    VISIT_CONFIRMATION = "VISIT_CONFIRMATION"
    HUMAN_HANDOFF = "HUMAN_HANDOFF"
    COMPLETED = "COMPLETED"

class ObjectionCategory(str, Enum):
    PRICE = "PRICE"
    LOCATION = "LOCATION"
    TIMING = "TIMING"
    TRUST = "TRUST"
    FAMILY_APPROVAL = "FAMILY_APPROVAL"
    ALREADY_COMPARING = "ALREADY_COMPARING"
    JUST_EXPLORING = "JUST_EXPLORING"
    INFORMATION_GAP = "INFORMATION_GAP"
    OTHER = "OTHER"

class PropertyConfiguration(BaseModel):
    type: str = Field(..., example="2 BHK")
    carpet_area_sqft: int = Field(..., example=750)
    starting_price_lakhs: float = Field(..., example=85.0)
    available_units: int = Field(default=10)

class PricingInformation(BaseModel):
    base_price_sqft: float = Field(..., example=11000.0)
    starting_price_text: str = Field(..., example="₹85 Lakhs onwards")
    maintenance_charges: str = Field(default="₹4/sqft monthly")
    offers: List[str] = Field(default_factory=lambda: ["Festive discount: Zero stamp duty on bookings this week."])

class AppointmentSlotConfig(BaseModel):
    site_visit_enabled: bool = True
    available_days: List[str] = Field(default_factory=lambda: ["Saturday", "Sunday", "Wednesday"])
    available_slots: List[str] = Field(default_factory=lambda: ["10:00", "13:00", "16:00"])
    timezone: str = "Asia/Kolkata"
    site_address: str = "Mahalaxmi Heights Site Office, Baner-Pashan Link Road, Pune"

class PronunciationOverride(BaseModel):
    original: str = Field(..., example="3BHK")
    spoken_form: str = Field(..., example="three B-H-K")

class AgencyConfigCreate(BaseModel):
    agency_name: str = Field(..., example="Skyline Realty Developers")
    developer_name: str = Field(..., example="Skyline Group")
    contact_phone: str = Field(..., example="+912012345678")
    contact_email: EmailStr = Field(..., example="sales@skylinerealty.com")
    website: str = Field(default="https://skylinerealty.com")
    office_address: str = Field(..., example="101 Business Tower, MG Road, Pune")
    city: str = Field(default="Pune")
    state: str = Field(default="Maharashtra")
    country: str = Field(default="India")
    timezone: str = Field(default="Asia/Kolkata")
    default_language: SupportedLanguage = SupportedLanguage.HINGLISH
    supported_languages: List[SupportedLanguage] = Field(
        default_factory=lambda: [SupportedLanguage.HINGLISH, SupportedLanguage.HINDI, SupportedLanguage.MARATHI, SupportedLanguage.ENGLISH]
    )
    
    project_name: str = Field(..., example="Mahalaxmi Heights")
    project_address: str = Field(..., example="Baner-Pashan Link Road, Baner, Pune")
    project_description: str = Field(..., example="Premium 2 & 3 BHK luxury residences with modern high-rise amenities.")
    project_facts: List[str] = Field(
        default_factory=lambda: ["2 Towers, 22 Floors", "Possession by December 2028", "RERA Approved: P52100099999"]
    )
    project_amenities: List[str] = Field(
        default_factory=lambda: ["Clubhouse", "Infinity Pool", "Children Play Area", "EV Charging", "Fitness Center"]
    )
    configurations: List[PropertyConfiguration] = Field(default_factory=list)
    pricing_information: PricingInformation
    possession_information: str = Field(default="December 2028")
    location_benefits: List[str] = Field(
        default_factory=lambda: ["5 mins from Mumbai-Pune Expressway", "10 mins from Hinjewadi IT Park Phase 1"]
    )
    approved_sales_points: List[str] = Field(
        default_factory=lambda: ["Prime connectivity to IT hubs", "0% Stamp Duty offer for limited period", "High resale & rental yields"]
    )
    site_visit_config: AppointmentSlotConfig = Field(default_factory=AppointmentSlotConfig)
    human_sales_contact: str = Field(default="+919876543211")
    human_transfer_enabled: bool = True
    google_sheet_enabled: bool = False
    google_sheet_spreadsheet_id: str = Field(default="")
    google_sheet_worksheet_name: str = Field(default="Leads")
    pronunciation_dictionary: Dict[str, str] = Field(
        default_factory=lambda: {
            "3BHK": "three B-H-K",
            "2BHK": "two B-H-K",
            "1.5BHK": "one point five B-H-K",
            "Baner": "Baa-ner",
            "Wakad": "Waa-kad",
            "Pashan": "Paa-shaan"
        }
    )

class AgencyConfigResponse(AgencyConfigCreate):
    agency_id: str
    created_at: str
    updated_at: str

class AgencyConfigUpdate(BaseModel):
    agency_name: Optional[str] = None
    project_name: Optional[str] = None
    project_description: Optional[str] = None
    pricing_information: Optional[PricingInformation] = None
    site_visit_config: Optional[AppointmentSlotConfig] = None
    pronunciation_dictionary: Optional[Dict[str, str]] = None

class SpeechPreparationRequest(BaseModel):
    text: str = Field(..., example="Price is ₹1,25,00,000 for a 3BHK unit in Baner with possession in Dec 2028.")
    language: SupportedLanguage = SupportedLanguage.HINGLISH

class SpeechPreparationResponse(BaseModel):
    original_text: str
    speech_ready_text: str
    language: SupportedLanguage
    pronunciation_overrides_applied: List[Dict[str, str]]

class PronunciationTerm(BaseModel):
    original: str
    spoken_form: str

class LLMStructuredResponse(BaseModel):
    reply_text: str = Field(description="Customer facing natural response")
    speech_text: str = Field(description="Normalized text prepared specifically for TTS execution")
    language: SupportedLanguage = SupportedLanguage.HINGLISH
    next_state: ConversationState = ConversationState.DISCOVERY
    intent: str = Field(default="GENERAL_INQUIRY")
    pronunciation_terms: List[PronunciationTerm] = Field(default_factory=list)
    tool_action: Optional[Dict[str, Any]] = None
    needs_clarification: bool = False
    handoff_required: bool = False

class BuyerProfile(BaseModel):
    customer_name: Optional[str] = None
    buying_purpose: Optional[str] = None  # end_user, investment, family
    preferred_config: Optional[str] = None  # 1 BHK, 2 BHK, 3 BHK
    budget_min_lakhs: Optional[float] = None
    budget_max_lakhs: Optional[float] = None
    preferred_location: Optional[str] = None
    buying_timeline: Optional[str] = None
    decision_makers: Optional[str] = None
    already_visited: bool = False
    key_priority: Optional[str] = None  # price, location, amenities, possession

class CallSession(BaseModel):
    call_id: str
    caller_phone: str
    agency_id: str
    preferred_language: SupportedLanguage = SupportedLanguage.HINGLISH
    current_state: ConversationState = ConversationState.GREETING
    last_user_transcript: str = ""
    conversation_history: List[Dict[str, str]] = Field(default_factory=list)
    buyer_profile: BuyerProfile = Field(default_factory=BuyerProfile)
    objection_history: List[str] = Field(default_factory=list)
    last_agent_speech: str = ""
    is_agent_speaking: bool = False
    is_customer_speaking: bool = False
    pending_tool_action: Optional[Dict[str, Any]] = None
    appointment_state: Dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    last_activity_timestamp: float = Field(default_factory=time.time)

class ConversationTestRequest(BaseModel):
    agency_id: str
    user_message: str
    session_id: Optional[str] = None

class LeadRecord(BaseModel):
    lead_id: str
    created_at: str
    updated_at: str
    agency_id: str
    caller_phone: str
    customer_name: Optional[str] = "Prospect"
    preferred_language: str
    buying_purpose: Optional[str] = "Unknown"
    configuration: Optional[str] = "Unspecified"
    budget_min: Optional[float] = None
    budget_max: Optional[float] = None
    location_preference: Optional[str] = "Default"
    buying_timeline: Optional[str] = "Flexible"
    lead_stage: str = "QUALIFIED"
    last_call_id: str
    last_call_summary: str
    objections: List[str] = Field(default_factory=list)
    site_visit_requested: bool = False
    site_visit_status: str = "NOT_SCHEDULED"
    site_visit_date: Optional[str] = None
    site_visit_time: Optional[str] = None
    site_visit_location: Optional[str] = None
    human_handoff_required: bool = False

# =====================================================================
# SECTION 3: IN-MEMORY DATABASE & STATE STORAGE
# =====================================================================

agencies_db: Dict[str, AgencyConfigResponse] = {}
sessions_db: Dict[str, CallSession] = {}
leads_db: Dict[str, LeadRecord] = {}

def get_default_agency_config() -> AgencyConfigResponse:
    ag_id = "default-agency-001"
    if ag_id not in agencies_db:
        create_data = AgencyConfigCreate(
            agency_name="Pinnacle Developers",
            developer_name="Pinnacle Group",
            contact_phone="+912099998888",
            contact_email="sales@pinnacle.com",
            office_address="101 Business Tower, Baner, Pune",
            project_name="Pinnacle Grandeur",
            project_address="Baner High Street, Pune",
            project_description="Ultra-luxury 2 and 3 BHK residential apartments with premium amenities.",
            pricing_information=PricingInformation(
                base_price_sqft=11500.0,
                starting_price_text="₹85 Lakhs onwards",
                maintenance_charges="₹4.5/sqft monthly",
                offers=["Zero Stamp Duty", "Free Modular Kitchen on spot booking"]
            ),
            configurations=[
                PropertyConfiguration(type="2 BHK", carpet_area_sqft=780, starting_price_lakhs=85.0),
                PropertyConfiguration(type="3 BHK", carpet_area_sqft=1150, starting_price_lakhs=125.0)
            ]
        )
        now = datetime.utcnow().isoformat()
        agencies_db[ag_id] = AgencyConfigResponse(
            **create_data.model_dump(),
            agency_id=ag_id,
            created_at=now,
            updated_at=now
        )
    return agencies_db[ag_id]

# Initialize default agency
get_default_agency_config()

# =====================================================================
# SECTION 4: SPEECH PREPARATION ENGINE & PRONUNCIATION NORMALIZER
# =====================================================================

class SpeechPrepEngine:
    """
    Deterministic text normalization layer converting database figures, acronyms,
    currencies, and numbers into spoken-form scripts for Sarvam / TTS.
    """
    
    @staticmethod
    def normalize_numbers_and_currency(text: str, lang: SupportedLanguage) -> str:
        # Currency: ₹1,25,00,000 or ₹1.25 Crore
        def replace_crore_lakh(match):
            val_str = match.group(1).replace(",", "")
            try:
                val = float(val_str)
                if val >= 10000000:
                    crores = val / 10000000
                    if crores.is_integer():
                        return f"{int(crores)} crore rupees"
                    return f"{crores:.2f} crore rupees".rstrip('0').rstrip('.')
                elif val >= 100000:
                    lakhs = val / 100000
                    if lakhs.is_integer():
                        return f"{int(lakhs)} lakh rupees"
                    return f"{lakhs:.2f} lakh rupees".rstrip('0').rstrip('.')
                else:
                    return f"{int(val)} rupees"
            except ValueError:
                return match.group(0)

        # Indian currency Regex pattern
        text = re.sub(r'₹\s*([0-9,]+(?:\.[0-9]+)?)', replace_crore_lakh, text)
        text = re.sub(r'Rs\.?\s*([0-9,]+(?:\.[0-9]+)?)', replace_crore_lakh, text)

        # Property terms normalization
        text = re.sub(r'\b3\s*BHK\b', 'three B-H-K', text, flags=re.IGNORECASE)
        text = re.sub(r'\b2\s*BHK\b', 'two B-H-K', text, flags=re.IGNORECASE)
        text = re.sub(r'\b1\s*BHK\b', 'one B-H-K', text, flags=re.IGNORECASE)
        text = re.sub(r'\b1\.5\s*BHK\b', 'one point five B-H-K', text, flags=re.IGNORECASE)
        text = re.sub(r'\b4\s*BHK\b', 'four B-H-K', text, flags=re.IGNORECASE)

        # Date & time normalization
        text = re.sub(r'\bDec\s+([0-9]{4})\b', r'December \1', text, flags=re.IGNORECASE)
        text = re.sub(r'\b13:00\b', 'one P-M', text)
        text = re.sub(r'\b10:00\b', 'ten A-M', text)
        text = re.sub(r'\b16:00\b', 'four P-M', text)

        # Clean excess spacing
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    @classmethod
    def prepare_speech(
        cls, 
        text: str, 
        lang: SupportedLanguage = SupportedLanguage.HINGLISH,
        agency_dict: Optional[Dict[str, str]] = None
    ) -> Tuple[str, List[Dict[str, str]]]:
        
        applied_overrides = []
        normalized = cls.normalize_numbers_and_currency(text, lang)
        
        # Apply agency specific custom pronunciation overrides
        if agency_dict:
            for term, spoken in agency_dict.items():
                pattern = re.compile(rf'\b{re.escape(term)}\b', re.IGNORECASE)
                if pattern.search(normalized):
                    normalized = pattern.sub(spoken, normalized)
                    applied_overrides.append({"original": term, "spoken_form": spoken})

        return normalized, applied_overrides

# =====================================================================
# SECTION 5: EXTERNAL PROVIDER ADAPTERS
# =====================================================================

class LLMAdapter:
    """OpenAI-compatible LLM client targeting GPT OSS 20B or direct configured endpoints."""
    
    @staticmethod
    async def generate_response(
        system_prompt: str,
        user_message: str,
        history: List[Dict[str, str]]
    ) -> LLMStructuredResponse:
        
        if settings.LLM_API_KEY == "mock-llm-key":
            logger.info("Using LLM Mock Adapter Engine.")
            return LLMAdapter._mock_llm_response(user_message, history)

        headers = {
            "Authorization": f"Bearer {settings.LLM_API_KEY}",
            "Content-Type": "application/json"
        }
        
        messages = [{"role": "system", "content": system_prompt}]
        for item in history[-6:]:  # Maintain reasonable context window
            messages.append({"role": item["role"], "content": item["content"]})
        messages.append({"role": "user", "content": user_message})

        payload = {
            "model": settings.LLM_MODEL,
            "messages": messages,
            "temperature": 0.3,
            "response_format": {"type": "json_object"}
        }

        async with httpx.AsyncClient(timeout=15.0) as client:
            try:
                response = await client.post(
                    f"{settings.LLM_BASE_URL.rstrip('/')}/chat/completions",
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()
                data = response.json()
                raw_content = data["choices"][0]["message"]["content"]
                parsed_json = json.loads(raw_content)
                return LLMStructuredResponse(**parsed_json)
            except Exception as e:
                logger.error(f"LLM Provider call failed: {str(e)}. Falling back to safe recovery mode.")
                return LLMStructuredResponse(
                    reply_text="Ji sir, main aapki baat samajh rahi hoon. Ek moment dijiye, main details verify kar leti hoon.",
                    speech_text="Ji sir, main aapki baat samajh rahi hoon. Ek moment dijiye, main details verify kar leti hoon.",
                    language=SupportedLanguage.HINGLISH,
                    next_state=ConversationState.DISCOVERY,
                    intent="ERROR_RECOVERY"
                )

    @staticmethod
    def _mock_llm_response(user_message: str, history: List[Dict[str, str]]) -> LLMStructuredResponse:
        msg = user_message.lower()
        if "price" in msg or "cost" in msg or "kitna" in msg or "budget" in msg:
            return LLMStructuredResponse(
                reply_text="Pinnacle Grandeur mein 2 BHK starting ₹85 Lakhs se hai, aur 3 BHK ₹1.25 Crore se. Aapka preferred budget range kitna hai?",
                speech_text="Pinnacle Grandeur mein two B-H-K starting eighty-five lakh rupees se hai, aur three B-H-K one crore twenty-five lakh rupees se. Aapka preferred budget range kitna hai?",
                language=SupportedLanguage.HINGLISH,
                next_state=ConversationState.PROPERTY_PRESENTATION,
                intent="PRICE_INQUIRY"
            )
        elif "visit" in msg or "saturday" in msg or "sunday" in msg or "time" in msg:
            return LLMStructuredResponse(
                reply_text="Aap Saturday ko 1 PM site visit karna convenient samjhenge? Main aapke liye appointment reserve kar deti hoon.",
                speech_text="Aap Saturday ko one P-M site visit karna convenient samjhenge? Main aapke liye appointment reserve kar deti hoon.",
                language=SupportedLanguage.HINGLISH,
                next_state=ConversationState.VISIT_PROPOSAL,
                intent="SITE_VISIT_REQUEST",
                tool_action={"tool_name": "check_visit_availability", "arguments": {"day": "Saturday", "time": "13:00"}}
            )
        elif "marathi" in msg or "namaskar" in msg:
            return LLMStructuredResponse(
                reply_text="Namaskar! Pinnacle Grandeur madhe aaple swagat aahe. Aaplyala 2 BHK paahije ki 3 BHK?",
                speech_text="Namaskar! Pinnacle Grandeur madhe aaple swagat aahe. Aaplyala two B-H-K paahije ki three B-H-K?",
                language=SupportedLanguage.MARATHI,
                next_state=ConversationState.DISCOVERY,
                intent="LANGUAGE_SWITCH"
            )
        else:
            return LLMStructuredResponse(
                reply_text="Namaste sir, main Riya bol rahi hoon Pinnacle Group se. Aapne hamare Baner project ke baare mein inquiry ki thi. Aap apne family ke liye dekh rahe hain ya investment purpose se?",
                speech_text="Namaste sir, main Riya bol rahi hoon Pinnacle Group se. Aapne hamare Baner project ke baare mein inquiry ki thi. Aap apne family ke liye dekh rahe hain ya investment purpose se?",
                language=SupportedLanguage.HINGLISH,
                next_state=ConversationState.DISCOVERY,
                intent="GREETING"
            )

class SarvamTTSAdapter:
    """Text-To-Speech Adapter calling Sarvam AI speech synthesis API."""

    @staticmethod
    async def synthesize(text: str, language: str = "hi-IN", voice: str = "meera") -> bytes:
        if settings.SARVAM_API_KEY == "mock-sarvam-key":
            logger.info("Executing Sarvam TTS Mock Adapter mode.")
            return b"RIYA_MOCK_TTS_AUDIO_BYTES_PCM16"

        headers = {
            "api-subscription-key": settings.SARVAM_API_KEY,
            "Content-Type": "application/json"
        }
        payload = {
            "inputs": [text],
            "target_language_code": language,
            "speaker": voice,
            "model": settings.SARVAM_TTS_MODEL,
            "enable_preprocessing": True
        }

        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                res = await client.post("https://api.sarvam.ai/text-to-speech", headers=headers, json=payload)
                res.raise_for_status()
                # Sarvam returns audio base64 encoded or raw byte stream depending on endpoint
                data = res.json()
                import base64
                return base64.b64decode(data.get("audios", [""])[0])
            except Exception as e:
                logger.error(f"Sarvam TTS API failed: {str(e)}")
                return b"FALLBACK_AUDIO_BYTES"

class GoogleSheetsCRMAdapter:
    """Google Sheets CRM Integration with Mock operational fallback."""

    @staticmethod
    async def sync_lead(lead: LeadRecord) -> bool:
        if not settings.GOOGLE_SHEETS_ENABLED:
            logger.info(f"[MOCK CRM] Lead {lead.lead_id} recorded successfully in Mock Mode.")
            return True

        try:
            import gspread
            from google.oauth2.service_account import Credentials
            
            creds_json = json.loads(settings.GOOGLE_SERVICE_ACCOUNT_JSON)
            scopes = ["https://www.googleapis.com/auth/spreadsheets"]
            credentials = Credentials.from_service_account_info(creds_json, scopes=scopes)
            client = gspread.authorize(credentials)
            
            sheet = client.open_by_key(settings.GOOGLE_SHEETS_SPREADSHEET_ID).worksheet(settings.GOOGLE_SHEETS_WORKSHEET_NAME)
            
            row = [
                lead.lead_id,
                lead.created_at,
                lead.updated_at,
                lead.agency_id,
                lead.caller_phone,
                lead.customer_name or "Unknown",
                lead.preferred_language,
                lead.buying_purpose or "",
                lead.configuration or "",
                lead.budget_min or 0,
                lead.budget_max or 0,
                lead.location_preference or "",
                lead.buying_timeline or "",
                lead.lead_stage,
                lead.last_call_id,
                lead.last_call_summary,
                ", ".join(lead.objections),
                str(lead.site_visit_requested),
                lead.site_visit_status,
                lead.site_visit_date or "",
                lead.site_visit_time or "",
                lead.site_visit_location or "",
                str(lead.human_handoff_required)
            ]
            
            # Search for existing lead by caller phone
            cell = sheet.find(lead.caller_phone)
            if cell:
                sheet.update(f"A{cell.row}:W{cell.row}", [row])
                logger.info(f"Updated existing Google Sheet row {cell.row} for phone {lead.caller_phone}")
            else:
                sheet.append_row(row)
                logger.info(f"Appended new Google Sheet lead row for phone {lead.caller_phone}")
            return True
        except Exception as e:
            logger.error(f"Failed to update Google Sheets CRM: {str(e)}")
            return False

# =====================================================================
# SECTION 6: SERVER-SIDE TOOL DEFINITIONS & ENGINE
# =====================================================================

class SalesToolEngine:
    """Tool execution harness enforcing strict validated tool calls."""
    
    @staticmethod
    async def execute_tool(
        tool_name: str, 
        arguments: Dict[str, Any], 
        session: CallSession, 
        agency: AgencyConfigResponse
    ) -> Dict[str, Any]:
        
        logger.info(f"Executing Server-Side Tool: {tool_name} with args: {arguments}")
        
        if tool_name == "get_project_information":
            return {
                "project_name": agency.project_name,
                "description": agency.project_description,
                "amenities": agency.project_amenities,
                "facts": agency.project_facts
            }

        elif tool_name == "get_price_information":
            return {
                "starting_price": agency.pricing_information.starting_price_text,
                "configurations": [c.model_dump() for c in agency.configurations],
                "offers": agency.pricing_information.offers
            }

        elif tool_name == "check_visit_availability":
            day = arguments.get("day", "Saturday")
            time_slot = arguments.get("time", "13:00")
            
            valid_day = day in agency.site_visit_config.available_days
            valid_slot = time_slot in agency.site_visit_config.available_slots
            
            return {
                "available": valid_day and valid_slot,
                "requested_day": day,
                "requested_slot": time_slot,
                "site_address": agency.site_visit_config.site_address
            }

        elif tool_name == "book_site_visit":
            day = arguments.get("day", "Saturday")
            time_slot = arguments.get("time", "13:00")
            cust_name = arguments.get("customer_name", session.buyer_profile.customer_name or "Valued Client")
            
            session.appointment_state = {
                "status": "CONFIRMED",
                "day": day,
                "time": time_slot,
                "location": agency.site_visit_config.site_address
            }
            
            # Sync update to CRM
            lead = LeadRecord(
                lead_id=f"lead_{session.caller_phone.replace('+', '')}",
                created_at=session.created_at,
                updated_at=datetime.utcnow().isoformat(),
                agency_id=agency.agency_id,
                caller_phone=session.caller_phone,
                customer_name=cust_name,
                preferred_language=session.preferred_language.value,
                buying_purpose=session.buyer_profile.buying_purpose,
                configuration=session.buyer_profile.preferred_config,
                last_call_id=session.call_id,
                last_call_summary=f"Site visit booked for {day} at {time_slot}.",
                site_visit_requested=True,
                site_visit_status="CONFIRMED",
                site_visit_date=day,
                site_visit_time=time_slot,
                site_visit_location=agency.site_visit_config.site_address
            )
            leads_db[lead.lead_id] = lead
            await GoogleSheetsCRMAdapter.sync_lead(lead)
            
            return {
                "booking_status": "SUCCESS",
                "confirmation_details": {
                    "customer_name": cust_name,
                    "day": day,
                    "time": time_slot,
                    "location": agency.site_visit_config.site_address
                }
            }

        elif tool_name == "request_human_handoff":
            session.current_state = ConversationState.HUMAN_HANDOFF
            return {
                "status": "HANDOFF_INITIATED",
                "sales_contact": agency.human_sales_contact
            }

        else:
            return {"status": "ERROR", "message": f"Unknown tool name: {tool_name}"}

# =====================================================================
# SECTION 7: SYSTEM PROMPT ENGINE & ORCHESTRATION
# =====================================================================

def build_riya_system_prompt(agency: AgencyConfigResponse, session: CallSession) -> str:
    return f"""
You are Riya, a female AI real estate sales executive working directly for {agency.developer_name}.
Your single focus and business objective is: CONVERT PROPERTY INQUIRIES INTO CONFIRMED REAL-LIFE SITE VISITS.

PERSONALITY & VOICE:
- Warm, professional, polite, confident, Indian sales professional tone.
- Conversational, natural, non-robotic.
- Speak naturally in Hinglish, Hindi, Marathi, or English based on buyer preference.
- Current language mode: {session.preferred_language.value.upper()}.

PROJECT CONTEXT:
- Project Name: {agency.project_name}
- Developer: {agency.developer_name}
- Address/Location: {agency.project_address}
- Pricing: {agency.pricing_information.starting_price_text}
- Configurations Available: {', '.join([c.type for c in agency.configurations])}
- Key Amenities: {', '.join(agency.project_amenities[:4])}
- Key Location Benefits: {', '.join(agency.location_benefits[:2])}
- Approved Special Offers: {', '.join(agency.pricing_information.offers)}

CURRENT SALES STATE: {session.current_state.value}
BUYER PROFILE SO FAR:
- Preferred Config: {session.buyer_profile.preferred_config or 'Unknown'}
- Budget Range: {session.buyer_profile.budget_min_lakhs or 'Unspecified'}
- Purpose: {session.buyer_profile.buying_purpose or 'Unknown'}

SALES STRATEGY RULES:
1. Speak in short 1-2 sentence spoken conversational bites.
2. Answer the user's specific question, then pivot towards discovery or site-visit proposal.
3. If user expresses price concern, explain value, mention flexible financing or current zero stamp duty offer, and suggest site visit.
4. When user shows interest, propose a specific slot: "Aap Saturday ko 1 PM site visit ke liye comfortable samjhenge?"
5. Do NOT invent fake discounts, legal guarantees, or unknown project details.
6. Return structured JSON strictly adhering to schema.

JSON RESPONSE FORMAT:
{{
  "reply_text": "<Customer facing response text>",
  "speech_text": "<Normalized text prepared specifically for TTS execution>",
  "language": "hinglish|hindi|marathi|english",
  "next_state": "GREETING|DISCOVERY|PROPERTY_PRESENTATION|OBJECTION_HANDLING|VISIT_PROPOSAL|VISIT_CONFIRMATION|HUMAN_HANDOFF|COMPLETED",
  "intent": "<INQUIRY_INTENT>",
  "pronunciation_terms": [{{"original": "3BHK", "spoken_form": "three B-H-K"}}],
  "tool_action": null | {{"tool_name": "<tool>", "arguments": {{}}}},
  "needs_clarification": false,
  "handoff_required": false
}}
"""

# =====================================================================
# SECTION 8: CONVERSATION MANAGER & CALL ORCHESTRATION
# =====================================================================

class CallSessionManager:
    
    @staticmethod
    def get_or_create_session(call_id: str, agency_id: str, caller_phone: str) -> CallSession:
        if call_id not in sessions_db:
            sessions_db[call_id] = CallSession(
                call_id=call_id,
                agency_id=agency_id,
                caller_phone=caller_phone
            )
        return sessions_db[call_id]

    @staticmethod
    async def process_user_turn(call_id: str, user_transcript: str) -> Tuple[LLMStructuredResponse, bytes]:
        start_time = time.time()
        session = sessions_db.get(call_id)
        if not session:
            raise HTTPException(status_code=404, detail="Call session not found")

        agency = agencies_db.get(session.agency_id) or get_default_agency_config()
        session.last_user_transcript = user_transcript
        session.last_activity_timestamp = time.time()
        
        # Build prompt & query LLM
        prompt = build_riya_system_prompt(agency, session)
        llm_resp = await LLMAdapter.generate_response(prompt, user_transcript, session.conversation_history)

        # Apply Server Speech Preparation Engine
        prep_speech, _ = SpeechPrepEngine.prepare_speech(
            llm_resp.speech_text or llm_resp.reply_text,
            lang=llm_resp.language,
            agency_dict=agency.pronunciation_dictionary
        )
        llm_resp.speech_text = prep_speech

        # Execute Tool if specified
        if llm_resp.tool_action and llm_resp.tool_action.get("tool_name"):
            tool_res = await SalesToolEngine.execute_tool(
                llm_resp.tool_action["tool_name"],
                llm_resp.tool_action.get("arguments", {}),
                session,
                agency
            )
            logger.info(f"Tool Result for {llm_resp.tool_action['tool_name']}: {tool_res}")

        # Update Session History & State
        session.current_state = llm_resp.next_state
        session.preferred_language = llm_resp.language
        session.conversation_history.append({"role": "user", "content": user_transcript})
        session.conversation_history.append({"role": "assistant", "content": llm_resp.reply_text})
        session.last_agent_speech = llm_resp.speech_text

        # Generate Audio via TTS
        audio_bytes = await SarvamTTSAdapter.synthesize(
            llm_resp.speech_text, 
            language="hi-IN" if llm_resp.language in [SupportedLanguage.HINDI, SupportedLanguage.HINGLISH] else "mr-IN"
        )

        total_latency = (time.time() - start_time) * 1000
        logger.info(f"[CALL {call_id}] Processing Turn Complete | Latency: {total_latency:.2f}ms | Next State: {session.current_state.value}")
        
        return llm_resp, audio_bytes

# =====================================================================
# SECTION 9: FASTAPI REST ENDPOINTS & WEBSOCKET ROUTING
# =====================================================================

app = FastAPI(
    title="Riya — AI Real Estate Sales Executive API",
    description="Multilingual, conversion-focused voice AI sales agent designed for residential property developers.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["System"])
async def root():
    return {
        "identity": "Riya — AI Real Estate Sales Executive",
        "status": "online",
        "version": "1.0.0",
        "docs": "/docs",
        "telephony_provider": settings.TELEPHONY_PROVIDER
    }

@app.get("/health", tags=["System"])
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "providers": {
            "llm_model": settings.LLM_MODEL,
            "stt_provider": "Deepgram",
            "tts_provider": "Sarvam AI",
            "telephony": settings.TELEPHONY_PROVIDER,
            "google_sheets_crm": "Enabled" if settings.GOOGLE_SHEETS_ENABLED else "Mock Mode"
        }
    }

# --- AGENCY MANAGEMENT ENDPOINTS ---

@app.post("/agencies", response_model=AgencyConfigResponse, tags=["Agency Management"], status_code=status.HTTP_201_CREATED)
async def create_agency(config: AgencyConfigCreate):
    agency_id = f"agency-{uuid.uuid4().hex[:8]}"
    now = datetime.utcnow().isoformat()
    res = AgencyConfigResponse(
        **config.model_dump(),
        agency_id=agency_id,
        created_at=now,
        updated_at=now
    )
    agencies_db[agency_id] = res
    logger.info(f"Created new real estate agency configuration: {agency_id} ({config.agency_name})")
    return res

@app.get("/agencies/{agency_id}", response_model=AgencyConfigResponse, tags=["Agency Management"])
async def get_agency(agency_id: str):
    if agency_id not in agencies_db:
        raise HTTPException(status_code=404, detail="Agency configuration not found")
    return agencies_db[agency_id]

@app.put("/agencies/{agency_id}", response_model=AgencyConfigResponse, tags=["Agency Management"])
async def update_agency(agency_id: str, config: AgencyConfigCreate):
    if agency_id not in agencies_db:
        raise HTTPException(status_code=404, detail="Agency configuration not found")
    now = datetime.utcnow().isoformat()
    updated = AgencyConfigResponse(
        **config.model_dump(),
        agency_id=agency_id,
        created_at=agencies_db[agency_id].created_at,
        updated_at=now
    )
    agencies_db[agency_id] = updated
    return updated

@app.patch("/agencies/{agency_id}", response_model=AgencyConfigResponse, tags=["Agency Management"])
async def patch_agency(agency_id: str, patch_data: AgencyConfigUpdate):
    if agency_id not in agencies_db:
        raise HTTPException(status_code=404, detail="Agency configuration not found")
    existing = agencies_db[agency_id].model_dump()
    updates = patch_data.model_dump(exclude_unset=True)
    existing.update(updates)
    existing["updated_at"] = datetime.utcnow().isoformat()
    updated = AgencyConfigResponse(**existing)
    agencies_db[agency_id] = updated
    return updated

# --- PRONUNCIATION ENGINE ENDPOINTS ---

@app.post("/agencies/{agency_id}/pronunciation", tags=["Speech Engine"])
async def add_pronunciation_override(agency_id: str, override: PronunciationOverride):
    agency = agencies_db.get(agency_id)
    if not agency:
        raise HTTPException(status_code=404, detail="Agency not found")
    agency.pronunciation_dictionary[override.original] = override.spoken_form
    agency.updated_at = datetime.utcnow().isoformat()
    return {"status": "success", "pronunciation_dictionary": agency.pronunciation_dictionary}

@app.get("/agencies/{agency_id}/pronunciation", tags=["Speech Engine"])
async def get_pronunciation_dictionary(agency_id: str):
    agency = agencies_db.get(agency_id)
    if not agency:
        raise HTTPException(status_code=404, detail="Agency not found")
    return {"agency_id": agency_id, "pronunciation_dictionary": agency.pronunciation_dictionary}

@app.post("/agencies/{agency_id}/test-speech", response_model=SpeechPreparationResponse, tags=["Speech Engine"])
async def test_speech_preparation(agency_id: str, req: SpeechPreparationRequest):
    agency = agencies_db.get(agency_id) or get_default_agency_config()
    speech_text, applied = SpeechPrepEngine.prepare_speech(
        req.text, 
        lang=req.language, 
        agency_dict=agency.pronunciation_dictionary
    )
    return SpeechPreparationResponse(
        original_text=req.text,
        speech_ready_text=speech_text,
        language=req.language,
        pronunciation_overrides_applied=applied
    )

@app.post("/agencies/{agency_id}/test-conversation", tags=["Testing & Simulation"])
async def test_conversation_turn(agency_id: str, req: ConversationTestRequest):
    agency = agencies_db.get(agency_id)
    if not agency:
        raise HTTPException(status_code=404, detail="Agency not found")
    
    session_id = req.session_id or f"test-session-{uuid.uuid4().hex[:6]}"
    session = CallSessionManager.get_or_create_session(session_id, agency_id, "+919876543210")
    
    llm_resp, audio_bytes = await CallSessionManager.process_user_turn(session_id, req.user_message)
    
    return {
        "session_id": session_id,
        "current_state": session.current_state.value,
        "llm_response": llm_resp.model_dump(),
        "audio_bytes_generated": len(audio_bytes),
        "buyer_profile": session.buyer_profile.model_dump()
    }

# --- CRM & APPOINTMENT ENDPOINTS ---

@app.get("/agencies/{agency_id}/leads", tags=["CRM Integration"])
async def get_agency_leads(agency_id: str):
    results = [lead for lead in leads_db.values() if lead.agency_id == agency_id]
    return {"agency_id": agency_id, "count": len(results), "leads": results}

@app.post("/agencies/{agency_id}/appointments/check", tags=["Appointments"])
async def check_appointment_slot(agency_id: str, day: str, time_slot: str):
    agency = agencies_db.get(agency_id) or get_default_agency_config()
    is_day_valid = day in agency.site_visit_config.available_days
    is_slot_valid = time_slot in agency.site_visit_config.available_slots
    return {
        "agency_id": agency_id,
        "requested_day": day,
        "requested_time": time_slot,
        "available": is_day_valid and is_slot_valid,
        "site_address": agency.site_visit_config.site_address
    }

# --- TELEPHONY WEBHOOKS & REAL-TIME AUDIO STREAMING ---

@app.post("/telephony/inbound", tags=["Telephony Integration"])
async def handle_inbound_call(request: Request):
    """Webhook handling incoming calls from Vobiz/Telphony/Twilio providers."""
    body = await request.json() if request.headers.get("content-type") == "application/json" else {}
    call_id = body.get("call_id") or f"vobiz-{uuid.uuid4().hex[:8]}"
    caller_phone = body.get("from") or "+919876543210"
    agency_id = body.get("agency_id") or "default-agency-001"
    
    session = CallSessionManager.get_or_create_session(call_id, agency_id, caller_phone)
    logger.info(f"Incoming call initialized: {call_id} from {caller_phone}")

    # Standard XML/JSON response instructing provider to connect stream
    return {
        "status": "accept",
        "call_id": call_id,
        "websocket_url": f"ws://{settings.APP_HOST}:{settings.APP_PORT}/telephony/media-stream?call_id={call_id}"
    }

@app.websocket("/telephony/media-stream")
async def media_stream_websocket(websocket: WebSocket, call_id: Optional[str] = None):
    """
    Bidirectional streaming WebSocket endpoint handling telephony PCM audio frames,
    Deepgram STT transcription events, and Sarvam TTS audio output.
    """
    await websocket.accept()
    cid = call_id or f"ws-call-{uuid.uuid4().hex[:6]}"
    session = CallSessionManager.get_or_create_session(cid, "default-agency-001", "+919876543210")
    logger.info(f"Telephony Media Stream Connected for Session: {cid}")

    try:
        # Initial greeting execution on connection
        greeting_text = "Namaste sir, main Riya bol rahi hoon Pinnacle Group se. Aapne hamare Baner project ke baare mein inquiry ki thi. Main aapki kis tarah help kar sakti hoon?"
        audio_payload = await SarvamTTSAdapter.synthesize(greeting_text)
        await websocket.send_bytes(audio_payload)

        while True:
            # Receive audio frame or event from telephony provider
            message = await websocket.receive()
            if "text" in message:
                event_data = json.loads(message["text"])
                if event_data.get("event") == "user_speech_final":
                    user_transcript = event_data.get("transcript", "")
                    if user_transcript.strip():
                        llm_resp, reply_audio = await CallSessionManager.process_user_turn(cid, user_transcript)
                        await websocket.send_json({"event": "agent_transcript", "text": llm_resp.reply_text})
                        await websocket.send_bytes(reply_audio)
                elif event_data.get("event") == "interruption":
                    logger.info(f"Customer interrupted Riya on session {cid}. Halting current output.")
            elif "bytes" in message:
                # Handle raw audio chunk stream from phone line
                pass

    except WebSocketDisconnect:
        logger.info(f"Telephony Media Stream Disconnected for Session: {cid}")
    except Exception as e:
        logger.error(f"Error in Media Stream WebSocket: {str(e)}")
        await websocket.close()

# =====================================================================
# SECTION 10: APPLICATION ENTRYPOINT
# =====================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        reload=(settings.APP_ENV == "development")
    )