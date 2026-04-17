"""
PII Masking Module
Redacts personally identifiable information before LLM calls.
"""

import re


class PIIMasker:
    """Masks PII in text and data before sending to LLM."""

    # Common Indian name patterns
    NAME_PATTERN = re.compile(
        r'\b(?:[A-Z][a-z]+ ){1,2}[A-Z][a-z]+\b'
    )

    # Phone numbers (Indian)
    PHONE_PATTERN = re.compile(
        r'\b(?:\+91[-\s]?)?[6-9]\d{9}\b'
    )

    # Email
    EMAIL_PATTERN = re.compile(
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    )

    # Aadhaar-like numbers
    AADHAAR_PATTERN = re.compile(
        r'\b\d{4}\s?\d{4}\s?\d{4}\b'
    )

    # PAN
    PAN_PATTERN = re.compile(
        r'\b[A-Z]{5}\d{4}[A-Z]\b'
    )

    @staticmethod
    def mask_customer_profile(profile: dict) -> dict:
        """Create an anonymised customer profile safe for LLM consumption."""
        masked = {}

        # Keep customer_id (not PII)
        masked["customer_id"] = profile.get("customer_id", "UNKNOWN")

        # Anonymise salary to range
        salary = profile.get("monthly_salary", 0)
        if salary < 25000:
            masked["salary_range"] = "below_25k"
        elif salary < 50000:
            masked["salary_range"] = "25k_to_50k"
        elif salary < 100000:
            masked["salary_range"] = "50k_to_100k"
        else:
            masked["salary_range"] = "above_100k"

        # Anonymise age to bracket
        age = profile.get("age", 0)
        if age < 30:
            masked["age_bracket"] = "under_30"
        elif age < 45:
            masked["age_bracket"] = "30_to_45"
        elif age < 60:
            masked["age_bracket"] = "45_to_60"
        else:
            masked["age_bracket"] = "above_60"

        # Keep occupation (not exact PII)
        masked["occupation"] = profile.get("occupation", "Unknown")

        # Keep city (general location, not exact address)
        masked["city"] = profile.get("city", "Unknown")

        # Financial details (ranges, not exact)
        credit_score = profile.get("credit_score", 0)
        if credit_score < 600:
            masked["credit_tier"] = "poor"
        elif credit_score < 700:
            masked["credit_tier"] = "fair"
        elif credit_score < 750:
            masked["credit_tier"] = "good"
        else:
            masked["credit_tier"] = "excellent"

        masked["has_active_loan"] = profile.get("loan_amount", 0) > 0
        masked["product_type"] = profile.get("product_type", "Unknown")

        # Explicitly exclude name
        # Do NOT include: name, phone, email, aadhaar, pan, exact salary

        return masked

    @staticmethod
    def redact_text(text: str) -> str:
        """Remove PII patterns from text."""
        text = PIIMasker.PHONE_PATTERN.sub("[PHONE_REDACTED]", text)
        text = PIIMasker.EMAIL_PATTERN.sub("[EMAIL_REDACTED]", text)
        text = PIIMasker.AADHAAR_PATTERN.sub("[ID_REDACTED]", text)
        text = PIIMasker.PAN_PATTERN.sub("[PAN_REDACTED]", text)
        return text

    @staticmethod
    def redact_name_from_message(message: str, customer_name: str = None) -> str:
        """Remove customer name from outreach messages."""
        if customer_name:
            # Remove exact name
            message = message.replace(customer_name, "valued customer")
            # Also remove first name and last name separately
            parts = customer_name.split()
            for part in parts:
                if len(part) > 2:  # Avoid replacing very short words
                    message = message.replace(part, "")
        # Clean up double spaces
        message = re.sub(r'\s+', ' ', message).strip()
        return message
