"""
LLM Client Module
Abstraction layer supporting mock, anthropic, and openai backends.
"""

import json
import os
import re
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class LLMClient:
    """Multi-backend LLM client for intervention decision-making."""

    def __init__(self, mode=None):
        with open(os.path.join(ROOT, "config", "llm_config.yaml"), "r") as f:
            config = yaml.safe_load(f)

        self.mode = mode or os.environ.get("LLM_MODE", config["llm"]["mode"])
        self.model_name = config["llm"]["model_name"]
        self.max_retries = config["llm"]["max_retries"]

    def invoke(self, system_prompt, user_prompt):
        """Invoke the LLM with system and user prompts."""
        try:
            if self.mode == "mock":
                return self._mock_response(user_prompt)
            elif self.mode == "anthropic":
                return self._call_anthropic(system_prompt, user_prompt)
            elif self.mode == "openai":
                return self._call_openai(system_prompt, user_prompt)
            else:
                return self._mock_response(user_prompt)
        except Exception as e:
            print(f"  LLM call failed ({self.mode}): {e}")
            return self._mock_response(user_prompt)

    def _mock_response(self, prompt):
        """Return deterministic response based on risk score bands."""
        # Parse risk score from prompt
        risk_score = 0.5
        risk_match = re.search(r'Risk Score:\s*([\d.]+)', prompt)
        if risk_match:
            risk_score = float(risk_match.group(1))

        if risk_score >= 0.80:
            response = {
                "intervention": "PAYMENT_HOLIDAY",
                "channel": "CALL",
                "reason": "Customer shows severe financial distress with multiple risk indicators",
                "message": "We noticed some changes in your payment patterns. We'd like to help - you may be eligible for a temporary payment pause. Please call us."
            }
        elif risk_score >= 0.65:
            response = {
                "intervention": "RESTRUCTURING_OFFER",
                "channel": "EMAIL",
                "reason": "Customer shows significant stress signals, needs repayment plan adjustment",
                "message": "We're here to help you manage your finances better. Let's explore flexible repayment options that work for you. Reach out anytime."
            }
        elif risk_score >= 0.55:
            response = {
                "intervention": "FINANCIAL_COUNSELING",
                "channel": "SMS",
                "reason": "Early stress signals detected, preventive counseling recommended",
                "message": "We care about your financial wellness. Our free counseling service can help plan your finances better. Book a session today."
            }
        else:
            response = {
                "intervention": "SMS_OUTREACH",
                "channel": "SMS",
                "reason": "Low-level concern flagged, gentle check-in appropriate",
                "message": "Stay on top of your finances with our free budgeting tools. We're here to support you every step of the way."
            }

        return json.dumps(response)

    def _call_anthropic(self, system_prompt, user_prompt):
        """Call Anthropic API."""
        try:
            from langchain_anthropic import ChatAnthropic
            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            if not api_key:
                return self._mock_response(user_prompt)

            model = ChatAnthropic(model_name="claude-3-haiku-20240307",
                                   api_key=api_key)
            response = model.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ])
            return response.content
        except Exception as e:
            print(f"  Anthropic API error: {e}")
            return self._mock_response(user_prompt)

    def _call_openai(self, system_prompt, user_prompt):
        """Call OpenAI API."""
        try:
            from langchain_openai import ChatOpenAI
            api_key = os.environ.get("OPENAI_API_KEY", "")
            if not api_key:
                return self._mock_response(user_prompt)

            model = ChatOpenAI(model_name="gpt-4o-mini", api_key=api_key)
            response = model.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ])
            return response.content
        except Exception as e:
            print(f"  OpenAI API error: {e}")
            return self._mock_response(user_prompt)
