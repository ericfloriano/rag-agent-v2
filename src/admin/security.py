"""
Security module for the Admin Panel.
Implements OWASP recommendations:
- Rate Limiting (SlowAPI)
- Turnstile Captcha verification
- HttpOnly Cookie session validation (Hash-based)
"""

import hashlib
import logging
import httpx
from fastapi import Request, HTTPException, status, Depends

from src.config import ADMIN_SECRET_KEY, TURNSTILE_SECRET

logger = logging.getLogger(__name__)

# --- Rate Limiter ---
from slowapi import Limiter
from slowapi.util import get_remote_address

# IP-based rate limiter to prevent Brute Force and DDoS attempts
limiter = Limiter(key_func=get_remote_address)


# --- Session Validation ---
# SHA256 hash of the password used as cookie value to avoid exposing the original secret
EXPECTED_COOKIE_HASH = hashlib.sha256(ADMIN_SECRET_KEY.encode("utf-8")).hexdigest()
COOKIE_NAME = "recare_admin_session"


async def verify_admin_session(request: Request):
    """
    FastAPI Dependency to protect admin routes.
    Checks if the HttpOnly cookie is present and valid.
    """
    cookie_value = request.cookies.get(COOKIE_NAME)
    
    if not cookie_value or cookie_value != EXPECTED_COOKIE_HASH:
        logger.warning(f"🛡️ Unauthorized admin access attempt from {request.client.host}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session expired or invalid.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return True


# --- Cloudflare Turnstile ---
async def verify_turnstile(token: str, ip: str) -> bool:
    """
    Call Cloudflare Turnstile API to verify the challenge token.
    If TURNSTILE_SECRET is not set in .env, bypass verification (useful for local dev).
    """
    if not TURNSTILE_SECRET:
        logger.warning("🛡️ Turnstile secret not configured. Bypassing captcha.")
        return True

    if not token:
        logger.warning("🛡️ Missing Turnstile token.")
        return False

    url = "https://challenges.cloudflare.com/turnstile/v0/siteverify"
    payload = {
        "secret": TURNSTILE_SECRET,
        "response": token,
        "remoteip": ip
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, data=payload, timeout=5.0)
            result = response.json()
            
            if result.get("success"):
                return True
            else:
                logger.warning(f"🛡️ Turnstile verification failed: {result.get('error-codes')}")
                return False
    except Exception as e:
        logger.error(f"❌ Turnstile API Error: {e}")
        return False
