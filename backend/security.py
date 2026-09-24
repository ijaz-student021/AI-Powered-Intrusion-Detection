import hmac
import os
import secrets

from fastapi import Header, HTTPException

API_KEY = os.environ.get("API_KEY")
if not API_KEY:
    API_KEY = secrets.token_urlsafe(32)
    print(f"No API_KEY set. Generated a temporary key for this session: {API_KEY}")
    print("Set the API_KEY environment variable to keep a stable key across restarts.")


def require_api_key(x_api_key: str = Header(default="")):
    if not hmac.compare_digest(x_api_key, API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")
