import os
import logging
from typing import Optional

import resend
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

RESEND_API_KEY = os.getenv("RESEND_API_KEY", "").strip()
MAIL_FROM = os.getenv("MAIL_FROM", "noreply@mindfold3d.com")
MAIL_REPLY_TO = os.getenv("MAIL_REPLY_TO", "support@mindfold3d.com")
APP_BASE_URL = os.getenv("APP_BASE_URL", "http://localhost:3001").rstrip("/")

if RESEND_API_KEY:
    resend.api_key = RESEND_API_KEY


def email_enabled() -> bool:
    return bool(RESEND_API_KEY)


def send_password_reset_email(to_email: str, reset_token: str) -> bool:
    if not email_enabled():
        logger.warning("Resend API key not set; skipping email send")
        return False

    reset_link = f"{APP_BASE_URL}/reset-password?token={reset_token}"

    html_body = f"""
    <div style="font-family: Arial, sans-serif; max-width: 560px; margin: 0 auto;">
      <h2 style="color: #333;">Reset your MindFold 3D password</h2>
      <p>We received a request to reset the password on your MindFold 3D account.
         Click the link below to set a new password. This link expires in 24 hours.</p>
      <p style="margin: 24px 0;">
        <a href="{reset_link}" style="background:#007acc; color:#fff; padding:12px 24px; text-decoration:none; border-radius:4px; display:inline-block;">Reset password</a>
      </p>
      <p style="color:#666; font-size:13px;">If the button doesn't work, copy this link into your browser:<br>
        <a href="{reset_link}">{reset_link}</a>
      </p>
      <p style="color:#666; font-size:13px;">If you didn't request a password reset, you can safely ignore this message.</p>
    </div>
    """

    text_body = (
        "We received a request to reset the password on your MindFold 3D account.\n\n"
        f"Reset link (expires in 24 hours): {reset_link}\n\n"
        "If you didn't request a password reset, you can safely ignore this message."
    )

    try:
        resend.Emails.send({
            "from": MAIL_FROM,
            "to": to_email,
            "reply_to": MAIL_REPLY_TO,
            "subject": "Reset your MindFold 3D password",
            "html": html_body,
            "text": text_body,
        })
        return True
    except Exception as e:
        logger.error(f"Failed to send password reset email to {to_email}: {e}")
        return False
