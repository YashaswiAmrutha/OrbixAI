"""
Mail content generator.
Prefers content already produced by the orchestrator (gpraneeth555/llama-3-13k).
Falls back to a direct Ollama call only when no pre-generated content is available.
"""

import logging
from llm.ollama_client import generate_response

logger = logging.getLogger(__name__)


class MailGenerator:

    @staticmethod
    def generate_mail_content(
        user_prompt: str,
        recipient_name: str = None,
        meeting_link: str = None,
        prefilled: dict = None          # ← content already from the HF orchestrator
    ) -> dict:
        """
        Return {'subject': ..., 'body': ...}.

        If `prefilled` dict already has both keys (from the orchestrator JSON),
        those values are used directly — injecting the meeting_link if needed.
        Otherwise a fresh Ollama call is made.
        """

        # ── 1. Use orchestrator-generated content if available ──────────────
        if prefilled:
            subject = (prefilled.get("subject") or "").strip()
            body    = (prefilled.get("body")    or "").strip()
            if subject and body:
                if meeting_link and meeting_link not in body:
                    body += f"\n\nGoogle Meet Link: {meeting_link}"
                logger.info("[MailGenerator] Using orchestrator-generated email content")
                return {"subject": subject, "body": body}

        # ── 2. Fallback — ask Ollama to generate subject + body ─────────────
        logger.info("[MailGenerator] Generating email content via Ollama (fallback)")
        try:
            context_lines = [f"Generate a professional email for: {user_prompt}"]
            if recipient_name:
                context_lines.append(f"Recipient: {recipient_name}")
            if meeting_link:
                context_lines.append(f"Include this Google Meet link in the email: {meeting_link}")

            system = (
                "You are an email content generator. Return ONLY:\n"
                "SUBJECT: <one-line subject>\n"
                "BODY: <email body>"
            )
            full_prompt = system + "\n\n" + "\n".join(context_lines)

            response = generate_response(full_prompt)

            subject = ""
            body    = ""
            lines   = response.split("\n")
            for i, line in enumerate(lines):
                if line.startswith("SUBJECT:"):
                    subject = line.replace("SUBJECT:", "").strip()
                elif line.startswith("BODY:"):
                    body = "\n".join(lines[i:]).replace("BODY:", "").strip()
                    break

            if not subject or not body:
                subject = "Meeting Request" if meeting_link else "Message from OrbixAI"
                body    = user_prompt
                if meeting_link:
                    body += f"\n\nGoogle Meet Link: {meeting_link}"

            return {"subject": subject, "body": body}

        except Exception as e:
            logger.error(f"[MailGenerator] Error generating content: {e}")
            return {
                "subject": "Meeting Invitation" if meeting_link else "Message from OrbixAI",
                "body":    user_prompt + (f"\n\nMeet: {meeting_link}" if meeting_link else "")
            }
