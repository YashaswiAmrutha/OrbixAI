from llm.ollama_client import generate_response
from llm.prompt import build_prompt
import logging

logger = logging.getLogger(__name__)

class MailGenerator:
    """Generate email subject and body using LLM"""
    
    @staticmethod
    def generate_mail_content(user_prompt, recipient_name=None, meeting_link=None):
        """
        Generate email subject and body based on user prompt
        
        Args:
            user_prompt: The user's intention for the email
            recipient_name: Optional recipient name
            meeting_link: Optional Google Meet link to include
        
        Returns:
            dict with 'subject' and 'body' keys
        """
        try:
            # Build a structured prompt for the LLM
            system_instruction = """You are an email content generator. Generate professional, concise email content.
            Return response in this exact format:
            SUBJECT: [email subject line]
            BODY: [email body text]
            
            Keep the subject to one line. Keep the body concise but complete."""
            
            context = f"Generate an email for: {user_prompt}"
            if recipient_name:
                context += f"\nRecipient: {recipient_name}"
            if meeting_link:
                context += f"\nInclude this Google Meet link in the email: {meeting_link}"
            
            full_prompt = build_prompt(context)
            
            # Generate response
            response = generate_response(full_prompt)
            
            # Parse the response
            subject = ""
            body = ""
            
            lines = response.split('\n')
            for i, line in enumerate(lines):
                if line.startswith("SUBJECT:"):
                    subject = line.replace("SUBJECT:", "").strip()
                elif line.startswith("BODY:"):
                    body = '\n'.join(lines[i:]).replace("BODY:", "").strip()
                    break
            
            if not subject or not body:
                # Fallback if parsing fails
                subject = "Meeting Request"
                body = user_prompt
                if meeting_link:
                    body += f"\n\nGoogle Meet Link: {meeting_link}"
            
            logger.info(f"Email content generated successfully")
            return {
                'subject': subject,
                'body': body
            }
        except Exception as e:
            logger.error(f"Error generating email content: {str(e)}")
            return {
                'subject': 'Meeting Request',
                'body': 'Please join our meeting.'
            }

    @staticmethod
    def generate_meeting_invitation(recipient_name, meeting_time=None, meeting_purpose=None):
        """Generate a professional meeting invitation"""
        try:
            prompt = f"Generate a professional meeting invitation email to {recipient_name}"
            if meeting_time:
                prompt += f" for {meeting_time}"
            if meeting_purpose:
                prompt += f" about {meeting_purpose}"
            
            return MailGenerator.generate_mail_content(prompt, recipient_name)
        except Exception as e:
            logger.error(f"Error generating meeting invitation: {str(e)}")
            return {
                'subject': f'Meeting Invitation - {meeting_purpose or "Discussion"}',
                'body': f'Hi {recipient_name},\n\nI would like to schedule a meeting with you.\n\nBest regards'
            }
