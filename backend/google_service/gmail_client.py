import os
import json
import base64
from datetime import datetime
from google.auth.transport.requests import Request
from google.oauth2.service_account import Credentials
from google.oauth2.credentials import Credentials as UserCredentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import logging

logger = logging.getLogger(__name__)

# Gmail API scope
SCOPES = ['https://www.googleapis.com/auth/gmail.modify', 'https://www.googleapis.com/auth/calendar']

class GmailClient:
    def __init__(self, credentials_file='credentials.json', token_file='token.json'):
        self.service = None
        self.calendar_service = None
        self.credentials = None
        self.credentials_file = credentials_file
        self.token_file = token_file
        self._initialize_service()
    
    def _initialize_service(self):
        """Initialize Gmail and Calendar API services using OAuth2"""
        try:
            # Try to load existing credentials
            if os.path.exists(self.token_file):
                self.credentials = UserCredentials.from_authorized_user_file(self.token_file, SCOPES)
                logger.info(f"Loaded existing credentials from {self.token_file}")
            
            # If no valid credentials, create new ones
            if not self.credentials or not self.credentials.valid:
                if self.credentials and self.credentials.expired and self.credentials.refresh_token:
                    logger.info("Refreshing expired credentials")
                    self.credentials.refresh(Request())
                else:
                    # Check if credentials file exists
                    if not os.path.exists(self.credentials_file):
                        logger.error(f"Credentials file '{self.credentials_file}' not found!")
                        raise FileNotFoundError(
                            f"Please download credentials.json from Google Cloud Console "
                            f"and save it to: {os.path.abspath(self.credentials_file)}"
                        )
                    
                    logger.info(f"Starting OAuth2 flow using {self.credentials_file}")
                    flow = InstalledAppFlow.from_client_secrets_file(
                        self.credentials_file,
                        SCOPES
                    )
                    # Try port 8080 first, which should match your Google Cloud Console settings
                    logger.info("Attempting to use port 8080 for OAuth callback...")
                    self.credentials = flow.run_local_server(
                        port=8080, 
                        open_browser=True,
                        host='localhost',
                        access_type='offline',
                        prompt='consent'
                    )
                    logger.info("OAuth2 authentication successful on port 8080")
                
                # Save credentials for next run
                with open(self.token_file, 'w') as token:
                    token.write(self.credentials.to_json())
                logger.info(f"Credentials saved to {self.token_file}")
            
            self.service = build('gmail', 'v1', credentials=self.credentials)
            self.calendar_service = build('calendar', 'v3', credentials=self.credentials)
            logger.info("Gmail and Calendar services initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Gmail client: {str(e)}")
            raise

    def get_latest_emails(self, max_results=5):
        """Fetch the latest emails from inbox"""
        try:
            results = self.service.users().messages().list(
                userId='me',
                q='in:inbox',
                maxResults=max_results,
                pageToken=None
            ).execute()
            
            messages = results.get('messages', [])
            emails = []
            
            for message in messages:
                msg = self.service.users().messages().get(
                    userId='me',
                    id=message['id'],
                    format='full'
                ).execute()
                
                headers = msg['payload']['headers']
                email_data = {
                    'id': message['id'],
                    'from': next((h['value'] for h in headers if h['name'] == 'From'), 'Unknown'),
                    'subject': next((h['value'] for h in headers if h['name'] == 'Subject'), 'No Subject'),
                    'date': next((h['value'] for h in headers if h['name'] == 'Date'), ''),
                    'snippet': msg['snippet'],
                    'type': 'received'
                }
                emails.append(email_data)
            
            return emails
        except HttpError as error:
            logger.error(f"An error occurred: {error}")
            return []

    def get_all_recent_emails(self, max_results=10):
        """Fetch both received and sent emails using lightweight metadata format"""
        try:
            all_emails = []
            
            # Fetch inbox and sent message IDs
            inbox_results = self.service.users().messages().list(
                userId='me',
                q='in:inbox',
                maxResults=max_results
            ).execute()
            
            sent_results = self.service.users().messages().list(
                userId='me',
                q='in:sent',
                maxResults=max_results
            ).execute()
            
            inbox_ids = inbox_results.get('messages', [])
            sent_ids = sent_results.get('messages', [])
            
            # De-duplicate (same message can appear in both)
            seen = set()
            
            for message in inbox_ids:
                mid = message['id']
                if mid in seen:
                    continue
                seen.add(mid)
                try:
                    msg = self.service.users().messages().get(
                        userId='me',
                        id=mid,
                        format='metadata',
                        metadataHeaders=['From', 'To', 'Subject', 'Date']
                    ).execute()
                    
                    headers = {h['name']: h['value'] for h in msg.get('payload', {}).get('headers', [])}
                    all_emails.append({
                        'id': mid,
                        'from': headers.get('From', 'Unknown'),
                        'to': headers.get('To', ''),
                        'subject': headers.get('Subject', 'No Subject'),
                        'date': headers.get('Date', ''),
                        'snippet': msg.get('snippet', ''),
                        'type': 'received'
                    })
                except Exception as e:
                    logger.error(f"Error processing inbox message {mid}: {str(e)}")
                    continue
            
            for message in sent_ids:
                mid = message['id']
                if mid in seen:
                    continue
                seen.add(mid)
                try:
                    msg = self.service.users().messages().get(
                        userId='me',
                        id=mid,
                        format='metadata',
                        metadataHeaders=['From', 'To', 'Subject', 'Date']
                    ).execute()
                    
                    headers = {h['name']: h['value'] for h in msg.get('payload', {}).get('headers', [])}
                    all_emails.append({
                        'id': mid,
                        'from': headers.get('From', 'Unknown'),
                        'to': headers.get('To', ''),
                        'subject': headers.get('Subject', 'No Subject'),
                        'date': headers.get('Date', ''),
                        'snippet': msg.get('snippet', ''),
                        'type': 'sent'
                    })
                except Exception as e:
                    logger.error(f"Error processing sent message {mid}: {str(e)}")
                    continue
            
            # Sort by date, newest first
            def parse_email_date(date_str):
                from email.utils import parsedate_to_datetime
                try:
                    return parsedate_to_datetime(date_str)
                except:
                    return None
            
            all_emails.sort(key=lambda x: parse_email_date(x['date']) or datetime.min, reverse=True)
            
            return all_emails[:max_results * 2]
        except Exception as error:
            logger.error(f"Error fetching emails: {error}")
            return []

    def send_email(self, to_email, subject, body, html_body=None):
        """Send an email message"""
        try:
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            
            if html_body:
                message = MIMEMultipart('alternative')
                part1 = MIMEText(body, 'plain')
                part2 = MIMEText(html_body, 'html')
                message.attach(part1)
                message.attach(part2)
            else:
                message = MIMEText(body)
            
            message['to'] = to_email
            message['subject'] = subject
            
            raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
            send_message = {'raw': raw_message}
            
            result = self.service.users().messages().send(
                userId='me',
                body=send_message
            ).execute()
            
            logger.info(f"Email sent successfully to {to_email}")
            return {
                'success': True,
                'message_id': result['id'],
                'to': to_email,
                'subject': subject,
                'type': 'sent'
            }
        except HttpError as error:
            logger.error(f"An error occurred: {error}")
            return {'success': False, 'error': str(error)}

    def create_google_meet(self, event_title, event_description, attendee_email, start_time=None):
        """Create a Google Calendar event with Google Meet link"""
        try:
            from datetime import datetime, timedelta
            
            if not start_time:
                start_time = datetime.utcnow() + timedelta(hours=1)
            end_time = start_time + timedelta(hours=1)
            
            event = {
                'summary': event_title,
                'description': event_description,
                'start': {
                    'dateTime': start_time.isoformat() + 'Z',
                    'timeZone': 'UTC',
                },
                'end': {
                    'dateTime': end_time.isoformat() + 'Z',
                    'timeZone': 'UTC',
                },
                'conferenceData': {
                    'createRequest': {
                        'requestId': f"meet-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{id(self)}",
                        'conferenceSolutionKey': {
                            'type': 'hangoutsMeet'
                        }
                    }
                },
                'attendees': [
                    {'email': attendee_email}
                ]
            }
            
            created_event = self.calendar_service.events().insert(
                calendarId='primary',
                body=event,
                conferenceDataVersion=1
            ).execute()
            
            meet_link = created_event.get('conferenceData', {}).get('entryPoints', [{}])[0].get('uri', 'N/A')
            
            logger.info(f"Google Meet created: {meet_link}")
            return {
                'success': True,
                'event_id': created_event['id'],
                'meet_link': meet_link,
                'event_title': event_title,
                'attendee_email': attendee_email
            }
        except HttpError as error:
            logger.error(f"An error occurred: {error}")
            return {'success': False, 'error': str(error)}

    def get_email_for_contact(self, contact_name):
        """Extract email from contact name (helper function)"""
        # This is a simple helper; you might want to integrate with Google Contacts API
        # For now, we'll return a formatted email
        return None
