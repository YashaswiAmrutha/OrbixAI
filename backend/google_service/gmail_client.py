import os
import json
import base64
from datetime import datetime
from google.auth.transport.requests import Request
from google.oauth2.service_account import Credentials
from google.oauth2.credentials import Credentials as UserCredentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import logging

logger = logging.getLogger(__name__)

# Gmail API scope
SCOPES = ['https://www.googleapis.com/auth/gmail.modify', 'https://www.googleapis.com/auth/calendar']


class GmailClient:
    def __init__(self, credentials_file='credentials.json', token_file='token.json', redirect_uri='http://127.0.0.1:8001/auth/callback'):
        self.service = None
        self.calendar_service = None
        self.credentials = None
        self.credentials_file = credentials_file
        self.token_file = token_file
        self.redirect_uri = redirect_uri
        self._try_load_token()

    def _try_load_token(self):
        """Try to load and refresh existing token. Returns True if successful."""
        try:
            if os.path.exists(self.token_file):
                self.credentials = UserCredentials.from_authorized_user_file(self.token_file, SCOPES)
                logger.info(f"Loaded existing credentials from {self.token_file}")

                if self.credentials and self.credentials.valid:
                    self._build_services()
                    return True

                if self.credentials and self.credentials.expired and self.credentials.refresh_token:
                    try:
                        logger.info("Refreshing expired credentials")
                        self.credentials.refresh(Request())
                        self._save_token()
                        self._build_services()
                        return True
                    except Exception as e:
                        logger.warning(f"Token refresh failed: {e}. Deleting stale token.")
                        os.remove(self.token_file)
                        self.credentials = None

            return False
        except Exception as e:
            logger.error(f"Error loading token: {e}")
            self.credentials = None
            return False

    def _build_services(self):
        """Build Gmail and Calendar API services from current credentials."""
        self.service = build('gmail', 'v1', credentials=self.credentials)
        self.calendar_service = build('calendar', 'v3', credentials=self.credentials)
        logger.info("Gmail and Calendar services initialized successfully")

    def _save_token(self):
        """Save current credentials to token file."""
        with open(self.token_file, 'w') as token:
            token.write(self.credentials.to_json())
        logger.info(f"Credentials saved to {self.token_file}")

    def is_authenticated(self):
        """Check if client has valid credentials and services."""
        return self.service is not None and self.credentials is not None and self.credentials.valid

    def get_auth_url(self):
        """Generate the Google OAuth authorization URL."""
        if not os.path.exists(self.credentials_file):
            raise FileNotFoundError(
                f"Please download credentials.json from Google Cloud Console "
                f"and save it to: {os.path.abspath(self.credentials_file)}"
            )

        flow = Flow.from_client_secrets_file(
            self.credentials_file,
            scopes=SCOPES,
            redirect_uri=self.redirect_uri
        )
        auth_url, state = flow.authorization_url(
            access_type='offline',
            prompt='consent'
        )
        logger.info(f"Generated auth URL with state: {state}")
        return auth_url, state

    def handle_auth_callback(self, authorization_response_url):
        """Handle the OAuth callback and save credentials."""
        flow = Flow.from_client_secrets_file(
            self.credentials_file,
            scopes=SCOPES,
            redirect_uri=self.redirect_uri
        )
        flow.fetch_token(authorization_response=authorization_response_url)
        self.credentials = flow.credentials
        self._save_token()
        self._build_services()
        logger.info("OAuth2 authentication successful via callback")

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

    def _ensure_fresh_credentials(self):
        """Refresh credentials if expired before making API calls."""
        if self.credentials and self.credentials.expired and self.credentials.refresh_token:
            try:
                logger.info("Pre-call credential refresh")
                self.credentials.refresh(Request())
                self._save_token()
                self._build_services()
            except Exception as e:
                logger.warning(f"Pre-call refresh failed: {e}. Deleting stale token.")
                if os.path.exists(self.token_file):
                    os.remove(self.token_file)
                self.credentials = None
                self.service = None
                raise Exception("needs_auth")

    def get_all_recent_emails(self, max_results=10):
        """Fetch both received and sent emails using lightweight metadata format"""
        try:
            self._ensure_fresh_credentials()
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
        return None
