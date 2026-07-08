import os
import json
import base64
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from google.auth.transport.requests import Request
from google.oauth2.service_account import Credentials
from google.oauth2.credentials import Credentials as UserCredentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import logging

logger = logging.getLogger(__name__)

# Google API scopes. Gmail + Calendar power the gsuite server; drive.readonly lets the
# optional Google Drive integration reuse this same sign-in. Adding a scope here does
# NOT invalidate an existing token (Gmail/Calendar keep working); Drive access is only
# granted the next time the user re-runs Connect with Google.
SCOPES = ['https://www.googleapis.com/auth/gmail.modify',
          'https://www.googleapis.com/auth/calendar',
          'https://www.googleapis.com/auth/drive.readonly']


_EMAIL_CACHE_TTL = 25  # seconds — slightly less than the 30s auto-refresh interval

class GmailClient:
    def __init__(self, credentials_file='credentials.json', token_file='token.json', redirect_uri='http://127.0.0.1:8001/auth/callback'):
        self.service = None
        self.calendar_service = None
        self.credentials = None
        self.credentials_file = credentials_file
        self.token_file = token_file
        self.redirect_uri = redirect_uri
        self._flow = None
        # In-memory email cache (avoids repeated full fetches on every 30s auto-refresh)
        self._email_cache = None
        self._email_cache_time = None
        self._email_cache_lock = threading.Lock()
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
        """Build Gmail and Calendar API services using google-auth credentials directly."""
        self.service          = build('gmail',    'v1', credentials=self.credentials)
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

        self._flow = Flow.from_client_secrets_file(
            self.credentials_file,
            scopes=SCOPES,
            redirect_uri=self.redirect_uri
        )
        auth_url, state = self._flow.authorization_url(
            access_type='offline',
            prompt='consent'
        )
        logger.info(f"Generated auth URL with state: {state}")
        return auth_url, state

    def handle_auth_callback(self, authorization_response_url):
        """Handle the OAuth callback and save credentials."""
        if self._flow is None:
            # Fallback: recreate flow without PKCE (state won't be verified)
            self._flow = Flow.from_client_secrets_file(
                self.credentials_file,
                scopes=SCOPES,
                redirect_uri=self.redirect_uri
            )
        self._flow.fetch_token(authorization_response=authorization_response_url)
        self.credentials = self._flow.credentials
        self._flow = None
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
        """
        Fetch inbox + sent emails using direct REST calls (requests library) for
        parallel fetches — avoids httplib2 SSL corruption under concurrent threads.
        """
        import requests as _req
        from email.utils import parsedate_to_datetime

        # ── 1. Return cached result if still fresh ───────────────────────────
        with self._email_cache_lock:
            if (self._email_cache is not None and
                    self._email_cache_time is not None and
                    (datetime.utcnow() - self._email_cache_time).total_seconds() < _EMAIL_CACHE_TTL):
                logger.info("Returning cached emails (cache hit)")
                return self._email_cache

        try:
            self._ensure_fresh_credentials()

            token    = self.credentials.token
            base_url = "https://gmail.googleapis.com/gmail/v1/users/me/messages"
            auth_hdr = {"Authorization": f"Bearer {token}"}
            per_folder = max(5, max_results // 2)

            # ── 2. Fetch inbox + sent IDs in parallel (requests, thread-safe) ─
            def list_folder(q):
                resp = _req.get(base_url, headers=auth_hdr,
                                params={"q": q, "maxResults": per_folder,
                                        "fields": "messages/id"},
                                timeout=10)
                resp.raise_for_status()
                return resp.json().get("messages", [])

            with ThreadPoolExecutor(max_workers=2) as pool:
                inbox_f = pool.submit(list_folder, "in:inbox")
                sent_f  = pool.submit(list_folder, "in:sent")
                inbox_ids = inbox_f.result()
                sent_ids  = sent_f.result()

            # Build deduplicated task list
            seen, tasks = set(), []
            for m in inbox_ids:
                if m["id"] not in seen:
                    seen.add(m["id"])
                    tasks.append((m["id"], "received"))
            for m in sent_ids:
                if m["id"] not in seen:
                    seen.add(m["id"])
                    tasks.append((m["id"], "sent"))

            # ── 3. Fetch each message's metadata in parallel ─────────────────
            # Each worker uses its own requests.Session — fully thread-safe.
            META_PARAMS = [
                ("format", "metadata"),
                ("metadataHeaders", "From"),
                ("metadataHeaders", "To"),
                ("metadataHeaders", "Subject"),
                ("metadataHeaders", "Date"),
                ("fields", "id,snippet,payload/headers"),
            ]

            def fetch_message(id_type):
                mid, mtype = id_type
                try:
                    with _req.Session() as s:
                        r = s.get(f"{base_url}/{mid}", headers=auth_hdr,
                                  params=META_PARAMS, timeout=8)
                        r.raise_for_status()
                    msg  = r.json()
                    hdrs = {h["name"]: h["value"]
                            for h in msg.get("payload", {}).get("headers", [])}
                    return {
                        "id":      mid,
                        "from":    hdrs.get("From", "Unknown"),
                        "to":      hdrs.get("To", ""),
                        "subject": hdrs.get("Subject", "No Subject"),
                        "date":    hdrs.get("Date", ""),
                        "snippet": msg.get("snippet", ""),
                        "type":    mtype,
                    }
                except Exception as e:
                    logger.warning(f"Skipping message {mid}: {e}")
                    return None

            workers = min(8, max(1, len(tasks)))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(fetch_message, tasks))

            all_emails = [r for r in results if r is not None]

            # ── 4. Sort newest-first ─────────────────────────────────────────
            def _parse_date(date_str):
                try:
                    return parsedate_to_datetime(date_str)
                except Exception:
                    return None

            all_emails.sort(
                key=lambda x: _parse_date(x["date"]) or datetime.min,
                reverse=True
            )
            final = all_emails[:max_results * 2]

            # ── 5. Update cache ──────────────────────────────────────────────
            with self._email_cache_lock:
                self._email_cache      = final
                self._email_cache_time = datetime.utcnow()
            logger.info(f"Email fetch complete: {len(final)} emails cached")
            return final

        except Exception as error:
            logger.error(f"Error fetching emails: {error}")
            with self._email_cache_lock:
                if self._email_cache is not None:
                    logger.info("Returning stale cache due to fetch error")
                    return self._email_cache
            raise

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

    def get_user_profile(self):
        """
        Return {"email", "name"} for the authenticated (logged-in) user.

        Email comes from Gmail's getProfile. The display name is read from the
        'From' header of one of the user's own sent messages (works with the
        existing gmail.modify scope — no extra consent needed). Falls back to a
        name derived from the email local-part.
        """
        import re as _re
        from email.utils import parseaddr

        profile = {"email": None, "name": None}
        try:
            self._ensure_fresh_credentials()

            # ── Email address ────────────────────────────────────────────────
            try:
                p = self.service.users().getProfile(userId='me').execute()
                profile["email"] = p.get("emailAddress")
            except Exception as e:
                logger.warning(f"getProfile failed: {e}")

            # ── Real display name from a sent message's From header ───────────
            try:
                lst = self.service.users().messages().list(
                    userId='me', q='in:sent', maxResults=1
                ).execute()
                msgs = lst.get('messages', [])
                if msgs:
                    m = self.service.users().messages().get(
                        userId='me', id=msgs[0]['id'],
                        format='metadata', metadataHeaders=['From']
                    ).execute()
                    hdrs = {h['name']: h['value']
                            for h in m.get('payload', {}).get('headers', [])}
                    realname, _addr = parseaddr(hdrs.get('From', ''))
                    if realname:
                        profile["name"] = realname.strip()
            except Exception as e:
                logger.warning(f"Sent-name lookup failed: {e}")

            # ── Fallback: derive a friendly name from the email local-part ────
            if not profile["name"] and profile["email"]:
                local = profile["email"].split("@")[0]
                token = _re.split(r'[._\-+]', local)[0]
                token = _re.sub(r'\d+', '', token)
                if token:
                    profile["name"] = token[:1].upper() + token[1:]

            logger.info(f"User profile resolved: name={profile['name']}, email={profile['email']}")
            return profile
        except Exception as e:
            logger.error(f"get_user_profile error: {e}")
            return profile

    def get_email_for_contact(self, contact_name):
        """Extract email from contact name (helper function)"""
        return None
