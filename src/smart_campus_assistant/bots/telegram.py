import re
import requests
import telebot
import threading
import logging

# Updated Project Imports
from src.smart_campus_assistant.config.settings import settings
from src.smart_campus_assistant.bots.whisper import transcribe_audio
from src.smart_campus_assistant.agents.supervisor import run_supervisor

logger = logging.getLogger(__name__)

# Initialize the bot object using your token
bot = telebot.TeleBot(settings.TELEGRAM_BOT_TOKEN)

# ==========================================
# FORMATTING HELPER
# ==========================================
def clean_markdown_for_telegram(text: str) -> str:
    """
    Converts standard AI Markdown (GitHub Flavored) into Telegram's strict Markdown format.
    Automatically intercepts Markdown tables and converts them into mobile-friendly lists.
    """
    if not text:
        return text
        
    # 1. Replace horizontal rules (---, ***, ___) with a safe Unicode line
    text = re.sub(r'^\s*[-*_]{3,}\s*$', '━━━━━━━━━━━━━━━━━━━━', text, flags=re.MULTILINE)
    
    # 2. Escape underscores in snake_case words (like intel_iommu) to prevent italic parser crashes
    text = re.sub(r'([a-zA-Z0-9])_([a-zA-Z0-9])', r'\1\\_\2', text)

    # 3. Handle bullet points (covers both asterisks and hyphens)
    text = re.sub(r'^(\s*)\*\s+', r'\1* ', text, flags=re.MULTILINE)
    text = re.sub(r'^(\s*)-\s+', r'\1- ', text, flags=re.MULTILINE)
        
    # 4. Handle Headers & Bold text
    text = re.sub(r'\*\*(.*?)\*\*', r'*\1*', text)
    text = re.sub(r'^###\s+(.*)', r'*\1*', text, flags=re.MULTILINE)
    text = re.sub(r'^##\s+(.*)', r'*\1*', text, flags=re.MULTILINE)
    text = re.sub(r'^#\s+(.*)', r'*\1*', text, flags=re.MULTILINE)

    lines = text.split('\n')
    formatted_lines = []
    
    table_buffer = []
    
    def process_table_buffer(buffer):
        if len(buffer) < 3:
            return buffer
        headers = [col.strip() for col in buffer[0].split('|')[1:-1]]
        cards = []
        for row in buffer[2:]:
            cols = [col.strip() for col in row.split('|')[1:-1]]
            card_lines = []
            for i, col_val in enumerate(cols):
                header_name = headers[i] if i < len(headers) else "Data"
                if i == 0:
                    card_lines.append(f"- *{col_val}*")
                else:
                    card_lines.append(f"  - *{header_name}:* {col_val}")
            cards.append("\n".join(card_lines))
        return ["\n" + "\n\n".join(cards) + "\n"]

    for line in lines:
        if line.strip().startswith('|') and line.strip().endswith('|'):
            table_buffer.append(line.strip())
        else:
            if table_buffer:
                formatted_lines.extend(process_table_buffer(table_buffer))
                table_buffer = []
            formatted_lines.append(line)
            
    if table_buffer:
        formatted_lines.extend(process_table_buffer(table_buffer))
        
    return '\n'.join(formatted_lines)

# ==========================================
# PUSH NOTIFICATION (For Cron Jobs/Alerts)
# ==========================================
def send_telegram_alert(text: str) -> None:
    formatted_text = clean_markdown_for_telegram(text)
    url = f"https://api.telegram.org/bot{settings.TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": settings.TELEGRAM_ALLOWED_USER_ID,
        "text": formatted_text,
        "parse_mode": "Markdown"
    }
    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 400 and "parse entities" in response.text.lower():
            logger.warning("Telegram rejected Markdown in alert. Retrying as plain text.")
            payload.pop("parse_mode") 
            payload["text"] = f"[Markdown stripped]\n\n{formatted_text}"
            response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to send Telegram alert: {e}")

# ==========================================
# SHARED AGENT LOGIC
# ==========================================
def process_query_with_agent(message, user_query: str):
    """
    Executes the main Supervisor LLM and returns the formatted response.
    """
    logger.info(f"Incoming query from User {message.from_user.id}: '{user_query}'")
    
    stop_typing = threading.Event()
    typing_thread = threading.Thread(
        target=keep_chat_action_alive, 
        args=(message.chat.id, stop_typing, 'typing')
    )
    typing_thread.start()

    try:
        # Call the Supervisor Agent
        logger.info("Invoking Supervisor Agent...")
        reply_text = run_supervisor(user_query)
            
        if not reply_text or not str(reply_text).strip():
            reply_text = "Error: The AI returned an empty response."
        
        final_text = clean_markdown_for_telegram(str(reply_text))
        
        try:
            bot.reply_to(message, final_text, parse_mode="Markdown")
            logger.info("Sent Bot Message Successfully.")
        except Exception as send_err:
            logger.error(f"Markdown Send Error: {send_err}")
            bot.reply_to(message, f"[Format Error Retrying...]\n\n{final_text}")
        
    except Exception as e:
        logger.error(f"CRITICAL ERROR during agent execution: {str(e)}")
        bot.reply_to(message, f"*Error:*\n`{str(e)}`", parse_mode="Markdown")
        
    finally:
        stop_typing.set()
        typing_thread.join()
        
# ==========================================
# Handlers
# ==========================================
@bot.message_handler(content_types=['text'])
def handle_incoming_text(message):
    if message.from_user.id != settings.TELEGRAM_ALLOWED_USER_ID:
        logger.warning(f"Unauthorized text access attempt from user: {message.from_user.id}")
        bot.reply_to(message, "Access denied.")
        return
    process_query_with_agent(message, message.text)

@bot.message_handler(content_types=['voice', 'audio'])
def handle_incoming_voice(message):
    if message.from_user.id != settings.TELEGRAM_ALLOWED_USER_ID:
        logger.warning(f"Unauthorized voice access attempt from user: {message.from_user.id}")
        bot.reply_to(message, "Access denied.")
        return
    
    stop_recording = threading.Event()
    recording_thread = threading.Thread(target=keep_chat_action_alive, args=(message.chat.id, stop_recording, 'record_voice'))
    recording_thread.start()
    
    try:
        file_id = message.voice.file_id if message.content_type == 'voice' else message.audio.file_id
        file_info = bot.get_file(file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        transcription = transcribe_audio(downloaded_file)
        
        stop_recording.set()
        recording_thread.join()
        
        if not transcription:
            logger.error("Transcription returned an empty string.")
            bot.reply_to(message, "Transcription failed.")
            return
            
        bot.reply_to(message, f"*I heard:* _{transcription}_", parse_mode="Markdown")
        process_query_with_agent(message, transcription)
        
    except Exception as e:
        stop_recording.set()
        recording_thread.join()
        logger.error(f"Voice Error: {e}")
        bot.reply_to(message, f"Audio Error: {str(e)}")

def keep_chat_action_alive(chat_id, stop_event, action='typing'):
    while not stop_event.is_set():
        try:
            bot.send_chat_action(chat_id, action)
        except Exception:
            pass
        stop_event.wait(4)

def start_bot_daemon():
    logger.info(f"Starting Telegram Bot Daemon for User ID: {settings.TELEGRAM_ALLOWED_USER_ID}...")
    bot.infinity_polling()