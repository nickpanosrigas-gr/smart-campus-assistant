import logging
from src.smart_campus_assistant.bots.telegram import start_bot_daemon

if __name__ == "__main__":
    # Setup global logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("=======================================")
    print(" Smart Campus Assistant Initializing")
    print("=======================================")
    
    # Start the long-polling Telegram bot
    try:
        start_bot_daemon()
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    except Exception as e:
        print(f"\nFatal Error: {e}")