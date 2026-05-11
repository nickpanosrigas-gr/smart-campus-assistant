import sys
import logging
from src.smart_campus_assistant.bots.telegram import start_bot_daemon
from src.smart_campus_assistant.utils.initialization import run_initialization

if __name__ == "__main__":
    # Setup global logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("=======================================")
    print(" Smart Campus Assistant Initializing")
    print("=======================================")
    
    # Run service health checks and Vector DB sync
    init_success = run_initialization()
    if not init_success:
        print("\n[!] CRITICAL: Initialization failed. Please check the logs and ensure Docker containers (Qdrant, Ollama, Whisper) are running.")
        sys.exit(1)
    
    print("\n=======================================")
    print(" Systems GO. Starting Telegram Bot...")
    print("=======================================\n")
    
    # Start the long-polling Telegram bot
    try:
        start_bot_daemon()
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    except Exception as e:
        print(f"\nFatal Error: {e}")