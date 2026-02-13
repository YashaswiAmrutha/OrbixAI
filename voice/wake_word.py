import time

WAKE_WORD = "orbi"

def listen_for_wake_word():
    """
    Temporary wake word simulation.
    Replace later with Porcupine once audio stack is stable.
    """
    while True:
        typed = input("Type wake word ('orbi'): ").lower().strip()
        if WAKE_WORD in typed:
            print("🟢 Wake word detected")
            return
