import time
import sys
from threading import Thread
from popup_module.popup1 import show_popup  # Import Popup

def screen_runtime_tracker():
    start_time = time.time()
    alert_interval_first = 5       # First popup after 5 seconds
    alert_interval_next = 300      # Next popups every 5 minutes (300 seconds)
    first_popup_shown = False      # Track if the first popup has been shown

    try:
        while True:
            elapsed_time = int(time.time() - start_time)
            hours = elapsed_time // 3600
            minutes = (elapsed_time % 3600) // 60
            seconds = elapsed_time % 60

            print(f"\rScreen running time: {hours:02d}:{minutes:02d}:{seconds:02d}", end="")

            # First popup after 5 seconds
            if not first_popup_shown and elapsed_time >= alert_interval_first:
                print("\nTimes up!! (First Popup)")
                Thread(target=show_popup).start()
                first_popup_shown = True
                start_time = time.time()  # Reset time after first popup

            # Subsequent popups every 5 minutes
            elif first_popup_shown and elapsed_time >= alert_interval_next:
                print("\nTimes up!! (Next Popup)")
                Thread(target=show_popup).start()
                start_time = time.time()  # Reset time after each popup

            time.sleep(1)

    except KeyboardInterrupt:
        print("\nTracking stopped.")
        sys.exit(0)

if __name__ == "__main__":
    screen_runtime_tracker()
