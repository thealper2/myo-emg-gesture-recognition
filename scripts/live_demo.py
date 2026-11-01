"""Start the dashboard (demo = simulated stream, live = physical Myo) or run the headless Myo check.

    python scripts/live_demo.py dashboard --mode demo
    python scripts/live_demo.py dashboard --mode live [--mac AA:BB:..] [--auto-connect]
    python scripts/live_demo.py myo-check --seconds 10
"""

from myo_emg.app.main import main

if __name__ == "__main__":
    main()
