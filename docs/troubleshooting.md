# Troubleshooting

| symptom | cause / fix |
|---|---|
| `make bt-status`: "No Bluetooth adapter in WSL" | dongle not attached: `make bt-attach`. If usbipd says the device is not shared, run `usbipd bind --hardware-id 0a12:0001` once in an **administrator** PowerShell. |
| `make bt-attach`: "usbipd.exe: not found" | usbipd-win not installed or in another folder: `winget install usbipd`, or `make bt-attach USBIPD="/mnt/c/.../usbipd.exe"`. |
| `bt-attach` works but no `hci0` | check `dmesg | grep -i bluetooth`; `btusb` must load (it is a module in the WSL kernel). Unplug/replug the dongle and attach again. |
| `make myo-scan` finds no Myo | Myo asleep: move it or plug in USB until the logo LED is on, keep it within ~2 m. Make sure it is not connected to another host (Myo Connect, phone). |
| live/myo-check: "No Myo found within 20 s" | same as above; or the wrong MAC. Increase `--set device.scan_timeout_s=40`. |
| "Myo disconnected (BLE link lost)" | range, low battery (the test armband was at 7 %), or interference. Charge it and reconnect with the Connect button. |
| dashboard shows "no data for N s" | stream stalled; the link may be about to drop. Disconnect / Connect. |
| rate warning in dashboard | measured rate differs from 200 Hz by >10 %. ~212 Hz is normal for the tested armband (see hardware_setup.md). |
| predictions mostly `uncertain` | expected for a new wearer without calibration. Check placement (forearm below the elbow, logo up), wait for warm-up, try `channel_shift` 1..7 if the band is rotated, or lower the threshold. |
| `noGesture` for movements that are not one of the 5 gestures | `noGesture` only means relaxed hand. The model has no "other movement" class. |
| `make live`: "No trained model at ..." | run `make pipeline` (or `make evaluate`) first, or set `MYO_EMG_MODEL_DIR`. |
| `make demo`: "Replay data not found" | `data/processed/epn612_test.npz` missing: `make prepare-data`. |
| port 7860 busy | `make demo PORT=7861` / `make live PORT=7861`. |
| dashboard not reachable from Windows | WSL2 forwards `localhost`. Open `http://localhost:7860`. If localhost forwarding is disabled in `.wslconfig`, use `ARGS="--set dashboard.host=0.0.0.0"` and the WSL IP. |
| `make evaluate` refuses to run | test metrics already exist; after retraining use `make evaluate FORCE=1` (this is a new use of the test set). |
