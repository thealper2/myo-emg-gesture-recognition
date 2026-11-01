# Hardware setup: Myo Armband in WSL2 via dl-myo

Verified on 2026-11-01 with: Windows 11 + WSL2 Ubuntu 24.04 (kernel 6.18 microsoft-standard-WSL2),
usbipd-win 5.3.0, a USB Bluetooth dongle (CSR `0a12:0001`, BT 4.0 with LE), BlueZ 5.72,
Bleak 0.20.2, local dl-myo at `/mnt/d/work2/dl-myo`, Myo "Can's Myo" `D7:50:04:F0:88:9A`.

## Why a USB dongle + usbipd

WSL2 has no access to the PC's Bluetooth radio. A USB Bluetooth dongle can be forwarded into WSL
with usbipd-win; the WSL kernel then drives it with `btusb` and BlueZ, which Bleak (dl-myo's backend)
uses over D-Bus. The PC's built-in adapter (`Realtek 0bda:b850`, used by the headphones) stays
with Windows. Do not attach that one.

## One-time setup (Windows, administrator PowerShell)

```powershell
winget install usbipd            # already installed here (5.3.0)
usbipd list                      # find the dongle, e.g. 1-5  0a12:0001  Generic Bluetooth Radio
usbipd bind --hardware-id 0a12:0001
```

`bind` (shown as "Shared") survives reboots. In Windows Device Manager the CSR dongle shows an error
state; that does not matter because Linux drives it.

## Every time (after reboot / replug)

```bash
make bt-attach      # usbipd.exe attach --wsl --hardware-id 0a12:0001, then make bt-status
```

Expected `make bt-status` output: `adapter: hci0`, `bluetooth.service: active`, `Powered: yes`.
`btusb` is auto-loaded and `bluetooth.service` starts automatically. No `sudo` is needed.
To give the dongle back to Windows: `make bt-detach`. To re-attach automatically after
replugging, run `usbipd attach --wsl --hardware-id 0a12:0001 --auto-attach` in PowerShell
(this keeps running).

## Wear the armband like the training data

EMG-EPN-612 users wore the Myo on the forearm a few centimetres below the elbow
(`userInfo.distanceFromElbowToMyoInCm`, e.g. 6 cm for user1), with the logo/LED module on the
upper side of the forearm (see `photo.png` in the dataset user folders). Channel `ch1..ch8` are the
Myo sensor indices as delivered by dl-myo (`EMGData.sample1[0..7]`), the same order as the dataset
JSON. If you consistently wear the band rotated, `device.channel_shift` in
`configs/inference.yaml` (or `--set device.channel_shift=k`) rotates the channels in software. The
default is 0. Before streaming, wear the Myo for a few minutes so it warms up and makes good skin
contact.

## Check the connection

```bash
make myo-scan                         # lists BLE devices, marks Myo armbands ("<-- MYO")
make myo-check MAC=D7:50:04:F0:88:9A  # 10 s headless stream, writes artifacts/reports/hardware/myo_check_*.json
make live MAC=D7:50:04:F0:88:9A       # dashboard at http://localhost:7860, connects on start
```

Without `MAC=...` dl-myo scans for any device advertising the Myo service UUID
`d5060001-a904-deb9-4748-2c7f4a124842`. You can also set `MYO_MAC` in the environment.

## How dl-myo is used

`src/myo_emg/device/dl_myo_adapter.py` imports `myo` from `DL_MYO_DIR` (the local checkout; the
report shows `dl_myo_module: /mnt/d/work2/dl-myo/myo/__init__.py`). It subclasses `myo.MyoClient`,
calls `with_device(mac)` (wrapped in a timeout, because dl-myo retries forever), then
`setup(ClassifierMode.DISABLED, EMGMode.SEND_EMG, IMUMode.NONE)` and `start()`. It receives
`on_emg_data(EMGData)` callbacks with two int8 samples each. The asyncio loop runs in its own thread
and pushes `(2, 8)` batches into a bounded queue. Inference and the UI run in other threads. Link
loss is detected by polling the Bleak client (dl-myo has no disconnect callback). On stop it calls
`client.stop()` and `client.disconnect()`. The library itself is not modified.

## Measured hardware results (2026-11-01)

| check | result |
|---|---|
| connect + setup | OK in 4–7 s (scan, connect, LED/vibration setup), battery read 7 % |
| 15 s stream (`make myo-check`) | 3194 samples, 0 invalid packets, 0 queue drops, saturation 0.008 % |
| 12 s raw packet capture | 1275 packets = 106.2 packets/s = **212.5 samples/s**, 0 duplicated packets |
| BLE delivery | bursty: median packet gap 0.02 ms, p99 48 ms, max 92 ms |
| live dashboard (`make live`) | streaming, ~22 predictions/s, end-to-end latency mean 3.2 ms / p95 10.3 ms, 0 missing |
| disconnect button | clean `stop()` + `disconnect()` |

**Sample-rate discrepancy.** The dataset metadata says 200 Hz, and a nominal 5 s recording has ~996
samples. The host receives ~212 samples/s from this Myo. Host timing cannot tell whether the
armband's internal EMG clock is 200 Hz or ~212 Hz. Both the dataset and this stream come from the
Myo firmware EMG stream, so **no resampling is applied**. A 40-sample window covers 188 ms if the true
rate is 212 Hz. The dashboard shows the measured rate and warns if it differs from 200 Hz by more
than 10 %.

These checks show that the device is connected and streaming. They do **not** measure how well
gestures are recognised for the wearer. That has not been evaluated.
