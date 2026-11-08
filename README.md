# SiegeScout

SiegeScout is a Windows computer-vision automation bot for Hero Siege. It
captures the game screen, detects map and resource features locally, and
provides a Tkinter control panel for movement, combat, mining, and loot.

This project is provided as-is for experimentation and community development.
Use it only where automation is allowed, and do not use it to bypass
anti-cheat or platform rules.

There is no account or license server required to run the open-source build.

## Community

SiegeScout is open source and community-driven. Join the
[SiegeScout Discord server](https://discord.gg/VVM2UwdJQ4) to discuss the bot,
report issues, share ideas, and coordinate development.

## Requirements

- Windows 10 or later
- Python 3.10+
- A 16:9 display is recommended
- Hero Siege configured for windowed fullscreen

## Setup

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python gui.py
```

Alternatively, run `run.bat` from an elevated terminal when live stats need
the additional permissions requested by the application.

Live stats (gold/hr, zone buffs, item drops) are optional. They need
[Npcap](https://npcap.com) plus `python -m pip install "scapy>=2.5" "psutil>=5.9"`;
without them the bot runs normally.

## Development

Run the test suite with:

```powershell
python -m pytest
```

Local model training instructions are in [training/README.md](training/README.md).
Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.
The minimap training data is available from the
[SiegeScout Roboflow project](https://universe.roboflow.com/test-jmbou/minimapvision-2).

## Configuration and secrets

Roboflow is optional and disabled by default. If you enable it, provide
`ROBOFLOW_API_KEY` as an environment variable or create the ignored
`roboflow_key.txt` file locally. Never commit API keys, tokens, logs,
screenshots, or local model-training exports.

## License

Released under the [MIT License](LICENSE).
