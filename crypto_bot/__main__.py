"""Entry when running: python -m crypto_bot"""

from crypto_bot._bootstrap import ensure_project_venv

ensure_project_venv()

from crypto_bot.main import main

if __name__ == "__main__":
    main()
