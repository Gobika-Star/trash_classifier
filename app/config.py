import os

DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///bot.db')
SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key')
DISCORD_CLIENT_ID = os.getenv('DISCORD_CLIENT_ID')
DISCORD_CLIENT_SECRET = os.getenv('DISCORD_CLIENT_SECRET')
DISCORD_REDIRECT_URI = os.getenv('DISCORD_REDIRECT_URI')
BOT_TOKEN = os.getenv('BOT_TOKEN')
