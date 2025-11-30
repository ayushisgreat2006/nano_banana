import asyncio
import aiohttp
import logging
import json
import re
import io
import secrets
import hashlib
import os
from typing import Dict, List, Optional, Any
from aiogram import Bot, Dispatcher, types, F, Router, BaseMiddleware
from aiogram.filters import Command
from aiogram.types import BufferedInputFile, InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import StatesGroup, State
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import DuplicateKeyError
from datetime import datetime, timezone
from seleniumbase import SB
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

API_TOKEN = os.getenv("API_TOKEN")
BEARER_TOKEN = os.getenv("BEARER_TOKEN")
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
OWNER_ID = int(os.getenv("OWNER_ID", "0"))
FORCE_JOIN_CHANNEL = os.getenv("FORCE_JOIN_CHANNEL", "@yourchannel")
LOG_GROUP_ID = int(os.getenv("LOG_GROUP_ID")) if os.getenv("LOG_GROUP_ID") else None

COOKIE_FILE_CONTENT = os.getenv("COOKIE_FILE_CONTENT", """# Netscape HTTP Cookie File
geminigen.ai	FALSE	/	FALSE	1779779317	ext_name	ojplmecpdpgccookcobabopnaifgidhf
geminigen.ai	FALSE	/	FALSE	1779741622	i18n_redirected	en
.geminigen.ai	TRUE	/	TRUE	1779741772	cf_clearance	6azc623mvyLqCfSRQZvLt3JCLs_lqXVIlYCUOAE3770-1764189771-1.2.1.1-dTH3sePAT0USkZbzKNjwE1dzzgJ5V6p7iuW6TMuQ_6sYmZsxVpJREHoDuolv9gfwvOKlURyCynaKbUOLS0aHsZj1pe72wdtYZUAOqkQ1sIFrBREfEoJh.s763UkmcFZdXlNdWOLaTmeo4TSFgyKkCVmxPUfWtNYlrxXsYG18B.HmBYgT.9EkTVduLdVeD7QqCClAlvuYU7JXp7TYBih8XtAEsMv78zBirZLxrEkyvvI
""")

NSFW_WORDS = {"nude", "nudes", "xxx", "naked", "remove cloth", "remove clothes", "nsfw", "porn", "sex", "explicit", "nudity", "undress", "strip"}
NSFW_PATTERN = re.compile(r'\b(?:' + '|'.join(re.escape(word) for word in NSFW_WORDS) + r')\b', re.IGNORECASE)

def parse_netscape_cookies(content: str) -> dict:
    cookies = {}
    for line in content.splitlines():
        line = line.strip()
        if line and not line.startswith('#') and '\t' in line:
            try:
                parts = line.split('\t')
                if len(parts) >= 7:
                    name, value = parts[5], parts[6]
                    cookies[name] = value
            except Exception:
                continue
    return cookies

def validate_config():
    required = {"API_TOKEN": API_TOKEN, "BEARER_TOKEN": BEARER_TOKEN, "MONGODB_URI": MONGODB_URI}
    for name, value in required.items():
        if not value or value == f"YOUR_{name}":
            logger.error(f"❌ Configuration error: {name} is not set!")
            raise ValueError(f"Configuration error: {name} must be set!")

class Database:
    def __init__(self, uri: str):
        try:
            logger.info("🔌 Connecting to MongoDB...")
            self.client = MongoClient(uri, serverSelectionTimeoutMS=5000)
            self.client.server_info()
            self.db = self.client.gemini_bot
            self.users: Collection = self.db.users
            self.groups: Collection = self.db.groups
            self.coupons: Collection = self.db.coupons
            self.referrals: Collection = self.db.referrals
            self._create_indexes()
            logger.info("✅ MongoDB connected successfully")
        except Exception as e:
            logger.error(f"❌ MongoDB connection failed: {e}")
            raise
    
    def _create_indexes(self):
        try:
            self.users.create_index("user_id", unique=True)
            self.groups.create_index("chat_id", unique=True)
            self.coupons.create_index("code", unique=True)
            self.referrals.create_index("code", unique=True)
            self.referrals.create_index("referred_by")
        except Exception as e:
            logger.warning(f"Index creation warning: {e}")
    
    def get_user(self, user_id: int) -> Optional[Dict[str, Any]]:
        return self.users.find_one({"user_id": user_id})
    
    def create_user(self, user_id: int, username: str = None) -> Dict[str, Any]:
        user = {
            "user_id": user_id,
            "username": username,
            "joined_at": datetime.now(timezone.utc),
            "credits": {"image_daily": 10, "video_daily": 4, "image_one_time": 0, "video_one_time": 0},
            "generation_count": {"images": 0, "videos": 0},
            "referred_by": None,
            "is_admin": False,
            "blocked": False,
            "nsfw_warnings": 0,
            "last_used_date": str(datetime.now(timezone.utc).date())
        }
        try:
            self.users.insert_one(user)
        except DuplicateKeyError:
            return self.get_user(user_id)
        return user
    
    def update_credits(self, user_id: int, credit_type: str, amount: int):
        valid_types = ["image_daily", "video_daily", "image_one_time", "video_one_time"]
        if credit_type not in valid_types:
            raise ValueError(f"Invalid credit type: {credit_type}")
        self.users.update_one({"user_id": user_id}, {"$inc": {f"credits.{credit_type}": amount}})
    
    def use_credit(self, user_id: int, media_type: str) -> bool:
        if media_type not in ["image", "video"]:
            raise ValueError(f"Invalid media type: {media_type}")
        user = self.get_user(user_id)
        if not user or user.get("blocked"):
            return False
        today = datetime.now(timezone.utc).date()
        last_used = user.get("last_used_date")
        if last_used != str(today):
            self.users.update_one({"user_id": user_id}, {"$set": {"credits.image_daily": 10, "credits.video_daily": 4, "last_used_date": str(today)}})
            user = self.get_user(user_id)
        credits = user["credits"]
        daily_key = f"{media_type}_daily"
        one_time_key = f"{media_type}_one_time"
        if credits.get(daily_key, 0) > 0:
            self.update_credits(user_id, daily_key, -1)
            self.users.update_one({"user_id": user_id}, {"$inc": {f"generation_count.{media_type}s": 1}})
            return True
        if credits.get(one_time_key, 0) > 0:
            self.update_credits(user_id, one_time_key, -1)
            self.users.update_one({"user_id": user_id}, {"$inc": {f"generation_count.{media_type}s": 1}})
            return True
        return False
    
    def ban_user(self, user_id: int, reason: str = "nsfw violation", banned_by: int = None):
        self.users.update_one({"user_id": user_id}, {"$set": {"blocked": True, "ban_reason": reason, "banned_at": datetime.now(timezone.utc), "banned_by": banned_by}})
    
    def unban_user(self, user_id: int, unbanned_by: int = None):
        self.users.update_one({"user_id": user_id}, {"$unset": {"ban_reason": "", "banned_at": "", "banned_by": ""}, "$set": {"blocked": False, "nsfw_warnings": 0, "unbanned_at": datetime.now(timezone.utc), "unbanned_by": unbanned_by}})
    
    def add_whitelist(self, user_id: int, media_type: str, value: int):
        if not self.get_user(user_id):
            self.create_user(user_id)
        key = f"credits.{media_type}"
        self.users.update_one({"user_id": user_id}, {"$inc": {key: value}})
    
    def create_coupon(self, code: str, media_type: str, value: int, created_by: int):
        self.coupons.insert_one({"code": code, "media_type": media_type, "value": value, "created_by": created_by, "created_at": datetime.now(timezone.utc), "used_by": None, "used_at": None})
    
    def redeem_coupon(self, user_id: int, code: str) -> bool:
        coupon = self.coupons.find_one({"code": code, "used_by": None})
        if not coupon:
            return False
        credit_map = {"pic": "image_one_time", "vdo": "video_one_time"}
        credit_type = credit_map.get(coupon["media_type"])
        if credit_type:
            self.update_credits(user_id, credit_type, coupon["value"])
            self.coupons.update_one({"code": code}, {"$set": {"used_by": user_id, "used_at": datetime.now(timezone.utc)}})
            return True
        return False
    
    def create_referral(self, user_id: int) -> str:
        code = secrets.token_urlsafe(8)
        self.referrals.insert_one({"code": code, "referred_by": user_id, "created_at": datetime.now(timezone.utc), "used_by": []})
        return code
    
    def use_referral(self, user_id: int, code: str) -> bool:
        if self.referrals.find_one({"used_by": user_id}):
            return False
        ref = self.referrals.find_one({"code": code, "referred_by": {"$ne": user_id}})
        if not ref:
            return False
        self.update_credits(ref["referred_by"], "image_one_time", 15)
        self.update_credits(ref["referred_by"], "video_one_time", 10)
        self.update_credits(user_id, "image_one_time", 10)
        self.update_credits(user_id, "video_one_time", 5)
        self.referrals.update_one({"code": code}, {"$push": {"used_by": user_id}})
        return True
    
    def get_all_users(self) -> List[int]:
        try:
            return [user["user_id"] for user in self.users.find({}, {"user_id": 1})]
        except Exception as e:
            logger.error(f"Failed to get all users: {e}")
            return []

db = Database(MONGODB_URI)

class UserManagerMiddleware(BaseMiddleware):
    async def __call__(self, handler, event, data):
        if not isinstance(event, (types.Message, types.CallbackQuery)):
            return await handler(event, data)
        user_id = None
        username = None
        chat_id = None
        if isinstance(event, types.Message):
            user_id = event.from_user.id if event.from_user else None
            username = event.from_user.username if event.from_user else None
            chat_id = event.chat.id if event.chat else None
        elif isinstance(event, types.CallbackQuery):
            user_id = event.from_user.id
            username = event.from_user.username
            chat_id = event.message.chat.id if event.message else None
        if chat_id and chat_id < 0:
            try:
                db.groups.update_one({"chat_id": chat_id}, {"$set": {"last_active": datetime.now(timezone.utc)}}, upsert=True)
            except Exception as e:
                logger.error(f"Failed to update group {chat_id}: {e}")
        if not user_id:
            return await handler(event, data)
        try:
            user = db.get_user(user_id)
            if not user:
                user = db.create_user(user_id, username)
            if user.get("blocked"):
                reason = user.get("ban_reason", "Violation of bot rules")
                if isinstance(event, types.Message):
                    await event.reply(f"You are banned ({reason}) contact owner for further classification")
                elif isinstance(event, types.CallbackQuery):
                    await event.answer(f"You are banned ({reason}) contact owner for further classification", show_alert=True)
                return
            data["user"] = user
            data["db"] = db
        except Exception as e:
            logger.error(f"User manager error for {user_id}: {e}")
        return await handler(event, data)

class ForceJoinMiddleware(BaseMiddleware):
    def __init__(self, channel_username: str):
        self.channel_username = channel_username
    async def __call__(self, handler, event, data):
        if isinstance(event, (types.Message, types.CallbackQuery)):
            if isinstance(event, types.Message) and event.text and (event.text.startswith("/help") or event.text.startswith("/start")):
                return await handler(event, data)
            user_id = event.from_user.id
            try:
                member = await bot.get_chat_member(self.channel_username, user_id)
                if member.status in ["member", "administrator", "creator"]:
                    return await handler(event, data)
            except Exception as e:
                logger.warning(f"Failed to check membership for {user_id}: {e}")
                return await handler(event, data)
            keyboard = InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text="Join Channel", url=f"https://t.me/{self.channel_username.strip('@')}")]])
            message_obj = event if isinstance(event, types.Message) else event.message
            if message_obj:
                await message_obj.reply(f"❌ You must join {self.channel_username} to use this bot!", reply_markup=keyboard)
            return
        return await handler(event, data)

class NSFWFilterMiddleware(BaseMiddleware):
    async def __call__(self, handler, event, data):
        if isinstance(event, types.Message) and event.text:
            if NSFW_PATTERN.search(event.text):
                user_id = event.from_user.id
                db.users.update_one({"user_id": user_id}, {"$inc": {"nsfw_warnings": 1}})
                user = db.get_user(user_id)
                if not user:
                    return await handler(event, data)
                warnings = user.get("nsfw_warnings", 0)
                if warnings >= 3:
                    db.ban_user(user_id, reason="nsfw violation")
                    await event.reply("❌ You have been permanently blocked for NSFW content!")
                    return
                else:
                    await event.reply(f"⚠️ NSFW content detected! Warning {warnings}/3\nDo not use NSFW prompt or you will be permanently blocked!")
                    return
        return await handler(event, data)

class BananaStates(StatesGroup):
    waiting_for_prompt = State()
    waiting_for_photo = State()
    waiting_for_edit_prompt = State()

class VideoStates(StatesGroup):
    waiting_for_video_prompt = State()

class GeminiGenAPI:
    def __init__(self, cookies: dict, bearer_token: str):
        self.cookies = cookies
        self.bearer_token = bearer_token
        self.base_url = "https://api.geminigen.ai"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
            "Accept": "application/json, text/plain, */*",
            "Origin": "https://geminigen.ai",
            "Referer": "https://geminigen.ai/",
            "Authorization": f"Bearer {self.bearer_token}",
        }
    
    async def generate_image(self, prompt: str, image_bytes: bytes = None, aspect_ratio: str = "16:9") -> str:
        async with aiohttp.ClientSession(cookies=self.cookies, headers=self.headers) as session:
            endpoint = f"{self.base_url}/api/generate_image"
            form = aiohttp.FormData()
            form.add_field('prompt', prompt)
            form.add_field('model', 'imagen-pro')
            form.add_field('aspect_ratio', aspect_ratio)
            form.add_field('style', 'None')
            if image_bytes:
                form.add_field('files', image_bytes, filename="reference.jpg", content_type='image/jpeg')
            async with session.post(endpoint, data=form) as resp:
                if resp.status not in (200, 202):
                    text = await resp.text()
                    if resp.status == 429:
                        raise Exception("Rate limited by API. Please try again in a few minutes.")
                    raise Exception(f"Generation failed: HTTP {resp.status}\nResponse: {text[:500]}")
                result = await resp.json()
                job_id = result.get("uuid") or result.get("id")
                if not job_id:
                    raise Exception(f"No job_id found: {result}")
                return job_id
    
    def _get_turnstile_token_sync(self) -> Optional[str]:
        try:
            with SB(uc=True, headless=True, disable_gpu=True, no_sandbox=True) as sb:
                sb.open("https://geminigen.ai/video")
                sb.wait_for_ready_state_complete()
                sb.sleep(5)
                token = sb.execute_script("""
                    const tokenInput = document.querySelector('input[name="turnstile_token"]');
                    if (tokenInput && tokenInput.value) return tokenInput.value;
                    const forms = document.querySelectorAll('form');
                    for (let form of forms) {
                        const hiddenInput = form.querySelector('input[name="turnstile_token"]');
                        if (hiddenInput && hiddenInput.value) return hiddenInput.value;
                    }
                    const turnstileElements = document.querySelectorAll('[name="cf-turnstile-response"]');
                    for (let el of turnstileElements) {
                        if (el.value) return el.value;
                    }
                    return null;
                """)
                if token:
                    return token
                sb.sleep(5)
                token = sb.execute_script("""
                    const tokenInput = document.querySelector('input[name="turnstile_token"]');
                    return tokenInput ? tokenInput.value : null;
                """)
                return token
        except Exception as e:
            logger.error(f"Browser automation failed: {e}")
            return None
    
    async def solve_turnstile_free(self) -> str:
        logger.info("🔄 Launching browser to solve Turnstile (15-30s)...")
        loop = asyncio.get_event_loop()
        token = await loop.run_in_executor(executor, self._get_turnstile_token_sync)
        if not token:
            raise Exception("❌ Failed to get Turnstile token via browser automation")
        logger.info("✅ Turnstile token obtained!")
        return token
    
    async def generate_video(self, prompt: str) -> str:
        async with aiohttp.ClientSession(cookies=self.cookies, headers=self.headers) as session:
            endpoint = f"{self.base_url}/api/video-gen/veo"
            turnstile_token = await self.solve_turnstile_free()
            form = aiohttp.FormData()
            form.add_field('prompt', prompt)
            form.add_field('model', 'veo-3-fast')
            form.add_field('duration', '8')
            form.add_field('resolution', '720p')
            form.add_field('aspect_ratio', '16:9')
            form.add_field('enhance_prompt', 'true')
            form.add_field('turnstile_token', turnstile_token)
            async with session.post(endpoint, data=form) as resp:
                if resp.status not in (200, 202):
                    text = await resp.text()
                    if resp.status == 429:
                        raise Exception("Rate limited by API. Please try again in a few minutes.")
                    raise Exception(f"Generation failed: HTTP {resp.status}\nResponse: {text[:500]}")
                result = await resp.json()
                job_id = result.get("uuid") or result.get("id")
                if not job_id:
                    raise Exception(f"No job_id found: {result}")
                return job_id
    
    async def poll_for_image(self, job_id: str, timeout: int = 300) -> str:
        async with aiohttp.ClientSession(cookies=self.cookies, headers=self.headers) as session:
            start = datetime.now()
            endpoint = f"{self.base_url}/api/history/{job_id}"
            await asyncio.sleep(5)
            retry_count = 0
            max_retries = 5
            while True:
                elapsed = (datetime.now() - start).total_seconds()
                if elapsed > timeout:
                    raise TimeoutError(f"Timeout after {timeout}s")
                try:
                    async with session.get(endpoint) as resp:
                        if resp.status != 200:
                            text = await resp.text()
                            retry_count += 1
                            if retry_count >= max_retries:
                                raise Exception(f"Max retries ({max_retries}) reached: HTTP {resp.status}")
                            await asyncio.sleep(min(10 * (retry_count + 1), 30))
                            continue
                        result = await resp.json()
                        image_url = None
                        if "generated_image" in result and isinstance(result["generated_image"], list):
                            for img_item in result["generated_image"]:
                                if isinstance(img_item, dict):
                                    possible_fields = ['image_url', 'file_download_url', 'download_url', 'url']
                                    for field in possible_fields:
                                        if field in img_item and img_item[field]:
                                            image_url = img_item[field]
                                            break
                                    if image_url:
                                        break
                        if not image_url:
                            top_fields = ['image_url', 'download_url', 'url', 'media_url', 'output_url']
                            for field in top_fields:
                                if field in result and result[field]:
                                    image_url = result[field]
                                    break
                        if not image_url:
                            result_str = json.dumps(result)
                            img_matches = re.findall(r'https?://[^\s"]+\.(?:png|jpg|jpeg|webp)(?:\?[^\s"]*)?', result_str, re.IGNORECASE)
                            if img_matches:
                                image_url = img_matches[0]
                        if image_url:
                            return image_url
                        status = result.get("status", "")
                        progress = result.get("status_percentage", 0)
                        error_message = result.get("error_message", "")
                        if status in [0, "failed", "error"]:
                            raise Exception(f"Generation failed with status: {status}, error: {error_message}")
                        if status in [1, "processing", "queued"] or progress < 100:
                            await asyncio.sleep(8)
                            continue
                        await asyncio.sleep(5)
                except Exception as e:
                    logger.exception(f"Polling error: {e}")
                    retry_count += 1
                    if retry_count >= max_retries:
                        raise Exception(f"Max retries ({max_retries}) reached: {str(e)}")
                    await asyncio.sleep(min(10 * (retry_count + 1), 30))
    
    async def poll_for_video(self, job_id: str, timeout: int = 300) -> str:
        async with aiohttp.ClientSession(cookies=self.cookies, headers=self.headers) as session:
            start = datetime.now()
            endpoint = f"{self.base_url}/api/history/{job_id}"
            await asyncio.sleep(5)
            retry_count = 0
            max_retries = 5
            while True:
                elapsed = (datetime.now() - start).total_seconds()
                if elapsed > timeout:
                    raise TimeoutError(f"Timeout after {timeout}s")
                try:
                    async with session.get(endpoint) as resp:
                        if resp.status != 200:
                            text = await resp.text()
                            retry_count += 1
                            if retry_count >= max_retries:
                                raise Exception(f"Max retries ({max_retries}) reached")
                            await asyncio.sleep(min(10 * (retry_count + 1), 30))
                            continue
                        result = await resp.json()
                        video_url = None
                        if "generated_video" in result and isinstance(result["generated_video"], list):
                            for video_item in result["generated_video"]:
                                if isinstance(video_item, dict):
                                    possible_fields = ['video_url', 'file_download_url', 'download_url', 'url']
                                    for field in possible_fields:
                                        if field in video_item and video_item[field]:
                                            video_url = video_item[field]
                                            break
                                    if video_url:
                                        break
                        if not video_url:
                            top_fields = ['video_url', 'download_url', 'url', 'media_url', 'output_url']
                            for field in top_fields:
                                if field in result and result[field]:
                                    video_url = result[field]
                                    break
                        if not video_url:
                            result_str = json.dumps(result)
                            mp4_matches = re.findall(r'https?://[^\s"]+\.mp4(?:\?[^\s"]*)?', result_str)
                            if mp4_matches:
                                video_url = mp4_matches[0]
                        if video_url:
                            return video_url
                        status = result.get("status", "")
                        progress = result.get("status_percentage", 0)
                        error_message = result.get("error_message", "")
                        if status in [0, "failed", "error"]:
                            raise Exception(f"Generation failed: {error_message}")
                        if status in [1, "processing", "queued"] or progress < 100:
                            await asyncio.sleep(5)
                            continue
                        await asyncio.sleep(3)
                except Exception as e:
                    logger.exception(f"Video polling error: {e}")
                    retry_count += 1
                    if retry_count >= max_retries:
                        raise Exception(f"Max retries ({max_retries}) reached: {str(e)}")
                    await asyncio.sleep(min(10 * (retry_count + 1), 30))
    
    async def download_media(self, url: str, max_size: int = 50 * 1024 * 1024) -> bytes:
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=300)) as resp:
                    if resp.status != 200:
                        raise Exception(f"Download failed: HTTP {resp.status}")
                    size = int(resp.headers.get('content-length', 0))
                    if size > max_size:
                        raise Exception(f"File too large: {size / 1024 / 1024:.1f}MB > {max_size / 1024 / 1024:.1f}MB limit")
                    return await resp.read()
            except asyncio.TimeoutError:
                raise Exception("Download timeout after 5 minutes")

bot = Bot(token=API_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp = Dispatcher()
router = Router()
dp.include_router(router)
dp.message.middleware(UserManagerMiddleware())
dp.callback_query.middleware(UserManagerMiddleware())
dp.message.middleware(ForceJoinMiddleware(FORCE_JOIN_CHANNEL))
dp.message.middleware(NSFWFilterMiddleware())

def banana_menu_keyboard():
    return InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text="🎨 Generate New Image", callback_data="banana_new")], [InlineKeyboardButton(text="🖼️ Edit Image (Ref.)", callback_data="banana_edit")]])

@router.message(Command("start"))
async def cmd_start(msg: types.Message, user: Dict):
    args = msg.text.split(maxsplit=1)[1:] if len(msg.text.split()) > 1 else []
    if args and args[0].startswith("ref_"):
        ref_code = args[0]
        try:
            if db.use_referral(msg.from_user.id, ref_code):
                await msg.answer("🎉 Referral successful! You received 10 image and 5 video credits!")
            else:
                await msg.answer("❌ Invalid referral code or already used!")
        except Exception as e:
            logger.error(f"Referral error: {e}")
            await msg.answer("❌ Error processing referral. Please try again.")
    await msg.reply("🎬 <b>GeminiGen Multi-Tool Bot</b>\n\n🍌 /banana - Nano Banana image generation\n📹 /video - Video generation (4 credits/day)\n🔗 /refer - Get your referral link\n🎟️ /predeem - Redeem image coupon\n🎟️ /vredeem - Redeem video coupon\n\n💬 Support: @ayushxchat_robot\n👨‍💻 Developed by: @mahadev_ki_iccha", parse_mode="HTML")

@router.message(Command("help"))
async def cmd_help(msg: types.Message):
    await msg.reply("🆘 <b>Help</b>\n\nThis bot generates images and videos using AI.\n\n<b>Commands:</b>\n/start - Start the bot\n/help - Show this help\n/banana - Generate images\n/video - Generate videos\n/refer - Get referral link\n/predeem - Redeem coupon\n/vredeem - Redeem video coupon\n\n⚠️ NSFW content is strictly prohibited!", parse_mode="HTML")

@router.message(Command("banana"))
async def cmd_banana(msg: types.Message):
    await msg.reply("🍌 <b>Nano Banana Menu</b>\n\nChoose what you want to do:", reply_markup=banana_menu_keyboard(), parse_mode="HTML")

@router.message(Command("video"))
async def cmd_video(msg: types.Message, state: FSMContext, user: Dict):
    if not db.use_credit(msg.from_user.id, "video"):
        await msg.answer("❌ No video credits left. Use /refer or redeem coupons to earn more!")
        return
    await msg.answer("📹 <b>Send me a prompt for video generation:</b>", parse_mode="HTML")
    await state.set_state(VideoStates.waiting_for_video_prompt)

@router.message(Command("refer"))
async def cmd_refer(msg: types.Message, user: Dict):
    if len(msg.text.split()) > 1:
        code = msg.text.split()[1]
        try:
            if db.use_referral(msg.from_user.id, code):
                await msg.answer("🎉 Referral successful! You received 10 image and 5 video credits!")
            else:
                await msg.answer("❌ Invalid referral code or already used!")
        except Exception as e:
            logger.error(f"Referral usage error: {e}")
            await msg.answer("❌ Error processing referral. Please try again.")
    else:
        try:
            existing = db.referrals.find_one({"referred_by": msg.from_user.id})
            if existing:
                code = existing["code"]
            else:
                code = db.create_referral(msg.from_user.id)
            bot_info = await bot.get_me()
            bot_username = bot_info.username
            link = f"https://t.me/{bot_username}?start=ref_{code}"
            await msg.answer(f"🔗 <b>Your Referral Link:</b>\n<code>{link}</code>\n\nShare this with friends! You get 15 image + 10 video credits when they join.\nThey get 10 image + 5 video credits.", parse_mode="HTML")
        except Exception as e:
            logger.error(f"Referral generation error: {e}")
            await msg.answer("❌ Error generating referral link. Please try again.")

@router.message(Command("predeem"))
async def cmd_predeem(msg: types.Message, user: Dict):
    if len(msg.text.split()) < 2:
        await msg.answer("Usage: /predeem <coupon_code>")
        return
    code = msg.text.split()[1]
    try:
        if db.redeem_coupon(msg.from_user.id, code):
            await msg.answer("🎉 Coupon redeemed successfully! Image credits added.")
        else:
            await msg.answer("❌ Invalid or already used coupon!")
    except Exception as e:
        logger.error(f"Coupon redemption error: {e}")
        await msg.answer("❌ Error redeeming coupon. Please try again.")

@router.message(Command("vredeem"))
async def cmd_vredeem(msg: types.Message, user: Dict):
    if len(msg.text.split()) < 2:
        await msg.answer("Usage: /vredeem <coupon_code>")
        return
    code = msg.text.split()[1]
    try:
        if db.redeem_coupon(msg.from_user.id, code):
            await msg.answer("🎉 Coupon redeemed successfully! Video credits added.")
        else:
            await msg.answer("❌ Invalid or already used coupon!")
    except Exception as e:
        logger.error(f"Video coupon redemption error: {e}")
        await msg.answer("❌ Error redeeming coupon. Please try again.")

def is_owner(user_id: int) -> bool:
    return user_id == OWNER_ID

def is_admin(user_id: int) -> bool:
    user = db.get_user(user_id)
    return user and (user.get("is_admin") or is_owner(user_id))

@router.message(Command("stats"))
async def cmd_stats(msg: types.Message, user: Dict):
    if not is_admin(msg.from_user.id):
        await msg.answer("❌ You don't have permission!")
        return
    try:
        total_users = db.users.count_documents({})
        total_groups = db.groups.count_documents({})
        total_images_result = db.users.aggregate([{"$group": {"_id": None, "total": {"$sum": "$generation_count.images"}}}])
        total_images = next(total_images_result, {}).get("total", 0)
        total_videos_result = db.users.aggregate([{"$group": {"_id": None, "total": {"$sum": "$generation_count.videos"}}}])
        total_videos = next(total_videos_result, {}).get("total", 0)
        await msg.answer(f"📊 <b>Bot Statistics</b>\n\n👥 Total Users: {total_users}\n👥 Total Groups: {total_groups}\n🖼️ Total Images: {total_images}\n📹 Total Videos: {total_videos}", parse_mode="HTML")
    except Exception as e:
        logger.error(f"Stats command error: {e}")
        await msg.answer("❌ Error retrieving statistics. Please try again.")

@router.message(Command("ban"))
async def cmd_ban(msg: types.Message, user: Dict):
    if not is_admin(msg.from_user.id):
        await msg.answer("❌ Admin only!")
        return
    try:
        parts = msg.text.split(maxsplit=2)
        if len(parts) < 2:
            await msg.answer("Usage: /ban <user_id> [reason]")
            return
        target_id = int(parts[1])
        reason = parts[2] if len(parts) > 2 else "nsfw violation"
        if target_id == OWNER_ID:
            await msg.answer("❌ Cannot ban the owner!")
            return
        if is_admin(target_id) and not is_owner(msg.from_user.id):
            await msg.answer("❌ You cannot ban other admins!")
            return
        if target_id == msg.from_user.id:
            await msg.answer("❌ You cannot ban yourself!")
            return
        db.ban_user(target_id, reason, banned_by=msg.from_user.id)
        await msg.answer(f"✅ User {target_id} has been banned.\nReason: {reason}")
        try:
            await bot.send_message(target_id, f"❌ You have been banned from using this bot.\n\nReason: {reason}\n\nIf you believe this is a mistake, contact the bot owner.")
        except Exception as e:
            logger.warning(f"Could not notify banned user {target_id}: {e}")
    except ValueError:
        await msg.answer("❌ Invalid user ID!")
    except Exception as e:
        logger.error(f"Ban error: {e}")
        await msg.answer("❌ Error banning user. Please try again.")

@router.message(Command("unban"))
async def cmd_unban(msg: types.Message, user: Dict):
    if not is_admin(msg.from_user.id):
        await msg.answer("❌ Admin only!")
        return
    try:
        parts = msg.text.split(maxsplit=2)
        if len(parts) < 2:
            await msg.answer("Usage: /unban <user_id> [reason]")
            return
        target_id = int(parts[1])
        unban_reason = parts[2] if len(parts) > 2 else "Appeal accepted"
        target_user = db.get_user(target_id)
        if not target_user or not target_user.get("blocked"):
            await msg.answer(f"❌ User {target_id} is not banned!")
            return
        db.unban_user(target_id, unbanned_by=msg.from_user.id)
        await msg.answer(f"✅ User {target_id} has been unbanned.\nReason: {unban_reason}")
        try:
            await bot.send_message(target_id, f"✅ You have been unbanned from the bot!\n\nReason: {unban_reason}\n\nHappy generating! 🎉")
        except Exception as e:
            logger.warning(f"Could not notify unbanned user {target_id}: {e}")
    except ValueError:
        await msg.answer("❌ Invalid user ID!")
    except Exception as e:
        logger.error(f"Unban error: {e}")
        await msg.answer("❌ Error unbanning user. Please try again.")

@router.message(Command("pic_whitelist"))
async def cmd_pic_whitelist(msg: types.Message, user: Dict):
    if not is_owner(msg.from_user.id):
        await msg.answer("❌ Only owner can use this!")
        return
    try:
        parts = msg.text.split()
        if len(parts) != 3:
            raise ValueError("Invalid format - expected 2 arguments")
        _, value, target_id = parts
        db.add_whitelist(int(target_id), "image_daily", int(value))
        await msg.answer(f"✅ Whitelisted {target_id} for {value} image credits/day")
    except ValueError as e:
        await msg.answer(f"❌ Usage: /pic_whitelist <value> <user_id>\nError: {e}")
    except Exception as e:
        logger.error(f"Pic whitelist error: {e}")
        await msg.answer("❌ Error updating whitelist. Please try again.")

@router.message(Command("vdo_whitelist"))
async def cmd_vdo_whitelist(msg: types.Message, user: Dict):
    if not is_owner(msg.from_user.id):
        await msg.answer("❌ Only owner can use this!")
        return
    try:
        parts = msg.text.split()
        if len(parts) != 3:
            raise ValueError("Invalid format - expected 2 arguments")
        _, value, target_id = parts
        db.add_whitelist(int(target_id), "video_daily", int(value))
        await msg.answer(f"✅ Whitelisted {target_id} for {value} video credits/day")
    except ValueError as e:
        await msg.answer(f"❌ Usage: /vdo_whitelist <value> <user_id>\nError: {e}")
    except Exception as e:
        logger.error(f"Video whitelist error: {e}")
        await msg.answer("❌ Error updating whitelist. Please try again.")

@router.message(Command("pic_coupon"))
async def cmd_pic_coupon(msg: types.Message, user: Dict):
    if not is_admin(msg.from_user.id):
        await msg.answer("❌ Admin only!")
        return
    try:
        _, value, name = msg.text.split(maxsplit=2)
        db.create_coupon(name, "pic", int(value), msg.from_user.id)
        await msg.answer(f"🎟️ Coupon created: <code>{name}</code>\nValue: {value} image credits", parse_mode="HTML")
    except ValueError:
        await msg.answer("❌ Usage: /pic_coupon <value> <name>")
    except Exception as e:
        logger.error(f"Pic coupon creation error: {e}")
        await msg.answer("❌ Error creating coupon. Please try again.")

@router.message(Command("vdo_coupon"))
async def cmd_vdo_coupon(msg: types.Message, user: Dict):
    if not is_admin(msg.from_user.id):
        await msg.answer("❌ Admin only!")
        return
    try:
        _, value, name = msg.text.split(maxsplit=2)
        db.create_coupon(name, "vdo", int(value), msg.from_user.id)
        await msg.answer(f"🎟️ Coupon created: <code>{name}</code>\nValue: {value} video credits", parse_mode="HTML")
    except ValueError:
        await msg.answer("❌ Usage: /vdo_coupon <value> <name>")
    except Exception as e:
        logger.error(f"Video coupon creation error: {e}")
        await msg.answer("❌ Error creating coupon. Please try again.")

@router.message(Command("broadcast"))
async def cmd_broadcast(msg: types.Message):
    if not is_owner(msg.from_user.id):
        await msg.answer("❌ Owner only!")
        return
    text = msg.text[11:].strip()
    if not text:
        await msg.answer("Usage: /broadcast <message>")
        return
    if len(text) > 4000:
        await msg.answer("❌ Message too long! Maximum 4000 characters.")
        return
    confirm_id = hashlib.sha256(f"{msg.message_id}:{text}".encode()).hexdigest()[:16]
    keyboard = InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text="✅ Confirm Broadcast", callback_data=f"broadcast_confirm_{confirm_id}")], [InlineKeyboardButton(text="❌ Cancel", callback_data="broadcast_cancel")]])
    await msg.answer(f"Broadcast to {db.users.count_documents({})} users?\n\n{text}", reply_markup=keyboard)

@router.callback_query(F.data.startswith("broadcast_confirm_"))
async def confirm_broadcast(cb: CallbackQuery):
    if not is_owner(cb.from_user.id):
        await cb.answer("❌ Not allowed!", show_alert=True)
        return
    if not cb.message:
        await cb.answer("❌ Message not found!", show_alert=True)
        return
    await cb.answer("📢 Broadcasting...", show_alert=False)
    await cb.message.edit_text("📢 Broadcasting in progress...\n\n⏳ Sending messages...")
    success = 0
    failed = 0
    user_ids = db.get_all_users()
    total = len(user_ids)
    for idx, user_id in enumerate(user_ids, 1):
        try:
            message_parts = cb.message.html_text.split('\n', 2)
            if len(message_parts) < 3:
                await cb.message.edit_text("❌ Invalid broadcast message format!")
                return
            message_text = message_parts[2]
            await bot.send_message(user_id, message_text, parse_mode="HTML")
            success += 1
            if idx % 50 == 0:
                await cb.message.edit_text(f"📢 Broadcasting in progress...\n\n⏳ Progress: {idx}/{total}\n✅ Sent: {success}\n❌ Failed: {failed}")
            await asyncio.sleep(0.05)
        except Exception as e:
            failed += 1
            logger.error(f"Broadcast failed to {user_id}: {e}")
    await cb.message.edit_text(f"✅ Broadcast complete!\n\n📊 Total: {total}\n✅ Sent: {success}\n❌ Failed: {failed}")

@router.callback_query(F.data == "broadcast_cancel")
async def cancel_broadcast(cb: CallbackQuery):
    if not is_owner(cb.from_user.id):
        await cb.answer("❌ Not allowed!", show_alert=True)
        return
    if not cb.message:
        await cb.answer("❌ Message not found!", show_alert=True)
        return
    await cb.answer("Cancelled", show_alert=False)
    await cb.message.edit_text("❌ Broadcast cancelled")

@router.callback_query(F.data == "banana_new")
async def process_new_image(cb: CallbackQuery, state: FSMContext, user: Dict):
    if not db.use_credit(cb.from_user.id, "image"):
        await cb.answer("❌ No image credits left!", show_alert=True)
        return
    await cb.answer()
    await cb.message.edit_text("🎨 <b>Send me a prompt for new image generation:</b>", parse_mode="HTML")
    await state.set_state(BananaStates.waiting_for_prompt)

@router.callback_query(F.data == "banana_edit")
async def process_edit_image(cb: CallbackQuery, state: FSMContext, user: Dict):
    if not db.use_credit(cb.from_user.id, "image"):
        await cb.answer("❌ No image credits left!", show_alert=True)
        return
    await cb.answer()
    await cb.message.edit_text("🖼️ <b>Send me the image you want to edit:</b>\n\n<i>Upload as a photo (not as file)</i>", parse_mode="HTML")
    await state.set_state(BananaStates.waiting_for_photo)

def detect_aspect_ratio(width: int, height: int) -> str:
    if width > height:
        return "16:9"
    elif height > width:
        return "9:16"
    else:
        return "1:1"

def add_caption_footer(original_caption: str = "") -> str:
    footer = ("Generated by @nano_banana_veobot\nYour image was successfully removed from our database !! 100% Private")
    if original_caption:
        return f"{original_caption}\n\n{footer}"
    return footer

@router.message(BananaStates.waiting_for_photo, F.photo)
async def handle_received_photo(msg: types.Message, state: FSMContext, user: Dict):
    photo = msg.photo[-1]
    if photo.file_size and photo.file_size > 10 * 1024 * 1024:
        await msg.reply("❌ Image too large! Please send a photo under 10MB.")
        return
    aspect_ratio = detect_aspect_ratio(photo.width, photo.height)
    await state.update_data(photo_file_id=photo.file_id, aspect_ratio=aspect_ratio)
    await msg.reply("✅ <b>Image received!</b>\n\n📐 <b>Detected aspect ratio:</b> <code>{aspect_ratio}</code>\n\nNow send me the edit prompt:", parse_mode="HTML")
    await state.set_state(BananaStates.waiting_for_edit_prompt)

@router.message(BananaStates.waiting_for_photo)
async def handle_no_photo(msg: types.Message):
    await msg.reply("❌ Please upload a photo (not a file). Try again:")

@router.message(BananaStates.waiting_for_edit_prompt, F.text)
async def handle_edit_prompt(msg: types.Message, state: FSMContext, user: Dict):
    prompt = msg.text.strip()
    if len(prompt) < 3:
        await msg.reply("❌ Prompt too short. Minimum 3 characters required. Try again:")
        return
    user_data = await state.get_data()
    photo_file_id = user_data.get("photo_file_id")
    aspect_ratio = user_data.get("aspect_ratio", "16:9")
    if not photo_file_id:
        await msg.reply("❌ Error: No image found. Start over with /banana")
        await state.clear()
        return
    status_msg = await msg.reply("📥 <b>Downloading your image...</b>", parse_mode="HTML")
    try:
        photo_file = await bot.get_file(photo_file_id)
        photo_bytes = await bot.download_file(photo_file.file_path)
        await status_msg.edit_text("🎨 <b>Generating edited image...</b> (~10-30s)", parse_mode="HTML")
        job_id = await api.generate_image(prompt, image_bytes=photo_bytes.getvalue(), aspect_ratio=aspect_ratio)
        await status_msg.edit_text("⏳ <b>Processing...</b>", parse_mode="HTML")
        image_url = await api.poll_for_image(job_id, timeout=300)
        await status_msg.edit_text("⬇️ <b>Downloading final image...</b>", parse_mode="HTML")
        image_bytes = await api.download_media(image_url)
        image_file = BufferedInputFile(image_bytes, filename=f"{job_id}.png")
        response_msg = await msg.answer_document(document=image_file, caption=add_caption_footer(f"✨ <b>Edited Image Ready!</b>\n\n📝 <b>Prompt:</b> <code>{prompt}</code>\n📐 <b>Ratio:</b> <code>{aspect_ratio}</code>\n🔖 <b>Job:</b> <code>{job_id[:8]}...</code>"), parse_mode="HTML")
        await status_msg.delete()
        if LOG_GROUP_ID:
            await bot.forward_message(LOG_GROUP_ID, msg.chat.id, response_msg.message_id)
    except Exception as e:
        logger.exception(f"Failed for edit prompt: {prompt}")
        await status_msg.edit_text(f"❌ <b>Error:</b>\n<code>{str(e)[:400]}</code>", parse_mode="HTML")
        if LOG_GROUP_ID:
            await bot.send_message(LOG_GROUP_ID, f"❌ Error for user {msg.from_user.id}:\n{str(e)[:400]}")
    await state.clear()

@router.message(BananaStates.waiting_for_edit_prompt)
async def handle_invalid_edit_prompt(msg: types.Message):
    await msg.reply("❌ Please send a text prompt. Try again:")

@router.message(BananaStates.waiting_for_prompt, F.text)
async def handle_new_image_prompt(msg: types.Message, state: FSMContext, user: Dict):
    prompt = msg.text.strip()
    if len(prompt) < 3:
        await msg.reply("❌ Prompt too short. Minimum 3 characters required. Try again:")
        return
    status_msg = await msg.reply("🎨 <b>Generating new image...</b> (~10-30s)", parse_mode="HTML")
    try:
        job_id = await api.generate_image(prompt, aspect_ratio="16:9")
        await status_msg.edit_text("⏳ <b>Processing...</b>", parse_mode="HTML")
        image_url = await api.poll_for_image(job_id, timeout=300)
        await status_msg.edit_text("⬇️ <b>Downloading final image...</b>", parse_mode="HTML")
        image_bytes = await api.download_media(image_url)
        image_file = BufferedInputFile(image_bytes, filename=f"{job_id}.png")
        response_msg = await msg.answer_document(document=image_file, caption=add_caption_footer(f"✨ <b>New Image Ready!</b>\n\n📝 <b>Prompt:</b> <code>{prompt}</code>\n🔖 <b>Job:</b> <code>{job_id[:8]}...</code>"), parse_mode="HTML")
        await status_msg.delete()
        if LOG_GROUP_ID:
            await bot.forward_message(LOG_GROUP_ID, msg.chat.id, response_msg.message_id)
    except Exception as e:
        logger.exception(f"Failed for new image prompt: {prompt}")
        await status_msg.edit_text(f"❌ <b>Error:</b>\n<code>{str(e)[:400]}</code>", parse_mode="HTML")
        if LOG_GROUP_ID:
            await bot.send_message(LOG_GROUP_ID, f"❌ Error for user {msg.from_user.id}:\n{str(e)[:400]}")
    await state.clear()

@router.message(BananaStates.waiting_for_prompt)
async def handle_invalid_prompt(msg: types.Message):
    await msg.reply("❌ Please send a text prompt. Try again:")

@router.message(VideoStates.waiting_for_video_prompt, F.text)
async def handle_video_prompt(msg: types.Message, state: FSMContext, user: Dict):
    prompt = msg.text.strip()
    if len(prompt) < 3:
        await msg.reply("❌ Prompt too short. Minimum 3 characters required. Try again:")
        return
    status_msg = await msg.reply("🎬 <b>Generating video...</b> (~30-90s)", parse_mode="HTML")
    try:
        job_id = await api.generate_video(prompt)
        await status_msg.edit_text("⏳ <b>Processing video...</b>", parse_mode="HTML")
        video_url = await api.poll_for_video(job_id, timeout=300)
        await status_msg.edit_text("⬇️ <b>Downloading final video...</b>", parse_mode="HTML")
        video_bytes = await api.download_media(video_url)
        video_file = BufferedInputFile(video_bytes, filename=f"{job_id}.mp4")
        response_msg = await msg.answer_video(video=video_file, caption=add_caption_footer(f"✨ <b>Video Ready!</b>\n\n📝 <b>Prompt:</b> <code>{prompt}</code>\n🔖 <b>Job:</b> <code>{job_id[:8]}...</code>"), parse_mode="HTML", width=1280, height=720, duration=8, supports_streaming=True)
        await status_msg.delete()
        if LOG_GROUP_ID:
            await bot.forward_message(LOG_GROUP_ID, msg.chat.id, response_msg.message_id)
    except Exception as e:
        logger.exception(f"Failed for video prompt: {prompt}")
        await status_msg.edit_text(f"❌ <b>Error:</b>\n<code>{str(e)[:400]}</code>", parse_mode="HTML")
        if LOG_GROUP_ID:
            await bot.send_message(LOG_GROUP_ID, f"❌ Error for user {msg.from_user.id}:\n{str(e)[:400]}")
    await state.clear()

executor = ThreadPoolExecutor(max_workers=2)
api = GeminiGenAPI(parse_netscape_cookies(COOKIE_FILE_CONTENT), BEARER_TOKEN)

async def on_startup():
    logger.info("=" * 60)
    logger.info("🤖 Nano Banana Bot Starting...")
    logger.info(f"Owner ID: {OWNER_ID}")
    logger.info(f"Force Join: {FORCE_JOIN_CHANNEL}")
    logger.info(f"Log Group ID: {LOG_GROUP_ID}")
    logger.info(f"Total Users: {db.users.count_documents({})}")
    logger.info("=" * 60)

async def on_shutdown():
    logger.info("🛑 Shutting down bot...")
    await bot.session.close()
    db.client.close()
    executor.shutdown(wait=True)
    logger.info("✅ Bot shutdown complete")

async def main():
    logger.info("🚀 Starting bot initialization...")
    try:
        await on_startup()
        logger.info("✅ Startup complete. Starting polling...")
        async def health_check():
            while True:
                await asyncio.sleep(30)
                logger.info("💓 Bot is alive and running...")
        asyncio.create_task(health_check())
        await dp.start_polling(bot, skip_updates=True)
    except Exception as e:
        logger.exception(f"❌ Fatal error in main: {e}")
        raise
    finally:
        await on_shutdown()

if __name__ == "__main__":
    try:
        logger.info("🎯 Starting bot process...")
        asyncio.run(main())
    except Exception as e:
        logger.exception(f"💥 Fatal error: {e}")
