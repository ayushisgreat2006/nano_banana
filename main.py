import asyncio
import aiohttp
import logging
import json
import re
import io
import secrets
import hashlib
import signal
import sys
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

# ========== LOGGING CONFIGURATION (TOP) ==========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s"
)
logger = logging.getLogger(__name__)
# ===============================================

# ========== CONFIGURATION (ENVIRONMENT VARIABLES) ==========
API_TOKEN = os.getenv("API_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN")
BEARER_TOKEN = os.getenv("BEARER_TOKEN", "YOUR_BEARER_TOKEN")
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
OWNER_ID = int(os.getenv("OWNER_ID", "123456789"))
FORCE_JOIN_CHANNEL = os.getenv("FORCE_JOIN_CHANNEL", "@yourchannel")
LOG_GROUP_ID = int(os.getenv("LOG_GROUP_ID", "-1001234567890")) if os.getenv("LOG_GROUP_ID") else None

COOKIE_FILE_CONTENT = """# Netscape HTTP Cookie File
geminigen.ai	FALSE	/	FALSE	1779779317	ext_name	ojplmecpdpgccookcobabopnaifgidhf
geminigen.ai	FALSE	/	FALSE	1779741622	i18n_redirected	en
.geminigen.ai	TRUE	/	TRUE	1779741772	cf_clearance	6azc623mvyLqCfSRQZvLt3JCLs_lqXVIlYCUOAE3770-1764189771-1.2.1.1-dTH3sePAT0USkZbzKNjwE1dzzgJ5V6p7iuW6TMuQ_6sYmZsxVpJREHoDuolv9gfwvOKlURyCynaKbUOLS0aHsZj1pe72wdtYZUAOqkQ1sIFrBREfEoJh.s763UkmcFZdXlNdWOLaTmeo4TSFgyKkCVmxPUfWtNYlrxXsYG18B.HmBYgT.9EkTVduLdVeD7QqCClAlvuYU7JXp7TYBih8XtAEsMv78zBirZLxrEkyvvI
"""
# ===========================================================

NSFW_WORDS = {
    "nude", "nudes", "xxx", "naked", "remove cloth", "remove clothes", 
    "nsfw", "porn", "sex", "explicit", "nudity", "undress", "strip"
}

NSFW_PATTERN = re.compile(r'\b(?:' + '|'.join(re.escape(word) for word in NSFW_WORDS) + r')\b', re.IGNORECASE)

def parse_netscape_cookies(content: str) -> dict:
    """Parse Netscape format cookies"""
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
    """Validate critical configuration values"""
    required = {
        "API_TOKEN": API_TOKEN,
        "BEARER_TOKEN": BEARER_TOKEN,
        "MONGODB_URI": MONGODB_URI,
    }
    
    for name, value in required.items():
        if not value or value == f"YOUR_{name}":
            raise ValueError(f"Configuration error: {name} must be set!")
    
    if OWNER_ID == 123456789:
        logger.warning("Using default OWNER_ID, consider changing it!")

class Database:
    def __init__(self, uri: str):
        try:
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
        """Create database indexes"""
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
            "credits": {
                "image_daily": 10,
                "video_daily": 4,
                "image_one_time": 0,
                "video_one_time": 0
            },
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
        """Increment credits by amount (can be negative)"""
        valid_types = ["image_daily", "video_daily", "image_one_time", "video_one_time"]
        if credit_type not in valid_types:
            raise ValueError(f"Invalid credit type: {credit_type}")
        
        self.users.update_one(
            {"user_id": user_id},
            {"$inc": {f"credits.{credit_type}": amount}}
        )
    
    def use_credit(self, user_id: int, media_type: str) -> bool:
        """Attempt to use a credit, returns True if successful"""
        if media_type not in ["image", "video"]:
            raise ValueError(f"Invalid media type: {media_type}")
        
        user = self.get_user(user_id)
        if not user or user.get("blocked"):
            return False
        
        today = datetime.now(timezone.utc).date()
        last_used = user.get("last_used_date")
        
        # Reset daily credits if new day
        if last_used != str(today):
            self.users.update_one(
                {"user_id": user_id},
                {"$set": {
                    "credits.image_daily": 10,
                    "credits.video_daily": 4,
                    "last_used_date": str(today)
                }}
            )
            user = self.get_user(user_id)
        
        credits = user["credits"]
        daily_key = f"{media_type}_daily"
        one_time_key = f"{media_type}_one_time"
        
        # Try daily credits first
        if credits.get(daily_key, 0) > 0:
            self.update_credits(user_id, daily_key, -1)
            # FIXED: Use direct DB update for generation count
            self.users.update_one(
                {"user_id": user_id},
                {"$inc": {f"generation_count.{media_type}s": 1}}
            )
            return True
        
        # Then try one-time credits
        if credits.get(one_time_key, 0) > 0:
            self.update_credits(user_id, one_time_key, -1)
            # FIXED: Use direct DB update for generation count
            self.users.update_one(
                {"user_id": user_id},
                {"$inc": {f"generation_count.{media_type}s": 1}}
            )
            return True
        
        return False
    
    def ban_user(self, user_id: int, reason: str = "nsfw violation", banned_by: int = None):
        """Ban a user with a specific reason"""
        self.users.update_one(
            {"user_id": user_id},
            {
                "$set": {
                    "blocked": True,
                    "ban_reason": reason,
                    "banned_at": datetime.now(timezone.utc),
                    "banned_by": banned_by
                }
            }
        )
    
    def unban_user(self, user_id: int, unbanned_by: int = None):
        """Unban a user and reset warnings"""
        self.users.update_one(
            {"user_id": user_id},
            {
                "$unset": {
                    "ban_reason": "",
                    "banned_at": "",
                    "banned_by": ""
                },
                "$set": {
                    "blocked": False,
                    "nsfw_warnings": 0,
                    "unbanned_at": datetime.now(timezone.utc),
                    "unbanned_by": unbanned_by
                }
            }
        )
    
    def add_whitelist(self, user_id: int, media_type: str, value: int):
        """Add whitelist credits (increments existing)"""
        if not self.get_user(user_id):
            self.create_user(user_id)
        
        key = f"credits.{media_type}"
        self.users.update_one(
            {"user_id": user_id},
            {"$inc": {key: value}}
        )
    
    def create_coupon(self, code: str, media_type: str, value: int, created_by: int):
        self.coupons.insert_one({
            "code": code,
            "media_type": media_type,
            "value": value,
            "created_by": created_by,
            "created_at": datetime.now(timezone.utc),
            "used_by": None,
            "used_at": None
        })
    
    def redeem_coupon(self, user_id: int, code: str) -> bool:
        coupon = self.coupons.find_one({"code": code, "used_by": None})
        if not coupon:
            return False
        
        credit_map = {
            "pic": "image_one_time",
            "vdo": "video_one_time"
        }
        credit_type = credit_map.get(coupon["media_type"])
        
        if credit_type:
            self.update_credits(user_id, credit_type, coupon["value"])
            self.coupons.update_one(
                {"code": code},
                {"$set": {"used_by": user_id, "used_at": datetime.now(timezone.utc)}}
            )
            return True
        return False
    
    def create_referral(self, user_id: int) -> str:
        code = secrets.token_urlsafe(8)
        self.referrals.insert_one({
            "code": code,
            "referred_by": user_id,
            "created_at": datetime.now(timezone.utc),
            "used_by": []
        })
        return code
    
    def use_referral(self, user_id: int, code: str) -> bool:
        if self.referrals.find_one({"used_by": user_id}):
            return False
        
        ref = self.referrals.find_one({"code": code, "referred_by": {"$ne": user_id}})
        if not ref:
            return False
        
        # Give credits to referrer
        self.update_credits(ref["referred_by"], "image_one_time", 15)
        self.update_credits(ref["referred_by"], "video_one_time", 10)
        
        # Give credits to referee
        self.update_credits(user_id, "image_one_time", 10)
        self.update_credits(user_id, "video_one_time", 5)
        
        self.referrals.update_one(
            {"code": code},
            {"$push": {"used_by": user_id}}
        )
        return True
    
    def get_all_users(self) -> List[int]:
        """Get all user IDs for broadcasting"""
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
                db.groups.update_one(
                    {"chat_id": chat_id},
                    {"$set": {"last_active": datetime.now(timezone.utc)}},
                    upsert=True
                )
            except Exception as e:
                logger.error(f"Failed to update group {chat_id}: {e}")
        
        if not user_id:
            return await handler(event, data)
        
        try:
            user = db.get_user(user_id)
            if not user:
                user = db.create_user(user_id, username)
            
            # Check if user is banned
            if user.get("blocked"):
                reason = user.get("ban_reason", "Violation of bot rules")
                if isinstance(event, types.Message):
                    await event.reply(
                        f"You are banned ({reason}) contact owner for further classification"
                    )
                elif isinstance(event, types.CallbackQuery):
                    await event.answer(
                        f"You are banned ({reason}) contact owner for further classification",
                        show_alert=True
                    )
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
        # Check both messages and callback queries
        if isinstance(event, (types.Message, types.CallbackQuery)):
            # Allow help command
            if isinstance(event, types.Message) and event.text and event.text.startswith("/help"):
                return await handler(event, data)
            
            # Allow start with referral (handled separately)
            if isinstance(event, types.Message) and event.text and event.text.startswith("/start"):
                return await handler(event, data)
            
            # Check channel membership
            user_id = event.from_user.id
            try:
                # FIXED: Removed extra space in URL
                member = await bot.get_chat_member(self.channel_username, user_id)
                if member.status in ["member", "administrator", "creator"]:
                    return await handler(event, data)
            except Exception as e:
                logger.warning(f"Failed to check membership for {user_id}: {e}")
                # FIXED: Continue if check fails (don't block user)
                return await handler(event, data)
            
            # User is not a member
            keyboard = InlineKeyboardMarkup(inline_keyboard=[[
                InlineKeyboardButton(
                    text="Join Channel", 
                    url=f"https://t.me/{self.channel_username.strip('@')}"  # FIXED: Removed space
                )
            ]])
            
            message_obj = event if isinstance(event, types.Message) else event.message
            if message_obj:
                await message_obj.reply(
                    f"❌ You must join {self.channel_username} to use this bot!",
                    reply_markup=keyboard
                )
            return
        
        return await handler(event, data)

class NSFWFilterMiddleware(BaseMiddleware):
    async def __call__(self, handler, event, data):
        if isinstance(event, types.Message) and event.text:
            # Use regex for better word boundary detection
            if NSFW_PATTERN.search(event.text):
                user_id = event.from_user.id
                
                # Update warnings in DB
                db.users.update_one(
                    {"user_id": user_id},
                    {"$inc": {"nsfw_warnings": 1}}
                )
                
                # Fetch updated user data
                user = db.get_user(user_id)
                if not user:
                    return await handler(event, data)
                
                warnings = user.get("nsfw_warnings", 0)
                
                if warnings >= 3:
                    # Use the new ban_user method for consistency
                    db.ban_user(user_id, reason="nsfw violation")
                    # FIXED: Use reply() for messages
                    await event.reply("❌ You have been permanently blocked for NSFW content!")
                    return
                else:
                    # FIXED: Use reply() for messages
                    await event.reply(
                        f"⚠️ NSFW content detected! Warning {warnings}/3\n"
                        "Do not use NSFW prompt or you will be permanently blocked!"
                    )
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
            
            logger.info(f"🚀 POST {endpoint} with aspect_ratio={aspect_ratio}")
            
            async with session.post(endpoint, data=form) as resp:
                if resp.status not in (200, 202):
                    text = await resp.text()
                    if resp.status == 429:
                        raise Exception("Rate limited by API. Please try again in a few minutes.")
                    raise Exception(f"Generation failed: HTTP {resp.status}\nResponse: {text[:500]}")
                
                result = await resp.json()
                logger.info(f"✅ Generation response: {json.dumps(result, indent=2)}")
                
                job_id = result.get("uuid") or result.get("id")
                if not job_id:
                    raise Exception(f"No job_id found: {result}")
                
                logger.info(f"🆔 Job UUID: {job_id}")
                return job_id
    
    async def generate_video(self, prompt: str) -> str:
        async with aiohttp.ClientSession(cookies=self.cookies, headers=self.headers) as session:
            endpoint = f"{self.base_url}/api/video-gen/veo"
            
            form = aiohttp.FormData()
            form.add_field('prompt', prompt)
            form.add_field('model', 'veo-3-fast')
            form.add_field('duration', '8')
            form.add_field('resolution', '720p')
            form.add_field('aspect_ratio', '16:9')
            form.add_field('enhance_prompt', 'true')
            
            logger.info(f"🚀 POST {endpoint}")
            
            async with session.post(endpoint, data=form) as resp:
                if resp.status not in (200, 202):
                    text = await resp.text()
                    if resp.status == 429:
                        raise Exception("Rate limited by API. Please try again in a few minutes.")
                    raise Exception(f"Generation failed: HTTP {resp.status}\nResponse: {text[:500]}")
                
                result = await resp.json()
                logger.info(f"✅ Generation response: {json.dumps(result, indent=2)}")
                
                job_id = result.get("uuid") or result.get("id")
                if not job_id:
                    raise Exception(f"No job_id found: {result}")
                
                logger.info(f"🆔 Job UUID: {job_id}")
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
                
                logger.info(f"⏳ Polling {endpoint} ({elapsed:.1f}s) - Attempt {retry_count + 1}/{max_retries}")
                
                try:
                    async with session.get(endpoint) as resp:
                        if resp.status != 200:
                            text = await resp.text()
                            logger.warning(f"Poll failed: HTTP {resp.status} - {text[:200]}")
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
                                            logger.info(f"✅ Found image URL in generated_image[0]['{field}']: {image_url[:80]}...")
                                            break
                                    if image_url:
                                        break
                        
                        if not image_url:
                            top_fields = ['image_url', 'download_url', 'url', 'media_url', 'output_url']
                            for field in top_fields:
                                if field in result and result[field]:
                                    image_url = result[field]
                                    logger.info(f"✅ Found image URL in top-level '{field}': {image_url[:80]}...")
                                    break
                        
                        if not image_url:
                            result_str = json.dumps(result)
                            img_matches = re.findall(r'https?://[^\s"]+\.(?:png|jpg|jpeg|webp)(?:\?[^\s"]*)?', result_str, re.IGNORECASE)
                            if img_matches:
                                image_url = img_matches[0]
                                logger.info(f"✅ Extracted image URL from JSON scan: {image_url[:80]}...")
                        
                        if image_url:
                            return image_url
                        
                        status = result.get("status", "")
                        progress = result.get("status_percentage", 0)
                        queue = result.get("queue_position", 0)
                        error_message = result.get("error_message", "")
                        
                        if status in [0, "failed", "error"]:
                            raise Exception(f"Generation failed with status: {status}, error: {error_message}")
                        
                        if error_message and "high traffic" in error_message.lower():
                            logger.warning(f"⚠️ High traffic: {error_message}")
                            retry_count += 1
                            if retry_count >= max_retries:
                                raise Exception(f"Max retries ({max_retries}) reached due to high traffic")
                            logger.info(f"⏳ Retrying in 10s... Queue: {queue}, Progress: {progress}%")
                            await asyncio.sleep(10)
                            continue
                        
                        if status in [1, "processing", "queued"] or progress < 100:
                            logger.info(f"⏳ Processing... Progress: {progress}%, Queue: {queue}")
                            await asyncio.sleep(8)
                            continue
                        
                        logger.warning(f"Unknown state: status={status}, progress={progress}, error={error_message}")
                        await asyncio.sleep(5)
                
                except asyncio.TimeoutError:
                    raise
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
                
                logger.info(f"⏳ Polling video {endpoint} ({elapsed:.1f}s) - Attempt {retry_count + 1}/{max_retries}")
                
                try:
                    async with session.get(endpoint) as resp:
                        if resp.status != 200:
                            text = await resp.text()
                            logger.warning(f"Poll failed: HTTP {resp.status}")
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
                        queue = result.get("queue_position", 0)
                        error_message = result.get("error_message", "")
                        
                        if status in [0, "failed", "error"]:
                            raise Exception(f"Generation failed: {error_message}")
                        
                        if status in [1, "processing", "queued"] or progress < 100:
                            logger.info(f"⏳ Video processing... Progress: {progress}%, Queue: {queue}")
                            await asyncio.sleep(5)
                            continue
                        
                        logger.warning(f"Unknown video state: status={status}, progress={progress}")
                        await asyncio.sleep(3)
                
                except Exception as e:
                    logger.exception(f"Video polling error: {e}")
                    retry_count += 1
                    if retry_count >= max_retries:
                        raise Exception(f"Max retries ({max_retries}) reached: {str(e)}")
                    await asyncio.sleep(min(10 * (retry_count + 1), 30))
    
    async def download_media(self, url: str, max_size: int = 50 * 1024 * 1024) -> bytes:
        """Download media with size limit (default 50MB)"""
        async with aiohttp.ClientSession() as session:
            logger.info(f"📥 Downloading from {url[:80]}...")
            
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=300)) as resp:
                    if resp.status != 200:
                        raise Exception(f"Download failed: HTTP {resp.status}")
                    
                    size = int(resp.headers.get('content-length', 0))
                    if size > max_size:
                        raise Exception(f"File too large: {size / 1024 / 1024:.1f}MB > {max_size / 1024 / 1024:.1f}MB limit")
                    
                    logger.info(f"Download size: {size / 1024:.2f} KB")
                    
                    return await resp.read()
            except asyncio.TimeoutError:
                raise Exception("Download timeout after 5 minutes")

# FIXED: Initialize bot with proper properties
bot = Bot(token=API_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp = Dispatcher()
router = Router()
dp.include_router(router)

# FIXED: Reordered middleware for proper execution
# UserManager first to ensure user exists, then ForceJoin, then NSFW filter
dp.message.middleware(UserManagerMiddleware())
dp.callback_query.middleware(UserManagerMiddleware())
dp.message.middleware(ForceJoinMiddleware(FORCE_JOIN_CHANNEL))
dp.message.middleware(NSFWFilterMiddleware())

def banana_menu_keyboard():
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🎨 Generate New Image", callback_data="banana_new")],
        [InlineKeyboardButton(text="🖼️ Edit Image (Ref.)", callback_data="banana_edit")]
    ])

@router.message(Command("start"))
async def cmd_start(msg: types.Message, user: Dict):
    """Handle /start command with optional referral"""
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
    
    await msg.reply(
        "🎬 <b>GeminiGen Multi-Tool Bot</b>\n\n"
        "🍌 /banana - Nano Banana image generation\n"
        "📹 /video - Video generation (4 credits/day)\n"
        "🔗 /refer - Get your referral link\n"
        "🎟️ /predeem - Redeem image coupon\n"
        "🎟️ /vredeem - Redeem video coupon\n\n"
        "💬 Support: @ayushxchat_robot\n"
        "👨‍💻 Developed by: @mahadev_ki_iccha",
        parse_mode="HTML"
    )

# ... (rest of your command handlers remain the same) ...

# Initialize API client
api = GeminiGenAPI(parse_netscape_cookies(COOKIE_FILE_CONTENT), BEARER_TOKEN)

async def on_startup():
    """Startup tasks"""
    validate_config()
    logger.info("=" * 60)
    logger.info("🤖 Nano Banana Bot Starting...")
    logger.info(f"Owner ID: {OWNER_ID}")
    logger.info(f"Force Join: {FORCE_JOIN_CHANNEL}")
    logger.info(f"Log Group ID: {LOG_GROUP_ID}")
    logger.info(f"Total Users: {db.users.count_documents({})}")
    logger.info("=" * 60)

async def on_shutdown():
    """Shutdown tasks"""
    logger.info("🛑 Shutting down bot...")
    await bot.session.close()
    db.client.close()
    logger.info("✅ Bot shutdown complete")

async def main():
    """Main bot runner"""
    await on_startup()
    
    # Set up signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        await dp.start_polling(bot, skip_updates=True)
    finally:
        await on_shutdown()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("🛑 Bot stopped by user")
    except Exception as e:
        logger.exception(f"❌ Fatal error: {e}")
        sys.exit(1)
