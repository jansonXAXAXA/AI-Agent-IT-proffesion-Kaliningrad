
import json
import os
import uuid
import re
import asyncio
from typing import List, Optional, Dict
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import Message, InlineKeyboardButton, InlineKeyboardMarkup
from aiogram.utils.keyboard import InlineKeyboardBuilder
import logging

import hashlib
import re
import time
import json
import os
from collections import OrderedDict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from urllib.parse import urlparse, urlunparse, unquote
import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if not os.path.exists('user_data'):
    os.makedirs('user_data')

DATA_PATH = "data/dataset-it-profession.csv"
BOT_TOKEN = "YOUR_BOT_TOKEN_HERE"

bot = Bot(token=BOT_TOKEN)
storage = MemoryStorage()
dp = Dispatcher(storage=storage)

class RegistrationStates(StatesGroup):
    waiting_for_full_name = State()
    waiting_for_email = State()
    waiting_for_position = State()
    waiting_for_search_query = State()

class ITEventSemanticSearch:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.df = None
        self.index = None
        self.model = None
        self.is_initialized = False
        
        try:
            self._initialize()
        except Exception as e:
            logger.error(f"Ошибка при инициализации поиска: {e}")
    
    def _initialize(self):
        logger.info("Загрузка данных и модели для семантического поиска...")
        
        self.df = pd.read_csv(self.csv_path, sep=',', encoding='utf-8')
        
        text_columns = ['Event Name', 'Description', 'Category', 'Location']
        for col in text_columns:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna('').astype(str)
        
        self.df['search_text'] = (
            self.df['Event Name'] + ". " +
            self.df['Description'] + ". " +
            self.df.get('Category', '') + ". " +
            self.df.get('Location', '')
        )
        
        self.model = SentenceTransformer('cointegrated/rubert-tiny2')
        logger.info("Модель загружена успешно")
        
        self._build_vector_index()
        self.is_initialized = True
        logger.info("Семантический поиск успешно инициализирован")
    
    def _build_vector_index(self):
        texts = self.df['search_text'].tolist()
        logger.info(f"Создание эмбеддингов для {len(texts)} мероприятий...")
        
        embeddings = self.model.encode(texts, batch_size=32, show_progress_bar=False)
        
        embeddings = embeddings.astype(np.float32)
        faiss.normalize_L2(embeddings)
        
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)
        logger.info(f"Индекс создан, добавлено {self.index.ntotal} векторов")
    
    def search(self, query: str, top_k: int = 5) -> List[str]:
        if not self.is_initialized:
            return ["Сервис поиска временно недоступен. Попробуйте позже."]
        
        if not isinstance(query, str) or len(query.strip()) < 2:
            return ["Пожалуйста, введите запрос длиной не менее 2 символов."]
        
        try:
            query = query.strip()
            logger.info(f"Поиск по запросу: '{query}'")
            
            query_embedding = self.model.encode([query], convert_to_numpy=True)
            query_embedding = query_embedding.astype(np.float32)
            faiss.normalize_L2(query_embedding)
            
            distances, indices = self.index.search(query_embedding, top_k)
            
            results = []
            seen_events = set()
            for i, idx in enumerate(indices[0]):
                if 0 <= idx < len(self.df):
                    event_info = self.df.iloc[idx]['End Date']
                    if event_info and event_info not in seen_events:
                        seen_events.add(event_info)
                        results.append(f"• {event_info}")
                        if len(results) >= top_k:
                            break
            
            if not results:
                return ["По вашему запросу не найдено мероприятий. Попробуйте изменить формулировку."]
            
            return results[:top_k]
        
        except Exception as e:
            logger.error(f"Ошибка при поиске: {e}")
            return ["Произошла ошибка при поиске. Попробуйте позже."]

event_search = ITEventSemanticSearch(DATA_PATH)

def is_valid_email(email):
    pattern = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
    return re.match(pattern, email) is not None

@dp.message(Command("start"))
async def cmd_start(message: Message, state: FSMContext):
    user_id = message.from_user.id
    
    user_file = f"user_data/user_{user_id}.json"
    
    if os.path.exists(user_file):
        with open(user_file, 'r', encoding='utf-8') as f:
            user_data = json.load(f)
        await message.answer(
            f"Вы уже зарегистрированы!\n\n"
            f"Ваши данные:\n"
            f"ФИО: {user_data['full_name']}\n"
            f"Почта: {user_data['email']}\n"
            f"Должность: {user_data['position']}\n"
            f"Уникальный ID: {user_data['unique_id']}\n\n"
            f"Чтобы найти мероприятия, используйте команду /search или просто напишите ваш запрос."
        )
    else:
        await message.answer("Добро пожаловать! Давайте начнем регистрацию.\n\n"
                           "Пожалуйста, введите ваше ФИО:")
        await state.set_state(RegistrationStates.waiting_for_full_name)

@dp.message(RegistrationStates.waiting_for_full_name)
async def process_full_name(message: Message, state: FSMContext):
    full_name = message.text.strip()
    
    if len(full_name) < 3:
        await message.answer("ФИО слишком короткое. Пожалуйста, введите корректное ФИО:")
        return
    
    await state.update_data(full_name=full_name)
    await message.answer("Отлично! Теперь введите ваш email:")
    await state.set_state(RegistrationStates.waiting_for_email)

@dp.message(RegistrationStates.waiting_for_email)
async def process_email(message: Message, state: FSMContext):
    email = message.text.strip()
    
    if not is_valid_email(email):
        await message.answer("Некорректный email. Пожалуйста, введите правильный email:")
        return
    
    await state.update_data(email=email)
    await message.answer("Отлично! Теперь введите вашу должность:")
    await state.set_state(RegistrationStates.waiting_for_position)

@dp.message(RegistrationStates.waiting_for_position)
async def process_position(message: Message, state: FSMContext):
    position = message.text.strip()
    
    if len(position) < 2:
        await message.answer("Должность слишком короткая. Пожалуйста, введите корректную должность:")
        return
    
    data = await state.get_data()
    full_name = data['full_name']
    email = data['email']
    
    unique_id = str(uuid.uuid4())
    
    user_data = {
        'user_id': message.from_user.id,
        'full_name': full_name,
        'email': email,
        'position': position,
        'unique_id': unique_id,
        'registration_date': message.date.isoformat(),
        'username': message.from_user.username if message.from_user.username else None
    }
    
    user_file = f"user_data/user_{message.from_user.id}.json"
    with open(user_file, 'w', encoding='utf-8') as f:
        json.dump(user_data, f, ensure_ascii=False, indent=2)
    
    unique_file = f"user_data/{unique_id}.json"
    with open(unique_file, 'w', encoding='utf-8') as f:
        json.dump(user_data, f, ensure_ascii=False, indent=2)
    
    await state.clear()
    
    welcome_text = (
        f"Регистрация успешно завершена! 🎉\n\n"
        f"Ваши данные сохранены:\n"
        f"ФИО: {full_name}\n"
        f"Почта: {email}\n"
        f"Должность: {position}\n\n"
        f"Уникальный ID: {unique_id}\n\n"
        f"Теперь вы можете искать IT-мероприятия!\n"
        f"Используйте команду /search или просто напишите, какие мероприятия вас интересуют.\n\n"
        f"Примеры запросов:\n"
        f"• IT конференция в СПбГУ\n"
        f"• Хакатон по машинному обучению\n"
        f"• Вебинар по Python\n"
        f"• Митап по искусственному интеллекту"
    )
    
    await message.answer(welcome_text)

@dp.message(Command("search"))
async def cmd_search(message: Message, state: FSMContext):
    user_id = message.from_user.id
    user_file = f"user_data/user_{user_id}.json"
    
    if not os.path.exists(user_file):
        await message.answer("Для использования поиска необходимо сначала зарегистрироваться. Используйте команду /start")
        return
    
    await message.answer("🔍 Введите ваш запрос для поиска IT-мероприятий:")
    await state.set_state(RegistrationStates.waiting_for_search_query)

@dp.message()
async def handle_text_search(message: Message, state: FSMContext):
    current_state = await state.get_state()
    
    if current_state == RegistrationStates.waiting_for_search_query:
        query = message.text.strip()
        await state.clear()
        
        user_id = message.from_user.id
        user_file = f"user_data/user_{user_id}.json"
        
        if not os.path.exists(user_file):
            await message.answer("Ошибка: вы не зарегистрированы. Используйте /start")
            return
        
        results = event_search.search(query, top_k=5)
        
        response = f"🎯 Результаты поиска по запросу '{query}':\n\n"
        
        if results and len(results) > 0:
            for i, result in enumerate(results, 1):
                response += f"{i}. {result}\n"
        else:
            response += "К сожалению, по вашему запросу не найдено мероприятий."
        
        response += "\n\n🔍 Чтобы выполнить новый поиск, используйте команду /search или просто напишите новый запрос."
        
        await message.answer(response)
        return
    
    user_id = message.from_user.id
    user_file = f"user_data/user_{user_id}.json"
    
    if not os.path.exists(user_file):
        await message.answer("Пожалуйста, сначала зарегистрируйтесь с помощью команды /start")
        return
    
    query = message.text.strip()
    
    if len(query) < 2:
        await message.answer("Пожалуйста, введите запрос длиной не менее 2 символов для поиска мероприятий.")
        return
    
    results = event_search.search(query, top_k=5)
    
    response = f"🎯 Результаты поиска по запросу '{query}':\n\n"
    
    if results and len(results) > 0:
        for i, result in enumerate(results, 1):
            response += f"{i}. {result}\n"
    else:
        response += "К сожалению, по вашему запросу не найдено мероприятий."
    
    response += "\n\n🔍 Чтобы выполнить новый поиск, используйте команду /search или просто напишите новый запрос."
    
    await message.answer(response)

@dp.message(Command("mydata"))
async def cmd_mydata(message: Message):
    user_id = message.from_user.id
    user_file = f"user_data/user_{user_id}.json"
    
    if os.path.exists(user_file):
        with open(user_file, 'r', encoding='utf-8') as f:
            user_data = json.load(f)
        
        await message.answer(
            f"Ваши регистрационные данные:\n\n"
            f"👤 ФИО: {user_data['full_name']}\n"
            f"📧 Почта: {user_data['email']}\n"
            f"💼 Должность: {user_data['position']}\n"
            f"🆔 Уникальный ID: {user_data['unique_id']}\n"
            f"📅 Дата регистрации: {user_data['registration_date'][:10]}"
        )
    else:
        await message.answer("Вы не зарегистрированы. Используйте команду /start для регистрации.")

@dp.message(Command("help"))
async def cmd_help(message: Message):
    help_text = (
        "Помощь по боту:\n\n"
        "🚀 /start - Начать регистрацию или показать данные\n"
        "🔍 /search - Найти IT-мероприятия\n"
        "📋 /mydata - Показать ваши регистрационные данные\n"
        "🆘 /help - Показать эту справку\n\n"
        "💡 Как использовать поиск:\n"
        "• Просто напишите ваш запрос в чат\n"
        "• Или используйте команду /search\n"
        "• Примеры запросов:\n"
        "  - IT конференция в СПбГУ\n"
        "  - Хакатон по машинному обучению\n"
        "  - Вебинар по Python\n"
        "  - Митап по AI в Санкт-Петербурге"
    )
    await message.answer(help_text)

@dp.message(Command("reregister"))
async def cmd_reregister(message: Message, state: FSMContext):
    user_id = message.from_user.id
    user_file = f"user_data/user_{user_id}.json"
    
    if os.path.exists(user_file):
        os.remove(user_file)
        await message.answer("Ваши старые данные удалены. Давайте начнем регистрацию заново.\n\nВведите ваше ФИО:")
        await state.set_state(RegistrationStates.waiting_for_full_name)
    else:
        await message.answer("Вы еще не зарегистрированы. Используйте команду /start для регистрации.")

@dp.errors()
async def error_handler(update, exception):
    logger.error(f"Произошла ошибка: {exception}")
    if update.message:
        await update.message.answer("Произошла ошибка при обработке вашего запроса. Пожалуйста, попробуйте позже.")
    return True

def get_feed(user_id):
    user_file = f"user_data/user_{user_id}.json"
    query = None
    
    if os.path.exists(user_file):
        with open(user_file, 'r', encoding='utf-8') as f:
            user_data = json.load(f)
        
        status = user_data['position']
        query = f"IT мероприятия для {status} в Санкт-Петербурге"

        SEARCH(query, user_id)
        recs_file = f"events{user_id}.json"

        with open(recs_file, 'r', encoding='utf-8') as f:
            recs_data = json.load(f)

async def main():
    logger.info("Запуск бота...")
    
    if not event_search.is_initialized:
        logger.warning("RAG система не инициализирована. Поиск мероприятий будет недоступен.")
    
    await dp.start_polling(bot)

if __name__ == "__main__":
    import asyncio
    logger.info("🔧 Инициализация бота...")
    asyncio.run(main())

class DuckDuckGoSearch:
    def __init__(self, cache_size: int = 200, cache_ttl: int = 1800):
        self.cache: OrderedDict = OrderedDict()
        self.cache_ttl = cache_ttl
        self.cache_timestamps: Dict[str, datetime] = {}
        self.cache_size = cache_size
        self.session = requests.Session()
        
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
            'Accept-Language': 'ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7',
            'Referer': 'https://duckduckgo.com/',
            'Upgrade-Insecure-Requests': '1'
        })
    
    def _generate_cache_key(self, query: str) -> str:
        salt = "ddg_v2"
        normalized = re.sub(r'\s+', ' ', query.lower().strip())
        return hashlib.sha256(f"{normalized}{salt}".encode('utf-8')).hexdigest()
    
    def _is_ru_domain(self, url: str) -> bool:
        """Проверка, что сайт относится к RU сегменту"""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            allowed = ['.ru', '.su', '.рф', '.moscow', '.tech', '.com', '.org', '.net']
            return any(domain.endswith(tld) or f".{tld}" in domain for tld in allowed)
        except:
            return False

    def _clean_url(self, raw_url: str) -> Optional[str]:
        try:
            if 'duckduckgo.com/l/?uddg=' in raw_url:
                match = re.search(r'uddg=([^&]+)', raw_url)
                if match:
                    raw_url = unquote(match.group(1))
            
            raw_url = raw_url.strip()
            if not raw_url.startswith(('http://', 'https://')):
                return None
                
            parsed = urlparse(raw_url)
            if not parsed.netloc:
                return None
                
            if 'duckduckgo' in parsed.netloc or 'yandex' in parsed.netloc or 'google' in parsed.netloc:
                return None
                
            return raw_url
        except:
            return None
    
    def search(self, query: str) -> List[str]:
        cache_key = self._generate_cache_key(query)
        if cache_key in self.cache and (datetime.now() - self.cache_timestamps.get(cache_key, datetime.min)).total_seconds() < self.cache_ttl:
            return self.cache[cache_key]

        url = "https://html.duckduckgo.com/html/"
        data = {'q': query, 'kl': 'ru-ru', 'df': 'y'}


        try:
            resp = self.session.post(url, data=data, timeout=15)
            resp.raise_for_status()
            
            urls = []
            seen = set()
            
            soup = BeautifulSoup(resp.text, 'html.parser')
            
            links = soup.find_all('a', class_='result__a')
            
            for link_tag in links:
                href = link_tag.get('href')
                if not href: continue
                
                clean = self._clean_url(href)
                if clean and clean not in seen:
                    if self._is_ru_domain(clean):
                        urls.append(clean)
                        seen.add(clean)
            
            if not urls:
                raw_links = re.findall(r'href=["\'](https?://[^"\']+)["\']', resp.text)
                for href in raw_links:
                    clean = self._clean_url(href)
                    if clean and clean not in seen and self._is_ru_domain(clean):
                        urls.append(clean)
                        seen.add(clean)

            self.cache[cache_key] = urls[:10]
            self.cache_timestamps[cache_key] = datetime.now()
            return urls[:10]

        except Exception as e:
            print(f"Search failed: {e}")
            return []

class EventParser:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; EventBot/1.0)',
            'Accept-Language': 'ru-RU,ru;q=0.9',
        })
        
        self.months = {
            'янв': 1, 'января': 1, 'январь': 1,
            'фев': 2, 'февраля': 2, 'февраль': 2,
            'мар': 3, 'марта': 3, 'март': 3,
            'апр': 4, 'апреля': 4, 'апрель': 4,
            'мая': 5, 'май': 5,
            'июн': 6, 'июня': 6, 'июнь': 6,
            'июл': 7, 'июля': 7, 'июль': 7,
            'авг': 8, 'августа': 8, 'август': 8,
            'сен': 9, 'сентября': 9, 'сентябрь': 9,
            'окт': 10, 'октября': 10, 'октябрь': 10,
            'ноя': 11, 'ноября': 11, 'ноябрь': 11,
            'дек': 12, 'декабря': 12, 'декабрь': 12,
        }

    def get_soup(self, url: str) -> Optional[BeautifulSoup]:
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            if response.encoding is None:
                response.encoding = 'utf-8'
            return BeautifulSoup(response.text, 'html.parser')
        except Exception:
            return None

    def _get_meta(self, soup: BeautifulSoup, attrs_list: List[Dict]) -> str:
        for attrs in attrs_list:
            tag = soup.find('meta', attrs=attrs)
            if tag and tag.get('content'):
                return tag['content'].strip()
        return ""

    def _extract_text_by_keyword(self, soup: BeautifulSoup, keywords: List[str], length=300) -> str:
        body_text = soup.get_text(" ", strip=True)
        for kw in keywords:
            if kw.lower() in body_text.lower():
                idx = body_text.lower().find(kw.lower())
                start = max(0, idx)
                end = min(len(body_text), idx + length)
                return body_text[start:end].strip() + "..."
        return ""

    def _parse_date(self, date_str: str, year: str = None) -> Optional[datetime]:
        try:
            match = re.search(r'(\d{1,2})\s+([а-яА-Я]+)\s+(\d{4})', date_str)
            if match:
                day = int(match.group(1))
                month_str = match.group(2).lower()
                year = int(match.group(3))
                
                for key, month in self.months.items():
                    if key in month_str:
                        return datetime(year, month, day)
            
            match = re.search(r'(\d{1,2})\s+([а-яА-Я]+)', date_str)
            if match and year:
                day = int(match.group(1))
                month_str = match.group(2).lower()
                year_int = int(year)
                
                for key, month in self.months.items():
                    if key in month_str:
                        return datetime(year_int, month, day)
            
            match = re.search(r'(\d{1,2})\.(\d{1,2})\.(\d{4})', date_str)
            if match:
                day, month, year = int(match.group(1)), int(match.group(2)), int(match.group(3))
                return datetime(year, month, day)
                
        except (ValueError, AttributeError):
            pass
        
        return None

    def _is_valid_title(self, title: str) -> bool:
        if not title or len(title) < 5:
            return False
        
        has_cyrillic = bool(re.search('[а-яА-ЯёЁ]', title))
        has_latin = bool(re.search('[a-zA-Z]', title))
        
        has_garbage = bool(re.search(r'[ÐÑÐ]{3,}', title))
        
        return (has_cyrillic or has_latin) and not has_garbage

    def parse(self, url: str) -> Dict[str, Any]:
        soup = self.get_soup(url)
        data = {k: '' for k in ['Year', 'Start Date', 'End Date', 'Event Name', 'Event Type', 
                                'Description', 'Participants Count', 'Speakers/Organizers', 
                                'Partners', 'Category', 'Location', 'Source URL', 'Parsed Date']}
        data['Source URL'] = url
        
        if not soup:
            return data

        data['Event Name'] = self._get_meta(soup, [
            {'property': 'og:title'}, {'name': 'twitter:title'}, {'name': 'title'}
        ])
        if not data['Event Name'] and soup.title:
            data['Event Name'] = soup.title.string or ''

        if not self._is_valid_title(data['Event Name']):
            return data 

        data['Description'] = self._get_meta(soup, [
            {'property': 'og:description'}, {'name': 'description'}
        ])

        data['Location'] = self._get_meta(soup, [{'property': 'og:locality'}])
        if not data['Location']:
            text = soup.get_text()
            if "Санкт-Петербург" in text or "СПб" in text or "Питер" in text:
                data['Location'] = "Санкт-Петербург"

        text = soup.get_text(" ", strip=True)
        
        year_match = re.search(r'202[4-9]', text)
        if year_match:
            data['Year'] = year_match.group(0)
        else:
            data['Year'] = str(datetime.now().year)
        
        date_patterns = [
            r'(\d{1,2}\s+(?:янв|фев|мар|апр|мая|июн|июл|авг|сен|окт|ноя|дек)[а-я]*\s+\d{4})',
            r'(\d{1,2}\s+(?:янв|фев|мар|апр|мая|июн|июл|авг|сен|окт|ноя|дек)[а-я]*)',
            r'(\d{1,2}\.\d{1,2}\.\d{4})'
        ]
        
        for pattern in date_patterns:
            date_match = re.search(pattern, text.lower())
            if date_match:
                date_str = date_match.group(0)
                data['Start Date'] = date_str
                
                parsed_date = self._parse_date(date_str, data['Year'])
                if parsed_date:
                    data['Parsed Date'] = parsed_date
                break

        data['Speakers/Organizers'] = self._extract_text_by_keyword(soup, ['Спикеры', 'Speakers', 'Докладчики', 'Ведущие'])
        data['Partners'] = self._extract_text_by_keyword(soup, ['Партнеры', 'Спонсоры', 'Partners'])

        title_lower = (data['Event Name'] or '').lower()
        if 'конференц' in title_lower: data['Event Type'] = 'Конференция'
        elif 'митап' in title_lower or 'meetup' in title_lower: data['Event Type'] = 'Митап'
        elif 'хакатон' in title_lower: data['Event Type'] = 'Хакатон'
        else: data['Event Type'] = 'Мероприятие'

        return data

class JsonWriter:
    def __init__(self, filename: str = 'events.json'):
        self.filename = filename
        self.data = []
        
        # Загружаем существующие данные, если файл есть
        if os.path.exists(self.filename):
            try:
                with open(self.filename, 'r', encoding='utf-8') as f:
                    self.data = json.load(f)
                    if not isinstance(self.data, list):
                        self.data = []
            except (json.JSONDecodeError, Exception):
                self.data = []
                print(f"Предупреждение: файл {self.filename} поврежден или пуст. Создается новый.")
    
    def append(self, row: Dict):
        clean_row = {}
        for k, v in row.items():
            if k == 'Parsed Date':
                continue  # Не сохраняем в JSON
            if isinstance(v, datetime):
                clean_row[k] = v.strftime('%Y-%m-%d')
            else:
                clean_row[k] = str(v).strip() if v else ''
        self.data.append(clean_row)
    
    def save(self):
        with open(self.filename, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
        print(f"\nДанные сохранены в {self.filename} ({len(self.data)} записей)")

def SEARCH(query, user_id):
    today = datetime.now()

    searcher = DuckDuckGoSearch()
    urls = searcher.search(query)
    
    if not urls:
        print("Ссылки не найдены. Попробуйте изменить запрос или проверить соединение.")
        return

    print(f"Найдено ссылок: {len(urls)}")
    for i, url in enumerate(urls, 1):
        print(f"  {i}. {url}")
    
    parser = EventParser()
    writer = JsonWriter(f'events{user_id}.json')
    
    added_count = 0
    skipped_count = 0
    
    for i, url in enumerate(urls, 1):
        try:
            print(f"[{i}/{len(urls)}] Обработка: {url}")
            event_data = parser.parse(url)
            
            if not event_data['Event Name'] or not parser._is_valid_title(event_data['Event Name']):
                print(f"  ⚠ Пропуск: некорректное название или кракозябры")
                skipped_count += 1
                continue
            
            parsed_date = event_data.get('Parsed Date')
            if parsed_date:
                if parsed_date < today:
                    print(f"Пропуск: событие прошло ({parsed_date.strftime('%d.%m.%Y')})")
                    skipped_count += 1
                    continue
                else:
                    print(f"  📅 Дата: {parsed_date.strftime('%d.%m.%Y')}")
            else:
                print(f"Дата не найдена, добавляем с предупреждением")
            
            writer.append(event_data)
            title_display = event_data['Event Name'][:60]
            print(f"Добавлено: {title_display}...")
            added_count += 1
            
            time.sleep(1)
            
        except Exception as e:
            print(f"Ошибка: {e}")
            skipped_count += 1
    writer.save()
