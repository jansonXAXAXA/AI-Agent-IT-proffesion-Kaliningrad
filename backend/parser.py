import hashlib
import re
import time
import csv
import os
from collections import OrderedDict
from datetime import datetime
from typing import Dict, List, Optional, Any
from urllib.parse import urlparse, urlunparse, unquote
import requests
from bs4 import BeautifulSoup

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
            
            # Убираем лишнее
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

        print(f"Searching DDG for: {query}...")
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
        """Безопасное извлечение мета-тегов"""
        for attrs in attrs_list:
            tag = soup.find('meta', attrs=attrs)
            if tag and tag.get('content'):
                return tag['content'].strip()
        return ""

    def _extract_text_by_keyword(self, soup: BeautifulSoup, keywords: List[str], length=300) -> str:
        """Ищет текст рядом с ключевыми словами (эвристика)"""
        body_text = soup.get_text(" ", strip=True)
        for kw in keywords:
            if kw.lower() in body_text.lower():
                idx = body_text.lower().find(kw.lower())
                start = max(0, idx)
                end = min(len(body_text), idx + length)
                return body_text[start:end].strip() + "..."
        return ""

    def parse(self, url: str) -> Dict[str, Any]:
        soup = self.get_soup(url)
        data = {k: '' for k in ['Year', 'Start Date', 'End Date', 'Event Name', 'Event Type', 
                                'Description', 'Participants Count', 'Speakers/Organizers', 
                                'Partners', 'Category', 'Location', 'Source URL']}
        data['Source URL'] = url
        
        if not soup:
            return data

        # 1. Название (Title)
        data['Event Name'] = self._get_meta(soup, [
            {'property': 'og:title'}, {'name': 'twitter:title'}, {'name': 'title'}
        ])
        if not data['Event Name'] and soup.title:
            data['Event Name'] = soup.title.string

        # 2. Описание
        data['Description'] = self._get_meta(soup, [
            {'property': 'og:description'}, {'name': 'description'}
        ])



        # 3. Локация
        data['Location'] = self._get_meta(soup, [{'property': 'og:locality'}])
        if not data['Location']:
            # Проверяем, есть ли Калининград в тексте
            if "Калининград" in soup.get_text():
                data['Location'] = "Калининград"

        # 4. Даты (Поиск паттернов)
        text = soup.get_text(" ", strip=True)
        # Паттерн: число + месяц (15 сентября, 20.10.2025)
        date_match = re.search(r'(\d{1,2})\s+(янв|фев|мар|апр|мая|июн|июл|авг|сен|окт|ноя|дек)[а-я]*\s+(\d{4})?', text.lower())
        if date_match:
            data['Start Date'] = date_match.group(0)
            data['Year'] = date_match.group(3) if date_match.group(3) else '2025'
        
        # Если год не нашли, ищем просто 4 цифры 202X
        if not data['Year']:
            year_match = re.search(r'202[4-6]', text)
            if year_match:
                data['Year'] = year_match.group(0)

        # 5. Спикеры / Партнеры
        data['Speakers/Organizers'] = self._extract_text_by_keyword(soup, ['Спикеры', 'Speakers', 'Докладчики', 'Ведущие'])
        data['Partners'] = self._extract_text_by_keyword(soup, ['Партнеры', 'Спонсоры', 'Partners'])

        # 6. Тип события
        title_lower = (data['Event Name'] or '').lower()
        if 'конференц' in title_lower: data['Event Type'] = 'Конференция'
        elif 'митап' in title_lower or 'meetup' in title_lower: data['Event Type'] = 'Митап'
        elif 'хакатон' in title_lower: data['Event Type'] = 'Хакатон'
        else: data['Event Type'] = 'Мероприятие'

        return data

# --- ЗАПИСЬ В CSV ---

class CsvWriter:
    def __init__(self, filename: str):
        self.filename = filename
        self.fieldnames = ['Year', 'Start Date', 'End Date', 'Event Name', 'Event Type', 
                           'Description', 'Participants Count', 'Speakers/Organizers', 
                           'Partners', 'Category', 'Location', 'Source URL']
        
        # Создаем файл с заголовками, если его нет
        if not os.path.exists(self.filename):
            with open(self.filename, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def append(self, row: Dict):
        with open(self.filename, 'a', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            # Очищаем строки от переносов для корректного CSV
            clean_row = {k: str(v).replace('\n', ' ').replace('\r', '').strip() for k, v in row.items() if k in self.fieldnames}
            writer.writerow(clean_row)

# --- MAIN ---

def main():
    # Уточненный запрос с годом, чтобы найти актуальное
    query = "IT мероприятия Санкт-Петербург 2024-2025"
    
    searcher = DuckDuckGoSearch()
    urls = searcher.search(query)
    
    if not urls:
        print("Ссылки не найдены. Попробуйте изменить запрос или проверить соединение.")
        return

    print(f"Найдено ссылок: {len(urls)}")
    print(urls)

    parser = EventParser()
    writer = CsvWriter('events.csv')

    print("\nНачинаем парсинг...")
    for url in urls:
        try:
            print(f"Обработка: {url}")
            event_data = parser.parse(url)
            
            # Фильтр мусора: если нет названия, пропускаем
            if not event_data['Event Name']:
                print("  -> Пропуск (не удалось извлечь название)")
                continue
                
            writer.append(event_data)
            print(f"  -> Записано: {event_data['Event Name'][:40]}...")
            time.sleep(1) # Вежливость
            
        except Exception as e:
            print(f"  -> Ошибка: {e}")

    print("\nГотово! Данные сохранены в events.csv")

if __name__ == "__main__":
    main()
