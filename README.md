# Voice AI Bot for Discord

Голосовой AI-собеседник для Discord-каналов. Слушает речь в реальном времени, распознаёт, отвечает голосом.

**Pipeline:** Discord audio → DAVE/MLS decrypt → Silero VAD → faster-whisper STT → Anthropic Claude → Yandex SpeechKit TTS → Discord voice

## Особенности

- **Realtime разговор** — латентность ~1-2с от конца твоей фразы до начала ответа
- **DAVE/MLS receive patch** — наше решение для приёма audio с включённым E2EE Discord. Без этого боты получают только зашифрованный шум на современных каналах. **Реализовано вручную через `davey.DaveSession.decrypt()`** — публичных библиотек с этой фичей пока нет
- **3 режима реакции:**
  - `all` — отвечает на каждую фразу
  - `name` — только когда обращаются по имени ("эй бот, как дела")
  - `listener` — слушает разговор, иногда вставляет реплику со случайным шансом
- **Voice Activity Detection** — Silero v5 ONNX, режет фразы по паузам
- **Whisper STT на GPU** — `large-v3-turbo` через faster-whisper, RTF≈0.04
- **Claude Haiku 4.5** для умных ответов (~1с/реплика, отлично знает русский)
- **Yandex SpeechKit** для голоса — натуральные нейросетевые голоса (Захар, Алёна, Жанна и др.)
- **Краткосрочная история диалога** — последние 20 реплик в RAM
- **6 встроенных персон** — токсичный, саркастичный, спокойный, гиперактивный, злой, нейтральный

## Требования

- **Python 3.10+** (тестировано на 3.10 и 3.14)
- **NVIDIA GPU** с минимум 4GB VRAM (для Whisper). RTX 30/40/50 серии.
- **CUDA Toolkit** или CUDA-runtime DLL'ы (ставятся через pip)
- **FFmpeg** в PATH (нужен discord.py для проигрывания)
- **API-ключи** (см. ниже)

### API-ключи

| Сервис | Что для чего | Цена |
|---|---|---|
| Discord Bot Token | Сам бот | Бесплатно |
| **Anthropic Claude API** | LLM (мозг) | Платный, ~$0.001 за реплику |
| **Yandex SpeechKit** | TTS (голос) | Бесплатные кредиты для новых пользователей |

## Установка

```bash
git clone https://github.com/Zaifat/discord-voice-ai-bot.git
cd voice-ai-bot
python setup.py
```

`setup.py` автоматически:
- Установит зависимости из `requirements.txt`
- Поставит `discord-ext-voice-recv` напрямую с GitHub (для DAVE-патча)
- Скачает Silero VAD модель (~2MB) в `models/`
- Создаст `.env` из `.env.example`

После этого открой `.env` и заполни ключи (см. ниже).

### 1. Discord Bot Token

1. Зайди на https://discord.com/developers/applications
2. **New Application** → введи имя
3. Слева → **Bot** → **Reset Token** → копируй
4. В **Privileged Gateway Intents** включи:
   - **MESSAGE CONTENT INTENT**
   - **SERVER MEMBERS INTENT**
5. Слева → **OAuth2** → **URL Generator** → отметь scope **`bot`**, права:
   - View Channels, Connect, Speak, Use Voice Activity, Send Messages
6. Открой получившуюся ссылку и пригласи бота на свой сервер
7. Вставь токен в `.env` → `DISCORD_TOKEN=...`

### 2. Anthropic Claude API

1. https://console.anthropic.com/settings/keys → **Create Key** → копируй
2. https://console.anthropic.com/settings/billing → пополни **минимум $5**
3. Вставь в `.env` → `ANTHROPIC_API_KEY=sk-ant-api03-...`

### 3. Yandex SpeechKit

1. https://console.cloud.yandex.ru/ → войди / зарегистрируйся (новым пользователям обычно дают бесплатные кредиты)
2. Открой **Cloud → твой каталог → IAM → Сервисные аккаунты** → **Создать**:
   - Имя: `voice-bot`
   - Роль: **`ai.speechkit-tts.user`**
3. Зайди в созданный аккаунт → **API-ключи** → **Создать API-ключ** → копируй (показывается **один раз**)
4. Вставь в `.env` → `YANDEX_API_KEY=AQVN...`

## Запуск

**Windows:**
```bat
run_bot.bat
```

**Linux/Mac:**
```bash
python bot.py
```

Бот залогинится в Discord и (если в `.env` указан `AUTO_JOIN_CHANNEL`) автоматически зайдёт в этот голосовой канал и начнёт слушать.

## Команды (текстовый чат на сервере)

### Голосовой канал
| Команда | Описание |
|---|---|
| `!join` / `!зайди` | Зайти в твой текущий голосовой канал |
| `!leave` / `!выйди` | Выйти |

### Режим реакции
| Команда | Описание |
|---|---|
| `!mode` | Показать текущий режим и список доступных |
| `!mode all` | Отвечать на все фразы (по умолчанию) |
| `!mode name` | Отвечать только когда обращаются по имени ("эй бот", "voiceai", "ИИ"). После ответа — 10 секунд follow-up без имени |
| `!mode listener 0.15` | Слушать разговор, вставлять реплику с шансом 15% (можно настраивать 0.0..1.0) |

### Персонажи
| Команда | Описание |
|---|---|
| `!persona` | Показать текущего и список доступных |
| `!personas` | Подробное описание всех персон |
| `!persona toxic` | Токсичный циник с матом |
| `!persona sarcastic` | Сухой саркастичный остряк |
| `!persona calm` | Спокойный, мудрый |
| `!persona hyper` | Гиперактивный энтузиаст |
| `!persona angry` | Постоянно раздражённый ворчун |
| `!persona default` | Нейтральный дружелюбный |

### Память (три уровня)
| Команда | Описание |
|---|---|
| `!history` или `!history 20` | Показать последние N реплик из persistent JSONL-лога |
| `!clear` | Очистить **краткосрочную** RAM-историю (persistent лог сохраняется) |
| `!clear all` | Очистить и RAM, и persistent JSONL-лог |
| `!remember <факт>` | Явно сохранить факт в **долгосрочную** vector-память (ChromaDB) |
| `!forget` | Очистить долгосрочную vector-память |
| `!status` / `!статус` | Состояние бота (режим, персона, размер всех 3 уровней памяти) |

## Память — как устроена

| Уровень | Где хранится | Когда используется |
|---|---|---|
| **Краткосрочная (RAM)** | `deque` в памяти процесса, последние 30 реплик | Передаётся Claude как context window каждый запрос |
| **Persistent JSONL** | `chat_logs/<guild_id>.jsonl` (на диске) | При старте бот загружает последние 30 реплик отсюда — диалог продолжается после перезапуска |
| **Долгосрочная (vector)** | `memory_db/` (ChromaDB + bge-m3 embeddings) | На каждой новой фразе делается семантический поиск — релевантные старые факты добавляются в system prompt |

После перезапуска бот **помнит** что ты говорил вчера: краткие реплики восстанавливаются из JSONL, а семантически близкие факты находятся через vector search.

## Структура проекта

```
voice-ai-bot/
├── bot.py                  ← главный файл, объединяет pipeline
├── config.py               ← глобальный конфиг (читает .env)
├── setup.py                ← скрипт автоустановки
├── run_bot.bat             ← запуск на Windows
├── requirements.txt
├── .env.example
│
├── audio/
│   ├── dave_patch.py       ← патч для discord-ext-voice-recv (DAVE/MLS receive)
│   ├── vad.py              ← Silero VAD streaming (ONNX)
│   ├── stt.py              ← faster-whisper (GPU CUDA)
│   └── tts.py              ← Yandex SpeechKit HTTP клиент
│
├── llm/
│   └── client.py           ← Anthropic Claude AsyncClient
│
├── memory/
│   └── store.py            ← ConversationHistory + ChromaDB long-term
│
├── personality/
│   └── engine.py           ← 6 предустановленных персон
│
└── models/
    └── silero_vad.onnx     ← VAD модель (скачивается setup-скриптом)
```

## DAVE/MLS receive patch — главное достижение проекта

Discord в 2024 включил **DAVE (Discord Audio/Video End-to-End Encryption)** на голосовых каналах. Вторая шифровка поверх transport encryption через MLS-протокол. **Боты, которые используют `discord-ext-voice-recv`, получают только зашифрованный мусор** — opus_decode выдаёт corrupted stream или garbage PCM.

Решение в [`audio/dave_patch.py`](audio/dave_patch.py):

```python
# До opus_decode перехватываем decrypted_data из voice_recv,
# прогоняем через davey.DaveSession.decrypt(sender_id, audio, payload)
# и подменяем decrypted_data на расшифрованный плейн.
voice_client._connection.dave_session.decrypt(
    int(sender_id), davey.MediaType.audio, bytes(packet.decrypted_data)
)
```

После этого все 100% пакетов корректно декодируются в чистый PCM.

## Производительность

На RTX 5070 (12GB VRAM):
| Этап | Latency |
|---|---|
| Silero VAD | ~30мс/чанк |
| Whisper large-v3-turbo (GPU fp16) | ~150-200мс на 2-секундную фразу |
| Claude Haiku 4.5 (API) | ~1000-1500мс |
| Yandex TTS (API) | ~500-1000мс |
| Discord audio delivery | ~200мс |
| **Total — конец фразы → начало ответа** | **~2-3 секунды** |

VRAM: ~1.5GB (только Whisper).

## Известные ограничения

- **Yandex SpeechKit стоит денег** после free tier (~$4 за 1M символов)
- **Anthropic Claude API стоит денег** (Haiku ~$0.005/1K input)
- **Цензура у Claude** — на запросах вроде "напиши инструкцию по убийству" откажет (это alignment в весах). На разговорный мат — норм
- **Whisper иногда галлюцинирует** на коротких/тихих фразах. Обходится постфильтрацией повторов
- **DAVE epoch transitions** — иногда несколько пакетов теряются с `CryptoError`. Один разовый сбой за час диалога. Не критично

## Альтернативные варианты под капотом

В коде заложена возможность переключиться:

- **LLM**: вместо Claude можно использовать локальный Ollama (gemma3, qwen3, llama3, dolphin) — поправь `LLM_MODEL` в `llm/client.py` и используй разные backend модули
- **TTS**: можно вернуть на локальный Piper TTS / Silero / edge-tts / XTTS-v2 — заменить `audio/tts.py`
- **STT**: faster-whisper можно заменить на whisper.cpp / vosk / Yandex SpeechKit ASR

## Лицензия

MIT (см. LICENSE)

## Authors

Made with persistent debugging and a lot of patience by [Zaifat](https://github.com/Zaifat)
