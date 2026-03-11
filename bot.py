from __future__ import annotations

import html
import json
import re
import sys
import textwrap
import warnings
import subprocess
from dataclasses import dataclass
from datetime import datetime
from getpass import GetPassWarning, getpass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

try:
    from openai import OpenAI
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: openai. Install it with: pip install openai"
    ) from exc

try:
    from urllib3.exceptions import NotOpenSSLWarning
except ImportError:
    NotOpenSSLWarning = None


APP_DIR = Path(__file__).resolve().parent
CONFIG_PATH = APP_DIR / "bot.config.json"
DASHBOARD_PATH = APP_DIR / "dashboard.html"

MODEL_OPTIONS: dict[str, dict[str, str]] = {
    "1": {
        "provider": "openai",
        "model": "gpt-5-mini",
        "display": "OpenAI - gpt-5-mini",
    },
    "2": {
        "provider": "openai",
        "model": "gpt-5",
        "display": "OpenAI - gpt-5",
    },
    "3": {
        "provider": "groq",
        "model": "llama-3.3-70b-versatile",
        "display": "Groq - llama-3.3-70b-versatile",
    },
}


@dataclass
class ThreadContent:
    url: str
    platform: str
    title: str
    content: str


TONE_OPTIONS: dict[str, dict[str, str]] = {
    "1": {
        "label": "helpful",
        "instruction": "Be warm, practical, constructive, and sincerely helpful.",
    },
    "2": {
        "label": "funny",
        "instruction": "Use light humor and wit, but keep it natural, friendly, and not cringe or forced.",
    },
    "3": {
        "label": "socratic questioning",
        "instruction": "Reply mainly through thoughtful questions that help the other person reflect, clarify, or refine their thinking.",
    },
    "4": {
        "label": "serious",
        "instruction": "Use a grounded, direct, thoughtful, and no-nonsense tone.",
    },
    "5": {
        "label": "provocative",
        "instruction": "Be bold, sharp, and slightly challenging, but do not be abusive, hateful, or needlessly inflammatory.",
    },
}

LENGTH_OPTIONS: dict[str, dict[str, str]] = {
    "1": {
        "label": "short",
        "instruction": "Keep the reply very short, around 1 sentence, or 2 short sentences at most.",
    },
    "2": {
        "label": "medium",
        "instruction": "Keep the reply concise, around 1 to 3 sentences.",
    },
    "3": {
        "label": "long",
        "instruction": "Write a more developed reply with some detail and nuance, around 4 to 7 sentences.",
    },
}

PLATFORM_DISPLAY_NAMES: dict[str, str] = {
    "reddit": "Reddit",
    "linkedin": "LinkedIn",
}


def configure_warnings() -> None:
    if NotOpenSSLWarning is not None:
        warnings.filterwarnings("ignore", category=NotOpenSSLWarning)


def ensure_app_dir() -> None:
    APP_DIR.mkdir(parents=True, exist_ok=True)


def load_config() -> dict[str, Any] | None:
    if not CONFIG_PATH.exists():
        return None
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def save_config(config: dict[str, Any]) -> None:
    ensure_app_dir()
    CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")


def prompt_secret(prompt: str) -> str:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", GetPassWarning)
            value = getpass(prompt)
    except GetPassWarning:
        print("Warning: secure hidden input is not available in this terminal.")
        value = input(prompt)
    return value.strip()


def prompt_nonempty(prompt: str, default: str | None = None) -> str:
    while True:
        suffix = f" [{default}]" if default else ""
        value = input(f"{prompt}{suffix}: ").strip()
        if value:
            return value
        if default is not None:
            return default
        print("Please enter a value.")


def prompt_choice(prompt: str, options: dict[str, dict[str, str]], default_key: str) -> dict[str, str]:
    print(prompt)
    for key, option in options.items():
        print(f"{key}. {option['label'].title()}")

    while True:
        choice = input(f"Enter number [{default_key}]: ").strip()
        if not choice:
            choice = default_key
        if choice in options:
            return options[choice]
        print("Invalid choice. Please try again.")


def print_setup_intro() -> None:
    message = """
    First-time setup

    This tool does NOT post anything.
    It reads one Reddit or LinkedIn post URL, drafts one reply, and writes the result to a local HTML file.
    """
    print(textwrap.dedent(message).strip())
    print()


def prompt_first_run_config() -> dict[str, Any]:
    print_setup_intro()
    print("Choose which AI provider/model to use:\n")

    for key, option in MODEL_OPTIONS.items():
        print(f"{key}. {option['display']}")

    while True:
        choice = input("\nEnter number: ").strip()
        if choice in MODEL_OPTIONS:
            selected = MODEL_OPTIONS[choice]
            break
        print("Invalid choice. Please try again.")

    provider = selected["provider"]
    model = selected["model"]
    print(f"\nYou selected: {selected['display']}")

    api_label = "OpenAI" if provider == "openai" else "Groq"
    api_key = prompt_secret(f"Enter your {api_label} API key: ")

    config = {
        "provider": provider,
        "model": model,
        "api_key": api_key,
    }

    save_config(config)
    print(f"\nConfiguration saved to {CONFIG_PATH}\n")
    return config


def get_config() -> dict[str, Any]:
    if "--reset-config" in sys.argv and CONFIG_PATH.exists():
        CONFIG_PATH.unlink()
        print("Deleted existing config. Starting setup again.\n")

    config = load_config()
    if config is not None:
        return config
    return prompt_first_run_config()


def validate_config(config: dict[str, Any]) -> None:
    required_keys = ["provider", "model", "api_key"]
    missing_keys = [key for key in required_keys if not str(config.get(key, "")).strip()]
    if missing_keys:
        missing_text = ", ".join(missing_keys)
        raise SystemExit(
            f"Config is missing required values: {missing_text}. Run with --reset-config and enter them again."
        )


def build_openai_client(config: dict[str, Any]) -> OpenAI:
    provider = config["provider"]
    api_key = config["api_key"]

    if provider == "openai":
        return OpenAI(api_key=api_key)
    if provider == "groq":
        return OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")

    raise ValueError(f"Unsupported provider: {provider}")



def normalize_input_url(url: str) -> str:
    value = url.strip()
    if not value:
        raise SystemExit("Please provide a supported post URL.")
    if not value.startswith("http://") and not value.startswith("https://"):
        value = f"https://{value}"
    return value



def detect_platform(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.netloc.lower()

    if "reddit.com" in host or "redd.it" in host:
        return "reddit"
    if "linkedin.com" in host:
        return "linkedin"

    raise SystemExit(
        "Unsupported URL. Please provide a Reddit or LinkedIn post URL."
    )


def fetch_thread_html(url: str) -> str:
    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36"
        },
    )
    try:
        with urlopen(request, timeout=20) as response:
            return response.read().decode("utf-8", errors="replace")
    except Exception as exc:
        raise SystemExit(f"Could not load the thread URL: {exc}") from exc


def strip_tags(value: str) -> str:
    text = re.sub(r"<script.*?</script>", " ", value, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_between(pattern: str, content: str) -> str:
    match = re.search(pattern, content, flags=re.DOTALL | re.IGNORECASE)
    if not match:
        return ""
    return strip_tags(match.group(1))


# --- Added helper functions for improved extraction ---
def extract_all_between(pattern: str, content: str) -> list[str]:
    matches = re.findall(pattern, content, flags=re.DOTALL | re.IGNORECASE)
    return [strip_tags(match) for match in matches if strip_tags(match)]


def extract_json_string_field(content: str, field_name: str) -> list[str]:
    pattern = rf'"{re.escape(field_name)}"\s*:\s*"((?:\\.|[^"\\])*)"'
    matches = re.findall(pattern, content, flags=re.DOTALL)
    results: list[str] = []

    for match in matches:
        try:
            decoded = bytes(match, "utf-8").decode("unicode_escape")
        except Exception:
            decoded = match
        cleaned = strip_tags(decoded)
        if cleaned:
            results.append(cleaned)

    return results


def score_body_candidate(text: str) -> int:
    value = clean_text(text, 5000)
    lower = value.lower()
    score = 0

    length = len(value)
    if 80 <= length <= 4000:
        score += 40
    elif 40 <= length <= 6000:
        score += 20

    bad_phrases = [
        "reddit - the heart of the internet",
        "skip to main content",
        "open menu",
        "open navigation",
        "go to reddit home",
        "get the reddit app",
        "log in",
        "create your account",
        "user agreement",
        "privacy policy",
        "accessibility",
        "reddit, inc.",
        "all rights reserved",
        "expand navigation",
        "collapse navigation",
        "read more share",
        "public anyone can view",
    ]
    for phrase in bad_phrases:
        if phrase in lower:
            score -= 80

    good_signals = [
        " i ",
        " i'm ",
        " i’ve ",
        " i was ",
        " i feel ",
        " my ",
        " we ",
        " because ",
        " but ",
    ]
    for signal in good_signals:
        if signal in f" {lower} ":
            score += 8

    punctuation_count = sum(value.count(char) for char in ".,!?;:")
    score += min(punctuation_count * 2, 30)

    word_count = len(value.split())
    if word_count >= 40:
        score += 20
    elif word_count >= 20:
        score += 10

    return score


def pick_best_body_candidate(candidates: list[str]) -> str:
    cleaned_candidates: list[str] = []
    seen: set[str] = set()

    for candidate in candidates:
        cleaned = clean_text(candidate, 5000)
        normalized = re.sub(r"\s+", " ", cleaned).strip().lower()
        if not cleaned or normalized in seen:
            continue
        seen.add(normalized)
        cleaned_candidates.append(cleaned)

    if not cleaned_candidates:
        return ""

    ranked = sorted(
        cleaned_candidates,
        key=lambda item: (score_body_candidate(item), len(item)),
        reverse=True,
    )
    return ranked[0]


def clean_text(text: str, limit: int = 4000) -> str:
    value = (text or "").strip()
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."




# --- Platform-generic parsing and helpers ---
def clean_title_text(title: str, platform: str) -> str:
    value = clean_text(title or "Post", 300)
    lower = value.lower()

    if platform == "reddit":
        separators = [" : r/", " : "]
        for separator in separators:
            if separator in value:
                value = value.split(separator)[0].strip()
                break
    elif platform == "x":
        for separator in [" / X", " on X", " / Twitter", " on Twitter"]:
            if separator.lower() in lower:
                value = value[: lower.index(separator.lower())].strip()
                break
    elif platform == "linkedin":
        for separator in [" | LinkedIn", " on LinkedIn"]:
            if separator.lower() in lower:
                value = value[: lower.index(separator.lower())].strip()
                break

    return clean_text(value or "Post", 300)



def extract_meta_content(page_html: str, property_name: str) -> list[str]:
    return extract_all_between(
        rf'<meta[^>]+(?:property|name)="{re.escape(property_name)}"[^>]+content="(.*?)"[^>]*>',
        page_html,
    )



def parse_reddit_content(url: str, page_html: str) -> ThreadContent:
    title = extract_between(r"<title>(.*?)</title>", page_html)
    body_candidates: list[str] = []

    body_candidates.extend(
        extract_all_between(
            r'<div[^>]+slot="text-body"[^>]*>(.*?)</div>',
            page_html,
        )
    )
    body_candidates.extend(
        extract_all_between(
            r'<shreddit-post[\s\S]*?<div[^>]+slot="text-body"[^>]*>(.*?)</div>',
            page_html,
        )
    )
    body_candidates.extend(extract_json_string_field(page_html, "content"))
    body_candidates.extend(extract_json_string_field(page_html, "selftext"))
    body_candidates.extend(extract_json_string_field(page_html, "body"))
    body_candidates.extend(extract_meta_content(page_html, "og:description"))
    body_candidates.extend(extract_meta_content(page_html, "description"))

    body = pick_best_body_candidate(body_candidates)
    if not body:
        body = clean_text(strip_tags(page_html), 3000)

    return ThreadContent(
        url=url,
        platform="reddit",
        title=clean_title_text(title, "reddit"),
        content=clean_text(body, 3000),
    )






def parse_linkedin_content(url: str, page_html: str) -> ThreadContent:
    title = extract_between(r"<title>(.*?)</title>", page_html)
    content_candidates: list[str] = []

    content_candidates.extend(extract_meta_content(page_html, "og:description"))
    content_candidates.extend(extract_meta_content(page_html, "description"))
    content_candidates.extend(extract_json_string_field(page_html, "description"))
    content_candidates.extend(extract_json_string_field(page_html, "text"))
    content_candidates.extend(
        extract_all_between(
            r'<div[^>]+class="[^"]*break-words[^"]*"[^>]*>(.*?)</div>',
            page_html,
        )
    )

    content = pick_best_body_candidate(content_candidates)
    if not content:
        content = clean_text(strip_tags(page_html), 2500)

    return ThreadContent(
        url=url,
        platform="linkedin",
        title=clean_title_text(title, "linkedin"),
        content=clean_text(content, 2500),
    )



def parse_post_content(url: str, page_html: str, platform: str) -> ThreadContent:
    if platform == "reddit":
        return parse_reddit_content(url, page_html)
    if platform == "linkedin":
        return parse_linkedin_content(url, page_html)

    raise SystemExit(f"Unsupported platform: {platform}")


def draft_reply(
    client: OpenAI,
    config: dict[str, Any],
    thread: ThreadContent,
    tone_option: dict[str, str],
    length_option: dict[str, str],
) -> str:
    platform_name = PLATFORM_DISPLAY_NAMES.get(thread.platform, thread.platform.title())
    prompt = f"""
You are drafting one social media reply for HUMAN REVIEW ONLY.

Write a natural reply to this {platform_name} post.
Do not mention being an AI.
Do not be promotional.
Do not ask for likes, upvotes, reposts, or engagement.
Do not use emojis unless the requested tone would clearly justify it, and even then use them sparingly.
Match normal {platform_name} tone.
If the post lacks detail, acknowledge that gently.
Output only the reply text.

Tone requirement:
{tone_option['instruction']}

Length requirement:
{length_option['instruction']}

Post title:
{thread.title}

Post content:
{thread.content or '[No post content found]'}
""".strip()

    response = client.responses.create(model=config["model"], input=prompt)
    return response.output_text.strip()




# --- Persistent dashboard history functions ---
def render_history_entry(
    thread: ThreadContent,
    draft: str,
    tone_label: str,
    length_label: str,
    generated_at: str,
) -> str:
    return f"""
  <article class="card">
    <p><a href="{html.escape(thread.url)}" target="_blank" rel="noopener noreferrer">Open post</a></p>
    <h2>{html.escape(thread.title)}</h2>
    <p><strong>Platform:</strong> {html.escape(PLATFORM_DISPLAY_NAMES.get(thread.platform, thread.platform.title()))}</p>
    <p><strong>Tone:</strong> {html.escape(tone_label.title())}</p>
    <p><strong>Length:</strong> {html.escape(length_label.title())}</p>
    <h3>Post content</h3>
    <pre>{html.escape(thread.content or '[No post content found]')}</pre>
    <h3>Draft reply</h3>
    <pre>{html.escape(draft)}</pre>
    <p class="footer">Generated at {html.escape(generated_at)}</p>
  </article>
""".strip()



def build_dashboard_document(history_entries_html: str) -> str:
    return f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Reply Draft History</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      max-width: 1000px;
      margin: 40px auto;
      padding: 0 16px;
      background: #f6f7f8;
      color: #1f2328;
    }}
    .card {{
      background: white;
      border-radius: 12px;
      padding: 20px;
      box-shadow: 0 2px 10px rgba(0,0,0,0.06);
      margin-bottom: 20px;
    }}
    a {{
      color: #0969da;
      text-decoration: none;
    }}
    a:hover {{
      text-decoration: underline;
    }}
    pre {{
      white-space: pre-wrap;
      word-wrap: break-word;
      background: #f6f8fa;
      border-radius: 8px;
      padding: 12px;
      overflow-x: auto;
    }}
    .footer {{
      color: #57606a;
      font-size: 14px;
      margin-top: 24px;
    }}
  </style>
</head>
<body>
  <h1>Reply Draft History</h1>
  {history_entries_html}
</body>
</html>
""".strip()



def update_dashboard_history(
    thread: ThreadContent,
    draft: str,
    tone_label: str,
    length_label: str,
) -> None:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_entry = render_history_entry(
        thread=thread,
        draft=draft,
        tone_label=tone_label,
        length_label=length_label,
        generated_at=generated_at,
    )

    existing_entries = ""
    if DASHBOARD_PATH.exists():
        existing_html = DASHBOARD_PATH.read_text(encoding="utf-8")
        body_match = re.search(r"<body>(.*)</body>", existing_html, flags=re.DOTALL | re.IGNORECASE)
        if body_match:
            body_html = body_match.group(1)
            body_html = re.sub(r"<h1>.*?</h1>", "", body_html, count=1, flags=re.DOTALL | re.IGNORECASE).strip()
            existing_entries = body_html

    combined_entries = new_entry
    if existing_entries:
        combined_entries = f"{new_entry}\n\n{existing_entries}"

    DASHBOARD_PATH.write_text(
        build_dashboard_document(combined_entries),
        encoding="utf-8",
    )


# --- Clipboard copy helper ---
def copy_to_clipboard(text: str) -> None:
    try:
        subprocess.run(
            ["pbcopy"],
            input=text,
            text=True,
            check=True,
        )
    except Exception as exc:
        print(f"Warning: could not copy reply to clipboard: {exc}")


# --- Clear dashboard history helper ---
def clear_dashboard_history() -> None:
    if DASHBOARD_PATH.exists():
        DASHBOARD_PATH.unlink()
        print(f"Cleared dashboard history: {DASHBOARD_PATH}")
    else:
        print("No existing dashboard history to clear.")


def print_help() -> None:
    help_text = f"""
Usage:
  python bot.py
  python bot.py --clear-history
  python bot.py --reset-config

What it does:
  - Prompts for a Reddit or LinkedIn post URL
  - Detects the platform automatically
  - Reads the page content
  - Generates one draft reply with your chosen AI provider
  - Saves the result to a local HTML file
  - Can clear the saved dashboard history with --clear-history

Files:
  Config:    {CONFIG_PATH}
  Dashboard: {DASHBOARD_PATH}
"""
    print(textwrap.dedent(help_text).strip())


def main() -> None:
    if "--help" in sys.argv or "-h" in sys.argv:
        print_help()
        return

    configure_warnings()
    ensure_app_dir()
    if "--clear-history" in sys.argv:
        clear_dashboard_history()
        return
    config = get_config()
    validate_config(config)
    client = build_openai_client(config)

    input_url = normalize_input_url(prompt_nonempty("Enter Reddit or LinkedIn post URL"))
    platform = detect_platform(input_url)
    print("Loading post...")
    page_html = fetch_thread_html(input_url)
    thread = parse_post_content(input_url, page_html, platform)

    print()
    tone_option = prompt_choice("Choose reply tone:", TONE_OPTIONS, "1")
    print()
    length_option = prompt_choice("Choose reply length:", LENGTH_OPTIONS, "2")

    print("Drafting reply...")
    try:
        draft = draft_reply(client, config, thread, tone_option, length_option)
    except Exception as exc:
        raise SystemExit(f"Draft generation failed: {exc}") from exc

    update_dashboard_history(thread, draft, tone_option["label"], length_option["label"])
    copy_to_clipboard(draft)
    print(f"\nDone. The reply has been copied to your clipboard. The dashboard history has been updated:\n{DASHBOARD_PATH}")


if __name__ == "__main__":
    main()