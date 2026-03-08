from __future__ import annotations

import html
import json
import re
import sys
import textwrap
import warnings
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
    title: str
    body: str


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
        "instruction": "Keep the reply concise, around 1 to 3 sentences.",
    },
    "2": {
        "label": "medium",
        "instruction": "Keep the reply moderate in length, around 4 to 7 sentences.",
    },
    "3": {
        "label": "long",
        "instruction": "Write a more developed reply with detail and nuance, around 8 to 12 sentences.",
    },
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

    This tool does NOT post to Reddit.
    It reads one Reddit thread URL, drafts one reply, and writes the result to a local HTML file.
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


def normalize_reddit_url(url: str) -> str:
    value = url.strip()
    if not value:
        raise SystemExit("Please provide a Reddit thread URL.")
    if not value.startswith("http://") and not value.startswith("https://"):
        value = f"https://{value}"

    parsed = urlparse(value)
    if "reddit.com" not in parsed.netloc and "redd.it" not in parsed.netloc:
        raise SystemExit("Please provide a valid Reddit thread URL.")

    return value


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


def clean_text(text: str, limit: int = 4000) -> str:
    value = (text or "").strip()
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def parse_thread_content(url: str, page_html: str) -> ThreadContent:
    title = extract_between(r"<title>(.*?)</title>", page_html)
    if title.endswith(" : r/"):
        title = title[:-5].strip()
    if " : " in title:
        title = title.split(" : ")[0].strip()

    body = extract_between(r'<shreddit-post[\\s\\S]*?<div slot="text-body"[^>]*>(.*?)</div>', page_html)
    if not body:
        body = extract_between(r'"content"\s*:\s*"(.*?)"', page_html)
    if not body:
        body = strip_tags(page_html)

    body = clean_text(body, 3000)
    title = clean_text(title or "Reddit thread", 300)
    return ThreadContent(url=url, title=title, body=body)


def draft_reply(
    client: OpenAI,
    config: dict[str, Any],
    thread: ThreadContent,
    tone_option: dict[str, str],
    length_option: dict[str, str],
) -> str:
    prompt = f"""
You are drafting one Reddit reply for HUMAN REVIEW ONLY.

Write a natural reply to this Reddit thread.
Do not mention being an AI.
Do not be promotional.
Do not ask for upvotes or engagement.
Do not use emojis unless the requested tone would clearly justify it, and even then use them sparingly.
Match normal Reddit tone.
If the thread lacks detail, acknowledge that gently.
Output only the reply text.

Tone requirement:
{tone_option['instruction']}

Length requirement:
{length_option['instruction']}

Thread title:
{thread.title}

Thread body:
{thread.body or '[No body text found]'}
""".strip()

    response = client.responses.create(model=config["model"], input=prompt)
    return response.output_text.strip()


def render_html(thread: ThreadContent, draft: str, tone_label: str, length_label: str) -> str:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Reddit Reply Draft</title>
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
  <div class="card">
    <h1>Reddit Reply Draft</h1>
    <p><a href="{html.escape(thread.url)}" target="_blank" rel="noopener noreferrer">Open thread</a></p>
    <h2>{html.escape(thread.title)}</h2>
    <p><strong>Tone:</strong> {html.escape(tone_label.title())}</p>
    <p><strong>Length:</strong> {html.escape(length_label.title())}</p>
    <h3>Thread body</h3>
    <pre>{html.escape(thread.body or '[No body text found]')}</pre>
    <h3>Draft reply</h3>
    <pre>{html.escape(draft)}</pre>
    <p class="footer">Generated at {html.escape(generated_at)}</p>
  </div>
</body>
</html>
""".strip()


def print_help() -> None:
    help_text = f"""
Usage:
  python bot.py
  python bot.py --reset-config

What it does:
  - Prompts for a Reddit thread URL
  - Reads the page content
  - Generates one draft reply with your chosen AI provider
  - Saves the result to a local HTML file

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
    config = get_config()
    validate_config(config)
    client = build_openai_client(config)

    thread_url = normalize_reddit_url(prompt_nonempty("Enter Reddit thread URL"))
    print("Loading thread...")
    page_html = fetch_thread_html(thread_url)
    thread = parse_thread_content(thread_url, page_html)

    print()
    tone_option = prompt_choice("Choose reply tone:", TONE_OPTIONS, "1")
    print()
    length_option = prompt_choice("Choose reply length:", LENGTH_OPTIONS, "2")

    print("Drafting reply...")
    try:
        draft = draft_reply(client, config, thread, tone_option, length_option)
    except Exception as exc:
        raise SystemExit(f"Draft generation failed: {exc}") from exc

    html_output = render_html(thread, draft, tone_option["label"], length_option["label"])
    DASHBOARD_PATH.write_text(html_output, encoding="utf-8")

    print(f"\nDone. Open this file in your browser:\n{DASHBOARD_PATH}")


if __name__ == "__main__":
    main()