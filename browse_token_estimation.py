# type: ignore
# ruff: noqa: F704
# %%

import os
import csv
import time
import llm
from playwright.async_api import async_playwright, Page
from sclog import getLogger
import tiktoken  # <-- Added for token counting

logger = getLogger(__name__)

# %%
# Make sure screenshots directory exists
os.makedirs("screenshots", exist_ok=True)

# %%
START_URL = "https://www.oberlin.edu"

SYSTEM_PROMPT = """You are an AI-enabled program with excellent understanding of HTML/CSS and no personality.

I am providing you with the HTML of the page I'm currently on.
Using the tools available, navigate the website.
(You will need to click on things to leave the first page!)

YOUR GOAL:
Go to the Studio Art department. Go to the Studio Art people page. Find the Studio Art professor with the shortest last name. Enter their profile. Tell me where they earned their Bachelor’s Degree.

This is not an interactive session, so do not ask questions or expect responses.
You can navigate the site by clicking links and returning HTML after each navigation.
"""

# %%
# Initialize tokenizer
enc = tiktoken.get_encoding("cl100k_base")  # GPT/Gemini-compatible tokenizer
total_input_tokens = 0
total_output_tokens = 0

# %%
# Define tools for the LLM to use
class PlaywrightTools(llm.Toolbox):
    def __init__(self, page: Page):
        super().__init__()
        self.page = page
        self.history: list[tuple[str, str]] = []
        self.screenshot_index = 0

        # Initialize CSV logging
        self.csv_path = "page_log.csv"
        with open(self.csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "step", "action", "description", "url"])

    async def _get_html(self) -> str:
        """Return the HTML of the current page."""
        return await self.page.locator("body").inner_html()

    async def _take_screenshot(self, label: str):
        """Save a screenshot with timestamp and step index."""
        self.screenshot_index += 1
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"screenshots/{self.screenshot_index:03d}-{label}-{timestamp}.png"
        await self.page.screenshot(path=filename, full_page=True)
        logger.info(f"📸 Screenshot saved: {filename}")

    async def _log_step(self, action: str, description: str = ""):
        """Append current URL and action info to the CSV log."""
        url = self.page.url
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(self.csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, self.screenshot_index, action, description, url])
        logger.info(f"📝 Logged step: {action} | {description} | {url}")

    async def click(self, selector: str, description: str = "") -> str:
        logger.debug(f"Clicking on {description} ({selector})")
        await self.page.locator(selector).click()
        self.history.append(("click", selector))
        await self._take_screenshot("click")
        await self._log_step("click", description or selector)
        return await self._get_html()

    async def go_back(self) -> str:
        logger.debug("Going back")
        await self.page.go_back()
        self.history.append(("back", ""))
        await self._take_screenshot("back")
        await self._log_step("back")
        return await self._get_html()

    async def get_html(self) -> str:
        await self._take_screenshot("html")
        await self._log_step("get_html")
        return await self._get_html()


# %%
def should_continue(skip_count: int) -> tuple[bool, int]:
    """Prompt whether to continue executing LLM tool calls."""
    if skip_count > 0:
        return True, skip_count - 1
    try:
        confirm = input("Continue? (Y/n or number to skip) > ").lower().strip()
        if confirm.startswith("n"):
            return False, 0
        elif confirm.isdigit():
            return True, int(confirm)
    except EOFError:
        return False, 0
    return True, 0


###############################################################################
# Main logic
# %%
playwright = async_playwright()
p = await playwright.__aenter__()
browser = await p.chromium.launch(channel="chrome", headless=False)
context = await browser.new_context(viewport={"width": 1440, "height": 1700})
page = await context.new_page()
page.set_default_timeout(8000)

await page.goto(START_URL)
await page.screenshot(path="screenshots/000-start.png", full_page=True)

# Log the starting page URL
with open("page_log.csv", mode="a", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), 0, "start", "initial load", page.url])

# %%
# Set up LLM
MODEL = "gemini-2.5-flash"  # or "gpt-4.1-turbo" if available
model = llm.get_async_model(MODEL)

tools = PlaywrightTools(page)
conversation = model.conversation(tools=[tools])

# Initial query
html_prompt = await tools._get_html()
full_prompt = SYSTEM_PROMPT + "\n" + html_prompt

# Count input tokens
num_input_tokens = len(enc.encode(full_prompt))
total_input_tokens += num_input_tokens
print(f"[Token count] Initial prompt input tokens: {num_input_tokens}")

response = await conversation.prompt(
    prompt=html_prompt,
    system=SYSTEM_PROMPT,
    tools=[tools],
)
logger.debug("Making initial query to LLM")
response_text = await response.text()

# Count output tokens
num_output_tokens = len(enc.encode(response_text))
total_output_tokens += num_output_tokens
print(f"[Token count] Initial prompt output tokens: {num_output_tokens}")

# %%
# Agentic loop
skip_confirmation_for = 0
while True:
    tool_calls = await response.tool_calls()
    if not tool_calls:
        break

    tool_results = await response.execute_tool_calls()
    await tools._take_screenshot("step")  # Screenshot after each action

    do_continue, skip_confirmation_for = should_continue(skip_confirmation_for)
    if not do_continue:
        break

    # Build prompt for next iteration
    next_prompt_text = ""  # Replace if you pass new HTML/tool results here
    full_prompt = SYSTEM_PROMPT + "\n" + next_prompt_text + "\n" + str(tool_results)
    num_input_tokens = len(enc.encode(full_prompt))
    total_input_tokens += num_input_tokens
    print(f"[Token count] Loop iteration input tokens: {num_input_tokens}")

    response = conversation.prompt(
        system=SYSTEM_PROMPT,
        tool_results=tool_results,
        tools=[tools],
    )

    response_text = await response.text()
    num_output_tokens = len(enc.encode(response_text))
    total_output_tokens += num_output_tokens
    print(f"[Token count] Loop iteration output tokens: {num_output_tokens}")

# Final output
final_text = await response.text()
print(f"Final response:\n{final_text}\n")

num_output_tokens = len(enc.encode(final_text))
total_output_tokens += num_output_tokens
print(f"[Token count] Final output tokens: {num_output_tokens}")

print("\nSteps taken:")
for action, value in tools.history:
    print(f"- {action}: {value}")

await tools._take_screenshot("final")
print(f"\n[Token count] Total input tokens: {total_input_tokens}")
print(f"[Token count] Total output tokens: {total_output_tokens}")

await page.close()
await context.close()
await browser.close()
await playwright.__aexit__()
