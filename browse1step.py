# type: ignore
# ruff: noqa: F704
# %%

import os
import time
import llm
from playwright.async_api import async_playwright, Page
from sclog import getLogger

logger = getLogger(__name__)

# %%
# Make sure screenshots directory exists
os.makedirs("screenshots", exist_ok=True)

# %%
START_URL = "https://www.oberlin.edu/"  # 👈 start at Oberlin's main site

STEP_PROMPTS = [
    """Step 1:
You are currently on the Oberlin College main site.
Find and navigate to the Computer Science department page.
Once there, stop and return the page HTML.
When you have successfully reached the Computer Science department page, output the message:
"STEP 1 COMPLETE".""",

    """Step 2:
You are now on the Computer Science department page.
Find the section that lists faculty members.
Once the faculty page is reached, stop and return the page HTML.
When you have successfully reached the faculty list page, output the message:
"STEP 2 COMPLETE".""",

    """Step 3:
You are now on the faculty list page for the Computer Science department.
Extract the names of all emeriti (Emeritus/Emerita) faculty.
Output only the names, one per line, and then write:
"STEP 3 COMPLETE".""",
]

# %%
class PlaywrightTools(llm.Toolbox):
    def __init__(self, page: Page):
        super().__init__()
        self.page = page
        self.history: list[tuple[str, str]] = []
        self.screenshot_index = 0

    async def _get_html(self) -> str:
        return await self.page.locator("body").inner_html()

    async def _take_screenshot(self, label: str):
        self.screenshot_index += 1
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"screenshots/{self.screenshot_index:03d}-{label}-{timestamp}.png"
        await self.page.screenshot(path=filename, full_page=True)
        current_url = self.page.url
        logger.info(f"📸 Screenshot saved: {filename} | URL: {current_url}")
        with open("run_log.txt", "a") as log:
            log.write(f"{timestamp} | STEP={label} | URL={current_url}\n")

    async def click(self, selector: str, description: str = "") -> str:
        logger.debug(f"Clicking on {description} ({selector})")
        await self.page.locator(selector).click()
        self.history.append(("click", selector))
        await self._take_screenshot("click")
        return await self._get_html()

    async def go_back(self) -> str:
        logger.debug("Going back")
        await self.page.go_back()
        self.history.append(("back", ""))
        await self._take_screenshot("back")
        return await self._get_html()

    async def get_html(self) -> str:
        await self._take_screenshot("html")
        return await self._get_html()


# %%
def should_continue(skip_count: int) -> tuple[bool, int]:
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
# Top-level async code (works with async_run.py)
# %%

playwright = await async_playwright().start()
browser = await playwright.chromium.launch(channel="chrome", headless=False)
context = await browser.new_context(viewport={"width": 1440, "height": 1700})
page = await context.new_page()
page.set_default_timeout(8000)

await page.goto(START_URL)
await page.screenshot(path="screenshots/000-start.png", full_page=True)

MODEL = "gemini-2.5-flash"  # or "gpt-4.1-turbo"
model = llm.get_async_model(MODEL)

tools = PlaywrightTools(page)
conversation = model.conversation(tools=[tools])

logger.debug("Starting one-step-at-a-time guide")

current_step = 0
response = await conversation.prompt(
    prompt=await tools._get_html(),
    system=STEP_PROMPTS[current_step],
    tools=[tools],
)

while current_step < len(STEP_PROMPTS):
    logger.info(f"🚀 Starting Step {current_step + 1}")

    tool_calls = await response.tool_calls()
    if tool_calls:
        logger.debug(f"Tool calls found in Step {current_step + 1}")
        tool_results = await response.execute_tool_calls()
        await tools._take_screenshot(f"step{current_step+1}")

        # Handle both list and single ToolResult
        if isinstance(tool_results, list):
            # Access .response instead of .text()
            text_output = "\n".join([tr.response for tr in tool_results])
        else:
            text_output = tool_results.response
    else:
        text_output = await response.text()

    if f"STEP {current_step + 1} COMPLETE" in text_output:
        logger.info(f"✅ Step {current_step + 1} output:\n{text_output}")
        print(f"\n=== Step {current_step + 1} Output ===\n{text_output}\n")

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        with open("run_log.txt", "a") as log:
            log.write(f"{timestamp} | STEP {current_step+1} COMPLETE | URL={page.url}\n")

        current_step += 1
        if current_step < len(STEP_PROMPTS):
            response = await conversation.prompt(
                system=STEP_PROMPTS[current_step],
                tools=[tools],
                prompt=await tools._get_html(),
            )

final_text = await response.text()
print(f"Final response:\n{final_text}\n")

print("\nSteps taken:")
for action, value in tools.history:
    print(f"- {action}: {value}")

await tools._take_screenshot("final")

await page.close()
await context.close()
await browser.close()
await playwright.stop()