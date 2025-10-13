# type: ignore
# ruff: noqa: F704
# %%

import llm
from playwright.async_api import async_playwright, Page
from sclog import getLogger

logger = getLogger(__name__)

# %%

START_URL = "https://www.oberlin.edu/"

SYSTEM_PROMPT = """
You are an autonomous web navigation agent with strong knowledge of HTML and CSS.

Your goal: Starting from the Oberlin College homepage, navigate to the **College of Arts and Sciences** page.

You have access to these tools:
- `click(selector, description)` → Click an element on the page using a CSS or XPath selector.
- `go_back()` → Go to the previous page.

**Instructions:**
1. Inspect the HTML of the current page (it is provided to you as input).
2. Identify the link or button that leads to the College of Arts and Sciences page.
3. Use `click()` with an unambiguous selector (e.g., `css=a[href*="arts-and-sciences"]`) to navigate.
4. Once the page title or heading clearly refers to “College of Arts and Sciences,” respond with **"Success"**.
5. If you cannot find it after reasonable attempts, respond with **"Failure"**.
6. Do not ask questions or expect user input — act autonomously using the tools.

Return only:
- Tool calls (if actions needed), or
- "Success"/"Failure" (as plain text).
"""


# %%
# Define tools that the LLM will be able to use
# Note: the methods' docstrings will be passed to the LLM too
class PlaywrightTools(llm.Toolbox):
    def __init__(self, page: Page):
        super().__init__()
        self.page = page
        self.history: list[tuple[str, str]] = []

    async def _get_html(self) -> str:
        """
        Return the HTML of the current page.
        This method is not passed to the LLM because its name starts with _.
        """
        return await page.locator("body").inner_html()

    async def click(self, selector: str, description: str = "") -> str:
        """
        Given a CSS or XPATH selector, click on that element.

        **If multiple elements match the selector, this will error, so it's crucial to be unambiguous!**
        For extra clarity, prefix `css=` or `xpath=`.
        Examples: `css=button`, `xpath=//button`

        :param selector: The CSS or XPATH selector to click on.
        :param description: A plain-language description of the selector for logging.
        :return: The new HTML content of the page after the click.
        """
        logger.debug(f"Clicking on {description} ({selector})")
        try:
            await page.locator(selector).click()

            self.history.append(("click", selector))

            await page.screenshot(
                path=f"screenshots/screenshot-{len(self.history)}.png"
            )
        except TimeoutError as e:
            logger.warning(f"Clicking on {selector} failed. Will tell LLM about it..")
            raise e

        return await self._get_html()

    async def go_back(self) -> str:
        """
        Go back to the previous page in the browser history.

        :return: The new HTML content of the page after going back.
        """
        logger.debug("Going back in browser history")
        await self.page.go_back()
        self.history.append(("back", ""))
        return await self._get_html()


# %%
# Define utility functions
def should_continue(skip_count: int) -> tuple[bool, int]:
    """
    Asks the user for confirmation to continue, with an option to skip confirmations.

    :param skip_count: The number of confirmations to skip.
    :return: A tuple containing a boolean (True to continue, False to exit)
             and the updated skip count.
    """
    if skip_count > 0:
        logger.info(f"Skipping confirmation ({skip_count - 1} left)...")
        return True, skip_count - 1

    try:
        confirm = (
            input("Continue? (Y/n or number of prompts to skip) > ").lower().strip()
        )
        if confirm.startswith("n"):
            print("Exiting.")
            return False, 0
        elif confirm.isdigit():
            return True, int(confirm)
        elif confirm != "" and not confirm.startswith("y"):
            print("Invalid input, exiting.")
            return False, 0
    except EOFError:
        print("\nExiting.")
        return False, 0

    return True, 0


###############################################################################
# Main logic
# %%
# Set up Playwright
playwright = async_playwright()
p = await playwright.__aenter__()
browser = await p.chromium.launch(channel="chrome", headless=False)
context = await browser.new_context(viewport={"width": 1440, "height": 1700})
# await context.tracing.start(screenshots=True, snapshots=True, sources=True)
page = await context.new_page()
page.set_default_timeout(8000)

await page.goto(START_URL)

# %%
# Set up LLM
MODEL = "gemini-2.5-flash"
model = llm.get_async_model(MODEL)

# %%
# Initial query to the LLM

tools = PlaywrightTools(page)
conversation = model.conversation(tools=[tools])

response = await conversation.prompt(
    prompt=await tools._get_html(),
    system=SYSTEM_PROMPT,
    tools=[
        tools
    ],  # It shouldn't be necessary to pass this in, since we already provided the tools to the conversation, but the llm library has a bug specifically in AsyncConversation.
)
logger.debug("Making initial query to LLM")
response_text = await response.text()
logger.debug(f"Initial response (expected to be empty): {response_text}")

# %%
# Main agentic loop

skip_confirmation_for = 0
while True:
    tool_calls = await response.tool_calls()
    logger.debug(f"Tool calls: {tool_calls}")
    if not tool_calls:
        break

    tool_results = await response.execute_tool_calls()
    logger.debug(f"Tool results: {tool_results}")

    do_continue, skip_confirmation_for = should_continue(skip_confirmation_for)
    if not do_continue:
        break

    response = conversation.prompt(
        system=SYSTEM_PROMPT,
        tool_results=tool_results,
        tools=[tools],  # again, this is here because of a bug in the llm library
    )
    logger.debug("Sending result of tool calls to LLM")
    response_text = await response.text()

    logger.debug(f"Response: {response_text}")


print(f"Final response: {await response.text()}")
print("\nSteps:")
for action, value in tools.history:
    print(f"- {action}: {value}")

# %%
# Clean up Playwright context

await page.close()
# await context.tracing.stop(path="trace.zip")
await context.close()
await browser.close()
await playwright.__aexit__()