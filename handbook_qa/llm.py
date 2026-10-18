"""The one place that talks to Claude."""

import os

import anthropic

from handbook_qa import config
from handbook_qa.index import Hit

# Claude replies with exactly this when the excerpts don't answer the question.
NOT_COVERED = "NOT_COVERED"

SYSTEM_PROMPT = f"""You answer employees' questions about their company handbook.

Use only the handbook excerpts in the user's message. Don't add outside knowledge, even when you know the answer.

Each excerpt contains [p. N] markers where a new page starts. Text after a marker is on that page until the next marker.

Cite the page after every claim, like this: [p. 412]. One page per bracket. Only cite pages that appear in the excerpts.

If the excerpts don't answer the question, reply with exactly {NOT_COVERED} and nothing else. If they answer part of it, answer that part and say what the handbook doesn't say.

Keep answers short and direct. Plain prose, no headings."""


class AnswerError(Exception):
    """Claude couldn't produce an answer. The message is safe to show to the user."""


class ClaudeUnavailable(AnswerError):
    pass


class RateLimited(AnswerError):
    def __init__(self, message: str, retry_after: float | None):
        super().__init__(message)
        self.retry_after = retry_after


class ClaudeConfigError(AnswerError):
    """Bad key, unknown model, or a request Claude rejects. Retrying won't help."""


def make_client() -> anthropic.Anthropic | None:
    """A Claude client, or None when no key is set (the app then runs in retrieval-only mode)."""
    if not os.environ.get("ANTHROPIC_API_KEY", "").strip():
        return None
    return anthropic.Anthropic()


def generate(client: anthropic.Anthropic, question: str, hits: list[Hit]) -> str:
    excerpts = "\n\n".join(
        f'<excerpt section="{" / ".join(h.chunk.sections)}">\n{h.chunk.text_with_page_markers()}\n</excerpt>'
        for h in hits
    )
    try:
        response = client.beta.messages.create(
            model=config.ANSWER_MODEL,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": f"{excerpts}\n\nQuestion: {question}"}],
            output_config={"effort": config.ANSWER_EFFORT},
            # If a safety classifier declines, the API retries on a fallback model in the same call.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
    except anthropic.RateLimitError as e:
        retry_after = e.response.headers.get("retry-after")
        raise RateLimited(
            "Claude is rate limiting requests. Wait a moment and try again.",
            float(retry_after) if retry_after else None,
        ) from e
    except anthropic.AuthenticationError as e:
        raise ClaudeConfigError("Claude rejected the API key. Check ANTHROPIC_API_KEY.") from e
    except anthropic.PermissionDeniedError as e:
        raise ClaudeConfigError("This API key isn't allowed to use Claude.") from e
    except anthropic.NotFoundError as e:
        raise ClaudeConfigError(
            f"Model {config.ANSWER_MODEL!r} wasn't found. Check HANDBOOK_QA_MODEL."
        ) from e
    except anthropic.BadRequestError as e:
        raise ClaudeConfigError(f"Claude rejected the request: {e.message}") from e
    except anthropic.APIStatusError as e:
        raise ClaudeUnavailable(f"Claude returned an error ({e.status_code}). Try again shortly.") from e
    except anthropic.APIConnectionError as e:
        raise ClaudeUnavailable("Couldn't reach Claude. Check your internet connection.") from e

    if response.stop_reason == "refusal":
        raise AnswerError("Claude declined to answer this question.")
    return "".join(block.text for block in response.content if block.type == "text").strip()
