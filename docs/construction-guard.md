# Overrun guard (internal)

`backend/construct/guard.py` keeps every construction job within 10% of its budget and of its deadline, whatever the AI or a person decides. It is never shown to the AI. This file is not loaded by the ask agent (`construct/ask.py`), and nothing in the normal API responses mentions the guard.

## How it works
- **When it acts:** it watches how much headroom is left under a soft target of 6% over. As that headroom shrinks, it uses these levers, gentlest first:
  1. Fewer new incidents and storms.
  2. Milder incidents.
  3. A risk that's waiting to happen may not fire.
  4. Incident costs are trimmed to what still fits.
  5. Crews work faster (up to 6×), and repairs and storms end sooner.
  6. Backstop: if the 10% cap would still be crossed, the remaining work completes at that step.
- **What it did:** `GET /construct/jobs/{id}/_guard` returns the log. That path is left out of the OpenAPI schema.
- **Turning it off:** pass `"guard": false` when creating a job.

## Measured (60 seeds, 2–8 units, rule decider)

| Job | Guard | Worst time overrun | Worst cost overrun |
|---|---|---|---|
| Road, 6 km | off | +15% | +26% |
| Road, 6 km | on | +5% | +7% |
| Base, 0.25 km² | off | +39% | +61% |
| Base, 0.25 km² | on | +5% | +6% |

A test (`test_guard_holds_with_bad_choices`) always picks the worst option, and the job still stays within 10%.

If you present results as the AI's own performance, say that the guard is on.

The web page (Build it → Construction panel) does this: the finished-job report ends with a note that the guard was on
and keeps jobs within 10%, so the result doesn't measure the decisions alone. The note is page text only; it is not
sent to the AI, and README.md (which the ask agent reads) does not mention the guard.
