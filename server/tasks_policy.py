# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Long-policy task bank for Prompt Golf.

These tasks exist to stress-test prompt compression on a real-world
pattern: an organization has a multi-page policy (ad standards, content
moderation, financial disclosures) and needs an LLM-based classifier
that decides whether a piece of content complies. The naive prompt is
the *entire policy verbatim* — easily 1000-2000 tokens. The trained
prompt golf agent must compress that into a tight classifier prompt
that preserves the load-bearing rules.

Why these tasks are valuable:
  - Long policy text = aggressive compression target (10×+ ratio).
  - Multi-class decisions test that the agent identifies the
    *hierarchy* of rules, not just keywords.
  - Real businesses pay for policy compression today — every byte of
    system prompt costs $$ at scale.

Each task ships with:
  - description: the full multi-clause policy (1000+ tokens)
  - train_examples: 3 (content_description, decision) pairs
  - test_examples: 6 hidden (content_description, decision) pairs
  - scorer: exact_label (decision is from a closed vocabulary)
  - budget_tokens: 250 (aggressive — verbose policy is ~1200 tokens)
"""

from __future__ import annotations

try:
    from .tasks import TaskSpec
except ImportError:
    from server.tasks import TaskSpec


TASKS_POLICY: dict[str, TaskSpec] = {}


def _add(task: TaskSpec) -> None:
    TASKS_POLICY[task.task_id] = task


def list_task_ids_policy() -> list[str]:
    return list(TASKS_POLICY.keys())


# ============================================================================
# 1. Ad creative policy compliance (MSN/Bing-style)
# ============================================================================

_MSN_AD_POLICY = """\
MSN AD CREATIVE POLICY (effective 2025)

This document describes the standards every ad creative must meet \
before serving on the MSN advertising network. Each submitted creative \
must be classified into exactly one of three buckets: ALLOW (compliant, \
serve as-is), DISALLOW (violates a hard prohibition, do not serve), or \
REVIEW (potentially compliant but requires manual editorial review or \
the addition of mandatory disclosures before serving).

SECTION A — HARD PROHIBITIONS (always DISALLOW)

A.1 Illegal goods or services. Any creative promoting goods or \
services that are illegal under United States federal law or in the \
country where the creative will serve is prohibited. This includes \
unlicensed pharmaceuticals, controlled substances, illegal weapons, \
counterfeit goods, and pirated content.

A.2 Tobacco, e-cigarettes, vaping products. Promotion of any tobacco \
or nicotine-delivery product is prohibited in the United States, \
European Union, Australia, and Singapore. Cessation aids prescribed by \
licensed clinicians are exempt.

A.3 Adult content. Sexually explicit material, escort services, and \
adult entertainment are prohibited.

A.4 Misleading medical claims. Any creative that claims to cure, \
prevent, or guarantee treatment outcomes for a serious medical \
condition (cancer, diabetes, HIV/AIDS) is prohibited unless backed by \
FDA approval cited in the creative.

A.5 Cryptocurrency speculation. Promotion of unregistered initial coin \
offerings, leveraged crypto trading without risk disclosure, or any \
'guaranteed returns' framing in the digital-asset category.

SECTION B — RESTRICTED CONTENT (REVIEW unless conditions met)

B.1 Alcohol. Permitted only when (a) creative includes 'Drink \
Responsibly' or jurisdiction-equivalent disclaimer, (b) campaign is \
restricted to ages 21+ in the US (18+ in countries with lower legal \
drinking age), and (c) target country permits alcohol advertising. If \
all three conditions are explicitly stated, ALLOW; otherwise REVIEW.

B.2 Gambling and games of chance. Permitted only with (a) state or \
country-level operator license number visible in the creative, and (b) \
problem-gambling helpline displayed. Both required for ALLOW.

B.3 Financial services and investment products. Stock trading, \
brokerage, and investment-management ads require an explicit risk \
disclosure clause ('investments may lose value' or equivalent). Loans \
and credit cards require APR ranges. If disclosure is present, ALLOW; \
otherwise REVIEW for editorial sign-off.

B.4 Political and issue advertising. Any creative referencing a \
political candidate, ballot measure, or contested social issue must \
have advertiser identity verified through MSN's political verification \
program. Without verification: REVIEW.

B.5 Health and wellness (non-medical). Supplements, fitness programs, \
and weight-loss products are permitted only when claims are factual and \
do not promise specific outcomes ('lose 20 lbs guaranteed' is \
prohibited; 'supports healthy weight management' is permitted).

SECTION C — FORMAT AND PRESENTATION STANDARDS

C.1 Creative must clearly identify the advertiser. Anonymous or \
deceptively branded creatives are REVIEW.

C.2 No fake-system-UI elements. Creatives that imitate browser \
warnings, OS notifications, or user-interface chrome to induce clicks \
are DISALLOW.

C.3 No autoplay audio with sound on by default.

C.4 All claims of superiority ('#1', 'best in class') must be \
independently verifiable; otherwise REVIEW.

DECISION FORMAT
For each input creative description, output exactly one label, \
lowercase, no punctuation, no explanation: 'allow', 'disallow', or \
'review'. The label must be on the first line of the output and \
nothing else.\
"""

_add(TaskSpec(
    task_id="policy_msn_ad_creative",
    category="policy_compression",
    description=_MSN_AD_POLICY,
    scorer="exact_label",
    train_examples=[
        ("Banner ad for Heineken beer, copy includes 'Drink Responsibly', "
         "campaign restricted to ages 21+, serving in the United States.",
         "allow"),
        ("Banner ad promoting an unregistered ICO promising 'guaranteed "
         "300% returns in 90 days'.",
         "disallow"),
        ("Banner ad for Robinhood stock trading app, copy mentions "
         "'commission-free trading' but no risk disclosure visible.",
         "review"),
    ],
    test_examples=[
        ("Banner ad for a vape pen brand, campaign serving in the US.",
         "disallow"),
        ("Native ad for an online casino, includes operator license "
         "number 'NV-2189' and 1-800-GAMBLER helpline.",
         "allow"),
        ("Display ad for Nike running shoes, no special claims.",
         "allow"),
        ("Sponsored post for a weight-loss supplement, copy says 'lose 20 "
         "pounds in 30 days, guaranteed!'.",
         "disallow"),
        ("Banner ad endorsing Senator Park's re-election campaign, no "
         "advertiser-verification badge present.",
         "review"),
        ("Display ad for Wells Fargo personal loans, copy mentions APR "
         "range '6.99%-19.99%' clearly.",
         "allow"),
    ],
    budget_tokens=250,
    difficulty="hard",
    tags=["policy", "compression", "ad-tech", "long-context"],
))


# ============================================================================
# 2. Content moderation policy (social-platform style)
# ============================================================================

_CONTENT_MOD_POLICY = """\
COMMUNITY CONTENT POLICY (v3.2)

Every user-submitted post on this platform is reviewed against the \
following rules. Each post is classified into exactly one of four \
buckets: KEEP (post complies, do not act), WARN (post is borderline; \
attach a sensitive-content interstitial but leave it visible), REMOVE \
(post violates a hard rule; delete it and notify the author), or \
ESCALATE (post may involve real-world harm or legal risk; route to a \
human reviewer immediately).

PART 1 — HARD REMOVAL CRITERIA (always REMOVE)

1.1 Doxxing. Posts that publish another user's private contact \
information (home address, personal phone, employer in a hostile \
context) without that user's consent.

1.2 CSAM and child exploitation. Any sexualized content involving \
minors, or content that grooms, recruits, or trafficks minors. These \
posts also trigger ESCALATE simultaneously.

1.3 Direct credible threats of violence. Posts that name a specific \
person, place, or scheduled event AND describe an intended violent act.

1.4 Coordinated harassment. Posts directing multiple users to attack a \
named target (pile-on, brigade, raid).

1.5 Illegal goods marketplace. Selling, buying, or arranging the \
exchange of regulated firearms without a license, controlled substances \
without a prescription, or services rendered for criminal purposes.

PART 2 — ESCALATE TO HUMAN REVIEWER (always ESCALATE)

2.1 First-person suicidal ideation or imminent self-harm. Route to \
crisis-resource workflow within 15 minutes.

2.2 Reports of ongoing real-world emergencies (active shooter, \
kidnapping in progress, hostage situation) — route to law enforcement \
liaison.

2.3 Coordinated influence operations or election-interference patterns \
flagged by the platform-integrity model.

2.4 Legal compliance gray zones (DMCA, GDPR right-to-erasure, court \
preservation orders).

PART 3 — WARN (interstitial but keep)

3.1 Graphic but newsworthy content. Photojournalism from conflict \
zones, accident scenes, or medical procedures shown for educational \
purposes. Apply sensitive-content overlay; allow click-through.

3.2 Suicide / self-harm references in the third person, news, fiction, \
or recovery contexts (NOT first-person ideation — that's ESCALATE).

3.3 Extremely strong language directed at a public figure (politicians, \
celebrities) but stopping short of credible threat or doxxing.

3.4 Adult nudity in clearly artistic, educational, or medical contexts \
(NOT pornographic).

PART 4 — KEEP DEFAULTS

Posts that don't trigger Part 1, 2, or 3 are KEEP. This includes \
ordinary disagreement, strong political opinion, profanity not directed \
at a specific user, satire and parody, and historical or scientific \
discussion of sensitive topics.

OUTPUT FORMAT
For each post description, output exactly one label, lowercase, no \
punctuation, no explanation: 'keep', 'warn', 'remove', or 'escalate'. \
First line only.\
"""

_add(TaskSpec(
    task_id="policy_content_moderation",
    category="policy_compression",
    description=_CONTENT_MOD_POLICY,
    scorer="exact_label",
    train_examples=[
        ("Post that names a specific journalist by full name and posts "
         "their home address, calling for 'accountability'.",
         "remove"),
        ("News photograph from a conflict zone showing wounded civilians, "
         "captioned with factual reporting context.",
         "warn"),
        ("Post containing strong profanity directed at a politician's "
         "policy stance, no doxxing or threats.",
         "keep"),
    ],
    test_examples=[
        ("First-person post saying 'I have a plan and pills, tonight is "
         "the night' with no recovery framing.",
         "escalate"),
        ("Satirical post mocking a CEO's recent quarterly earnings call, "
         "uses sarcasm but no slurs or threats.",
         "keep"),
        ("Post offering to sell an unregistered AR-15 to whoever sends "
         "the highest bid via DM.",
         "remove"),
        ("Documentary-style post about historical lynchings in the 20th "
         "century US, includes one period photograph.",
         "warn"),
        ("Post coordinating dozens of accounts to flood a single user's "
         "mentions with insults over a public dispute.",
         "remove"),
        ("Post sharing a recipe for homemade lemon cake with no other "
         "context.",
         "keep"),
    ],
    budget_tokens=250,
    difficulty="hard",
    tags=["policy", "compression", "moderation", "long-context"],
))


# ============================================================================
# 3. Financial-disclosure compliance (broker/dealer style)
# ============================================================================

_FINREG_POLICY = """\
FINRA-STYLE COMMUNICATION REVIEW POLICY (excerpted, simplified for \
classifier training)

A registered representative or firm-affiliated person at a broker-dealer \
must have all written communications with retail customers reviewed \
before they are sent. Each communication is classified into one of four \
buckets: APPROVED (compliant, send), HOLD (return to author for \
correction of a fixable issue), REJECT (substantive violation, do not \
send under any revision), or ESCALATE (requires compliance officer \
sign-off because of complexity, novel product, or potential customer \
harm).

PART I — REJECT CATEGORIES (always REJECT)

1.1 Performance guarantees. Any statement that the customer's \
investment 'cannot lose' or will 'guarantee' a specific return over a \
specific time horizon. Phrases like 'risk-free' applied to anything \
other than US Treasuries are REJECT.

1.2 Selective performance highlighting (cherry-picking). Citing only \
the best historical years of a fund without stating the comparable \
benchmark return AND the worst year in the same period.

1.3 Promissory language about future appreciation. 'Will' double, \
'must' rise, 'has to' recover.

1.4 Unsuitable product recommendations to vulnerable customers. \
Recommending leveraged ETFs, options strategies, or crypto futures to \
customers under 25 or over 75, or with stated 'capital preservation' \
objectives, in a public-facing communication.

PART II — HOLD (revisable)

2.1 Missing standard disclosures. Mutual fund pitches missing the \
'past performance does not guarantee future results' line, or annuity \
pitches missing the surrender-charge schedule reference.

2.2 Missing FINRA Rule 2210 footer (firm name, registration status, \
and contact info).

2.3 Fee disclosures present but not 'clear and prominent' (font size \
or contrast inadequate).

PART III — ESCALATE (compliance officer required)

3.1 New-issue securities communications.

3.2 Communications mentioning private placements, structured products, \
or non-traded REITs.

3.3 References to options strategies more complex than basic covered \
calls.

3.4 Communications targeting accounts opened in the past 90 days \
(new-customer suitability review).

PART IV — APPROVED DEFAULTS

Communications that include all required disclosures, do not promise \
returns, do not cherry-pick performance data, and concern only \
plain-vanilla products (broad-market index funds, money market funds, \
US Treasuries) for customers in the firm's standard suitability bands. \
APPROVED.

OUTPUT FORMAT
Output exactly one lowercase label, no punctuation, no explanation: \
'approved', 'hold', 'reject', or 'escalate'. First line only.\
"""

_add(TaskSpec(
    task_id="policy_finreg_communication_review",
    category="policy_compression",
    description=_FINREG_POLICY,
    scorer="exact_label",
    train_examples=[
        ("Email pitch promising clients that the firm's growth fund "
         "'will deliver double-digit returns over the next decade'.",
         "reject"),
        ("Brochure for an S&P 500 index fund, includes the standard "
         "'past performance does not guarantee future results' "
         "disclosure and FINRA footer.",
         "approved"),
        ("Newsletter about a non-traded REIT being added to the firm's "
         "platform, includes no obvious violations but is a new "
         "structured-product offering.",
         "escalate"),
    ],
    test_examples=[
        ("Mailer for a money-market fund, omits the 'past performance' "
         "line but otherwise compliant; FINRA footer present.",
         "hold"),
        ("Email to a 78-year-old retiree with stated 'capital "
         "preservation' goal, recommending a 3x leveraged tech ETF.",
         "reject"),
        ("Email pitching a new private placement to a sophisticated "
         "investor, with all standard risk disclosures present.",
         "escalate"),
        ("Quarterly note about US Treasury yields, no recommendations, "
         "all firm disclosures present.",
         "approved"),
        ("Brochure highlighting only the fund's three best years out of "
         "ten, omitting the worst year and benchmark return.",
         "reject"),
        ("Email pitching a covered-call strategy to a customer who "
         "opened the account 30 days ago.",
         "escalate"),
    ],
    budget_tokens=250,
    difficulty="hard",
    tags=["policy", "compression", "finance", "long-context"],
))


if __name__ == "__main__":
    from collections import Counter
    print(f"tasks_policy: {len(TASKS_POLICY)} tasks")
    for tid, spec in TASKS_POLICY.items():
        desc_words = len(spec.description.split())
        print(f"  {tid:42s} category={spec.category:22s} "
              f"budget={spec.budget_tokens} desc_words≈{desc_words}")
