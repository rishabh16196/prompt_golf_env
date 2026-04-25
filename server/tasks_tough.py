# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tough-scenarios task bank for Prompt Golf (v3).

Goal: scenarios where the *original* (verbose, hand-written) prompt that
naturally steers the target is 150-300 tokens long, but the MINIMUM
effective prompt is much shorter and non-obvious. The agent's job is to
find that compressed prompt — i.e. learn which fragments of the verbose
specification are load-bearing for the target model.

This file is the seed batch (10 scenarios — domain classifiers). The
remaining 42 will be added in later commits across:
  - Structured extraction
  - Format-strict generation
  - Persona + constraint
  - Multi-step reasoning
  - Adversarial / calibration

Why classifiers first: they exercise the existing `exact_label` scorer
deterministically, so we can validate the whole base→trained CSV
pipeline before investing in the fuzzier tasks.

Each scenario follows the existing TaskSpec contract from server/tasks.py
so it merges into _ALL_TASKS without code changes elsewhere.
"""

from __future__ import annotations

try:
    from .tasks import TaskSpec
except ImportError:
    from server.tasks import TaskSpec


TASKS_TOUGH: dict[str, TaskSpec] = {}


def _add(task: TaskSpec) -> None:
    TASKS_TOUGH[task.task_id] = task


def list_task_ids_tough() -> list[str]:
    return list(TASKS_TOUGH.keys())


# ============================================================================
# Domain classifiers (10)
#
# All use scorer="exact_label". Expected output is exactly one token from a
# closed vocabulary (lowercase, hyphenated, no punctuation, no explanation).
# ============================================================================

_add(TaskSpec(
    task_id="tough_fallacy_classify",
    category="classification_tough",
    description=(
        "Read the short argument and identify the dominant logical fallacy "
        "it commits. The target must output exactly one label from this "
        "closed vocabulary, in lowercase with hyphens, with no punctuation "
        "and no explanation:\n"
        "  - ad-hominem (attacking the person, not the argument)\n"
        "  - straw-man (misrepresenting an opponent's position to refute it)\n"
        "  - false-dilemma (presenting only two options when more exist)\n"
        "  - slippery-slope (claiming one event inevitably leads to extreme "
        "consequences without evidence)\n"
        "  - appeal-to-authority (citing an irrelevant or unqualified "
        "authority as proof)\n"
        "  - circular-reasoning (the conclusion is assumed in the premises)\n"
        "  - hasty-generalization (drawing a broad conclusion from a small "
        "or biased sample)\n"
        "  - red-herring (introducing an irrelevant topic to distract)\n"
        "If multiple fallacies are present, choose the one most central to "
        "the argument's structure. Output ONLY the label."
    ),
    scorer="exact_label",
    train_examples=[
        ("You can't trust Maria's economic analysis — she failed math in "
         "high school.", "ad-hominem"),
        ("Either we ban all cars or we accept that cities will be unlivable "
         "forever.", "false-dilemma"),
        ("My grandfather smoked his whole life and lived to 95, so smoking "
         "isn't really dangerous.", "hasty-generalization"),
    ],
    test_examples=[
        ("If we let students redo one exam, soon they'll demand to redo "
         "every assignment and graduation will be meaningless.", "slippery-slope"),
        ("Senator Park says climate policy is hurting jobs. He's been "
         "divorced twice — why would anyone listen to him?", "ad-hominem"),
        ("Of course the new drug works. It works because it's effective at "
         "treating the condition.", "circular-reasoning"),
        ("My opponent wants modest gun-safety reform. So she wants to "
         "confiscate every firearm in America.", "straw-man"),
        ("A famous actor endorses this supplement, so it must be "
         "medically sound.", "appeal-to-authority"),
        ("You ask about the budget overruns? Let's talk about how much "
         "the previous administration wasted.", "red-herring"),
    ],
    budget_tokens=120,
    difficulty="hard",
    tags=["classification", "tough", "reasoning"],
))


_add(TaskSpec(
    task_id="tough_bias_detect",
    category="classification_tough",
    description=(
        "Identify the cognitive bias most clearly demonstrated by the "
        "scenario. Output exactly one label from this closed vocabulary "
        "(lowercase, hyphenated, no punctuation, no explanation):\n"
        "  - confirmation (seeking/weighing only evidence that supports a "
        "prior belief)\n"
        "  - anchoring (over-relying on the first number or fact "
        "encountered)\n"
        "  - availability (judging probability by how easily examples come "
        "to mind)\n"
        "  - sunk-cost (continuing because of past investment rather than "
        "future value)\n"
        "  - survivorship (drawing conclusions from successful cases while "
        "ignoring failed ones)\n"
        "  - dunning-kruger (low-skill overconfidence; high-skill "
        "under-confidence)\n"
        "  - hindsight (believing past events were predictable after the "
        "fact)\n"
        "  - recency (overweighting the most recent data point)\n"
        "Output ONLY the label."
    ),
    scorer="exact_label",
    train_examples=[
        ("After watching three plane-crash documentaries, Priya is now "
         "afraid to fly even though she drives daily.", "availability"),
        ("The first house Raj saw was listed at $800k. Every other house "
         "now feels overpriced or like a steal compared to that number.",
         "anchoring"),
        ("Studied successful CEOs all dropped out of college, so dropping "
         "out is the path to success.", "survivorship"),
    ],
    test_examples=[
        ("I've already spent two years on this PhD topic — even though I "
         "don't believe in it anymore, I have to finish.", "sunk-cost"),
        ("After his stock dropped 8% yesterday, Arun is sure the whole "
         "market is collapsing despite a steady year.", "recency"),
        ("She only reads news outlets that agree with her political views "
         "and dismisses the rest as biased.", "confirmation"),
        ("After the company went bankrupt, every analyst said the warning "
         "signs were obvious all along.", "hindsight"),
        ("A first-year coder confidently tells the senior team their "
         "architecture is wrong; she's never shipped to production.",
         "dunning-kruger"),
        ("He only studies founders of unicorn startups to figure out how "
         "to build a unicorn, ignoring the thousands that failed.",
         "survivorship"),
    ],
    budget_tokens=120,
    difficulty="hard",
    tags=["classification", "tough", "psychology"],
))


_add(TaskSpec(
    task_id="tough_rhetorical_device",
    category="classification_tough",
    description=(
        "Identify the dominant rhetorical device used in the sentence. "
        "Output exactly one label from this closed vocabulary (lowercase, "
        "no punctuation, no explanation):\n"
        "  - anaphora (repeating the same word/phrase at the start of "
        "successive clauses)\n"
        "  - metaphor (implicit comparison, no 'like' or 'as')\n"
        "  - hyperbole (deliberate, obvious exaggeration)\n"
        "  - irony (saying the opposite of what is meant)\n"
        "  - alliteration (repeated initial consonant sounds)\n"
        "  - antithesis (juxtaposed contrasting ideas in parallel "
        "structure)\n"
        "  - rhetorical-question (a question asked for effect, not an "
        "answer)\n"
        "  - chiasmus (reversed grammatical structure: A-B-B-A)\n"
        "Pick the device most central to the rhetorical effect. Output "
        "ONLY the label."
    ),
    scorer="exact_label",
    train_examples=[
        ("We shall fight on the beaches. We shall fight on the landing "
         "grounds. We shall fight in the fields.", "anaphora"),
        ("Time is a thief that steals our youth.", "metaphor"),
        ("Ask not what your country can do for you — ask what you can do "
         "for your country.", "chiasmus"),
    ],
    test_examples=[
        ("I've told you a million times to clean your room.", "hyperbole"),
        ("Peter Piper picked a peck of pickled peppers.", "alliteration"),
        ("It is the best of times, it is the worst of times.", "antithesis"),
        ("Oh great, another Monday — exactly what I was hoping for.",
         "irony"),
        ("Do we really need another committee to study the obvious?",
         "rhetorical-question"),
        ("The classroom was a zoo during the substitute's lesson.",
         "metaphor"),
    ],
    budget_tokens=120,
    difficulty="hard",
    tags=["classification", "tough", "literature"],
))


_add(TaskSpec(
    task_id="tough_medical_urgency",
    category="classification_tough",
    description=(
        "Read the short patient complaint and triage it into one of four "
        "urgency tiers. Output exactly one label, lowercase, no "
        "punctuation, no explanation. This is a TRIAGE classifier — not "
        "medical advice — so be conservative on red-flag symptoms.\n"
        "  - emergency: chest pain, stroke signs, severe bleeding, loss of "
        "consciousness, anaphylaxis, suicidal intent — call ambulance now.\n"
        "  - urgent: significant injury, high fever with stiffness, severe "
        "pain, infection signs, pregnancy complications — same-day care.\n"
        "  - routine: persistent but stable symptoms, follow-ups, "
        "medication refills — schedule within 1-2 weeks.\n"
        "  - self-care: minor cuts, common cold, mild headache, fatigue "
        "without alarm features — rest, OTC, monitor.\n"
        "Default to the higher tier when symptoms are ambiguous. Output "
        "ONLY the label."
    ),
    scorer="exact_label",
    train_examples=[
        ("Crushing chest pressure radiating to my left arm, sweating, "
         "started 20 minutes ago.", "emergency"),
        ("Sore throat and runny nose for two days, no fever, eating "
         "normally.", "self-care"),
        ("Rash on forearm that's been spreading slowly for three weeks, "
         "no pain.", "routine"),
    ],
    test_examples=[
        ("Sudden numbness on one side of my face and slurred speech for "
         "the last 10 minutes.", "emergency"),
        ("Deep cut on my hand from a kitchen knife, bleeding has slowed "
         "but it might need stitches.", "urgent"),
        ("Ongoing knee stiffness in the mornings for the past month, "
         "manageable.", "routine"),
        ("Mild headache after a long day on screens, no other symptoms.",
         "self-care"),
        ("High fever 39.5C, stiff neck, and a new pinpoint rash that "
         "started this evening.", "emergency"),
        ("Persistent cough for four days, low-grade fever, achy but "
         "drinking fluids and resting.", "urgent"),
    ],
    budget_tokens=140,
    difficulty="hard",
    tags=["classification", "tough", "medical"],
))


_add(TaskSpec(
    task_id="tough_code_smell",
    category="classification_tough",
    description=(
        "Read the short code description and identify the dominant code "
        "smell. Output exactly one label from this closed vocabulary "
        "(lowercase, hyphenated, no punctuation, no explanation):\n"
        "  - long-method (a single function does too many things over too "
        "many lines)\n"
        "  - god-class (one class accumulates unrelated responsibilities)\n"
        "  - duplicate-code (the same logic appears in multiple places)\n"
        "  - dead-code (unused variables, branches, or functions)\n"
        "  - magic-number (unexplained literal constants in logic)\n"
        "  - primitive-obsession (using strings/ints where a small type "
        "would clarify intent)\n"
        "  - feature-envy (a method uses another class's data more than "
        "its own)\n"
        "  - shotgun-surgery (one logical change requires edits across "
        "many files)\n"
        "Output ONLY the label."
    ),
    scorer="exact_label",
    train_examples=[
        ("`processOrder()` is 600 lines long and handles validation, "
         "pricing, payment, shipping, email, and audit logging in one "
         "function.", "long-method"),
        ("`if total > 4500: applyDiscount(0.07)` — neither number is "
         "explained.", "magic-number"),
        ("Adding a new currency requires editing the database schema, "
         "three services, the UI, and two config files.",
         "shotgun-surgery"),
    ],
    test_examples=[
        ("`UserManager` handles authentication, profile editing, billing, "
         "email sending, audit logs, and CSV export.", "god-class"),
        ("The same 30-line block computing tax appears in CheckoutService, "
         "InvoiceService, and ReportService.", "duplicate-code"),
        ("`Order.calculateShipping()` reads 8 fields from `Customer` and "
         "uses only 1 from its own object.", "feature-envy"),
        ("There's a private helper `oldFormatLegacy()` that nothing in "
         "the repo references anymore.", "dead-code"),
        ("Phone numbers, emails, postal codes, and currency amounts are "
         "all stored as plain `str` everywhere.", "primitive-obsession"),
        ("A single function `handleRequest()` parses input, validates, "
         "queries DB, formats output, logs, and emails — 400 lines.",
         "long-method"),
    ],
    budget_tokens=140,
    difficulty="hard",
    tags=["classification", "tough", "software"],
))


_add(TaskSpec(
    task_id="tough_news_framing",
    category="classification_tough",
    description=(
        "Read the short news headline and identify its dominant framing "
        "technique. Output exactly one label from this closed vocabulary "
        "(lowercase, hyphenated, no punctuation, no explanation):\n"
        "  - episodic (focuses on a single event or individual case)\n"
        "  - thematic (focuses on broader trends, statistics, or "
        "context)\n"
        "  - conflict (frames the story as a clash between sides)\n"
        "  - human-interest (emotional angle on a person's experience)\n"
        "  - economic (frames consequences in financial / market terms)\n"
        "  - morality (frames the story in terms of right vs wrong, "
        "values)\n"
        "  - responsibility (assigns blame or credit to a specific "
        "actor)\n"
        "Pick the dominant frame even if minor frames are present. Output "
        "ONLY the label."
    ),
    scorer="exact_label",
    train_examples=[
        ("Single mother of three struggles to afford groceries as prices "
         "rise.", "human-interest"),
        ("National food-insecurity rate hits 12-year high, USDA report "
         "shows.", "thematic"),
        ("Senate Democrats and Republicans clash over food-stamp "
         "spending bill.", "conflict"),
    ],
    test_examples=[
        ("Local bakery owner closes shop after 30 years, blames soaring "
         "rent.", "episodic"),
        ("Inflation eats into household budgets as wages stagnate.",
         "economic"),
        ("Mayor accused of approving the contract that caused the water "
         "crisis.", "responsibility"),
        ("Is it ever right to lie to protect a friend? Readers weigh in.",
         "morality"),
        ("Climate-policy fight escalates as governors trade public "
         "letters.", "conflict"),
        ("Childhood obesity rates nationwide climbed 4% over the last "
         "decade.", "thematic"),
    ],
    budget_tokens=130,
    difficulty="hard",
    tags=["classification", "tough", "media"],
))


_add(TaskSpec(
    task_id="tough_claim_verifiability",
    category="classification_tough",
    description=(
        "Classify the claim by what kind of statement it is. Output "
        "exactly one label, lowercase, hyphenated, no punctuation, no "
        "explanation:\n"
        "  - verifiable (in principle checkable against publicly "
        "available facts or measurements)\n"
        "  - unverifiable (about private mental states, future events, "
        "or otherwise not externally checkable)\n"
        "  - value-judgment (expresses a preference, taste, or moral "
        "evaluation rather than a fact)\n"
        "  - tautology (true by definition; carries no empirical "
        "content)\n"
        "Distinguish carefully: an unverifiable empirical claim is NOT "
        "the same as a value-judgment. Output ONLY the label."
    ),
    scorer="exact_label",
    train_examples=[
        ("The Eiffel Tower is 330 meters tall.", "verifiable"),
        ("Chocolate ice cream is the best dessert ever invented.",
         "value-judgment"),
        ("All bachelors are unmarried.", "tautology"),
    ],
    test_examples=[
        ("The president secretly regrets signing the trade deal last "
         "year.", "unverifiable"),
        ("Mount Everest is taller than Mount Kilimanjaro.", "verifiable"),
        ("A triangle has three sides.", "tautology"),
        ("Modern art is shallow and pretentious.", "value-judgment"),
        ("Earth's average surface temperature has risen since 1900.",
         "verifiable"),
        ("Pluto will be reclassified as a planet again before 2050.",
         "unverifiable"),
    ],
    budget_tokens=120,
    difficulty="hard",
    tags=["classification", "tough", "epistemology"],
))


_add(TaskSpec(
    task_id="tough_argument_strength",
    category="classification_tough",
    description=(
        "Evaluate the short argument and classify its logical status. "
        "Output exactly one label, lowercase, hyphenated, no punctuation, "
        "no explanation:\n"
        "  - sound (valid form AND all premises are true / plausibly "
        "true)\n"
        "  - valid-but-unsound (the conclusion follows IF the premises "
        "are true, but at least one premise is false)\n"
        "  - invalid (the conclusion does NOT follow from the premises "
        "even if they were true)\n"
        "  - fallacious (commits a recognized informal fallacy that "
        "undermines the inference)\n"
        "Apply this order of priority: if the argument commits a clear "
        "informal fallacy, label it `fallacious` over `invalid`. Output "
        "ONLY the label."
    ),
    scorer="exact_label",
    train_examples=[
        ("All humans are mortal. Socrates is human. Therefore Socrates is "
         "mortal.", "sound"),
        ("All birds can fly. Penguins are birds. Therefore penguins can "
         "fly.", "valid-but-unsound"),
        ("Some dogs are brown. My cat is brown. Therefore my cat is a "
         "dog.", "invalid"),
    ],
    test_examples=[
        ("The new policy must be wrong because the senator proposing it "
         "had an affair last year.", "fallacious"),
        ("All squares have four sides. This shape is a square. Therefore "
         "it has four sides.", "sound"),
        ("If it rains, the streets get wet. The streets are wet. "
         "Therefore it rained.", "invalid"),
        ("Every prime number is odd. Seven is prime. Therefore seven is "
         "odd.", "valid-but-unsound"),
        ("Either you support our tax bill or you hate working families.",
         "fallacious"),
        ("All mammals are warm-blooded. Whales are mammals. Therefore "
         "whales are warm-blooded.", "sound"),
    ],
    budget_tokens=140,
    difficulty="hard",
    tags=["classification", "tough", "logic"],
))


_add(TaskSpec(
    task_id="tough_emotion_primary",
    category="classification_tough",
    description=(
        "Identify the dominant primary emotion expressed by the speaker, "
        "using Plutchik's eight basic emotions. Output exactly one label, "
        "lowercase, no punctuation, no explanation:\n"
        "  - joy (happiness, delight, contentment)\n"
        "  - trust (acceptance, confidence in someone/something)\n"
        "  - fear (apprehension, worry about a threat)\n"
        "  - surprise (unexpectedness, being caught off guard)\n"
        "  - sadness (sorrow, loss, dejection)\n"
        "  - disgust (revulsion, moral or physical aversion)\n"
        "  - anger (frustration, hostility, indignation)\n"
        "  - anticipation (expectation, looking forward)\n"
        "Pick the SINGLE strongest emotion even if blends are present. "
        "Output ONLY the label."
    ),
    scorer="exact_label",
    train_examples=[
        ("I can't believe she actually showed up — I had no idea she was "
         "in town!", "surprise"),
        ("My team has my back; I know they'll deliver no matter what.",
         "trust"),
        ("Everything I worked for these last five years is just gone.",
         "sadness"),
    ],
    test_examples=[
        ("I'm counting down the days until the trip — only two weeks "
         "left!", "anticipation"),
        ("How DARE they reroute my flight without a single email?",
         "anger"),
        ("Reading those emails made my skin crawl. I had to stop "
         "halfway.", "disgust"),
        ("What if the test results come back bad? I haven't slept in "
         "days.", "fear"),
        ("Got the offer, the salary, AND the team I wanted — best week "
         "ever.", "joy"),
        ("I keep replaying the call. She just isn't coming back.",
         "sadness"),
    ],
    budget_tokens=130,
    difficulty="hard",
    tags=["classification", "tough", "psychology"],
))


_add(TaskSpec(
    task_id="tough_policy_stance",
    category="classification_tough",
    description=(
        "Classify the speaker's stance on the policy proposal mentioned "
        "in the quote. Output exactly one label, lowercase, hyphenated, "
        "no punctuation, no explanation:\n"
        "  - support (clearly endorses the proposal)\n"
        "  - oppose (clearly rejects the proposal)\n"
        "  - neutral (declines to take a side, observes both views, or "
        "stays purely descriptive)\n"
        "  - conditional-support (would support IF certain conditions "
        "were met)\n"
        "  - conditional-oppose (would oppose UNLESS certain conditions "
        "were met)\n"
        "Distinguish carefully: a hedged endorsement that names "
        "preconditions is conditional-support, not neutral. A statement "
        "of mixed views without a stance is neutral. Output ONLY the "
        "label."
    ),
    scorer="exact_label",
    train_examples=[
        ("I'm fully behind the rent-cap proposal — it'll protect "
         "vulnerable tenants.", "support"),
        ("The mining permit is a disaster for the watershed and I will "
         "vote no.", "oppose"),
        ("Some economists like the tariff plan, others don't — the "
         "evidence is genuinely mixed.", "neutral"),
    ],
    test_examples=[
        ("I'd back the carbon-tax bill, but only if the revenue is "
         "rebated to households.", "conditional-support"),
        ("I cannot support the surveillance program unless judicial "
         "review is built in from day one.", "conditional-oppose"),
        ("The infrastructure package is exactly what this district has "
         "needed for a decade.", "support"),
        ("I won't comment on the merits of the bill; that's for the "
         "committee to weigh.", "neutral"),
        ("This zoning change will gut the neighborhood — count me as a "
         "firm no.", "oppose"),
        ("I'll support the immigration reform if it includes a real "
         "pathway to citizenship.", "conditional-support"),
    ],
    budget_tokens=140,
    difficulty="hard",
    tags=["classification", "tough", "politics"],
))


# ============================================================================
# Module-level helpers
# ============================================================================

if __name__ == "__main__":
    print(f"tasks_tough: {len(TASKS_TOUGH)} scenarios")
    for tid, spec in TASKS_TOUGH.items():
        print(f"  {tid:36s} {spec.category:24s} budget={spec.budget_tokens}")
