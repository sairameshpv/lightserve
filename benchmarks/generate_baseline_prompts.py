"""Generates benchmarks/baseline_prompts.jsonl from templates.

Regenerate after editing this file:
    python3 benchmarks/generate_baseline_prompts.py
"""
import json
import random
from pathlib import Path

random.seed(42)

OUT_PATH = Path(__file__).parent / "baseline_prompts.jsonl"

COUNTRIES = [
    "France", "Japan", "Brazil", "Egypt", "Canada", "Australia", "Germany", "India",
    "Mexico", "Kenya", "Norway", "Thailand", "Peru", "Poland", "Vietnam", "Chile",
    "Portugal", "Greece", "Morocco", "Finland", "Argentina", "Nigeria", "Sweden",
    "Turkey", "Indonesia", "Colombia", "Iceland", "Ireland", "Switzerland", "Austria",
    "Denmark", "Ukraine", "Romania", "Hungary", "Croatia", "Serbia", "Bulgaria",
    "Slovakia", "Slovenia", "Estonia", "Latvia", "Lithuania", "Belgium", "Netherlands",
    "Spain", "Italy", "South Korea", "Malaysia", "Philippines", "Bangladesh",
    "Pakistan", "Sri Lanka", "Nepal", "Ethiopia", "Ghana", "Tanzania", "Uganda",
    "Zambia", "Zimbabwe", "Botswana", "Namibia", "Senegal", "Tunisia", "Algeria",
    "Jordan", "Lebanon", "Israel", "Qatar", "Kuwait", "Oman", "Yemen", "Iraq",
    "Iran", "Afghanistan", "Kazakhstan", "Uzbekistan", "Mongolia", "Cambodia",
    "Laos", "Myanmar", "New Zealand", "Fiji", "Cuba", "Jamaica", "Panama",
    "Costa Rica", "Ecuador", "Bolivia", "Paraguay", "Uruguay", "Venezuela",
    "Guatemala", "Honduras", "Nicaragua", "Dominican Republic", "Haiti",
    "Trinidad and Tobago", "Malta", "Cyprus", "Luxembourg", "Monaco",
]
assert len(COUNTRIES) == len(set(COUNTRIES)), "duplicate country in list"

MATH_OPS = [("+", lambda a, b: a + b), ("-", lambda a, b: a - b), ("x", lambda a, b: a * b)]

TOPICS = [
    "photosynthesis", "blockchain", "quantum computing", "the water cycle",
    "supply and demand", "neural networks", "plate tectonics", "compound interest",
    "the immune system", "machine learning overfitting", "renewable energy",
    "the French Revolution", "DNA replication", "inflation", "climate change",
    "the stock market", "black holes", "the Roman Empire", "vaccines",
    "cybersecurity", "the internet's history", "artificial intelligence ethics",
    "genetic engineering", "the periodic table", "evolution by natural selection",
    "cloud computing", "distributed systems", "container orchestration",
    "load balancing", "database indexing", "microservices architecture",
    "the CAP theorem", "TCP/IP networking", "public key cryptography",
    "garbage collection", "concurrency and parallelism", "caching strategies",
    "the CPU instruction cycle", "operating system schedulers", "API design",
    "version control systems", "software testing", "agile methodology",
    "data structures", "sorting algorithms", "graph theory", "linear algebra",
    "probability theory", "statistics and hypothesis testing", "game theory",
]

CODE_TASKS = [
    ("Python", "a function that reverses a linked list"),
    ("Python", "a function that checks if a string is a palindrome"),
    ("Python", "a function that finds all prime numbers up to n using the Sieve of Eratosthenes"),
    ("Python", "a function that merges two sorted lists into one sorted list"),
    ("Python", "a function that flattens a nested list of arbitrary depth"),
    ("JavaScript", "a function that debounces another function"),
    ("JavaScript", "a function that deep-clones a nested object"),
    ("Go", "a function that implements a thread-safe counter"),
    ("Go", "a function that reads a large file line by line without loading it all into memory"),
    ("SQL", "a query that finds the second-highest salary in an employees table"),
    ("SQL", "a query that finds duplicate rows in a table based on two columns"),
    ("Rust", "a function that implements a basic LRU cache"),
    ("Python", "a class implementing a binary search tree with insert and search methods"),
    ("Python", "a decorator that retries a function up to N times on exception"),
    ("Python", "a generator function that yields Fibonacci numbers indefinitely"),
    ("Python", "a function that detects a cycle in a linked list"),
    ("Python", "a function that implements binary search on a sorted array"),
    ("Python", "a function that computes the edit distance between two strings"),
    ("JavaScript", "a function that implements a simple event emitter"),
    ("JavaScript", "a function that throttles another function"),
    ("Go", "a worker pool that processes jobs from a channel concurrently"),
    ("Go", "a function that implements exponential backoff for retries"),
    ("SQL", "a query that computes a running total ordered by date"),
    ("SQL", "a query that pivots rows into columns for a monthly report"),
    ("Rust", "a function that implements a basic bounded queue using a mutex"),
    ("Python", "a class implementing a min-heap from scratch"),
    ("Python", "a function that validates balanced parentheses in a string"),
    ("Java", "a class implementing a thread-safe singleton"),
    ("Java", "a method that performs a deep merge of two nested maps"),
    ("TypeScript", "a generic function that memoizes another function's results"),
]
assert len(CODE_TASKS) == len(set(CODE_TASKS)), "duplicate code task"

CODE_FRAMINGS = [
    "Write {article} {lang} implementation of {task}. Include a brief explanation of the time complexity.",
    "Implement {task} in {lang}, optimized for readability. Add a short docstring/comment explaining the approach.",
    "Provide an idiomatic {lang} implementation of {task} with a brief complexity analysis.",
    "Write {task} in {lang} and include one simple test case demonstrating correct behavior.",
    "Implement {task} in {lang}. Call out any edge cases your implementation handles.",
]

REASONING_TEMPLATES = [
    "A train leaves station A at {a} mph and another leaves station B, {d} miles away, "
    "at the same time traveling at {b} mph toward station A. How long until they meet? "
    "Show your reasoning step by step.",
    "If {a} workers can build a wall in {d} days, how many days would it take {b} workers "
    "working at the same rate? Show your reasoning step by step.",
    "A shop marks up an item by {a}% and then offers a {b}% discount on the marked-up price. "
    "Is the final price higher or lower than the original, and by what percentage? Show your work.",
    "You have {a} liters of a solution that is {b}% acid. How many liters of pure water must "
    "you add to dilute it to half that concentration? Show your reasoning step by step.",
    "Two pipes can fill a tank in {a} hours and {b} hours respectively. How long would it take "
    "to fill the tank if both pipes are open at the same time? Show your reasoning.",
]

INSTRUCTION_TEMPLATES = [
    "Write a short professional email declining a meeting about {topic} due to a scheduling "
    "conflict, and propose two alternative times.",
    "Write a concise product description for a tool that helps teams understand {topic}.",
    "Rewrite the following sentence to be more concise: 'Due to the fact that we are currently "
    "experiencing a situation related to {topic}, we would like to inform you of the necessity "
    "to reconsider our approach.'",
    "Write a short LinkedIn post announcing a new internal workshop on {topic}.",
    "Draft three possible subject lines for an email newsletter about {topic}.",
    "Write a one-paragraph explanation of {topic} suitable for a non-technical audience.",
    "List five common misconceptions about {topic} and briefly correct each one.",
]

LONG_TEMPLATES = [
    "Summarize the following in 3 bullet points:\n\n"
    "{topic_cap} is a foundational concept that many teams misunderstand at first. "
    "In practice, the core idea behind {topic} can be broken down into a handful of "
    "principles that, once internalized, make the rest of the subject much easier to reason "
    "about. Engineers new to {topic} often struggle because the terminology overlaps with "
    "other fields, and the tradeoffs involved are rarely explained clearly in introductory "
    "material. A deeper understanding usually comes from working through concrete examples, "
    "observing failure modes firsthand, and comparing how different systems approach the same "
    "underlying problem differently, each making distinct tradeoffs between simplicity, "
    "performance, and correctness under real-world constraints and operational pressure.",
    "The following is a customer support transcript. Identify the customer's core issue and "
    "suggest a resolution:\n\n"
    "Customer: Hi, I've been trying to understand our billing related to {topic} for the past "
    "week but every time I check the dashboard the numbers don't match what I expected. I've "
    "tried refreshing, checking on a different device, and reading the documentation twice.\n"
    "Agent: I'm sorry to hear that. Can you tell me roughly when you first noticed the "
    "discrepancy?\n"
    "Customer: About five days ago, right after we made changes related to {topic}. I've "
    "already spent an hour on this and would appreciate a real fix rather than generic "
    "troubleshooting steps.",
    "Write a brief incident postmortem summary for an outage caused by a misconfiguration "
    "related to {topic}. Include: what happened, root cause, impact, and two follow-up action "
    "items to prevent recurrence.",
    "The following are raw meeting notes. Turn them into a clean summary with action items:\n\n"
    "Notes: discussed {topic} rollout timeline, blocked on review from infra team, someone "
    "mentioned cost concerns, need a decision by Friday, action items unclear, follow up with "
    "stakeholders, also revisit the doc once feedback comes in, unclear owner for next steps.",
    "Review the following product spec excerpt and list three concerns a senior engineer "
    "might raise before approving it:\n\n"
    "Spec: We propose adopting {topic} across all services within one sprint. The migration "
    "will be handled by each team independently with no shared tooling, rollback plan, or "
    "staging environment validation, and we expect zero downtime during the transition.",
]

PREFIX_PERSONAS = [
    "You are a helpful assistant for a software engineering team. Answer concisely and technically.",
    "You are a patient tutor explaining concepts to a first-year computer science student.",
    "You are a terse senior engineer doing a quick code review. Be blunt and specific.",
    "You are a technical writer drafting clear documentation for external developers.",
]

PREFIX_QUESTIONS = [
    "What is the difference between a mutex and a semaphore?",
    "Explain the CAP theorem in one paragraph.",
    "What's the difference between TCP and UDP?",
    "When would you use a hash map instead of a tree map?",
    "What is the difference between a process and a thread?",
    "Explain what a race condition is and how to prevent one.",
    "What is the difference between horizontal and vertical scaling?",
    "Explain eventual consistency versus strong consistency.",
    "What's the difference between authentication and authorization?",
    "Explain what a memory leak is in a garbage-collected language.",
    "What is the difference between REST and gRPC?",
    "Explain what idempotency means for an API endpoint.",
    "What's the difference between a shallow copy and a deep copy?",
    "Explain what backpressure means in a streaming system.",
    "What is the difference between optimistic and pessimistic locking?",
    "Explain what a bloom filter is and when to use one.",
    "What's the difference between synchronous and asynchronous I/O?",
    "Explain what a deadlock is and one way to prevent it.",
    "What is the difference between SQL and NoSQL databases?",
    "Explain what connection pooling is and why it matters.",
    "What's the difference between a load balancer and a reverse proxy?",
    "Explain what sharding means for a database.",
    "What is the difference between at-least-once and exactly-once delivery?",
    "Explain what a circuit breaker pattern does.",
    "What's the difference between vertical and horizontal partitioning?",
    "Explain what consistent hashing is used for.",
    "What is the difference between a monolith and a microservices architecture?",
    "Explain what a webhook is compared to polling.",
    "What's the difference between latency and throughput?",
    "Explain what a leader election algorithm does in a distributed system.",
    "What is the difference between soft and hard real-time systems?",
    "Explain what write-ahead logging is used for in databases.",
    "What's the difference between TLS and SSL?",
    "Explain what a content delivery network does and why it reduces latency.",
    "What is the difference between stateful and stateless services?",
    "Explain what the N+1 query problem is.",
    "What's the difference between a primary key and a foreign key?",
    "Explain what rate limiting is and one common algorithm for it.",
    "What is the difference between a data warehouse and a data lake?",
    "Explain what a canary deployment is.",
    "What's the difference between blue-green and rolling deployments?",
    "Explain what a service mesh is used for.",
    "What is the difference between vertical scaling and autoscaling?",
    "Explain what a message queue is used for.",
    "What's the difference between an ORM and raw SQL?",
    "Explain what event sourcing means.",
    "What is the difference between a monolithic and distributed transaction?",
    "Explain what a heartbeat mechanism is used for in distributed systems.",
    "What's the difference between vertical and horizontal sharding?",
    "Explain what graceful degradation means for a service.",
]


def _dedup_check(prompts, label):
    assert len(prompts) == len(set(prompts)), f"duplicate prompt generated in {label}"
    return prompts


def build_short(target: int):
    prompts = [f"What is the capital of {c}?" for c in COUNTRIES]
    seen = set(prompts)
    while len(prompts) < target:
        a, b = random.randint(2, 99), random.randint(2, 99)
        op_symbol, _ = random.choice(MATH_OPS)
        p = f"What is {a} {op_symbol} {b}?"
        if p not in seen:
            seen.add(p)
            prompts.append(p)
    return _dedup_check(prompts[:target], "short")


def build_medium_reasoning(target: int):
    prompts = []
    seen = set()
    while len(prompts) < target:
        template = random.choice(REASONING_TEMPLATES)
        a, b, d = random.randint(2, 90), random.randint(2, 90), random.randint(10, 500)
        p = template.format(a=a, b=b, d=d)
        if p not in seen:
            seen.add(p)
            prompts.append(p)
    return _dedup_check(prompts, "medium-reasoning")


def build_medium_instruction(target: int):
    combos = [(t, topic) for t in INSTRUCTION_TEMPLATES for topic in TOPICS]
    assert len(combos) >= target, f"instruction combo space too small: {len(combos)} < {target}"
    random.shuffle(combos)
    prompts = [template.format(topic=topic) for template, topic in combos[:target]]
    return _dedup_check(prompts, "medium-instruction")


def build_medium_code(target: int):
    combos = [(task, framing) for task in CODE_TASKS for framing in CODE_FRAMINGS]
    assert len(combos) >= target, f"code combo space too small: {len(combos)} < {target}"
    random.shuffle(combos)
    prompts = []
    for (lang, task), framing in combos[:target]:
        article = "an" if lang[0] in "AEIOU" else "a"
        prompts.append(framing.format(article=article, lang=lang, task=task))
    return _dedup_check(prompts, "medium-code")


def build_long(target: int):
    combos = [(t, topic) for t in LONG_TEMPLATES for topic in TOPICS]
    assert len(combos) >= target, f"long combo space too small: {len(combos)} < {target}"
    random.shuffle(combos)
    prompts = [
        template.format(topic=topic, topic_cap=topic.capitalize())
        for template, topic in combos[:target]
    ]
    return _dedup_check(prompts, "long")


def build_prefix_reuse(target: int):
    combos = [(p, q) for p in PREFIX_PERSONAS for q in PREFIX_QUESTIONS]
    assert len(combos) >= target, f"prefix combo space too small: {len(combos)} < {target}"
    random.shuffle(combos)
    prompts = [f"{persona}\n\nQuestion: {question}" for persona, question in combos[:target]]
    return _dedup_check(prompts, "prefix-reuse")


def main():
    records = []

    for p in build_short(250):
        records.append(("short", p, 20))
    for p in build_medium_reasoning(120):
        records.append(("medium-reasoning", p, 150))
    for p in build_medium_instruction(120):
        records.append(("medium-instruction", p, 150))
    for p in build_medium_code(120):
        records.append(("medium-code", p, 220))
    for p in build_long(200):
        records.append(("long", p, 180))
    for p in build_prefix_reuse(200):
        records.append(("prefix-reuse", p, 100))

    random.shuffle(records)

    all_prompts = [p for _, p, _ in records]
    assert len(all_prompts) == len(set(all_prompts)), "duplicate prompt across categories"

    with OUT_PATH.open("w") as f:
        for i, (category, prompt, max_tokens) in enumerate(records, start=1):
            line = {
                "id": f"{category}-{i:04d}",
                "category": category,
                "prompt": prompt,
                "max_tokens": max_tokens,
            }
            f.write(json.dumps(line) + "\n")

    print(f"Wrote {len(records)} prompts to {OUT_PATH}")


if __name__ == "__main__":
    main()