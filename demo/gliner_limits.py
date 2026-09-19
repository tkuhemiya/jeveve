# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "altair>=5.5",
#     "httpx>=0.28",
#     "marimo>=0.19",
#     "pandas>=2.2",
# ]
# ///
"""Live GLiNER2.5 demo against the production extract API.

    uvx marimo edit --sandbox demo/gliner_limits.py
    uvx marimo run --sandbox demo/gliner_limits.py

Cold start (web + one extractor pool) can take a few minutes. The client
follows Modal's 303 result redirects and retries 503s while a model loads.
"""

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="full", app_title="GLiNER2.5 — push the extract API")

with app.setup:
    import html
    import json
    import os
    import time
    from concurrent.futures import ThreadPoolExecutor
    from typing import Any

    import altair as alt
    import httpx
    import marimo as mo
    import pandas as pd

    DEFAULT_URL = os.environ.get(
        "GLINER_URL", "https://tkuhemiya--gliner-web.modal.run"
    )
    DEFAULT_API_KEY = (
        os.environ.get("GLINER_API_KEY")
        or os.environ.get("API_KEY")
        or "jv_t9XGrsBJ_mjTLkp64WD9l6RoXhwVMSdk1Vp5MLc3Jf0"
    )
    MODELS = ("small", "base", "multi")
    OVERLAP_POLICIES = ("allow", "nested", "flat", "disallow", "longest")
    PALETTE = (
        "#7dd3fc",
        "#f9a8d4",
        "#86efac",
        "#fcd34d",
        "#c4b5fd",
        "#fda4af",
        "#67e8f9",
        "#fdba74",
        "#a5b4fc",
        "#bef264",
        "#f0abfc",
        "#5eead4",
        "#fca5a5",
        "#93c5fd",
        "#d9f99d",
    )

    def color_for(label: str) -> str:
        return PALETTE[sum(map(ord, label)) % len(PALETTE)]

    def parse_labels(block: str) -> list[str] | dict[str, str]:
        names: list[str] = []
        described: dict[str, str] = {}
        for raw in block.splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                name, _, desc = line.partition(":")
                name, desc = name.strip(), desc.strip()
                if name:
                    described[name] = desc or name
            else:
                names.append(line)
        if described and names:
            for name in names:
                described.setdefault(name, name)
            return described
        return described or names

    def flatten_entities(payload: Any) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, (list, tuple)) and item:
                    text = str(item[0])
                    conf = float(item[1]) if len(item) > 1 else None
                    start = int(item[2]) if len(item) > 2 else None
                    end = int(item[3]) if len(item) > 3 else None
                    rows.append(
                        {
                            "label": "entity",
                            "text": text,
                            "start": start,
                            "end": end,
                            "confidence": conf,
                        }
                    )
                elif isinstance(item, dict):
                    rows.append(
                        {
                            "label": str(item.get("label") or "entity"),
                            "text": str(item.get("text") or ""),
                            "start": item.get("start"),
                            "end": item.get("end"),
                            "confidence": item.get("confidence"),
                        }
                    )
            return rows
        if not isinstance(payload, dict):
            return rows
        entities = payload.get("entities", payload)
        if not isinstance(entities, dict):
            return rows
        for label, items in entities.items():
            if not items:
                continue
            if isinstance(items, str):
                items = [items]
            for item in items:
                if isinstance(item, str):
                    rows.append(
                        {
                            "label": label,
                            "text": item,
                            "start": None,
                            "end": None,
                            "confidence": None,
                        }
                    )
                elif isinstance(item, dict):
                    rows.append(
                        {
                            "label": label,
                            "text": str(item.get("text") or ""),
                            "start": item.get("start"),
                            "end": item.get("end"),
                            "confidence": item.get("confidence"),
                        }
                    )
        return rows

    def highlight_html(text: str, rows: list[dict[str, Any]]) -> str:
        spans = [
            row
            for row in rows
            if isinstance(row.get("start"), int) and isinstance(row.get("end"), int)
        ]
        if not spans:
            return f"<p class='gliner-doc'>{html.escape(text)}</p>"
        cuts = {0, len(text)}
        for row in spans:
            start, end = int(row["start"]), int(row["end"])
            if 0 <= start < end <= len(text):
                cuts.add(start)
                cuts.add(end)
        ordered = sorted(cuts)
        parts: list[str] = []
        for left, right in zip(ordered, ordered[1:]):
            chunk = html.escape(text[left:right])
            covering = [
                row
                for row in spans
                if int(row["start"]) <= left and right <= int(row["end"])
            ]
            if not covering:
                parts.append(chunk)
                continue
            covering.sort(
                key=lambda row: (
                    int(row["end"]) - int(row["start"]),
                    -(float(row["confidence"] or 0.0)),
                )
            )
            top = covering[0]
            color = color_for(str(top["label"]))
            title = ", ".join(
                (
                    f"{row['label']} {float(row['confidence']):.2f}"
                    if row.get("confidence") is not None
                    else str(row["label"])
                )
                for row in covering
            )
            parts.append(
                "<mark class='ent' "
                f"style='background:{color}33;border-bottom:2px solid {color}' "
                f"title='{html.escape(title)}'>{chunk}</mark>"
            )
        return f"<p class='gliner-doc'>{''.join(parts)}</p>"

    def legend_html(labels: list[str]) -> str:
        chips = []
        for label in labels:
            color = color_for(label)
            chips.append(
                "<span class='chip' "
                f"style='border-color:{color};background:{color}22'>"
                f"{html.escape(label)}</span>"
            )
        return "<div class='legend'>" + "".join(chips) + "</div>"

    def _header_float(headers: Any, name: str) -> float | None:
        raw = headers.get(name)
        if raw is None or raw == "":
            return None
        return float(raw)

    def extract_entities(
        *,
        url: str,
        api_key: str,
        model: str,
        text: str,
        labels: list[str] | dict[str, str],
        include_confidence: bool = True,
        include_spans: bool = True,
        threshold: float | None = None,
        overlap_policy: str | None = None,
        format_results: bool = True,
        timeout: float = 600.0,
        attempts: int = 8,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": model,
            "text": text,
            "labels": labels,
            "include_confidence": include_confidence,
            "include_spans": include_spans,
            "format_results": format_results,
        }
        if threshold is not None:
            body["threshold"] = threshold
        if overlap_policy:
            body["overlap_policy"] = overlap_policy
        endpoint = url.rstrip("/") + "/v1/extract_entities"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        last_error = "extract failed"
        started = time.perf_counter()
        with httpx.Client(
            follow_redirects=True,
            timeout=httpx.Timeout(timeout, connect=30.0),
        ) as client:
            for attempt in range(attempts):
                response = client.post(endpoint, headers=headers, json=body)
                if response.status_code == 503:
                    wait = int(response.headers.get("Retry-After", "10"))
                    last_error = response.text
                    time.sleep(wait)
                    continue
                if response.status_code == 401:
                    raise RuntimeError("401 Unauthorized — check the API key")
                response.raise_for_status()
                payload = response.json()
                elapsed = time.perf_counter() - started
                return {
                    "payload": payload,
                    "rows": flatten_entities(payload),
                    "elapsed": elapsed,
                    "status": response.status_code,
                    "attempts": attempt + 1,
                    "model": model,
                    "text": text,
                    "labels": labels,
                    "body": body,
                    "timing": {
                        "load_s": _header_float(response.headers, "x-gliner-load-s"),
                        "infer_s": _header_float(response.headers, "x-gliner-infer-s"),
                        "wait_s": _header_float(response.headers, "x-gliner-wait-s"),
                        "cold": response.headers.get("x-gliner-cold") == "true",
                        "slow": response.headers.get("x-gliner-slow") == "true",
                        "extracts": response.headers.get("x-gliner-extracts"),
                    },
                }
        raise RuntimeError(f"extractor not ready after {attempts} tries: {last_error}")

    def extract_many(
        specs: list[dict[str, Any]],
        *,
        url: str,
        api_key: str,
        timeout: float = 600.0,
    ) -> list[dict[str, Any]]:
        def _run(spec: dict[str, Any]) -> dict[str, Any]:
            return extract_entities(url=url, api_key=api_key, timeout=timeout, **spec)

        workers = min(3, max(1, len(specs)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(_run, specs))

    def healthcheck(url: str, timeout: float = 60.0) -> dict[str, Any]:
        started = time.perf_counter()
        with httpx.Client(
            follow_redirects=True,
            timeout=httpx.Timeout(timeout, connect=15.0),
        ) as client:
            response = client.get(url.rstrip("/") + "/health")
            response.raise_for_status()
            return {
                "payload": response.json(),
                "elapsed": time.perf_counter() - started,
                "status": response.status_code,
            }

    def labels_to_block(labels: list[str] | dict[str, str]) -> str:
        if isinstance(labels, dict):
            return "\n".join(f"{name}: {desc}" for name, desc in labels.items())
        return "\n".join(labels)

    NESTED_TEXT = (
        "The New York City Police Department's 17th Precinct in Midtown Manhattan "
        "coordinated with the United States Department of Homeland Security and the "
        "Bank of America Tower security team after a bomb threat was phoned into "
        "1 Bryant Park. Mayor Eric Adams and NYPD Commissioner Edward Caban held a "
        "briefing at City Hall in Lower Manhattan, New York. The Federal Bureau of "
        "Investigation's New York Field Office in Federal Plaza dispatched agents "
        "from the Joint Terrorism Task Force. Witnesses near Grand Central Terminal "
        "and the Chrysler Building reported seeing NYPD Emergency Service Unit trucks "
        "on 42nd Street beside the United Nations Headquarters."
    )
    NESTED_LABELS = [
        "organization",
        "government agency",
        "location",
        "neighborhood",
        "building",
        "person",
        "job title",
        "street",
        "city",
        "facility",
        "unit",
    ]
    PACKED_NEWS = (
        "SAN FRANCISCO — On January 8, 2024, Apple Inc. CEO Tim Cook and OpenAI "
        "chief Sam Altman jointly unveiled a $13.6 billion partnership to bring "
        "ChatGPT-4o to iPhone 15 Pro and MacBook Pro devices at Apple Park in "
        "Cupertino, California. Microsoft Corp., already down more than $13 billion "
        "into OpenAI, will supply Azure H100 GPU clusters from data centers in Des "
        "Moines, Iowa. Nvidia CEO Jensen Huang called the deal \"the iPhone moment "
        "for generative AI\" during a CNBC interview from CES in Las Vegas. Shares "
        "of AAPL rose 4.6% to $198.43 while NVDA added 2.1%. The U.S. Department of "
        "Justice and the European Commission said they would review the agreement "
        "under the Digital Markets Act. Cook later flew to Seoul to meet Samsung "
        "Electronics vice chairman Han Jong-hee about OLED supply for Vision Pro 2, "
        "expected in Q3 2025. Reuters later reported that Alphabet's Google DeepMind "
        "and Amazon Web Services were drafting counter-offers out of London and "
        "Arlington, Virginia."
    )
    CLINICAL_NOTE = (
        "64M with PMH of T2DM, HFrEF (EF 30%), atrial fibrillation, and CKD stage 3 "
        "presented to Massachusetts General Hospital with acute decompensated heart "
        "failure. Home meds: metformin 1000 mg BID, lisinopril 20 mg daily, "
        "empagliflozin 10 mg qAM, apixaban 5 mg BID, and furosemide 40 mg. In the ED "
        "he received 80 mg IV Lasix and a nitroglycerin drip at 20 mcg/min. "
        "Troponin-T peaked at 45 ng/L; NT-proBNP 1,840 pg/mL. Echo showed LVEF 25% "
        "with moderate mitral regurgitation. Cardiology (Dr. Priya Shah) recommended "
        "sacubitril/valsartan 24/26 mg BID after holding lisinopril for 36 hours. He "
        "was discharged on a 2 g sodium diet with follow-up at the MGH Heart Failure "
        "Clinic in Boston on 12 March 2026. Allergies: penicillin (anaphylaxis)."
    )
    MULTILINGUAL = """\
    FR Le président Emmanuel Macron a reçu Bernard Arnault, PDG de LVMH, à l'Élysée à Paris pour discuter du rachat de Tiffany & Co. pour 15,8 milliards de dollars.
    DE Bundeskanzler Olaf Scholz besuchte das BMW-Werk in München und kündigte Subventionen für die Neue Klasse Elektroautos und den Chipstandort Magdeburg an.
    ES La presidenta Claudia Sheinbaum inauguró la Línea 1 del Cablebús en la Ciudad de México junto al rector de la UNAM y directivos de Grupo Bimbo.
    JA トヨタ自動車の佐藤恒治社長は東京・豊田市で新型プリウスとレクサス RX の値上げを発表し、円安と半導体不足に言及した。
    ZH 华为轮值董事长孟晚舟在深圳发布了盘古大模型 5.0，并与沙特阿美在利雅得签署云计算合同。
    AR أعلن رئيس الوزراء المصري مصطفى مدبولي من القاهرة عن توسعة قناة السويس بالتعاون مع موانئ دبي العالمية وشركة أوراسكوم.
    KO 삼성전자는 수원에서 갤럭시 S24 울트라와 엑시노스 2400을 공개했으며 이재용 회장이 기조연설을 했다.
    PT O presidente Luiz Inácio Lula da Silva assinou em Brasília um acordo com a Petrobras sobre exploração na Margem Equatorial e o pre-sal da Bacia de Santos.
    HI भारत के प्रधानमंत्री नरेंद्र मोदी ने नई दिल्ली में अदाणी ग्रीन और रिलायंस इंडस्ट्रीज के साथ 20 गीगावाट की सौर परियोजना की घोषणा की।
    RU Президент Владимир Путин посетил офис «Газпрома» в Санкт-Петербурге и обсудил «Северный поток» с Алексеем Миллером.
    MIX Después del keynote en CDMX, Tim Cook said the nuevo iPhone 16 Pro se fabricará en 郑州 with Foxconn, then flew 成田 for a 会見 with ソニー's Hiroki Totoki about PlayStation Portal."""
    PII_DUMP = (
        "Please update the chart: patient Jane Q. Public, DOB 14 July 1984, MRN "
        "MGH-00918432, SSN 078-05-1120, lives at 742 Evergreen Terrace, Springfield, "
        "IL 62704. Reach her at jane.public@example.com or +1-217-555-0199. Billing "
        "card Visa 4111 1111 1111 1111 exp 09/27 CVC 123. Workstation 10.20.44.81 "
        "pulled https://records.example.org/encounters/88421 after login from "
        "user jpublic. Her emergency contact is Robert Public +1 (415) 555-2671. "
        "Do not page Dr. Omar Haddad at omar.haddad@mgh.harvard.edu about HIV status "
        "or the clozapine 100 mg prescription filled at CVS #4421."
    )
    LEGAL = (
        "THIS MASTER SERVICES AGREEMENT is entered into as of 1 November 2025 "
        "(\"Effective Date\") by and between Helios Robotics, Inc., a Delaware "
        "corporation with offices at 88 King Street, San Francisco, California "
        "(\"Helios\"), and Nordhavn Logistics GmbH, registered in Hamburg, Germany "
        "(\"Nordhavn\"). Helios shall deliver the Atlas Forklift Autonomy Kit, "
        "including the Helios Perception Stack v3.2, for €4,750,000 payable in "
        "three installments. The Agreement is governed by the laws of England and "
        "Wales, exclusive of conflict-of-law rules, and disputes shall be seated "
        "at the London Court of International Arbitration. Limitation of liability "
        "caps at the fees paid in the twelve (12) months preceding the claim, "
        "except for breaches of Clause 9 (Confidentiality) or infringement of the "
        "EU General Data Protection Regulation. Notices to Helios go to "
        "legal@heliosrobotics.example with a copy to Wilson Sonsini Goodrich & "
        "Rosati. Nordhavn's DPO is Ingrid Bauer in Berlin."
    )
    SCHEMA_BLAST_TEXT = (
        "At 7:41 p.m. ET on Super Bowl LIX, Patrick Mahomes of the Kansas City "
        "Chiefs threw a 32-yard touchdown to Hollywood Brown, putting KC up 31-28 "
        "over the Philadelphia Eagles at Caesars Superdome. During the break, "
        "Apple's Tim Cook posted on @tim_cook about Vision Pro, #ShotOniPhone, and "
        "a $2 million donation to St. Jude. Meanwhile Nature published a paper on "
        "CRISPR-Cas9 editing of BRCA1 in HEK293 cells using olaparib, and Binance "
        "halted BTC-USDT after Bitcoin crashed 11.4% to $61,240. The FAA grounded "
        "Boeing 737 MAX 9 N704AL following an Alaska Airlines incident, citing 14 "
        "CFR Part 39. Email tips to tips@reuters.com or +1-800-555-0192. Real Madrid "
        "later beat Bayern Munich 2-1 in the UEFA Champions League at the Bernabéu, "
        "while SpaceX's Starship IFT-6 lifted from Boca Chica. Gene TP53 and the "
        "spike protein of SARS-CoV-2 were trending next to AAPL, MSFT, and the "
        "Sherman Antitrust Act. Handle @NASA posted 4K footage of the ISS."
    )
    SCHEMA_BLAST_LABELS = [
        "person",
        "job title",
        "company",
        "product",
        "location",
        "facility",
        "city",
        "country",
        "date",
        "time",
        "money",
        "percent",
        "quantity",
        "ticker",
        "sports team",
        "league",
        "score",
        "event",
        "aircraft",
        "law",
        "gene",
        "protein",
        "drug",
        "disease",
        "email",
        "phone number",
        "hashtag",
        "social handle",
        "url",
        "regulator",
        "spacecraft",
        "award",
    ]
    AMBIGUOUS = (
        "Apple pie cooled on the windowsill at Apple Park while Apple Music queued "
        "Fiona Apple, and an intern ate a Pink Lady apple beside a crate of Apple "
        "Vision Pro units. Amazon the river flooded a warehouse that Amazon.com "
        "uses to ship Kindle copies of The Jungle, and a jaguar (the cat) slept on "
        "the hood of a Jaguar F-Type outside Oracle Park, which is not owned by "
        "Oracle Corporation. Turkey the country banned turkey imports for "
        "Thanksgiving, while Nice, France was not nice about the UN Security "
        "Council's resolution."
    )
    EARNINGS_GRAPH = (
        "Operator: Good afternoon. Welcome to NVIDIA Corporation's fourth quarter "
        "and fiscal 2025 earnings call. Joining us are Jensen Huang, founder and "
        "CEO, and Colette Kress, executive vice president and CFO.\n\n"
        "Jensen Huang: Demand for Hopper H100 and H200 remains off the charts. "
        "Blackwell GB200 NVL72 racks are shipping to Microsoft Azure, Amazon AWS, "
        "Google Cloud, Oracle Cloud Infrastructure, CoreWeave, and Tesla. Meta "
        "is building a 100,000-GPU cluster in New Albany, Ohio. TSMC's CoWoS "
        "capacity in Taiwan is the bottleneck; we also qualified Samsung HBM3E "
        "and SK Hynix stacks. Data Center revenue was $35.6 billion, up 93% "
        "year over year. Gaming was $2.6 billion. Automotive, including Mercedes-Benz "
        "DRIVE PILOT and Toyota, contributed $570 million. Gross margin landed at "
        "73.5%. We returned $16 billion via buybacks. China remains constrained "
        "under the U.S. Bureau of Industry and Security export rules for H20.\n\n"
        "Colette Kress: Q4 revenue was $39.3 billion versus the $37.5 billion "
        "outlook. We guided Q1 fiscal 2026 to $42.0 billion plus or minus 2%. "
        "Operating expenses were $4.7 billion. We ended with $43.9 billion in cash. "
        "The 10-K will be filed with the Securities and Exchange Commission. "
        "Non-GAAP diluted EPS was $0.89. Share count was approximately 24.6 billion.\n\n"
        "Analyst: C.J. Muse, Cantor Fitzgerald. Can you size Rubin versus Blackwell "
        "and comment on sovereign deals with Saudi Arabia's HUMAIN, the UAE's G42, "
        "and Japan’s METI?\n\n"
        "Jensen Huang: Rubin samples in late 2026. Sovereign clouds in Riyadh, "
        "Abu Dhabi, Tokyo, Paris, and Berlin are ordering DGX SuperPODs. We work "
        "with OpenAI, Anthropic, xAI, Mistral, and DeepSeek. CUDA remains the "
        "moat. Inference tokens on NIM microservices doubled sequentially."
    )

    def long_earnings_call() -> str:
        extra = (
            "\n\nFollow-up from Toshiya Hari at Goldman Sachs on networking: "
            "Quantum-X800 InfiniBand and Spectrum-X Ethernet are attaching at "
            "nearly 100% to GB200 deployments at Microsoft in Des Moines and "
            "Amazon in northern Virginia. A question from Vivek Arya at Bank of "
            "America on automotive: DRIVE Thor launches with BYD, XPENG, and "
            "Jaguar Land Rover. A question from Matt Ramsay at TD Cowen on China: "
            "H20 revenues are immaterial after the BIS license pause. A question "
            "from Harlan Sur at JPMorgan on software: AI Enterprise, CUDA-X, and "
            "Omniverse subscriptions grew double digits. A question from Mark "
            "Lipacis at Jefferies on supply: Amkor, SPIL, and ASE added packaging "
            "lines in Incheon and Hsinchu. Capital expenditures at TSMC Fab 21 in "
            "Arizona and Fab 18 in Kumamoto are cited as 2026 relief valves."
        )
        body = EARNINGS_GRAPH + extra
        cycle = extra
        while len(body) < 12_000:
            body += "\n\n" + cycle
        return body[:12_000]

    CASES: dict[str, dict[str, Any]] = {
        "packed news — all three models": {
            "kind": "compare",
            "blurb": (
                "Same English dispatch, three checkpoints in parallel. `small` is "
                "the cheap DeBERTa-v3-xsmall; `base` should pick up tickers, laws, "
                "and titles more cleanly; `multi` is the mDeBERTa multilingual "
                "encoder running on English."
            ),
            "model": "base",
            "text": PACKED_NEWS,
            "labels": [
                "company",
                "person",
                "product",
                "location",
                "date",
                "money",
                "percent",
                "ticker",
                "job title",
                "regulator",
                "law",
                "event",
            ],
        },
        "nested overlap gauntlet": {
            "kind": "overlap",
            "blurb": (
                "NYPD ⊂ New York City ⊂ New York. Bank of America Tower is an org, "
                "a building, and contains a country name. Five overlap policies "
                "on `base`: allow (everything), nested (containment ok), "
                "flat/disallow (weighted interval schedule), longest (drop contained)."
            ),
            "model": "base",
            "text": NESTED_TEXT,
            "labels": NESTED_LABELS,
        },
        "clinical described schema": {
            "kind": "single",
            "blurb": (
                "Zero-shot medical NER with label *descriptions* — the trick that "
                "makes invented types like `ejection_fraction` work. Runs on `base`."
            ),
            "model": "base",
            "text": CLINICAL_NOTE,
            "labels": {
                "patient": "The person receiving care, including age/sex shorthand",
                "hospital": "A named medical facility or clinic",
                "clinician": "A named doctor or care team member",
                "condition": "Disease, syndrome, or problem list item",
                "medication": "Brand or generic drug names",
                "dosage": "Numeric dose plus unit such as 1000 mg",
                "route_or_frequency": "BID, daily, IV, drip rates, qAM",
                "lab_test": "Named assays such as Troponin-T or NT-proBNP",
                "lab_value": "A measured numeric result with units",
                "procedure": "Imaging, surgery, or diagnostic procedure",
                "allergy": "A substance causing a documented allergy",
                "diet_or_precaution": "Discharge diet or hold parameters",
                "date": "A calendar date",
                "location": "City or campus",
            },
        },
        "multilingual world tour": {
            "kind": "single",
            "blurb": (
                "`multi` (mDeBERTa-v3-base) on ten languages plus a nasty "
                "ES/EN/ZH/JA code-switch line. `small`/`base` are English-only "
                "encoders — this is the checkpoint you actually want here."
            ),
            "model": "multi",
            "text": MULTILINGUAL,
            "labels": [
                "person",
                "job title",
                "company",
                "product",
                "location",
                "city",
                "country",
                "facility",
                "money",
                "geopolitical entity",
            ],
        },
        "PII redaction dump": {
            "kind": "single",
            "blurb": (
                "Synthetic chart-update email packed with the identifiers a "
                "privacy filter must catch. All values are fake. `base` with a "
                "described PII schema."
            ),
            "model": "base",
            "text": PII_DUMP,
            "labels": {
                "person": "A named human",
                "date_of_birth": "A date of birth",
                "medical_record_number": "Hospital MRN or chart id",
                "us_ssn": "US social security number",
                "street_address": "Street, city, region, postal code",
                "email": "Email address",
                "phone_number": "Telephone number",
                "credit_card": "Payment card number",
                "ip_address": "IPv4 or IPv6 address",
                "url": "Web URL",
                "username": "Account or login name",
                "clinician": "Named doctor",
                "diagnosis": "Sensitive diagnosis or status",
                "medication": "Drug name",
                "pharmacy": "Pharmacy or store identifier",
            },
        },
        "legal MSA": {
            "kind": "single",
            "blurb": (
                "Contract-shaped prose: parties, jurisdictions, venues, statutes, "
                "caps, product names, money. Described labels on `base`."
            ),
            "model": "base",
            "text": LEGAL,
            "labels": {
                "party": "A legal entity that is a signatory",
                "person": "A named human, including a DPO",
                "product": "A named commercial product or stack",
                "date": "A calendar or relative contractual date",
                "money": "A currency amount",
                "jurisdiction": "Governing law or country of registration",
                "forum": "Court, arbitral body, or seat",
                "statute": "Named regulation or act",
                "clause": "A numbered clause or named section",
                "location": "A city or street address",
                "email": "An email address",
                "law_firm": "A law firm",
                "role": "A defined contractual role or defined term",
            },
        },
        "schema explosion (32 labels)": {
            "kind": "single",
            "blurb": (
                "One chaotic paragraph, 32 simultaneous zero-shot types: sports, "
                "genomics, aviation regs, tickers, handles. This is the 'can the "
                "prompt even fit' test."
            ),
            "model": "base",
            "text": SCHEMA_BLAST_TEXT,
            "labels": SCHEMA_BLAST_LABELS,
        },
        "ambiguity & homographs": {
            "kind": "single",
            "blurb": (
                "Apple the fruit vs Apple the company vs Fiona Apple; Amazon the "
                "river vs Amazon.com; Jaguar the cat vs the car. Zero-shot has to "
                "use context, not a gazetteer."
            ),
            "model": "base",
            "text": AMBIGUOUS,
            "labels": [
                "company",
                "person",
                "product",
                "food",
                "animal",
                "location",
                "river",
                "sports venue",
                "country",
                "organization",
                "holiday",
            ],
        },
        "long earnings call (~12k chars)": {
            "kind": "single",
            "blurb": (
                "API max is 50,000 characters; GLiNER2.5's encoder window is "
                "4,096 tokens. This call is ~12k chars — sitting on the window. "
                "Expect more misses in the tail if the model truncates."
            ),
            "model": "base",
            "text": long_earnings_call(),
            "labels": [
                "person",
                "job title",
                "company",
                "product",
                "location",
                "money",
                "percent",
                "date",
                "facility",
                "regulator",
                "country",
                "metric",
            ],
        },
        "English on multi vs base": {
            "kind": "compare_pair",
            "blurb": (
                "Identical English, `base` vs `multi`. The multilingual encoder "
                "is not free — watch for dropped tickers and titles."
            ),
            "models": ["base", "multi"],
            "text": PACKED_NEWS,
            "labels": [
                "company",
                "person",
                "product",
                "location",
                "ticker",
                "money",
                "law",
            ],
        },
        "non-English on small (should hurt)": {
            "kind": "compare_pair",
            "blurb": (
                "Japanese + Chinese + Arabic sent to English `small` and to "
                "`multi`. This is the negative control: `small` should fall apart."
            ),
            "models": ["small", "multi"],
            "text": "\n".join(MULTILINGUAL.splitlines()[3:7]),
            "labels": ["person", "company", "product", "location", "job title"],
        },
    }

    CSS = """
    <style>
      .gliner-doc { white-space: pre-wrap; line-height: 1.7; font-size: 0.98rem; }
      .ent { padding: 0 2px; border-radius: 3px; }
      .legend { display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0 12px; }
      .chip {
        font-size: 12px; padding: 2px 8px; border-radius: 999px;
        border: 1px solid; font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      }
      .statrow { display: flex; flex-wrap: wrap; gap: 12px; margin: 8px 0 14px; }
      .statpill {
        border: 1px solid color-mix(in srgb, currentColor 18%, transparent);
        border-radius: 10px; padding: 8px 12px; min-width: 120px;
      }
      .statpill .k { font-size: 11px; opacity: 0.7; text-transform: uppercase; letter-spacing: .04em; }
      .statpill .v { font-size: 1.25rem; font-weight: 650; }
    </style>
    """


@app.cell(hide_code=True)
def _():
    mo.Html(CSS)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # GLiNER2.5, live, at the wall

    Zero-shot NER: **you invent the labels**, the model points at the spans. This notebook
    calls production [`POST /v1/extract_entities`](https://tkuhemiya--gliner-web.modal.run/docs)
    with every checkpoint the Modal app actually serves.

    | `model` | Checkpoint | Encoder | Use |
    | --- | --- | --- | --- |
    | `small` | `fastino/gliner2.5-small-v1` | DeBERTa-v3-xsmall · 74M | English, cheapest |
    | `base` | `fastino/gliner2.5-base-v1` | DeBERTa-v3-base · 194M | English, cleaner spans |
    | `multi` | `fastino/gliner2.5-multi-v1` | mDeBERTa-v3-base · 287M | Anything not English |

    The interesting bits are not "person/location" on a press release. They are **nested
    orgs**, **described domain schemas**, **32 labels at once**, **homographs**, **ten
    languages plus code-switch**, and a **12k-character earnings call** sitting on the
    4,096-token encoder window. API cap is 50k characters; idle pools scale to zero
    after 60s — first hit boots web, then that model's extractor (Modal 303-redirects
    after 150s; the client follows them).

    Set `GLINER_API_KEY` / `GLINER_URL` if you want to override the defaults.
    """)
    return


@app.cell
def _():
    url = mo.ui.text(value=DEFAULT_URL, label="API URL", full_width=True)
    api_key = mo.ui.text(
        value=DEFAULT_API_KEY,
        label="API key",
        kind="password",
        full_width=True,
    )
    timeout = mo.ui.slider(
        start=60,
        stop=900,
        value=600,
        step=30,
        label="HTTP timeout (s)",
        show_value=True,
    )
    mo.hstack([url, api_key, timeout], justify="start", gap=1, wrap=True)
    return api_key, timeout, url


@app.cell
def _():
    ping = mo.ui.run_button(label="Ping /health")
    ignite = mo.ui.run_button(label="Ignite small + base + multi")
    mo.hstack([ping, ignite], justify="start", gap=1)
    return ignite, ping


@app.cell
def _(api_key, ignite, ping, timeout, url):
    status_bits: list[Any] = []
    if ping.value:
        try:
            health = healthcheck(url.value, timeout=min(90.0, float(timeout.value)))
            kind = "warn" if health["payload"].get("status") == "degraded" else "success"
            status_bits.append(
                mo.md(
                    f"**health** `{health['payload'].get('status')}` in `{health['elapsed']:.2f}s`\n\n"
                    f"```json\n{json.dumps(health['payload'], indent=2)}\n```"
                ).callout(kind=kind)
            )
        except Exception as exc:
            status_bits.append(mo.md(f"health failed: `{exc}`").callout(kind="danger"))
    if ignite.value:
        try:
            ignited = extract_many(
                [
                    {
                        "model": name,
                        "text": "Apple Inc. is based in Cupertino.",
                        "labels": ["company", "location"],
                    }
                    for name in MODELS
                ],
                url=url.value,
                api_key=api_key.value,
                timeout=float(timeout.value),
            )
            lines = []
            slow = False
            for item in ignited:
                timing = item.get("timing") or {}
                mark = " SLOW" if timing.get("slow") else ""
                slow = slow or bool(timing.get("slow"))
                load = timing.get("load_s")
                wait = timing.get("wait_s")
                lines.append(
                    f"- `{item['model']}`{mark} · {len(item['rows'])} spans · "
                    f"client `{item['elapsed']:.1f}s` · load `{load}`s · wait `{wait}`s"
                )
            status_bits.append(
                mo.md("**extractors are up**\n\n" + "\n".join(lines)).callout(
                    kind="warn" if slow else "success"
                )
            )
        except Exception as exc:
            status_bits.append(mo.md(f"ignite failed: `{exc}`").callout(kind="danger"))
    if not status_bits:
        status_bits.append(
            mo.md(
                "Idle pools are cold. **Ping** is cheap. **Ignite** fires three tiny "
                "extracts in parallel so the showcases below don't eat a 3-minute "
                "cold start each. `small` traffic does **not** keep `base` or `multi` warm."
            ).callout(kind="neutral")
        )
    mo.vstack(status_bits)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Catalog
    """)
    return


@app.cell
def _():
    case_name = mo.ui.dropdown(
        options=list(CASES.keys()),
        value="packed news — all three models",
        label="Limit test",
        full_width=True,
    )
    threshold_on = mo.ui.checkbox(value=False, label="Override threshold")
    threshold = mo.ui.slider(
        start=0.0,
        stop=1.0,
        value=0.35,
        step=0.05,
        label="threshold",
        show_value=True,
    )
    overlap = mo.ui.dropdown(
        options=["(checkpoint default)", *OVERLAP_POLICIES],
        value="(checkpoint default)",
        label="overlap_policy",
    )
    run = mo.ui.run_button(label="Run this limit test")
    mo.vstack(
        [
            case_name,
            mo.hstack(
                [threshold_on, threshold, overlap, run],
                justify="start",
                gap=1,
                wrap=True,
            ),
        ]
    )
    return case_name, overlap, run, threshold, threshold_on


@app.cell
def _(case_name):
    case = CASES[case_name.value]
    preview_labels = labels_to_block(case["labels"])
    preview_model = case.get("model") or ", ".join(case.get("models", []))
    preview_text = case["text"]
    preview_clip = preview_text[:1200] + ("…" if len(preview_text) > 1200 else "")
    mo.md(
        f"""
    **{case_name.value}** · kind `{case['kind']}` · model `{preview_model}`

    {case['blurb']}

    ```
    {preview_clip}
    ```

    `{len(preview_text)}` chars · labels:

    ```
    {preview_labels}
    ```
    """
    )
    return (case,)


@app.function
def stat_row(items: list[tuple[str, str]]) -> mo.Html:
    pills = "".join(
        f"<div class='statpill'><div class='k'>{html.escape(k)}</div>"
        f"<div class='v'>{html.escape(v)}</div></div>"
        for k, v in items
    )
    return mo.Html(f"<div class='statrow'>{pills}</div>")


@app.function
def confidence_chart(rows: list[dict[str, Any]], title: str) -> Any:
    data = [
        {
            "label": row["label"],
            "text": row["text"][:48],
            "confidence": float(row["confidence"])
            if row.get("confidence") is not None
            else None,
        }
        for row in rows
        if row.get("confidence") is not None
    ]
    if not data:
        return mo.md("_No confidence scores in this payload._")
    frame = pd.DataFrame(data)
    chart = (
        alt.Chart(frame)
        .mark_bar()
        .encode(
            x=alt.X("confidence:Q", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("text:N", sort="-x", title=None),
            color=alt.Color("label:N", legend=alt.Legend(title="label")),
            tooltip=["label", "text", "confidence"],
        )
        .properties(title=title, height=max(120, 18 * len(frame)))
    )
    return chart


@app.function
def timing_pills(timing: dict[str, Any]) -> list[tuple[str, str]]:
    pills: list[tuple[str, str]] = []
    load_s = timing.get("load_s")
    infer_s = timing.get("infer_s")
    wait_s = timing.get("wait_s")
    if wait_s is not None:
        pills.append(("server wait", f"{wait_s:.1f}s"))
    if load_s is not None:
        pills.append(("load", f"{load_s:.1f}s"))
    if infer_s is not None:
        pills.append(("infer", f"{infer_s:.2f}s"))
    if timing.get("cold"):
        pills.append(("start", "cold"))
    if timing.get("slow"):
        pills.append(("start", "SLOW"))
    return pills


@app.function
def render_single(result: dict[str, Any], heading: str | None = None) -> Any:
    rows = result["rows"]
    labels = sorted({str(row["label"]) for row in rows})
    bits: list[Any] = []
    if heading:
        bits.append(mo.md(f"### {heading}"))
    bits.append(
        stat_row(
            [
                ("model", str(result["model"])),
                ("spans", str(len(rows))),
                ("labels hit", str(len(labels))),
                ("client wait", f"{result['elapsed']:.1f}s"),
                *timing_pills(result.get("timing") or {}),
            ]
        )
    )
    timing = result.get("timing") or {}
    if timing.get("slow"):
        bits.append(
            mo.md(
                f"**slow start** load `{timing.get('load_s')}s` · infer "
                f"`{timing.get('infer_s')}s` · server wait `{timing.get('wait_s')}s` "
                f"· cold `{timing.get('cold')}`. Load >= 30s or wait >= 60s is degraded."
            ).callout(kind="warn")
        )
    bits.append(mo.Html(legend_html(labels)))
    bits.append(mo.Html(highlight_html(result["text"], rows)))
    if rows:
        table = pd.DataFrame(rows)
        if "confidence" in table.columns:
            table = table.sort_values(
                by=["confidence", "start"],
                ascending=[False, True],
                na_position="last",
            )
        bits.append(mo.ui.table(table, pagination=True, page_size=25))
        bits.append(confidence_chart(rows, f"{result['model']} confidence"))
    else:
        bits.append(mo.md("_No entities. Lower the threshold or loosen labels._"))
    bits.append(
        mo.accordion(
            {
                "raw JSON": mo.md(
                    f"```json\n{json.dumps(result['payload'], indent=2)[:8000]}\n```"
                )
            }
        )
    )
    return mo.vstack(bits)


@app.function
def render_compare(results: list[dict[str, Any]]) -> Any:
    columns = []
    summary_rows = []
    for result in results:
        rows = result["rows"]
        labels = sorted({str(row["label"]) for row in rows})
        columns.append(
            mo.vstack(
                [
                    mo.md(f"### `{result['model']}`"),
                    stat_row(
                        [
                            ("spans", str(len(rows))),
                            ("types", str(len(labels))),
                            ("client wait", f"{result['elapsed']:.1f}s"),
                            *timing_pills(result.get("timing") or {}),
                        ]
                    ),
                    mo.Html(legend_html(labels)),
                    mo.Html(highlight_html(result["text"], rows)),
                ]
            )
        )
        for row in rows:
            summary_rows.append(
                {
                    "model": result["model"],
                    "label": row["label"],
                    "text": row["text"],
                    "confidence": row.get("confidence"),
                    "start": row.get("start"),
                    "end": row.get("end"),
                }
            )
    stack: list[Any] = [mo.hstack(columns, justify="space-between", gap=1, wrap=True)]
    if summary_rows:
        frame = pd.DataFrame(summary_rows)
        counts = (
            frame.groupby(["model", "label"], dropna=False)
            .size()
            .reset_index(name="count")
        )
        chart = (
            alt.Chart(counts)
            .mark_bar()
            .encode(
                x=alt.X("model:N"),
                y=alt.Y("count:Q"),
                color=alt.Color("label:N"),
                tooltip=["model", "label", "count"],
            )
            .properties(title="span counts by model × label", height=280)
        )
        stack.append(chart)
        stack.append(mo.ui.table(frame, pagination=True, page_size=30))
    return mo.vstack(stack)


@app.cell
def _(
    api_key,
    case,
    case_name,
    overlap,
    run,
    threshold,
    threshold_on,
    timeout,
    url,
):
    mo.stop(
        not run.value,
        mo.md("Pick a limit test and hit **Run this limit test**.").callout(kind="neutral"),
    )
    policy = None if overlap.value.startswith("(") else overlap.value
    thresh = float(threshold.value) if threshold_on.value else None
    kind = case["kind"]
    try:
        if kind == "compare":
            specs = [
                {
                    "model": name,
                    "text": case["text"],
                    "labels": case["labels"],
                    "threshold": thresh,
                    "overlap_policy": policy,
                }
                for name in MODELS
            ]
            compare_results = extract_many(
                specs,
                url=url.value,
                api_key=api_key.value,
                timeout=float(timeout.value),
            )
            out = mo.vstack(
                [
                    mo.md(f"## {case_name.value}"),
                    render_compare(compare_results),
                ]
            )
        elif kind == "compare_pair":
            specs = [
                {
                    "model": name,
                    "text": case["text"],
                    "labels": case["labels"],
                    "threshold": thresh,
                    "overlap_policy": policy,
                }
                for name in case["models"]
            ]
            pair_results = extract_many(
                specs,
                url=url.value,
                api_key=api_key.value,
                timeout=float(timeout.value),
            )
            out = mo.vstack(
                [
                    mo.md(f"## {case_name.value}"),
                    render_compare(pair_results),
                ]
            )
        elif kind == "overlap":
            specs = [
                {
                    "model": case["model"],
                    "text": case["text"],
                    "labels": case["labels"],
                    "threshold": thresh,
                    "overlap_policy": name,
                }
                for name in OVERLAP_POLICIES
            ]
            overlap_results = extract_many(
                specs,
                url=url.value,
                api_key=api_key.value,
                timeout=float(timeout.value),
            )
            panels = []
            for item, name in zip(overlap_results, OVERLAP_POLICIES, strict=True):
                item = dict(item)
                item["model"] = f"{item['model']} / {name}"
                panels.append(item)
            out = mo.vstack(
                [
                    mo.md(f"## {case_name.value}"),
                    mo.md(
                        "Same tokens, five decoders. `allow` should be noisy; `flat` "
                        "is the GLiNER2.5 default (weighted interval scheduling); "
                        "`longest` keeps `New York City Police Department` and drops "
                        "`New York`."
                    ),
                    render_compare(panels),
                ]
            )
        else:
            single = extract_entities(
                url=url.value,
                api_key=api_key.value,
                model=case["model"],
                text=case["text"],
                labels=case["labels"],
                threshold=thresh,
                overlap_policy=policy,
                timeout=float(timeout.value),
            )
            out = mo.vstack(
                [
                    mo.md(f"## {case_name.value}"),
                    render_single(single),
                ]
            )
    except Exception as exc:
        out = mo.md(f"**extract failed:** `{type(exc).__name__}: {exc}`").callout(
            kind="danger"
        )
    out
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Playground

    Bring your own text. Labels are one per line. `name: description` turns the line
    into a described label (the thing that makes `ejection_fraction` or `us_ssn` work).
    Mix all three models if you want a bake-off.
    """)
    return


@app.cell
def _():
    play_text = mo.ui.text_area(
        value=PACKED_NEWS,
        label="text",
        rows=10,
        full_width=True,
    )
    play_labels = mo.ui.text_area(
        value="company: A legally registered business or brand\nperson: A named human\nproduct: A commercial device, model, or service\nlocation: A city, campus, or country\nticker: An equity ticker symbol\nmoney: A currency amount\nlaw: A named statute or regulation",
        label="labels (one per line, optional `: description`)",
        rows=10,
        full_width=True,
    )
    play_models = mo.ui.multiselect(
        options=list(MODELS),
        value=["base"],
        label="models",
    )
    play_threshold_on = mo.ui.checkbox(value=False, label="Override threshold")
    play_threshold = mo.ui.slider(
        start=0.0, stop=1.0, value=0.4, step=0.05, label="threshold", show_value=True
    )
    play_overlap = mo.ui.dropdown(
        options=["(checkpoint default)", *OVERLAP_POLICIES],
        value="(checkpoint default)",
        label="overlap_policy",
    )
    play_run = mo.ui.run_button(label="Extract playground")
    mo.vstack(
        [
            play_text,
            play_labels,
            mo.hstack(
                [play_models, play_threshold_on, play_threshold, play_overlap, play_run],
                justify="start",
                gap=1,
                wrap=True,
            ),
        ]
    )
    return (
        play_labels,
        play_models,
        play_overlap,
        play_run,
        play_text,
        play_threshold,
        play_threshold_on,
    )


@app.cell
def _(
    api_key,
    play_labels,
    play_models,
    play_overlap,
    play_run,
    play_text,
    play_threshold,
    play_threshold_on,
    timeout,
    url,
):
    mo.stop(not play_run.value, mo.md("Edit the playground and extract."))
    parsed = parse_labels(play_labels.value)
    mo.stop(not parsed, mo.md("Add at least one label.").callout(kind="warn"))
    mo.stop(not play_models.value, mo.md("Pick at least one model.").callout(kind="warn"))
    mo.stop(not play_text.value.strip(), mo.md("Text is empty.").callout(kind="warn"))
    play_policy = None if play_overlap.value.startswith("(") else play_overlap.value
    play_thresh = float(play_threshold.value) if play_threshold_on.value else None
    try:
        play_specs = [
            {
                "model": name,
                "text": play_text.value,
                "labels": parsed,
                "threshold": play_thresh,
                "overlap_policy": play_policy,
            }
            for name in play_models.value
        ]
        play_results = extract_many(
            play_specs,
            url=url.value,
            api_key=api_key.value,
            timeout=float(timeout.value),
        )
        play_out = (
            render_compare(play_results)
            if len(play_results) > 1
            else render_single(play_results[0], heading="playground")
        )
    except Exception as exc:
        play_out = mo.md(f"**playground failed:** `{exc}`").callout(kind="danger")
    play_out
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## What this is actually stressing

    - **Zero-shot schema, not a gazetteer.** Labels like `us_ssn`, `ejection_fraction`,
      `forum`, `social handle` were never a closed class. Descriptions are the only
      hint the encoder gets.
    - **Overlap policy is a decoder, not a model.** `New York City Police Department`
      contains four legitimate spans. `flat` (default) will throw some of them away
      on purpose.
    - **Language is a checkpoint, not a flag.** `small`/`base` will happily emit
      confident nonsense on Arabic. That comparison is included so you see it.
    - **Window vs API cap.** 50k characters is a FastAPI `Field` constraint.
      GLiNER2.5 still encodes 4,096 tokens. The earnings call is sized to that wall.
    - **Cold start is part of the product.** `min_containers=0`, `scaledown_window=60`.
      Ignite once, then the catalog is snappy.

    ```bash
    uvx marimo edit --sandbox demo/gliner_limits.py
    uvx marimo run --sandbox demo/gliner_limits.py
    ```
    """)
    return


if __name__ == "__main__":
    app.run()
