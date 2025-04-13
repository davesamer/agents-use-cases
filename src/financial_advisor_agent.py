from textwrap import dedent

from agno.agent import Agent
from agno.models.azure import AzureOpenAI
from agno.tools.yfinance import YFinanceTools
from dotenv import load_dotenv

load_dotenv("configs/.env")


finance_agent = Agent(
    model=AzureOpenAI(id="gpt-35-turbo"),
    tools=[
        YFinanceTools(
            stock_price=True,
            analyst_recommendations=True,
            stock_fundamentals=True,
            historical_prices=True,
            company_info=True,
            company_news=True,
        )
    ],
    instructions=dedent("""\
        > 🇩🇪🇦🇹🇨🇭 Du bist ein erfahrener Finanzanalyst mit tiefgreifender Expertise in
        den Aktienmärkten der DACH-Region! 🧠📊
        > Deine Aufgabe ist es, professionelle Aktienanalysen zu erstellen – basierend auf
        aktuellen Daten von Yahoo Finance – mit Fokus auf Unternehmen aus Deutschland,
        Österreich und der Schweiz.

        ---

        ### 🔍 Analyse-Schritte:

        #### 1. Marktüberblick 🇩🇪
        - Aktueller Aktienkurs (in Landeswährung)
        - 52-Wochen-Hoch und -Tief 📈📉
        - Börsenplatz und Tickersymbol

        #### 2. Finanzielle Kennzahlen 💼
        - Wichtige Metriken:
        - Kurs-Gewinn-Verhältnis (KGV)
        - Marktkapitalisierung
        - Gewinn pro Aktie (EPS)
        - Dividendenrendite (falls verfügbar)
        - Umsatz- und Gewinnwachstum im Jahresvergleich
        - Vergleich mit Branchendurchschnitt in DACH/EU

        #### 3. Expertenmeinungen 📣
        - Analysten-Empfehlungen (Kaufen/Halten/Verkaufen)
        - Kürzliche Änderungen von Ratings
        - Konsens über Kursziele (Spanne)

        #### 4. Marktumfeld & Wettbewerb 🏭
        - Branchentrends und wirtschaftliche Einordnung
        - Wettbewerbsanalyse (lokale und europäische Konkurrenten)
        - Einfluss makroökonomischer und regulatorischer Faktoren
        - Marktstimmung (z. B. RSI, Handelsvolumen, Nachrichtenlage)

        ---

        ### 📄 Stil der Berichterstattung:
        - Beginne mit einer **Executive Summary**
        - Verwende **klare Abschnittsüberschriften**
        - Stelle Daten in **übersichtlichen Tabellen** dar
        - Nutze **Trend-Emojis** (📈 positiv, 📉 negativ, ⚠️ Risiko)
        - Hebe wichtige Erkenntnisse in **Stichpunkten** hervor
        - **Vergleiche** Unternehmenskennzahlen mit Branchenwerten
        - Erkläre **Fachbegriffe** knapp und verständlich
        - Schließe mit einem **Ausblick auf die kommenden 6–12 Monate**

        ---

        ### ⚠️ Risikohinweise:
        - Weisen auf **marktbezogene Risiken** hin (Konjunktur, Geopolitik)
        - Berücksichtige **regulatorische Entwicklungen** in EU/DACH
        - Erwähne mögliche **Währungsschwankungen** (v. a. EUR/CHF)
        - Hebe **wirtschaftliche Unsicherheiten** hervor (z. B. EZB-Politik, Inflation)

    """),
    add_datetime_to_instructions=True,
    show_tool_calls=True,
    markdown=True,
)

# Example usage with detailed market analysis request



'''

finance_agent.print_response(
    dedent("""\
    "Führe eine kombinierte technische und fundamentale Analyse zur Andritz AG durch.
    Besteht aktuell ein günstiger Einstiegspunkt auf Basis historischer RSI-Daten und P/E-Entwicklung?"
    """),  # noqa: E501
    stream=True,
)

finance_agent.print_response(
    dedent("""\
    "Gibt es bei österreichischen Small- oder Mid-Caps
    (z. B. Marinomed, Frequentis) positive Momentum-Signale oder Analysten-Upgrades in den letzten 30 Tagen?"
    """),  # noqa: E501
    stream=True,
)

finance_agent.print_response(
    dedent("""\
    "Vergleiche Verbund AG, EVN und E.ON hinsichtlich Finanzkennzahlen, Marktsentiment und Analystenbewertungen.
    Welche Firma zeigt im aktuellen Energiemarkt (DACH) die robusteste Entwicklung?"
    """),  # noqa: E501
    stream=True,
)


finance_agent.print_response(
    dedent("""\
    "Analysiere die aktuelle Bewertung und Marktstellung der Palfinger AG (Wiener Börse).
    Gibt es Abweichungen zum Branchendurchschnitt im Maschinenbau-Sektor innerhalb der DACH-Region?
    Welche Analystenmeinungen liegen vor?"
    """),  # noqa: E501
    stream=True,
)
'''
