from typing import List


class TestData:
    """Centralized test data for merge function tests."""

    @staticmethod
    def valid_json_arrays() -> List[str]:
        """Return valid JSON array responses."""
        return [
            '```json\n[\n  {\n    "counterParty": "CompanyA Ltd",\n    "ibanCounterParty": "",\n    "transactionDate": "2025-04-30",\n    "currency": "EUR",\n    "amount": -19.00,\n    "localAmount": -19.00,\n    "reference": "Monthly closure 30.04.2025"\n  },\n  {\n    "counterParty": "ShippingCorp Inc",\n    "ibanCounterParty": "DE12345678901234567890",\n    "transactionDate": "2025-04-28",\n    "currency": "EUR",\n    "amount": -94.81,\n    "localAmount": -94.81,\n    "reference": "INV/SHP000341518"\n  }\n]\n```',
            '```json\n[\n  {\n    "counterParty": "TechSolutions LLC",\n    "ibanCounterParty": "",\n    "transactionDate": "2025-04-14",\n    "currency": "USD",\n    "amount": -7630.35,\n    "localAmount": -6758.27,\n    "reference": "TECH-ZAI010139"\n  }\n]\n```'
        ]

    @staticmethod
    def malformed_json_responses() -> List[str]:
        """Return responses with malformed JSON."""
        return [
            '```json\n[\n  {\n    "counterParty": "RetailStore AG",\n    "ibanCounterParty": "DE98765432109876543210",\n    "transactionDate": "2025-04-22",\n    "currency": "EUR",\n    "amount": -4.99,\n    "localAmount": -4.99,\n    "reference": "ORDER-1234563237"\n  }\n]\n```',
            '```json\n  {\n    "counterParty": "ServiceProvider GmbH",\n    "ibanCounterParty": "DE11223344556677889900",\n    "transactionDate": "2025-04-07",\n    "currency": "EUR",\n    "amount": -35.58,\n    "localAmount": -35.58,\n    "reference": "SRV-5323-5270"\n  },\n  {\n    "counterParty": "TaxOffice Berlin",\n    "ibanCounterParty": "DE99887766554433221100",\n    "transactionDate": "2025-04-07",\n    "currency": "EUR",\n    "amount": -1527.00,\n    "localAmount": -1527.00,\n    "reference": "TAX-652-02483"\n  }\n]\n```'
        ]

    @staticmethod
    def single_response() -> List[str]:
        """Return a single response."""
        return [
            '```json\n[\n  {\n    "counterParty": "SingleTransaction Corp",\n    "ibanCounterParty": "DE99999999999999999999",\n    "transactionDate": "2025-04-01",\n    "currency": "EUR",\n    "amount": -100.00,\n    "localAmount": -100.00,\n    "reference": "SINGLE-TRANS-001"\n  }\n]\n```'
        ]

    @staticmethod
    def complete_sample_data() -> List[str]:
        """Return the complete sample data set (anonymized)."""
        return [
            '```json\n[\n  {\n    "counterParty": "AccountClosure",\n    "ibanCounterParty": "",\n    "transactionDate": "2025-04-30",\n    "currency": "EUR",\n    "amount": -19.00,\n    "localAmount": -19.00,\n    "reference": "Account closure 30.04.2025"\n  },\n  {\n    "counterParty": "PostalService AG",\n    "ibanCounterParty": "DE12345678901234567801",\n    "transactionDate": "2025-04-28",\n    "currency": "EUR",\n    "amount": -94.81,\n    "localAmount": -94.81,\n    "reference": "INV/POST000341518"\n  },\n  {\n    "counterParty": "TelecomProvider Ltd",\n    "ibanCounterParty": "DE12345678901234567802",\n    "transactionDate": "2025-04-22",\n    "currency": "EUR",\n    "amount": -4.99,\n    "localAmount": -4.99,\n    "reference": "Bill-Nr.: 1234563237/8"\n  },\n  {\n    "counterParty": "TelecomProvider Ltd",\n    "ibanCounterParty": "DE12345678901234567802",\n    "transactionDate": "2025-04-16",\n    "currency": "EUR",\n    "amount": -6.99,\n    "localAmount": -6.99,\n    "reference": "Bill-Nr.: 1228647969/8"\n  }\n]\n```',
            '```json\n[\n  {\n    "counterParty": "BiotechResearch LLC",\n    "ibanCounterParty": "",\n    "transactionDate": "2025-04-14",\n    "currency": "USD",\n    "amount": -7630.35,\n    "localAmount": -6758.27,\n    "reference": "RESEARCH-010139"\n  },\n  {\n    "counterParty": "PostalService AG",\n    "ibanCounterParty": "DE12345678901234567803",\n    "transactionDate": "2025-04-10",\n    "currency": "EUR",\n    "amount": -0.95,\n    "localAmount": -0.95,\n    "reference": "REF 661147/260011"\n  },\n  {\n    "counterParty": "InternetProvider GmbH",\n    "ibanCounterParty": "DE12345678901234567804",\n    "transactionDate": "2025-04-10",\n    "currency": "EUR",\n    "amount": -29.94,\n    "localAmount": -29.94,\n    "reference": "BILL2025502658313"\n  },\n  {\n    "counterParty": "PharmaCorp",\n    "ibanCounterParty": "FR1234567890123456789012",\n    "transactionDate": "2025-04-08",\n    "currency": "EUR",\n    "amount": 3110.00,\n    "localAmount": 3110.00,\n    "reference": "INV-100047"\n  },\n  {\n    "counterParty": "ExpressDelivery GmbH",\n    "ibanCounterParty": "DE12345678901234567805",\n    "transactionDate": "2025-04-07",\n    "currency": "EUR",\n    "amount": -45.07,\n    "localAmount": -45.07,\n    "reference": "DELIVERY2505584343"\n  },\n  {\n    "counterParty": "BusinessServices GmbH",\n    "ibanCounterParty": "DE12345678901234567806",\n    "transactionDate": "2025-04-07",\n    "currency": "EUR",\n    "amount": -35.58,\n    "localAmount": -35.58,\n    "reference": "SRV.5337.5323.5270"\n  },\n  {\n    "counterParty": "TaxAuthority Munich",\n    "ibanCounterParty": "DE12345678901234567807",\n    "transactionDate": "2025-04-07",\n    "currency": "EUR",\n    "amount": -1527.00,\n    "localAmount": -1527.00,\n    "reference": "TAX/24/652/02483"\n  }\n]\n```',
            '```json\n  {\n    "counterParty": "BusinessServices GmbH",\n    "ibanCounterParty": "DE12345678901234567806",\n    "transactionDate": "2025-04-07",\n    "currency": "EUR",\n    "amount": -35.58,\n    "localAmount": -35.58,\n    "reference": "SRV.5337.5323.5270"\n  },\n  {\n    "counterParty": "TaxAuthority Munich",\n    "ibanCounterParty": "DE12345678901234567807",\n    "transactionDate": "2025-04-07",\n    "currency": "EUR",\n    "amount": -1527.00,\n    "localAmount": -1527.00,\n    "reference": "TAX/24/652/02483"\n  },\n  {\n    "counterParty": "ResearchInstitute GmbH",\n    "ibanCounterParty": "AT123456789012345678",\n    "transactionDate": "2025-04-04",\n    "currency": "EUR",\n    "amount": 690.00,\n    "localAmount": 690.00,\n    "reference": "INV-100049/03.02.2025"\n  },\n  {\n    "counterParty": "StateOffice Bavaria",\n    "ibanCounterParty": "DE12345678901234567808",\n    "transactionDate": "2025-04-02",\n    "currency": "EUR",\n    "amount": 8192.50,\n    "localAmount": 8192.50,\n    "reference": "GRANT402"\n  },\n  {\n    "counterParty": "FinanceBank",\n    "ibanCounterParty": "DE12345678901234567809",\n    "transactionDate": "2025-04-01",\n    "currency": "EUR",\n    "amount": -60.86,\n    "localAmount": -60.86,\n    "reference": "LOAN2022120692459R"\n  },\n  {\n    "counterParty": "InsuranceCorp SE",\n    "ibanCounterParty": "DE12345678901234567810",\n    "transactionDate": "2025-04-01",\n    "currency": "EUR",\n    "amount": -11.00,\n    "localAmount": -11.00,\n    "reference": "POLICY7224512711"\n  }\n]\n```'
        ]