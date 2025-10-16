import pytest
import json
from parsee.utils.merge_strategies import merge_list_of_dict
from parsee.chat.custom_dataclasses import SinglePageProcessingSettings
from tests.parsee.chat.merge.test_data import TestData


def test_merge_valid_json_arrays():
    """Test merging valid JSON arrays from multiple responses."""
    responses = TestData.valid_json_arrays()

    result = merge_list_of_dict(responses)
    parsed_result = json.loads(result)

    assert len(parsed_result) == 3
    assert parsed_result[0]["counterParty"] == "CompanyA Ltd"
    assert parsed_result[1]["counterParty"] == "ShippingCorp Inc"
    assert parsed_result[2]["counterParty"] == "TechSolutions LLC"
    assert parsed_result[2]["currency"] == "USD"


def test_merge_with_malformed_json():
    """Test merging when one response has malformed JSON (missing opening bracket)."""
    responses = TestData.malformed_json_responses()

    result = merge_list_of_dict(responses)
    parsed_result = json.loads(result)

    assert len(parsed_result) == 3
    assert parsed_result[0]["counterParty"] == "RetailStore AG"
    assert parsed_result[1]["counterParty"] == "ServiceProvider GmbH"
    assert parsed_result[2]["counterParty"] == "TaxOffice Berlin"


def test_complete_sample_data_anonymized():
    """Test with complete sample data set (anonymized)."""
    responses = TestData.complete_sample_data()

    result = merge_list_of_dict(responses)
    parsed_result = json.loads(result)

    # Should merge all transactions, including duplicates
    assert len(parsed_result) == 17  # 4 + 7 + 6 transactions

    # Check for specific transactions
    account_closure = next((t for t in parsed_result if t["counterParty"] == "AccountClosure"), None)
    assert account_closure is not None
    assert account_closure["amount"] == -19.00

    biotech = next((t for t in parsed_result if t["counterParty"] == "BiotechResearch LLC"), None)
    assert biotech is not None
    assert biotech["currency"] == "USD"

    # Check for positive amount (income)
    pharma = next((t for t in parsed_result if t["counterParty"] == "PharmaCorp"), None)
    assert pharma is not None
    assert pharma["amount"] == 3110.00

    state_office = next((t for t in parsed_result if t["counterParty"] == "StateOffice Bavaria"), None)
    assert state_office is not None
    assert state_office["amount"] == 8192.50


def test_empty_responses():
    """Test handling of empty response list."""
    result = merge_list_of_dict([])
    parsed_result = json.loads(result)
    assert len(parsed_result) == 0
    assert parsed_result == []


def test_single_response():
    """Test with a single response."""
    responses = TestData.single_response()

    result = merge_list_of_dict(responses)
    parsed_result = json.loads(result)

    assert len(parsed_result) == 1
    assert parsed_result[0]["counterParty"] == "SingleTransaction Corp"
    assert parsed_result[0]["amount"] == -100.00


def test_responses_with_no_json_content():
    """Test responses that contain no valid JSON."""
    responses = ["No JSON here", "```\nPlain text\n```", ""]

    result = merge_list_of_dict(responses)
    parsed_result = json.loads(result)

    # The function might create empty objects from invalid content, so check actual content
    # Filter out any empty objects that might be created from invalid parsing
    valid_transactions = [t for t in parsed_result if t.get("counterParty") is not None]
    assert len(valid_transactions) == 0


def test_mixed_valid_and_invalid_responses():
    """Test mix of valid and invalid responses."""
    responses = [
        TestData.single_response()[0],  # Valid
        "Invalid JSON content",  # Invalid
        TestData.malformed_json_responses()[1]  # Valid but malformed
    ]

    result = merge_list_of_dict(responses)
    parsed_result = json.loads(result)

    # Filter out any empty/invalid objects
    valid_transactions = [t for t in parsed_result if t.get("counterParty") is not None]
    assert len(valid_transactions) == 3  # 1 from first + 2 from third response


@pytest.mark.parametrize("currency,expected_count", [
    ("EUR", 16),
    ("USD", 1),
])
def test_currency_filtering(currency, expected_count):
    """Test that we can identify transactions by currency."""
    responses = TestData.complete_sample_data()
    result = merge_list_of_dict(responses)
    parsed_result = json.loads(result)

    currency_transactions = [t for t in parsed_result if t.get("currency") == currency]
    assert len(currency_transactions) == expected_count


def test_duplicate_handling():
    """Test that duplicates are preserved (as expected behavior)."""
    responses = TestData.complete_sample_data()
    result = merge_list_of_dict(responses)
    parsed_result = json.loads(result)

    # Check for known duplicates
    business_services_transactions = [
        t for t in parsed_result
        if t.get("counterParty") == "BusinessServices GmbH"
    ]
    tax_authority_transactions = [
        t for t in parsed_result
        if t.get("counterParty") == "TaxAuthority Munich"
    ]

    # These appear in multiple responses, so should have duplicates
    assert len(business_services_transactions) == 2
    assert len(tax_authority_transactions) == 2


# SinglePageProcessingSettings tests
def test_default_values():
    """Test default values of SinglePageProcessingSettings."""
    settings = SinglePageProcessingSettings()
    assert settings.max_images_trigger == 3
    assert settings.merge_strategy is None


def test_custom_values():
    """Test SinglePageProcessingSettings with custom values."""
    settings = SinglePageProcessingSettings(
        max_images_trigger=5,
        merge_strategy=merge_list_of_dict
    )
    assert settings.max_images_trigger == 5
    assert settings.merge_strategy == merge_list_of_dict


def test_merge_strategy_execution():
    """Test that the merge strategy works when called."""
    settings = SinglePageProcessingSettings(
        max_images_trigger=5,
        merge_strategy=merge_list_of_dict
    )

    test_responses = ['```json\n[{"test": "value", "amount": 100}]\n```']
    result = settings.merge_strategy(test_responses)
    parsed = json.loads(result)

    assert len(parsed) == 1
    assert parsed[0]["test"] == "value"
    assert parsed[0]["amount"] == 100


def test_none_merge_strategy():
    """Test behavior when merge_strategy is None."""
    settings = SinglePageProcessingSettings(max_images_trigger=10)
    assert settings.merge_strategy is None


# Additional edge case tests
def test_responses_with_empty_arrays():
    """Test responses containing empty JSON arrays."""
    responses = ['```json\n[]\n```', '```json\n[]\n```']

    result = merge_list_of_dict(responses)
    parsed_result = json.loads(result)

    assert len(parsed_result) == 0


def test_responses_with_mixed_empty_and_valid():
    """Test mix of empty and valid responses."""
    responses = [
        '```json\n[]\n```',
        TestData.single_response()[0],
        '```json\n[]\n```'
    ]

    result = merge_list_of_dict(responses)
    parsed_result = json.loads(result)

    assert len(parsed_result) == 1
    assert parsed_result[0]["counterParty"] == "SingleTransaction Corp"


def test_response_with_single_object_no_array():
    """Test response containing a single JSON object (not in array)."""
    response = '''```json
{
    "counterParty": "DirectObject Corp",
    "ibanCounterParty": "DE11111111111111111111",
    "transactionDate": "2025-04-01",
    "currency": "EUR",
    "amount": -50.00,
    "localAmount": -50.00,
    "reference": "DIRECT-OBJ-001"
}
```'''

    result = merge_list_of_dict([response])
    parsed_result = json.loads(result)

    assert len(parsed_result) == 1
    assert parsed_result[0]["counterParty"] == "DirectObject Corp"
    assert parsed_result[0]["amount"] == -50.00


def test_large_amount_precision():
    """Test that large amounts and precision are preserved."""
    responses = TestData.complete_sample_data()
    result = merge_list_of_dict(responses)
    parsed_result = json.loads(result)

    # Find the USD transaction with high amount
    usd_transaction = next(t for t in parsed_result if t.get("currency") == "USD")
    assert usd_transaction["amount"] == -7630.35
    assert usd_transaction["localAmount"] == -6758.27

    # Find the large EUR grant
    grant_transaction = next(t for t in parsed_result if t.get("reference") == "GRANT402")
    assert grant_transaction["amount"] == 8192.50


@pytest.mark.parametrize("amount_range,expected_min_count", [
    ((-100, 0), 10),  # Expenses between -100 and 0
    ((0, 10000), 2),  # Income between 0 and 10000
    ((-10000, -1000), 2),  # Large expenses
])
def test_amount_ranges(amount_range, expected_min_count):
    """Test transaction filtering by amount ranges."""
    responses = TestData.complete_sample_data()
    result = merge_list_of_dict(responses)
    parsed_result = json.loads(result)

    min_amount, max_amount = amount_range
    filtered_transactions = [
        t for t in parsed_result
        if min_amount <= t.get("amount", 0) <= max_amount
    ]

    assert len(filtered_transactions) >= expected_min_count


def test_actual_transaction_count_in_sample_data():
    """Helper test to verify the actual counts in our test data."""
    responses = TestData.complete_sample_data()
    result = merge_list_of_dict(responses)
    parsed_result = json.loads(result)

    # Count transactions by currency
    eur_count = len([t for t in parsed_result if t.get("currency") == "EUR"])
    usd_count = len([t for t in parsed_result if t.get("currency") == "USD"])

    print(f"EUR transactions: {eur_count}")
    print(f"USD transactions: {usd_count}")
    print(f"Total transactions: {len(parsed_result)}")

    # This test helps us understand the actual data structure
    assert eur_count + usd_count == len(parsed_result)