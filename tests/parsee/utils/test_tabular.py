"""
Comprehensive CSV Parser Test Suite
Run with: pytest tabular_tests.py -v
"""

import pytest
import tempfile
import os
from typing import List, Dict
from unittest.mock import patch

# Assuming your parser module is imported here
from parsee.utils.tabular import parse_csv, csv_delimiter_simple
from parsee.extraction.extractor_elements import StructuredTable, DocumentType

# Test data samples - all anonymized and safe for public use
CSV_SAMPLES = {
    "basic_comma": '''Name,Age,City
John,30,New York
Jane,25,Boston''',

    "basic_semicolon": '''Name;Age;City
John;30;New York
Jane;25;Boston''',

    "quoted_with_commas": '''Name,"Full Address",Phone
John,"123 Main St, Apt 2B",555-1234
Jane,"456 Oak Ave, Suite 100",555-5678''',

    "mixed_quotes": '''Name,Address,Phone
"John Doe","123 Main St",555-1234
Jane Smith,"456 Oak Ave","555-5678"
"Bob Johnson",789 Pine St,555-9999''',

    "empty_fields": '''Name,Email,Phone,Notes
John,john@test.com,,
Jane,,555-1234,Important
Bob,bob@test.com,555-5678,''',

    "metadata_header": '''# Export from System XYZ
# Date: 2025-01-07
# Format: CSV
Name,Department,Salary
Alice,Engineering,80000
Bob,Marketing,70000''',

    "complex_marketplace_like": '''Report Header Information;;
Additional metadata here;;
More info about the data;;
"Column1","Column2","Column3","Column4"
"Data1","Data with, comma","More data","123.45"
"Data2","Another field","Different data","67.89"''',

    "german_numbers": '''Product;Price;Tax;Total
Laptop;"1.299,99";"247,00";"1.546,99"
Mouse;"29,50";"5,61";"35,11"
Keyboard;"89,99";"17,10";"107,09"''',

    "pipe_separated": '''ID|Name|Department|Salary
1001|Alice Johnson|Engineering|75000
1002|Bob Smith|Marketing|65000
1003|Carol Davis|HR|55000''',

    "tab_separated": '''Name\tAge\tCity\tCountry
John\t30\tNew York\tUSA
Jane\t25\tLondon\tUK
Bob\t35\tToronto\tCanada''',

    "irregular_rows": '''Name,Age,City,Country
John,30,New York,USA
Jane,25,London
Bob,35,Toronto,Canada,Extra Field
Alice,28''',

    "special_characters": '''Name,City,Description
José,São Paulo,"Café specialist"
François,Montréal,"Fromage expert"
张三,北京,"软件工程师"
Müller,München,"Bäcker & Konditor"''',

    "multiline_quotes": '''Product,Description,Price
"Widget A","This is a long
description that spans
multiple lines",29.99
"Widget B","Single line description",19.99''',

    "double_quotes_escaped": '''Name,Message,Status
John,"""Hello World""",Active
Jane,"Say ""Hi"" to everyone",Active
Bob,"""Special"" characters",Inactive''',

    "anonymized_amazon_style": '''Including transactions for E-commerce Platform, Fulfillment and Webstore;;
All amounts in EUR unless otherwise noted;;
Definitions:;;
Collected Tax: Including tax collected from buyers for product sales, shipping and gift wrapping.;;
Platform fees: Includes variable closing fees and promotional cost reimbursements.;;
Other transaction fees: Including settlement adjustments for shipping and handling fees.;;
"Other: Contains amounts for transactions not related to an order. Details in Type and Description columns.";;
"Date/Time,""Settlement ID"",""Type"",""Order ID"",""SKU"",""Description"",""Qty"",""Marketplace"",""Fulfillment"",""City"",""State"",""ZIP"",""Tax Model"",""Sales"",""Tax"",""Ship Credit"",""Ship Tax"",""Gift Credit"",""Gift Tax"",""Promo"",""Promo Tax"",""Withheld Tax"",""Fees"",""FBA Fees"",""Other Fees"",""Other"",""Total""";;
"01.08.2025 10:30:00 UTC,""MOCK123456"",""Order"",""111-2222333-4444555"",""TEST-SKU-001"",""Mock Product A - Sample Item for Testing CSV Parser - Contains Commas, Quotes and Special Chars"",""1"",""test-marketplace.com"",""FBA"",""TestTown"","",""12345"","",""19,99"",""3,80"",""0"",""0"",""0"",""0"",""0"",""0"",""0"",""-2,15"",""-6,25"",""0"",""0"",""15,39""";;
"01.08.2025 11:45:00 UTC,""MOCK123456"",""Order"",""222-3333444-5555666"",""TEST-SKU-002"",""Mock Product B - Multi-Unit Sample - Test Description for Quantity Processing"",""3"",""test-marketplace.com"",""FBA"",""SampleCity"",""TestState"",""67890"","",""45,00"",""8,55"",""0"",""0"",""0"",""0"",""0"",""0"",""0"",""-4,95"",""-18,75"",""0"",""0"",""29,85""";;''',

    "financial_data": '''Financial Report - Q4 2024;;
Generated: 2025-01-07;;
Currency: USD;;
;;
"Date","Transaction ID","Type","Amount","Fee","Net"
"2024-12-01","TXN001","Sale","100.00","-2.50","97.50"
"2024-12-02","TXN002","Refund","-25.00","0.00","-25.00"
"2024-12-03","TXN003","Sale","75.50","-1.89","73.61"''',
}


class TestCSVParser:
    """Main CSV Parser test class."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Set up and clean up temporary files for each test."""
        self.temp_files = []
        yield
        # Clean up temporary files
        for temp_file in self.temp_files:
            try:
                os.unlink(temp_file)
            except FileNotFoundError:
                pass

    def create_temp_csv(self, content: str, encoding: str = 'utf-8') -> str:
        """Create a temporary CSV file with given content."""
        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', encoding=encoding)
        temp_file.write(content)
        temp_file.close()
        self.temp_files.append(temp_file.name)
        return temp_file.name

    def test_amazon_csv_format(self):
        """Test parsing the problematic Amazon-style CSV format with anonymized data."""
        temp_file = self.create_temp_csv(CSV_SAMPLES["anonymized_amazon_style"])
        result = parse_csv(temp_file)

        # Verify basic structure
        assert isinstance(result, StructuredTable)
        assert result.source.source_type == DocumentType.TABULAR

        # Should have header + 2 data rows = 3 rows total (after skipping metadata)
        assert len(result.rows) >= 3

        # Check that we have the expected number of columns (26 based on the header)
        if len(result.rows) > 0:
            # Find the header row (should have Date/Time as first column)
            header_row = None
            for i, row in enumerate(result.rows):
                if row.values[0].val.startswith("Date/Time"):
                    header_row = row
                    break

            assert header_row is not None, "Header row should be found"
            assert len(header_row.values) == 26

        # Verify some key data points from data rows
        data_rows = [row for row in result.rows if len(row.values) >= 5 and row.values[0].val.startswith("01.08.2025")]
        assert len(data_rows) >= 2

        # Check first data row
        first_data_row = data_rows[0]
        assert "01.08.2025" in first_data_row.values[0].val
        assert first_data_row.values[4].val == "TEST-SKU-001"

    def test_simple_comma_csv(self):
        """Test parsing a simple comma-separated CSV."""
        temp_file = self.create_temp_csv(CSV_SAMPLES["basic_comma"])
        result = parse_csv(temp_file)

        assert len(result.rows) == 3  # Header + 2 data rows
        assert len(result.rows[0].values) == 3  # 3 columns

        # Check header
        assert result.rows[0].values[0].val == "Name"
        assert result.rows[0].values[1].val == "Age"

        # Check data
        assert result.rows[1].values[0].val == "John"
        assert result.rows[1].values[1].val == "30"

    def test_semicolon_csv(self):
        """Test parsing semicolon-separated CSV."""
        temp_file = self.create_temp_csv(CSV_SAMPLES["basic_semicolon"])
        result = parse_csv(temp_file)

        assert len(result.rows) == 3
        assert len(result.rows[0].values) == 3

        # Verify delimiter detection worked
        assert result.rows[0].values[0].val == "Name"
        assert result.rows[1].values[1].val == "30"

    def test_quoted_fields_csv(self):
        """Test CSV with quoted fields containing commas."""
        temp_file = self.create_temp_csv(CSV_SAMPLES["quoted_with_commas"])
        result = parse_csv(temp_file)

        assert len(result.rows) == 3

        # Check that quoted fields with commas are handled correctly
        assert result.rows[1].values[1].val == "123 Main St, Apt 2B"
        assert result.rows[2].values[1].val == "456 Oak Ave, Suite 100"

    def test_empty_fields_csv(self):
        """Test CSV with empty fields."""
        temp_file = self.create_temp_csv(CSV_SAMPLES["empty_fields"])
        result = parse_csv(temp_file)

        assert len(result.rows) == 4

        # Check empty fields are handled
        assert result.rows[1].values[2].val == ""  # Empty phone
        assert result.rows[2].values[1].val == ""  # Empty email
        assert result.rows[2].values[3].val == ""  # Empty address

    @pytest.mark.parametrize("sample_name,expected_delimiter", [
        ("pipe_separated", "|"),
        ("tab_separated", "\t"),
    ])
    def test_different_delimiters(self, sample_name, expected_delimiter):
        """Test various delimiter types."""
        temp_file = self.create_temp_csv(CSV_SAMPLES[sample_name])
        result = parse_csv(temp_file)

        assert len(result.rows) == 4
        if sample_name == "pipe_separated":
            assert result.rows[1].values[0].val == "1001"
            assert result.rows[1].values[1].val == "Alice Johnson"
        elif sample_name == "tab_separated":
            assert result.rows[1].values[2].val == "New York"

    def test_csv_with_metadata_header(self):
        """Test CSV with metadata at the top."""
        temp_file = self.create_temp_csv(CSV_SAMPLES["metadata_header"])
        result = parse_csv(temp_file)

        # Should skip comment lines and find the actual data
        assert len(result.rows) >= 3

        # Find the row with "Name" as first column (header row)
        header_found = any(row.values[0].val == "Name" for row in result.rows)
        assert header_found, "Header row should be found"

    def test_mixed_encodings(self):
        """Test CSV files with different encodings."""
        temp_file = self.create_temp_csv(CSV_SAMPLES["special_characters"], encoding='utf-8')
        result = parse_csv(temp_file)

        assert len(result.rows) == 5
        assert result.rows[1].values[0].val == "José"

    def test_max_rows_parameter(self):
        """Test the max_rows parameter."""
        csv_content = '''Name,Age
Alice,25
Bob,30
Carol,35
Dave,40
Eve,45'''

        temp_file = self.create_temp_csv(csv_content)
        result = parse_csv(temp_file, max_rows=3)

        # Should only read 3 rows (header + 2 data rows)
        assert len(result.rows) == 3

    def test_irregular_csv(self):
        """Test CSV with irregular row lengths."""
        temp_file = self.create_temp_csv(CSV_SAMPLES["irregular_rows"])
        result = parse_csv(temp_file)

        # Should handle rows with different lengths
        assert len(result.rows) == 5

        # All rows should be padded to the same length (max columns)
        max_cols = max(len(row.values) for row in result.rows)
        for row in result.rows:
            assert len(row.values) == max_cols

    def test_empty_file(self):
        """Test parsing an empty CSV file."""
        temp_file = self.create_temp_csv("")
        result = parse_csv(temp_file)

        assert len(result.rows) == 0

    def test_single_column_csv(self):
        """Test CSV with only one column."""
        csv_content = '''Names
Alice
Bob
Carol'''

        temp_file = self.create_temp_csv(csv_content)
        result = parse_csv(temp_file)

        assert len(result.rows) == 4
        assert len(result.rows[0].values) == 1
        assert result.rows[0].values[0].val == "Names"
        assert result.rows[1].values[0].val == "Alice"

    def test_csv_with_newlines_in_quotes(self):
        """Test CSV with newlines inside quoted fields."""
        temp_file = self.create_temp_csv(CSV_SAMPLES["multiline_quotes"])
        result = parse_csv(temp_file)

        # Should handle multiline quoted fields
        assert len(result.rows) == 3
        # The description should contain the newlines
        description = result.rows[1].values[1].val
        assert "long" in description
        assert "multiple" in description

    def test_double_quotes_escaped(self):
        """Test CSV with escaped double quotes."""
        temp_file = self.create_temp_csv(CSV_SAMPLES["double_quotes_escaped"])
        result = parse_csv(temp_file)

        assert len(result.rows) == 4
        # Check that escaped quotes are handled
        assert "Hello World" in result.rows[1].values[1].val
        assert "Hi" in result.rows[2].values[1].val

    def test_financial_data_csv(self):
        """Test with financial data similar to marketplace format."""
        temp_file = self.create_temp_csv(CSV_SAMPLES["financial_data"])
        result = parse_csv(temp_file)

        # Should skip metadata and parse data correctly
        assert len(result.rows) > 3

        # Find the data rows
        data_rows = []
        for row in result.rows:
            if len(row.values) >= 6 and row.values[0].val.startswith("2024"):
                data_rows.append(row)

        assert len(data_rows) == 3
        assert data_rows[0].values[1].val == "TXN001"

    def test_german_number_format(self):
        """Test CSV with German number formatting (comma as decimal separator)."""
        temp_file = self.create_temp_csv(CSV_SAMPLES["german_numbers"])
        result = parse_csv(temp_file)

        assert len(result.rows) == 4
        # Check that German number format is preserved in raw data
        assert "1.299,99" in result.rows[1].values[1].val
        assert "29,50" in result.rows[2].values[1].val

    def test_malformed_csv(self):
        """Test handling of malformed CSV data."""
        malformed_csv = '''Name,Age,City
John,30,"Unclosed quote
Jane,25,Boston
Bob,35,Seattle'''

        temp_file = self.create_temp_csv(malformed_csv)
        # Should not raise an exception, but handle gracefully
        result = parse_csv(temp_file)
        assert isinstance(result, StructuredTable)

    def test_very_large_fields(self):
        """Test CSV with very large field content."""
        large_content = "x" * 1000  # Reduced size for faster testing
        csv_content = f'''Name,Description,Code
Product1,"{large_content}",ABC123
Product2,"Normal description",DEF456'''

        temp_file = self.create_temp_csv(csv_content)
        result = parse_csv(temp_file)

        assert len(result.rows) == 3
        assert len(result.rows[1].values[1].val) == 1000

    @pytest.mark.parametrize("sample_name", [
        "basic_comma", "basic_semicolon", "quoted_with_commas", "mixed_quotes",
        "empty_fields", "metadata_header", "complex_marketplace_like",
        "german_numbers", "pipe_separated", "tab_separated", "irregular_rows",
        "special_characters", "multiline_quotes", "double_quotes_escaped"
    ])
    def test_all_csv_samples_parseable(self, sample_name):
        """Test that all CSV samples can be parsed without errors."""
        csv_content = CSV_SAMPLES[sample_name]
        temp_file = self.create_temp_csv(csv_content)

        result = parse_csv(temp_file)

        # Basic assertions that should work for all samples
        assert isinstance(result, StructuredTable)
        assert result.source.source_type == DocumentType.TABULAR

        # Should have at least one row (even if just headers) for non-empty content
        if csv_content.strip():
            assert len(result.rows) > 0


class TestCSVDelimiterDetection:
    """Test the delimiter detection functionality."""

    @pytest.mark.parametrize("csv_content,expected_delimiter", [
        ("a,b,c\n1,2,3\n4,5,6", ","),
        ("a;b;c\n1;2;3\n4;5;6", ";"),
        ("a|b|c\n1|2|3\n4|5|6", "|"),
        ("a\tb\tc\n1\t2\t3\n4\t5\t6", "\t"),
    ])
    def test_csv_delimiter_simple(self, csv_content, expected_delimiter):
        """Test the delimiter detection function."""
        delimiter, share, total = csv_delimiter_simple(csv_content)
        assert delimiter == expected_delimiter
        assert share > 0.5

    def test_delimiter_with_quoted_content(self):
        """Test delimiter detection with quoted content containing delimiters."""
        quoted_csv = '"a,b","c,d","e,f"\n"1,2","3,4","5,6"'
        delimiter, share, total = csv_delimiter_simple(quoted_csv)
        # Should detect comma as delimiter despite commas in quotes
        assert delimiter == ","

    def test_delimiter_edge_cases(self):
        """Test delimiter detection edge cases."""
        # No delimiters
        no_delimiter = "abc\ndef\nghi"
        delimiter, share, total = csv_delimiter_simple(no_delimiter)
        assert total == 0

        # Equal delimiters
        equal_delimiters = "a,b;c,d;e,f"
        delimiter, share, total = csv_delimiter_simple(equal_delimiters)
        assert delimiter in [",", ";"]


class TestCSVParserEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Set up and clean up temporary files for each test."""
        self.temp_files = []
        yield
        # Clean up temporary files
        for temp_file in self.temp_files:
            try:
                os.unlink(temp_file)
            except FileNotFoundError:
                pass

    def create_temp_csv(self, content: str, encoding: str = 'utf-8') -> str:
        """Create a temporary CSV file with given content."""
        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', encoding=encoding)
        temp_file.write(content)
        temp_file.close()
        self.temp_files.append(temp_file.name)
        return temp_file.name

    def test_whitespace_only_file(self):
        """Test file with only whitespace."""
        temp_file = self.create_temp_csv("   \n  \n\t\n  ")
        result = parse_csv(temp_file)
        # Should handle gracefully
        assert isinstance(result, StructuredTable)

    def test_single_row_csv(self):
        """Test CSV with only header row."""
        csv_content = "Name,Age,City"
        temp_file = self.create_temp_csv(csv_content)
        result = parse_csv(temp_file)

        assert len(result.rows) == 1
        assert len(result.rows[0].values) == 3

    def test_inconsistent_quoting(self):
        """Test CSV with inconsistent quoting styles."""
        csv_content = '''Name,Age,"City"
"John",30,New York
Jane,"25","Boston"
"Bob",35,"Seattle"'''

        temp_file = self.create_temp_csv(csv_content)
        result = parse_csv(temp_file)

        assert len(result.rows) == 4
        assert result.rows[1].values[0].val == "John"
        assert result.rows[2].values[1].val == "25"

    def test_trailing_commas(self):
        """Test CSV with trailing commas."""
        csv_content = '''Name,Age,City,
John,30,New York,
Jane,25,Boston,'''

        temp_file = self.create_temp_csv(csv_content)
        result = parse_csv(temp_file)

        assert len(result.rows) == 3
        # Should handle trailing empty columns
        assert len(result.rows[0].values) == 4

    def test_special_characters_in_data(self):
        """Test CSV with various special characters."""
        csv_content = '''Name,Special,Notes
Test1,"Line1\nLine2","Tab\there"
Test2,"Quote""inside","Comma,here"
Test3,"Backslash\\test","Percent%here"'''

        temp_file = self.create_temp_csv(csv_content)
        result = parse_csv(temp_file)

        assert len(result.rows) == 4
        # Check that special characters are preserved
        assert "Line1" in result.rows[1].values[1].val
        assert "Quote" in result.rows[2].values[1].val


class TestCSVParserPerformance:
    """Performance and stress tests."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Set up and clean up temporary files for each test."""
        self.temp_files = []
        yield
        # Clean up temporary files
        for temp_file in self.temp_files:
            try:
                os.unlink(temp_file)
            except FileNotFoundError:
                pass

    def create_temp_csv(self, content: str, encoding: str = 'utf-8') -> str:
        """Create a temporary CSV file with given content."""
        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', encoding=encoding)
        temp_file.write(content)
        temp_file.close()
        self.temp_files.append(temp_file.name)
        return temp_file.name

    @pytest.mark.slow
    def test_large_csv_file(self):
        """Test parsing a large CSV file."""
        # Generate a large CSV with 1000 rows
        header = "ID,Name,Email,Department,Salary,Notes"
        rows = []
        for i in range(1000):
            row = f'{i},"User{i}","user{i}@test.com","Dept{i % 10}",{50000 + i},"Note for user {i}"'
            rows.append(row)

        csv_content = header + "\n" + "\n".join(rows)
        temp_file = self.create_temp_csv(csv_content)

        result = parse_csv(temp_file)

        assert len(result.rows) == 1001  # Header + 1000 data rows
        assert result.rows[1].values[0].val == "0"
        assert result.rows[1000].values[0].val == "999"

    def test_wide_csv_file(self):
        """Test CSV with many columns."""
        # Generate CSV with 100 columns
        header = ",".join([f"Col{i}" for i in range(100)])
        row1 = ",".join([f"Data1_{i}" for i in range(100)])
        row2 = ",".join([f"Data2_{i}" for i in range(100)])

        csv_content = f"{header}\n{row1}\n{row2}"
        temp_file = self.create_temp_csv(csv_content)

        result = parse_csv(temp_file)

        assert len(result.rows) == 3
        assert len(result.rows[0].values) == 100
        assert result.rows[1].values[0].val == "Data1_0"
        assert result.rows[1].values[99].val == "Data1_99"


class TestCSVParserIntegration:
    """Integration tests combining multiple features."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Set up and clean up temporary files for each test."""
        self.temp_files = []
        yield
        # Clean up temporary files
        for temp_file in self.temp_files:
            try:
                os.unlink(temp_file)
            except FileNotFoundError:
                pass

    def create_temp_csv(self, content: str, encoding: str = 'utf-8') -> str:
        """Create a temporary CSV file with given content."""
        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', encoding=encoding)
        temp_file.write(content)
        temp_file.close()
        self.temp_files.append(temp_file.name)
        return temp_file.name

    def test_complex_real_world_scenario(self):
        """Test a complex scenario combining multiple CSV features."""
        complex_csv = '''# Complex E-commerce Export
# Generated: 2025-01-07 10:30:00
# Encoding: UTF-8
# Separator: Semicolon
;;
"Order ID";"Customer Name";"Product Description";"Quantity";"Unit Price";"Total";"Notes"
"ORD-001";"John Smith";"Product A - ""Special Edition"" with extras";"2";"29,95";"59,90";"Rush order
Please handle with care"
"ORD-002";"José García";"Product B; Standard version";"1";"15,50";"15,50";""
"ORD-003";"李小明";"Multi-line product
description here";"3";"8,25";"24,75";"Customer VIP"'''

        temp_file = self.create_temp_csv(complex_csv)
        result = parse_csv(temp_file)

        # Should handle metadata, semicolon delimiter, quotes, multiline, special chars
        assert len(result.rows) >= 4

        # Find data rows (should contain order IDs)
        data_rows = [row for row in result.rows if len(row.values) >= 7 and row.values[0].val.startswith("ORD-")]
        assert len(data_rows) == 3

        # Check specific data integrity
        assert data_rows[0].values[1].val == "John Smith"
        assert data_rows[1].values[1].val == "José García"
        assert data_rows[2].values[1].val == "李小明"

        # Check that special characters and formatting are preserved
        assert "Special Edition" in data_rows[0].values[2].val
        assert "29,95" in data_rows[0].values[4].val

    def test_multilingual_financial_report(self):
        """Test multilingual financial data with various formatting."""
        financial_csv = '''Rapporte Financier / Financial Report / 财务报告;;
Date de génération / Generated / 生成日期: 2025-01-07;;
Devise / Currency / 货币: EUR;;
;;
"Date/日期";"ID Transaction";"Type/类型";"Montant/Amount/金额";"Frais/Fees/费用";"Net/净额";"Notes/备注"
"01.01.2025";"TXN-FR-001";"Vente/Sale/销售";"1.299,99";"-45,50";"1.254,49";"Client français"
"02.01.2025";"TXN-EN-002";"Refund/退款";"-250,00";"0,00";"-250,00";"English customer"
"03.01.2025";"TXN-CN-003";"Sale/销售";"899,95";"-31,50";"868,45";"中国客户"'''

        temp_file = self.create_temp_csv(financial_csv)
        result = parse_csv(temp_file)

        # Should handle multilingual headers and data
        assert len(result.rows) >= 4

        # Find data rows
        data_rows = [row for row in result.rows if len(row.values) >= 7 and row.values[0].val.startswith("0")]
        assert len(data_rows) == 3

        # Check multilingual data preservation
        assert "français" in data_rows[0].values[6].val
        assert "English" in data_rows[1].values[6].val
        assert "中国" in data_rows[2].values[6].val


if __name__ == "__main__":
    # Run tests when called directly
    pytest.main([__file__, "-v", "--tb=short"])