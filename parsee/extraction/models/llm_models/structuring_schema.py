from typing import List, Optional, Union, Tuple
import re
from decimal import Decimal

from parsee.templates.general_structuring_schema import StructuringItemSchema
from parsee.utils.enums import OutputType
from parsee.utils.helper import clean_numeric_value_llm, get_entity_value, get_date_regex, clean_numeric_value


class PromptSchemaItem:

    def __init__(self, possible_values: List[str], example: Optional[str] = None,
                 default_value: Optional[str] = None):
        self.possible_values = possible_values
        self.example = example
        self.default_value = default_value if default_value is not None and default_value.strip() != "" else None

    def is_valid_input(self, val: Union[str, None]) -> bool:
        return val is not None and val.strip() != ""

    def get_example(self, clean_value: bool = False) -> str:
        raise NotImplemented

    def get_possible_values_str(self) -> str:
        raise NotImplemented

    def get_default_value(self) -> Union[str, None]:
        return self.default_value if self.default_value is not None else None

    def get_value(self, value: str) -> Tuple[Union[str, None], bool]:
        return value, True

    def parsed_to_raw(self, value: str) -> str:
        return value


class ListClassificationItem(PromptSchemaItem):

    def __init__(self, possible_values: List[str], example: Optional[str] = None,
                 default_value: Optional[str] = None):
        super().__init__(possible_values, example, default_value)

        self.possible_lowercase = [x.lower() for x in possible_values]

    def format_list_choice(self, val):
        return f"$ {val} $"

    def get_example(self, clean_value: bool = False) -> str:
        return self.format_list_choice(self.possible_values[0]) if not clean_value else self.possible_values[0]

    def get_possible_values_str(self) -> str:
        return "possible values (separated by $ symbols): " + ", ".join(
            self.format_list_choice(x) for x in self.possible_values)

    def get_value(self, value: str) -> Tuple[Union[str, None], bool]:

        value = value.lower()
        search = re.search(r'(\$( |)(.+)( |)\$)', value)
        if search is not None and len(search.groups()) == 4:
            value = search.groups()[2].strip()
        if value not in self.possible_lowercase:
            # try simple string matching
            for val in self.possible_lowercase:
                search = re.search(rf'\b{val}\b', value)
                if search is not None:
                    value = val
                    break
            if value not in self.possible_lowercase:
                return self.get_default_value(), False
        idx = self.possible_lowercase.index(value)
        return self.possible_values[idx], True

    def parsed_to_raw(self, value: str) -> str:
        return self.format_list_choice(value)


class PositiveIntegerItem(PromptSchemaItem):

    def __init__(self, example: Optional[str] = None, default_value: Optional[str] = None):
        super().__init__([], example, default_value)

    def get_example(self, clean_value: bool = False) -> str:
        return self.example if self.is_valid_input(self.example) else "123"

    def get_default_value(self) -> Union[str, None]:
        return self.default_value if self.default_value is not None else None

    def get_possible_values_str(self) -> str:
        return "possible values: positive integer"

    def get_value(self, value: str) -> Tuple[Union[str, None], bool]:
        val = clean_numeric_value_llm(value)
        if val is None:
            return self.get_default_value(), False
        return str(abs(int(val))), True


class TextItem(PromptSchemaItem):

    def __init__(self, example: Optional[str] = None, default_value: Optional[str] = None):
        super().__init__([], example, default_value)

    def get_example(self, clean_value: bool = False) -> str:
        return self.example if self.is_valid_input(self.example) else "answer to question"

    def get_default_value(self) -> Union[str, None]:
        return self.default_value if self.default_value is not None else None

    def get_possible_values_str(self) -> str:
        return "possible values: any text"


class NumericItem(PromptSchemaItem):

    def __init__(self, example: Optional[str] = None, default_value: Optional[str] = None):
        super().__init__([], example, default_value)

    def get_example(self, clean_value: bool = False) -> str:
        return self.example if self.is_valid_input(self.example) else "123"

    def get_default_value(self) -> Union[str, None]:
        return self.default_value if self.default_value is not None else None

    def get_possible_values_str(self) -> str:
        return "possible values: any number (positive or negative, including decimals)"

    def get_value(self, value: str) -> Tuple[Union[str, None], bool]:
        val = clean_numeric_value_llm(value)
        if val is None:
            return self.get_default_value(), False
        return str(val), True


class PercentageItem(PromptSchemaItem):

    def __init__(self, example: Optional[str] = None, default_value: Optional[str] = None):
        super().__init__([], example, default_value)

    def get_example(self, clean_value: bool = False) -> str:
        example = clean_numeric_value(self.example) if self.is_valid_input(self.example) else Decimal(0.15)
        if example < 1:
            example = example*100
        return str(round(float(example)))+"%"

    def get_default_value(self) -> Union[str, None]:
        return self.default_value if self.default_value is not None else None

    def get_possible_values_str(self) -> str:
        return "possible values: any percentage value"

    def get_value(self, value: str) -> Tuple[Union[str, None], bool]:
        val = clean_numeric_value_llm(value)
        # check if value has to be multiplied
        mult = 1 if "%" in value or val > 1 else 100
        val = val * mult
        if val is None:
            return self.get_default_value(), False
        if val.is_integer():
            val = str(int(val))
        else:
            val = str(val)
        return val+"%", True


class EntityItem(PromptSchemaItem):

    def __init__(self, example: Optional[str] = None, default_value: Optional[str] = None):
        super().__init__([], example, default_value)

    def get_example(self, clean_value: bool = False) -> str:
        return self.example if self.is_valid_input(self.example) else "answer to question"

    def get_default_value(self) -> Union[str, None]:
        return self.default_value if self.default_value is not None else None

    def get_possible_values_str(self) -> str:
        return "possible values: any text"

    def get_value(self, value: str) -> Tuple[Union[str, None], bool]:
        val = get_entity_value(value)
        if val is None:
            return value.strip().rstrip(".").strip(), True
        return val.strip().rstrip(".").strip(), True


class DateItem(PromptSchemaItem):

    def __init__(self, example: Optional[str] = None, default_value: Optional[str] = None):
        super().__init__([], example, default_value)

    def get_example(self, clean_value: bool = False) -> str:
        return self.example if self.is_valid_input(self.example) else "2023-11-24"

    def get_default_value(self) -> Union[str, None]:
        return self.default_value if self.default_value is not None else None

    def get_possible_values_str(self) -> str:
        return "possible values: date in the format: YYYY-MM-DD"

    def get_value(self, value: str) -> Tuple[Union[str, None], bool]:
        val = get_date_regex(value)
        if val is None:
            return self.get_default_value(), False
        return str(val), True


def get_prompt_schema_item(item: StructuringItemSchema) -> PromptSchemaItem:
    if item.type == OutputType.LIST:
        return ListClassificationItem(item.valuesList, item.example, item.defaultValue)
    elif item.type == OutputType.INTEGER:
        return PositiveIntegerItem(item.example, item.defaultValue)
    elif item.type == OutputType.NUMERIC:
        return NumericItem(item.example, item.defaultValue)
    elif item.type == OutputType.TEXT:
        return TextItem(item.example, item.defaultValue)
    elif item.type == OutputType.ENTITY:
        return EntityItem(item.example, item.defaultValue)
    elif item.type == OutputType.DATE:
        return DateItem(item.example, item.defaultValue)
    elif item.type == OutputType.PERCENTAGE:
        return PercentageItem(item.example, item.defaultValue)
    raise Exception("item not found")
