from copy import deepcopy
from decimal import Decimal
from parsee.extraction.extractor_elements import FinalOutputTable, FinalOutputTableColumn, StructuredTable, StructuredRow, ParseeLocation, ExtractedSource, DocumentType, StructuredTableCell, MappingSchema
from parsee.templates.mappings import MappingBucket, AggregationMethod


def FreezeProperty(value):
    cache = deepcopy(value)
    return property(
        lambda self: deepcopy(cache)
    )


def FrozenSpace(**args):
    args = {k: FreezeProperty(v) for k, v in args.items()}
    args['__slots__'] = ()
    cls = type('FrozenSpace', (), args)
    return cls()


__source_var = ExtractedSource(DocumentType.PDF, None, None, 1, None)


samples = FrozenSpace(
    source = __source_var,
    table = FinalOutputTable("test", [
        FinalOutputTableColumn(ParseeLocation("test", 1.0, "test", 1.0, __source_var, []),
                               StructuredTable(__source_var, [
                                   StructuredRow("body",[
                                       StructuredTableCell("Some item"), StructuredTableCell("31234")
                                   ]),
                                   StructuredRow("body", [
                                        StructuredTableCell("Another item"), StructuredTableCell("591")
                                   ]),
                                   StructuredRow("body", [
                                       StructuredTableCell("Third line item"), StructuredTableCell("492")
                                   ]),
                               ]), 0, 0, 0
                               )
    ], "abc"),
    schema = MappingSchema("test", "Some test mapping", "Just for testing", [
        MappingBucket("id1", None, "Some bucket", None, 0, Decimal(1.0), {}, AggregationMethod.SUM),
        MappingBucket("id2", None, "Some other bucket", None, 0, Decimal(1.0), {}, AggregationMethod.SUM),
        MappingBucket("id3", None, "Third bucket", None, 0, Decimal(1.0), {}, AggregationMethod.SUM),
    ])
)