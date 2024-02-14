from typing import List, Dict

from parsee.extraction.extractor_elements import StandardDocumentFormat, ElementGroup, get_text_distance, FinalOutputTableColumn, FinalOutputTable
from parsee.extraction.extractor_dataclasses import ParseeMeta, ParseeLocation
from parsee.templates.job_template import JobTemplate
from parsee.utils.constants import *


def text_distance_table_groups(el1: ElementGroup, el2: ElementGroup, elements):
    el_idx1 = el1.base_el().source.element_index
    el_idx2 = el2.base_el().source.element_index
    indices_to_exclude = [x.source.element_index for x in el1.components] + [x.source.element_index for x in el2.components]

    return get_text_distance(el_idx1, el_idx2, elements, indices_to_exclude, True)


def assemble(job_template: JobTemplate, document: StandardDocumentFormat, locations: List[ParseeLocation]) -> List[ElementGroup]:

    detection_schema_by_class = {}
    for item in job_template.detection.items:
        detection_schema_by_class[item.id] = item
    
    # transform predictions slightly
    candidates_by_class: Dict[str, List[ElementGroup]] = {}
    for k, prediction in enumerate(locations):
        class_value = prediction.detected_class
        if class_value is not None:
            if class_value not in candidates_by_class:
                candidates_by_class[class_value] = []
            candidates_by_class[class_value].append(ElementGroup(class_value, prediction))

    predictions_by_element_index = {}
    for pred in locations:
        predictions_by_element_index[pred.source.element_index] = pred

    # decide which partial matches to merge
    for class_value, structured_elements in candidates_by_class.items():

        if len(structured_elements) > 1:
            # check if el can be merged with previous
            for k in range(len(structured_elements) - 1, 0, -1):

                loc = structured_elements[k].base_el()
                merge_candidate = structured_elements[k - 1].closest_el(structured_elements[k])

                # both have to have quite high probability
                if loc.prob < MERGE_MIN_CONFIDENCE or loc.prob < MERGE_MIN_CONFIDENCE:
                    continue

                # both partial prob have to be above threshold
                if loc.partial_prob < PARTIAL_MIN_CONFIDENCE or merge_candidate.partial_prob < PARTIAL_MIN_CONFIDENCE:
                    continue

                # tables need to be uninterrupted by other candidates
                min_index = min(loc.source.element_index, merge_candidate.source.element_index)
                max_index = max(loc.source.element_index, merge_candidate.source.element_index)
                found_inbetween = False
                for el_index in range(min_index + 1, max_index):
                    if el_index in predictions_by_element_index and predictions_by_element_index[el_index].prob > THRESHOLD_INBETWEEN_MERGE:
                        found_inbetween = True
                        break
                if found_inbetween:
                    continue

                # there can't be too much text between elements
                td = get_text_distance(loc.source.element_index, merge_candidate.source.element_index, document.elements, include_tables=True)

                if td > TEXT_DISTANCE_MERGE_THRESHOLD and loc.prob < 2:
                    continue

                # merge elements
                structured_elements[k - 1].merge_with(structured_elements[k])
                # delete from list
                structured_elements.pop(k)

    # decide which statements to keep if more than 1 close together
    for class_value, structured_elements in candidates_by_class.items():
        if class_value in detection_schema_by_class and detection_schema_by_class[class_value].takeBestInProximity:
            if len(structured_elements) > 1:
                # if some elements are close together, take one with highest score
                distances = []
                to_del = []
                for k in range(0, len(structured_elements) - 1):
                    for kk in range(k + 1, len(structured_elements)):
                        distance = text_distance_table_groups(structured_elements[k], structured_elements[kk], document.elements)
                        distances.append((distance, k, kk))
                        if distance <= TEXT_DISTANCE_CLOSE_STATEMENT_DETECTION:
                            # delete statement with lower probability
                            idx_to_delete = k if structured_elements[k].prob_combined() < structured_elements[kk].prob_combined() else kk
                            to_del.append(idx_to_delete)

                # make unique and sort
                to_del = sorted(list(set(to_del)), reverse=True)

                # delete
                for idx in to_del:
                    structured_elements.pop(idx)

    output: List[ElementGroup] = []
    for class_value, structured_elements in candidates_by_class.items():
        if class_value in detection_schema_by_class:
            output += structured_elements

    make_unique = {key: item for (key, item) in detection_schema_by_class.items() if item.takeBestInProximity}
    
    if len(make_unique.keys()) > 1:
        
        # make final selection based on distance of statements to each other
        # determine text distance of one statement to all others
        distances = []

        for k in range(0, len(output) - 1):
            for kk in range(k + 1, len(output)):
                distance = text_distance_table_groups(output[k], output[kk], document.elements)
                distances.append((distance, k, kk))

        # sort
        distances = list(sorted(distances, key=lambda x: x[0]))

        # combine one by one with closest distance
        final_groups = [{"dist": 0, "prob_score": 0, "indices": {key: None for key in make_unique.keys()}}]
        for dist_tuple in distances:
            # only combine if max distance is respected (or location was user defined -> prob == 2)
            if (dist_tuple[0] < MAX_DISTANCE_UNIQUE_PROXIMITY) or (output[dist_tuple[1]].prob_combined() == 2 and output[dist_tuple[2]].prob_combined() == 2):
                placed_item = False
                el1 = output[dist_tuple[1]]
                el2 = output[dist_tuple[2]]
                for final_group in final_groups:
                    # check if element can be added to group
                    if el1.detected_class != el2.detected_class and (final_group["indices"][el1.detected_class] is None or final_group["indices"][el1.detected_class] == dist_tuple[1]) and (
                            final_group["indices"][el2.detected_class] is None or final_group["indices"][el2.detected_class] == dist_tuple[2]):
                        final_group["indices"][el1.detected_class] = dist_tuple[1]
                        final_group["indices"][el2.detected_class] = dist_tuple[2]
                        final_group["dist"] += dist_tuple[0]
                        placed_item = True

                # create new group if item was not placed yet
                if not placed_item:
                    final_groups.append({"dist": 0, "prob_score": 0, "indices": {key: None for key in make_unique.keys()}})
                    final_groups[-1]["indices"][el1.detected_class] = dist_tuple[1]
                    final_groups[-1]["indices"][el2.detected_class] = dist_tuple[2]
                    final_groups[-1]["dist"] += dist_tuple[0]
            else:
                break

        # take the first group that has no None values
        groups_filtered = list(sorted([x for x in final_groups if None not in x["indices"].values()], key=lambda x: x['dist']))
        # compile matching score of final groups
        for g in groups_filtered:
            g['prob_score'] = 0
            for key in make_unique.keys():
                g['prob_score'] += output[g["indices"][key]].prob_combined()

        if len(groups_filtered) == 0:
            return []

        # take highest probabilities, then shortest distance
        groups_filtered = list(sorted(groups_filtered, key=lambda x: (-x['prob_score'], x['dist'])))
        group_chosen = groups_filtered[0]

        # delete values
        all_valid_indices = [group_chosen["indices"][key] for key in make_unique.keys()]
        for k in range(len(output) - 1, -1, -1):
            if k not in all_valid_indices:
                output.pop(k)
    
    return output


def get_structured_tables_from_locations(job_template: JobTemplate, document: StandardDocumentFormat, locations: List[ParseeLocation]) -> List[FinalOutputTableColumn]:

    element_groups = assemble(job_template, document, locations)

    # make final structured values
    output_values: List[FinalOutputTableColumn] = []
    for group in element_groups:
        output_values += group.structured_values(document.elements)

    return output_values


def final_tables_from_columns(columns: List[FinalOutputTableColumn]) -> List[FinalOutputTable]:

    by_identifier = {}

    for col in columns:
        table_id = (col.li_identifier, col.detected_class)
        if table_id not in by_identifier:
            by_identifier[table_id] = FinalOutputTable(col.detected_class, [col], col.li_identifier, [x[0] for x in col.key_value_pairs])
        else:
            by_identifier[table_id].columns.append(col)

    return list(by_identifier.values())
