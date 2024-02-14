

def run_job_with_single_model(doc: StandardDocumentFormat, job_template: JobTemplate, model: MlModelSpecification, storage: Optional[StorageManager] = None) -> Tuple[List[ParseeBucket], List[FinalOutputTableColumn], List[ParseeAnswer]]:
    storage = InMemoryStorageManager([model]) if storage is None else storage
    # update the models
    job_template.set_default_model(model)
    return structure_data(doc, job_template, storage, {})
