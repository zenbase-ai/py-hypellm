import pytest

from hypellm import Example


@pytest.fixture(scope="session")
def medical_classification_dataset():
    return [
        Example(
            inputs="The patient presents with elevated troponin levels (0.8 ng/mL) and ST-segment depression, but no chest pain or dyspnea.",
            outputs="unstable_angina",
        ),
        Example(
            inputs="Labs show WBC 15k/μL with 80% neutrophils, fever 39.2°C, and consolidation in right lower lobe on chest X-ray.",
            outputs="bacterial_pneumonia",
        ),
        Example(
            inputs="Sudden onset vertigo with horizontal nystagmus, normal head CT, negative Dix-Hallpike, no hearing loss.",
            outputs="vestibular_neuritis",
        ),
        Example(
            inputs="Progressive weakness in lower extremities, decreased DTRs, EMG shows demyelination pattern, CSF protein elevated.",
            outputs="guillain_barre",
        ),
        Example(
            inputs="Recurrent episodes of focal seizures with preserved awareness, MRI shows temporal lobe calcification.",
            outputs="mesial_temporal_sclerosis",
        ),
    ]
