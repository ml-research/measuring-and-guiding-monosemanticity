import pandas as pd
import logging
import warnings
import spacy

pd.options.plotting.backend = "plotly"
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import (
    TransformersNlpEngine,
    NerModelConfiguration,
    NlpEngine,
    SpacyNlpEngine,
)
from presidio_analyzer.predefined_recognizers import (
    EmailRecognizer,
    IpRecognizer,
    UsLicenseRecognizer,
    UsSsnRecognizer,
    UsPassportRecognizer,
    PhoneRecognizer,
)

logging.getLogger("presidio-analyzer").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="presidio_analyzer")


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

spacy.load("en_core_web_lg")


class CustomEmailRecognizer(EmailRecognizer):
    def analyze(self, text, entities, nlp_artifacts):
        
        results = super().analyze(text, entities, nlp_artifacts)

        
        for result in results:
            if result.entity_type == "EMAIL_ADDRESS":
                result.entity_type = "EMAIL"

        return results


class CustomIPRecognizer(IpRecognizer):
    def analyze(self, text, entities, nlp_artifacts):
        
        results = super().analyze(text, entities, nlp_artifacts)

        
        for result in results:
            if result.entity_type == "IP_ADDRESS":
                result.entity_type = "IP"

        return results


class CustomDriverLicenceRecognizer(UsLicenseRecognizer):
    def analyze(self, text, entities, nlp_artifacts):
        
        results = super().analyze(text, entities, nlp_artifacts)

        
        for result in results:
            if result.entity_type == "US_DRIVER_LICENSE":
                result.entity_type = "DRIVERLICENSE"

        return results


class CustomPassportRecognizer(UsPassportRecognizer):
    def analyze(self, text, entities, nlp_artifacts):
        
        results = super().analyze(text, entities, nlp_artifacts)

        
        for result in results:
            if result.entity_type == "US_PASSPORT":
                result.entity_type = "PASSPORT"

        return results


class CustomSSNRecognizer(UsSsnRecognizer):
    def analyze(self, text, entities, nlp_artifacts):
        
        results = super().analyze(text, entities, nlp_artifacts)

        
        for result in results:
            if result.entity_type == "US_SSN":
                result.entity_type = "SOCIALNUMBER"

        return results


class CustomPhoneRecognizer(PhoneRecognizer):
    def analyze(self, text, entities, nlp_artifacts):
        
        results = super().analyze(text, entities, nlp_artifacts)

        
        for result in results:
            if result.entity_type == "PHONE_NUMBER":
                result.entity_type = "TEL"

        return results


# Define which transformers model to use
model_config = [
    {
        "lang_code": "en",
        "model_name": {
            "spacy": "en_core_web_lg",  # use a large spaCy model for lemmas, tokens etc.
            "transformers": "yonigo/distilbert-base-cased-pii-en",
        },
    }
]

# Standardized entity mapping between the model's and Presidio's
mapping = {
    "TIME": "TIME",
    "USERNAME": "USERNAME",
    "EMAIL": "EMAIL",  # Ensure emails are mapped consistently
    "EMAIL_ADDRESS": "EMAIL",  # Handle variations in casing/naming
    "Email_address": "EMAIL",  # Standardizing model output
    "IDCARD": "IDCARD",
    "SOCIALNUMBER": "SOCIALNUMBER",
    "LASTNAME": "LASTNAME",
    "LASTNAME1": "LASTNAME",
    "LASTNAME2": "LASTNAME",
    "LASTNAME3": "LASTNAME",
    "FULLNAME": "NAME",
    "PASSPORT": "PASSPORT",
    "DRIVERLICENSE": "DRIVERLICENSE",
    "BOD": "BOD",
    "IP": "IP",
    "GIVENNAME": "GIVENNAME",
    "GIVENNAME1": "GIVENNAME",
    "GIVENNAME2": "GIVENNAME",
    "CITY": "CITY",
    "STATE": "STATE",
    "TITLE": "TITLE",
    "SEX": "SEX",
    "POSTCODE": "POSTCODE",
    "BUILDING": "BUILDING",
    "STREET": "STREET",
    "TEL": "TEL",
    "DATE": "DATE",
    "COUNTRY": "COUNTRY",
    "PASS": "PASS",
    "SECADDRESS": "SECADDRESS",
    "GEOCOORD": "GEOCOORD",
    "CARDISSUER": "CARDISSUER",
}

# Labels to ignore (ensures unwanted entities are skipped)
labels_to_ignore = [
    "O",  # 'O' represents non-entity tokens in NER models
    "IN_PAN",
    "MASKEDNUMBER",
    "USERAGENT",
    "URL",
]

# Define NerModelConfiguration with corrected mappings
ner_model_configuration = NerModelConfiguration(
    model_to_presidio_entity_mapping=mapping,  # Use standardized mappings
    alignment_mode="expand",  # "strict", "contract", "expand"
    aggregation_strategy="max",  # "simple", "first", "average", "max"
    labels_to_ignore=labels_to_ignore,  # Ignore unwanted entities
)

# Initialize the Transformer-based NLP engine
transformers_nlp_engine = TransformersNlpEngine(
    models=model_config, ner_model_configuration=ner_model_configuration
)

# Transformer-based analyzer
analyzer = AnalyzerEngine(
    nlp_engine=transformers_nlp_engine, supported_languages=["en"]
)

# 🔥 Replace the default EmailRecognizer with the CustomEmailRecognizer
analyzer.registry.remove_recognizer("UrlRecognizer")
analyzer.registry.remove_recognizer("EmailRecognizer")
analyzer.registry.remove_recognizer("IpRecognizer")

analyzer.registry.remove_recognizer("InVehicleRegistrationRecognizer")
# analyzer.registry.remove_recognizer("IbanRecognizer")
# analyzer.registry.remove_recognizer("CryptoRecognizer")
analyzer.registry.remove_recognizer("CreditCardRecognizer")
# analyzer.registry.remove_recognizer("DateRecognizer")
analyzer.registry.remove_recognizer("AuAcnRecognizer")
analyzer.registry.remove_recognizer("MedicalLicenseRecognizer")
analyzer.registry.remove_recognizer("AuMedicareRecognizer")
analyzer.registry.remove_recognizer("UsPassportRecognizer")
analyzer.registry.remove_recognizer("AuAbnRecognizer")
analyzer.registry.remove_recognizer("InAadhaarRecognizer")
analyzer.registry.remove_recognizer("SgFinRecognizer")
analyzer.registry.remove_recognizer("InPanRecognizer")
analyzer.registry.remove_recognizer("AuTfnRecognizer")
analyzer.registry.remove_recognizer("UkNinoRecognizer")
# analyzer.registry.remove_recognizer("InPassportRecognizer")
analyzer.registry.remove_recognizer("UsBankRecognizer")
analyzer.registry.remove_recognizer("PhoneRecognizer")
analyzer.registry.remove_recognizer("InVoterRecognizer")
analyzer.registry.remove_recognizer("UsLicenseRecognizer")
analyzer.registry.remove_recognizer("UsSsnRecognizer")
# analyzer.registry.remove_recognizer("TransformersRecognizer")
analyzer.registry.remove_recognizer("UsItinRecognizer")
analyzer.registry.remove_recognizer("NhsRecognizer")
analyzer.registry.add_recognizer(CustomEmailRecognizer())  # Add the custom version
analyzer.registry.add_recognizer(CustomIPRecognizer())  # Add the custom version
analyzer.registry.add_recognizer(CustomSSNRecognizer())  # Add the custom version
analyzer.registry.add_recognizer(CustomPassportRecognizer())  # Add the custom version
analyzer.registry.add_recognizer(CustomPhoneRecognizer())  # Add the custom version
analyzer.registry.add_recognizer(
    CustomDriverLicenceRecognizer()
)  # Add the custom version


if __name__ == "__main__":
    # Test with a sample text
    text = (
        "My email is john.doe@example.com and another Email_address is test@mail.com."
    )

    results = analyzer.analyze(text=text, language="en")

    print(results)  
