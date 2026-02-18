import os
from dotenv import load_dotenv

load_dotenv()

class EncryptionConfig:
    GCP_PROJECT_ID = os.getenv('GCP_PROJECT_ID', 'souloxy-webapp')
    GCP_KMS_LOCATION = os.getenv('GCP_KMS_LOCATION', 'in')
    GCP_KMS_KEYRING = os.getenv('GCP_KMS_KEYRING', 'souloxy-mk-test')
    GCP_KMS_KEY = os.getenv('GCP_KMS_KEY', 'souloxy-mk-test')
    SERVICE_ACCOUNT_KEY = './config/serviceAccountKey.json'
    ENCRYPTION_ENABLED = os.getenv('ENCRYPTION_ENABLED', 'true').lower().strip() == 'true'

encryption_config = EncryptionConfig()